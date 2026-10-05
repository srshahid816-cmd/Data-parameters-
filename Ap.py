import streamlit as st
import requests
import pandas as pd
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="Crypto Futures Dashboard", layout="wide")

# ==================== CONFIG ====================
BINANCE_FAPI = "https://fapi.binance.com"

st.title("📊 Crypto Futures Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== DATA FETCH FUNCTIONS ====================
@st.cache_data(ttl=300)  # 5 minute cache
def fetch_all_tickers():
    """Fetch all Binance Futures tickers"""
    try:
        resp = requests.get(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        df['priceChangePercent'] = pd.to_numeric(df['priceChangePercent'])
        df['lastPrice'] = pd.to_numeric(df['lastPrice'])
        df['volume'] = pd.to_numeric(df['volume'])
        # Filter USDT pairs only
        df = df[df['symbol'].str.endswith('USDT')]
        return df
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_klines(symbol, interval="1h", limit=100):
    """Fetch kline data for a symbol"""
    try:
        resp = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        return df
    except Exception as e:
        st.error(f"Error fetching klines for {symbol}: {e}")
        return pd.DataFrame()

def calculate_rsi(df, period=14):
    """Calculate RSI using pandas-ta-classic"""
    try:
        rsi = ta.rsi(df['close'], length=period)
        return rsi.iloc[-1] if len(rsi) > 0 else None
    except:
        return None

# ==================== MAIN LAYOUT ====================
tickers_df = fetch_all_tickers()

if tickers_df.empty:
    st.error("Could not fetch market data. Please check your internet connection.")
    st.stop()

# Sort for gainers/losers
sorted_df = tickers_df.sort_values('priceChangePercent', ascending=False)

# ==================== SECTION 1: TOP GAINERS / LOSERS ====================
st.header("📈 Top 10 Gainers & Losers (24h)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Top 10 Gainers")
    gainers = sorted_df.head(10)[['symbol', 'lastPrice', 'priceChangePercent', 'volume']].copy()
    gainers.columns = ['Symbol', 'Price', '24h %', 'Volume']
    gainers['24h %'] = gainers['24h %'].apply(lambda x: f"+{x:.2f}%")
    st.dataframe(gainers, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🔴 Top 10 Losers")
    losers = sorted_df.tail(10).sort_values('priceChangePercent')[['symbol', 'lastPrice', 'priceChangePercent', 'volume']].copy()
    losers.columns = ['Symbol', 'Price', '24h %', 'Volume']
    losers['24h %'] = losers['24h %'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(losers, use_container_width=True, hide_index=True)

# ==================== SECTION 2: BTC & MAJOR PAIRS ====================
st.header("₿ BTC & Major Pairs")

btc_row = tickers_df[tickers_df['symbol'] == 'BTCUSDT']
eth_row = tickers_df[tickers_df['symbol'] == 'ETHUSDT']

if not btc_row.empty:
    btc = btc_row.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC/USDT", f"${btc['lastPrice']:,.2f}", f"{btc['priceChangePercent']:.2f}%")
    c2.metric("24h Volume", f"${float(btc['quoteVolume']):,.0f}")
    
    if not eth_row.empty:
        eth = eth_row.iloc[0]
        c3.metric("ETH/USDT", f"${eth['lastPrice']:,.2f}", f"{eth['priceChangePercent']:.2f}%")
        c4.metric("ETH Volume", f"${float(eth['quoteVolume']):,.0f}")

# ==================== SECTION 3: RSI & OPEN INTEREST (BTC Example) ====================
st.header("📊 Technical Indicators (BTC/USDT)")

klines_1h = fetch_klines("BTCUSDT", "1h", 100)

if not klines_1h.empty:
    rsi_val = calculate_rsi(klines_1h)
    
    c1, c2 = st.columns(2)
    with c1:
        if rsi_val:
            rsi_color = "🔴" if rsi_val > 70 else ("🟢" if rsi_val < 30 else "🟡")
            st.metric(f"{rsi_color} RSI (1h, 14)", f"{rsi_val:.1f}")
            if rsi_val > 70:
                st.warning("RSI > 70: Overbought zone — potential pullback")
            elif rsi_val < 30:
                st.success("RSI < 30: Oversold zone — potential bounce")
            else:
                st.info("RSI Neutral zone")
    
    with c2:
        try:
            oi_resp = requests.get(
                f"{BINANCE_FAPI}/fapi/v1/openInterest",
                params={"symbol": "BTCUSDT"},
                timeout=10
            )
            oi_data = oi_resp.json()
            st.metric("Open Interest (BTC)", f"{float(oi_data['openInterest']):,.0f}")
        except:
            st.metric("Open Interest", "N/A")

# ==================== SECTION 4: PRICE CHART ====================
st.header("📈 BTC/USDT Price Chart (1h)")

if not klines_1h.empty:
    fig = go.Figure(data=[
        go.Candlestick(
            x=pd.to_datetime(klines_1h['open_time'], unit='ms'),
            open=klines_1h['open'],
            high=klines_1h['high'],
            low=klines_1h['low'],
            close=klines_1h['close']
        )
    ])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=400,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== AUTO REFRESH ====================
st.sidebar.header("⚙️ Controls")
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("Data auto-refreshes every 5 minutes (cache TTL)")

# ==================== DISCLAIMER ====================
st.sidebar.header("⚠️ Disclaimer")
st.sidebar.caption(
    "This dashboard is for educational purposes only. "
    "Not financial advice. Trade at your own risk."
)

# ==================== ACTIVE PARAMETERS ====================
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
- ✅ Top 10 Gainers/Losers (24h)
- ✅ BTC & ETH Metrics
- ✅ RSI (1h)
- ✅ Open Interest
- ✅ Price Chart
""")

st.sidebar.header("🚫 Not Yet Added")
st.sidebar.markdown("""
- ❌ GEX / Gamma Levels
- ❌ Volume Profile / Delta
- ❌ Absorption
- ❌ Spot/Futures Basis
- ❌ Token Supply
""")
