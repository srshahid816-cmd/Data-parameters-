import streamlit as st
import requests
import pandas as pd
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("📊 Crypto Futures Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ==================== MEXC API (US accessible) ====================
MEXC_BASE = "https://contract.mexc.com"

@st.cache_data(ttl=300)
def fetch_mexc_tickers():
    """MEXC USDT perpetuals ticker data"""
    try:
        resp = requests.get(f"{MEXC_BASE}/api/v1/contract/ticker", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            df = pd.DataFrame(data["data"])
            df = df[df['symbol'].str.endswith('_USDT')]
            df['riseFallRate'] = pd.to_numeric(df['riseFallRate'], errors='coerce') * 100
            df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
            df['volume24'] = pd.to_numeric(df['volume24'], errors='coerce')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"MEXC ticker error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_mexc_klines(symbol, interval="Min60", limit=100):
    """MEXC kline data"""
    try:
        url = f"{MEXC_BASE}/api/v1/contract/kline/{symbol}"
        resp = requests.get(url, params={"interval": interval}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") and data.get("data"):
            k = data["data"]
            df = pd.DataFrame({
                'time': pd.to_datetime(k['time'], unit='s'),
                'open': k['open'],
                'high': k['high'],
                'low': k['low'],
                'close': k['close'],
                'volume': k['vol']
            })
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df.tail(limit)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"MEXC kline error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_mexc_funding(symbol="BTC_USDT"):
    """MEXC funding rate"""
    try:
        resp = requests.get(f"{MEXC_BASE}/api/v1/contract/funding_rate/{symbol}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return float(data["data"]["fundingRate"]) * 100
        return None
    except:
        return None

def calculate_rsi(df, period=14):
    try:
        rsi = ta.rsi(df['close'], length=period)
        return rsi.iloc[-1] if len(rsi) > 0 else None
    except:
        return None

# ==================== MAIN ====================
tickers_df = fetch_mexc_tickers()

if tickers_df.empty:
    st.error("Data fetch nahi ho saka. Thodi der baad refresh karein.")
    st.stop()

sorted_df = tickers_df.sort_values('riseFallRate', ascending=False)

# Top Gainers/Losers
st.header("📈 Top 10 Gainers & Losers (24h)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Top Gainers")
    g = sorted_df.head(10)[['symbol', 'lastPrice', 'riseFallRate', 'volume24']].copy()
    g.columns = ['Symbol', 'Price', '24h %', 'Volume']
    g['24h %'] = g['24h %'].apply(lambda x: f"+{x:.2f}%")
    st.dataframe(g, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🔴 Top Losers")
    l = sorted_df.tail(10).sort_values('riseFallRate')[['symbol', 'lastPrice', 'riseFallRate', 'volume24']].copy()
    l.columns = ['Symbol', 'Price', '24h %', 'Volume']
    l['24h %'] = l['24h %'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(l, use_container_width=True, hide_index=True)

# BTC/ETH Section
st.header("₿ BTC & ETH")
btc = tickers_df[tickers_df['symbol'] == 'BTC_USDT']
eth = tickers_df[tickers_df['symbol'] == 'ETH_USDT']
if not btc.empty:
    b = btc.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC Price", f"${b['lastPrice']:,.2f}", f"{b['riseFallRate']:.2f}%")
    c2.metric("BTC Volume", f"${b['volume24']:,.0f}")
    if not eth.empty:
        e = eth.iloc[0]
        c3.metric("ETH Price", f"${e['lastPrice']:,.2f}", f"{e['riseFallRate']:.2f}%")
        c4.metric("ETH Volume", f"${e['volume24']:,.0f}")

# RSI + Funding
st.header("📊 Indicators (BTC_USDT)")
klines = fetch_mexc_klines("BTC_USDT", "Min60", 100)
if not klines.empty:
    rsi = calculate_rsi(klines)
    c1, c2 = st.columns(2)
    with c1:
        if rsi:
            icon = "🔴" if rsi > 70 else ("🟢" if rsi < 30 else "🟡")
            st.metric(f"{icon} RSI (1h)", f"{rsi:.1f}")
    with c2:
        fr = fetch_mexc_funding("BTC_USDT")
        if fr is not None:
            st.metric("Funding Rate", f"{fr:.4f}%")

# Chart
st.header("📈 BTC/USDT (1h)")
if not klines.empty:
    fig = go.Figure(data=[go.Candlestick(
        x=klines['time'], open=klines['open'], high=klines['high'],
        low=klines['low'], close=klines['close']
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
- ✅ Top 10 Gainers/Losers (MEXC)
- ✅ BTC & ETH Metrics
- ✅ RSI (1h)
- ✅ Funding Rate
- ✅ Price Chart
""")
st.sidebar.header("🚫 Not Yet Added")
st.sidebar.markdown("""
- ❌ GEX / Gamma Levels
- ❌ Volume Profile / Delta
- ❌ Spot/Futures Basis
""")
st.sidebar.button("🔄 Refresh", on_click=lambda: st.cache_data.clear())
