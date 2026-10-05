import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Pro Crypto Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Professional Crypto Futures Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: Binance + OKX + Bitget")

# ==================== EXCHANGE ENDPOINTS ====================
BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_SPOT = "https://api.binance.com"
OKX_BASE = "https://www.okx.com"
BITGET_BASE = "https://api.bitget.com"
MEXC_BASE = "https://contract.mexc.com"
COINGECKO = "https://api.coingecko.com/api/v3"

# ==================== BINANCE DATA ====================
@st.cache_data(ttl=300)
def fetch_binance_tickers():
    """Binance USDT perpetuals 24h ticker"""
    try:
        resp = requests.get(f"{BINANCE_FAPI}/fapi/v1/ticker/24hr", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        df = df[df['symbol'].str.endswith('USDT')]
        for col in ['priceChangePercent', 'lastPrice', 'quoteVolume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_binance_klines(symbol, interval="1h", limit=200):
    """Binance kline data"""
    try:
        resp = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_binance_oi(symbol="BTCUSDT"):
    """Binance Open Interest + History"""
    try:
        # Current OI
        resp = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=15
        )
        resp.raise_for_status()
        current_oi = float(resp.json()['openInterest'])

        # OI History (30 days)
        resp2 = requests.get(
            f"{BINANCE_FAPI}/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "1h", "limit": 168},
            timeout=15
        )
        resp2.raise_for_status()
        hist = pd.DataFrame(resp2.json())
        hist['sumOpenInterest'] = pd.to_numeric(hist['sumOpenInterest'])
        hist['timestamp'] = pd.to_datetime(hist['timestamp'], unit='ms')
        return current_oi, hist
    except Exception as e:
        return None, pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_binance_funding(symbol="BTCUSDT"):
    """Binance Funding Rate"""
    try:
        resp = requests.get(
            f"{BINANCE_FAPI}/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 8},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        df['fundingRate'] = pd.to_numeric(df['fundingRate']) * 100
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_binance_long_short(symbol="BTCUSDT"):
    """Binance Long/Short Ratios (Global + Top Accounts + Top Positions)"""
    results = {}
    endpoints = {
        'global': 'globalLongShortAccountRatio',
        'top_account': 'topLongShortAccountRatio',
        'top_position': 'topLongShortPositionRatio',
        'taker': 'takerlongshortRatio'
    }
    for key, endpoint in endpoints.items():
        try:
            resp = requests.get(
                f"{BINANCE_FAPI}/futures/data/{endpoint}",
                params={"symbol": symbol, "period": "1h", "limit": 24},
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            if data:
                df = pd.DataFrame(data)
                for col in ['longShortRatio', 'longAccount', 'shortAccount', 'buySellRatio', 'buyVol', 'sellVol']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                results[key] = df
        except:
            results[key] = pd.DataFrame()
    return results

# ==================== OKX DATA ====================
@st.cache_data(ttl=300)
def fetch_okx_taker_volume(symbol="BTC-USDT-SWAP"):
    """OKX Taker Buy/Sell Volume (CVD)"""
    try:
        resp = requests.get(
            f"{OKX_BASE}/api/v5/rubik/stat/taker-volume",
            params={"ccy": symbol.replace("-USDT-SWAP", ""), "instType": "CONTRACTS", "period": "1H"},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') == '0' and data.get('data'):
            df = pd.DataFrame(data['data'], columns=['timestamp', 'sellVol', 'buyVol'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df['sellVol'] = pd.to_numeric(df['sellVol'])
            df['buyVol'] = pd.to_numeric(df['buyVol'])
            df['CVD'] = df['buyVol'] - df['sellVol']
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# ==================== BITGET DATA ====================
@st.cache_data(ttl=300)
def fetch_bitget_taker(symbol="BTCUSDT"):
    """Bitget Taker Buy/Sell Volume"""
    try:
        resp = requests.get(
            f"{BITGET_BASE}/api/v2/mix/market/taker-buy-sell",
            params={"symbol": symbol, "productType": "usdt-futures", "period": "1H"},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') == '00000' and data.get('data'):
            df = pd.DataFrame(data['data'])
            df['buyVolume'] = pd.to_numeric(df['buyVolume'])
            df['sellVolume'] = pd.to_numeric(df['sellVolume'])
            df['CVD'] = df['buyVolume'] - df['sellVolume']
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# ==================== INDICATORS ====================
def calculate_rsi(df, period=14):
    try:
        rsi = ta.rsi(df['close'], length=period)
        return rsi.iloc[-1] if len(rsi) > 0 else None
    except:
        return None

def calculate_macd(df):
    try:
        macd = ta.macd(df['close'])
        if macd is not None and len(macd) > 0:
            return {
                'macd': macd.iloc[-1, 0],
                'signal': macd.iloc[-1, 1],
                'histogram': macd.iloc[-1, 2]
            }
        return None
    except:
        return None

def calculate_ema(df, period=20):
    try:
        ema = ta.ema(df['close'], length=period)
        return ema.iloc[-1] if len(ema) > 0 else None
    except:
        return None

def calculate_atr(df, period=14):
    try:
        atr = ta.atr(df['high'], df['low'], df['close'], length=period)
        return atr.iloc[-1] if len(atr) > 0 else None
    except:
        return None

def calculate_volume_profile(df, bins=50):
    """Simple Volume Profile: POC, VAH, VAL"""
    try:
        if df.empty or len(df) < 10:
            return None
        price_min = df['low'].min()
        price_max = df['high'].max()
        if price_min == price_max:
            return None
        bins_edges = np.linspace(price_min, price_max, bins + 1)
        df['price_bin'] = pd.cut(df['close'], bins=bins_edges, labels=False)
        volume_profile = df.groupby('price_bin')['volume'].sum()
        poc_bin = volume_profile.idxmax()
        poc = (bins_edges[poc_bin] + bins_edges[poc_bin + 1]) / 2

        # Value Area (70%)
        total_vol = volume_profile.sum()
        sorted_vol = volume_profile.sort_values(ascending=False)
        cumsum = 0
        va_bins = []
        for bin_idx, vol in sorted_vol.items():
            cumsum += vol
            va_bins.append(bin_idx)
            if cumsum >= total_vol * 0.7:
                break
        vah = bins_edges[max(va_bins) + 1]
        val = bins_edges[min(va_bins)]
        return {'POC': poc, 'VAH': vah, 'VAL': val}
    except:
        return None

def calculate_absorption(df):
    """Approximate Absorption: high volume with small price range"""
    try:
        if df.empty or len(df) < 20:
            return None
        df['range'] = df['high'] - df['low']
        df['vol_per_range'] = df['volume'] / (df['range'] + 1e-10)
        recent = df.tail(10)
        avg_vol = recent['volume'].mean()
        avg_range = recent['range'].mean()
        # Absorption: volume high but range small
        absorption_ratio = (recent['volume'] / avg_vol) / (recent['range'] / avg_range + 1e-10)
        return absorption_ratio.mean()
    except:
        return None

# ==================== SIGNAL ENGINE ====================
def generate_signal(symbol, price, rsi, macd, funding, oi_change, ls_ratio, cvd, vp, absorption):
    """
    Multi-parameter Signal Engine
    Returns: signal (BUY/SELL/HOLD/WAIT), confidence, entry, sl, tp1, tp2, tp3
    """
    score = 0
    reasons = []

    # 1. RSI
    if rsi:
        if rsi < 30:
            score += 2
            reasons.append("RSI Oversold")
        elif rsi > 70:
            score -= 2
            reasons.append("RSI Overbought")

    # 2. MACD
    if macd:
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            score += 1
            reasons.append("MACD Bullish")
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            score -= 1
            reasons.append("MACD Bearish")

    # 3. Funding Rate
    if funding is not None:
        if funding > 0.05:  # High positive funding = overleveraged longs
            score -= 1
            reasons.append("High Funding (Longs crowded)")
        elif funding < -0.05:
            score += 1
            reasons.append("Negative Funding (Shorts crowded)")

    # 4. Long/Short Ratio
    if ls_ratio is not None:
        if ls_ratio > 2.0:  # Too many longs = contrarian bearish
            score -= 1
            reasons.append("L/S Ratio Extreme Long")
        elif ls_ratio < 0.5:
            score += 1
            reasons.append("L/S Ratio Extreme Short")

    # 5. CVD (Taker Buy/Sell)
    if cvd is not None:
        if cvd > 0:
            score += 1
            reasons.append("Buyers Dominant (CVD+)")
        else:
            score -= 1
            reasons.append("Sellers Dominant (CVD-)")

    # 6. Volume Profile
    if vp and price:
        if price < vp['VAL']:
            score += 1
            reasons.append("Below Value Area (Oversold)")
        elif price > vp['VAH']:
            score -= 1
            reasons.append("Above Value Area (Overbought)")

    # 7. Absorption
    if absorption is not None:
        if absorption > 1.5:
            reasons.append("High Absorption (Institutional)")

    # ==================== FINAL SIGNAL ====================
    if score >= 3:
        signal = "BUY"
        confidence = min(95, 50 + score * 8)
    elif score <= -3:
        signal = "SELL"
        confidence = min(95, 50 + abs(score) * 8)
    elif -2 <= score <= 2:
        signal = "WAIT / HOLD"
        confidence = 40 + abs(score) * 5
    else:
        signal = "WAIT / HOLD"
        confidence = 40

    # ==================== SL / TP CALCULATION ====================
    entry = price
    if signal == "BUY":
        sl = entry * 0.98  # 2% SL
        tp1 = entry * 1.02
        tp2 = entry * 1.04
        tp3 = entry * 1.06
    elif signal == "SELL":
        sl = entry * 1.02
        tp1 = entry * 0.98
        tp2 = entry * 0.96
        tp3 = entry * 0.94
    else:
        sl = tp1 = tp2 = tp3 = None

    return {
        'signal': signal,
        'confidence': confidence,
        'score': score,
        'reasons': reasons,
        'entry': entry,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3
    }

# ==================== MAIN UI ====================
tickers_df = fetch_binance_tickers()

if tickers_df.empty:
    st.error("Binance data fetch nahi ho saka. Thodi der baad try karo.")
    st.stop()

sorted_df = tickers_df.sort_values('priceChangePercent', ascending=False)

# ==================== SECTION 1: TOP GAINERS / LOSERS ====================
st.header("📈 Top 10 Gainers & Losers (24h)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟢 Top Gainers")
    gainers = sorted_df.head(10)[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']].copy()
    gainers.columns = ['Symbol', 'Price', '24h %', 'Volume']
    gainers['24h %'] = gainers['24h %'].apply(lambda x: f"+{x:.2f}%")
    st.dataframe(gainers, use_container_width=True, hide_index=True)

with col2:
    st.subheader("🔴 Top Losers")
    losers = sorted_df.tail(10).sort_values('priceChangePercent')[['symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume']].copy()
    losers.columns = ['Symbol', 'Price', '24h %', 'Volume']
    losers['24h %'] = losers['24h %'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(losers, use_container_width=True, hide_index=True)

# ==================== SECTION 2: BTC & ETH ====================
st.header("₿ BTC & ETH Metrics")

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

# ==================== SECTION 3: SIGNAL ENGINE ====================
st.header("🎯 Trade Signal Engine (BTC/USDT)")

klines = fetch_binance_klines("BTCUSDT", "1h", 200)
current_oi, oi_hist = fetch_binance_oi("BTCUSDT")
funding_df = fetch_binance_funding("BTCUSDT")
ls_data = fetch_binance_long_short("BTCUSDT")
okx_cvd = fetch_okx_taker_volume("BTC-USDT-SWAP")
bitget_cvd = fetch_bitget_taker("BTCUSDT")

if not klines.empty:
    price = klines['close'].iloc[-1]
    rsi = calculate_rsi(klines)
    macd = calculate_macd(klines)
    vp = calculate_volume_profile(klines)
    absorption = calculate_absorption(klines)

    funding = funding_df['fundingRate'].iloc[-1] if not funding_df.empty else None
    oi_change = None
    if not oi_hist.empty and len(oi_hist) > 1:
        oi_change = ((oi_hist['sumOpenInterest'].iloc[-1] - oi_hist['sumOpenInterest'].iloc[0]) / oi_hist['sumOpenInterest'].iloc[0]) * 100

    ls_ratio = None
    if not ls_data.get('global', pd.DataFrame()).empty:
        ls_ratio = ls_data['global']['longShortRatio'].iloc[-1]

    cvd = None
    if not okx_cvd.empty:
        cvd = okx_cvd['CVD'].iloc[-1]
    elif not bitget_cvd.empty:
        cvd = bitget_cvd['CVD'].iloc[-1]

    # Generate Signal
    signal_result = generate_signal(
        "BTCUSDT", price, rsi, macd, funding, oi_change, ls_ratio, cvd, vp, absorption
    )

    # Display Signal
    sig_col1, sig_col2, sig_col3 = st.columns(3)
    with sig_col1:
        if signal_result['signal'] == 'BUY':
            st.success(f"🟢 **{signal_result['signal']}**")
        elif signal_result['signal'] == 'SELL':
            st.error(f"🔴 **{signal_result['signal']}**")
        else:
            st.warning(f"🟡 **{signal_result['signal']}**")
        st.metric("Confidence", f"{signal_result['confidence']}%")

    with sig_col2:
        if signal_result['entry']:
            st.metric("Entry", f"${signal_result['entry']:,.2f}")
        if signal_result['sl']:
            st.metric("Stop Loss", f"${signal_result['sl']:,.2f}")

    with sig_col3:
        if signal_result['tp1']:
            st.metric("TP1", f"${signal_result['tp1']:,.2f}")
            st.metric("TP2", f"${signal_result['tp2']:,.2f}")
            st.metric("TP3", f"${signal_result['tp3']:,.2f}")

    # Reasons
    with st.expander("📋 Signal Reasons"):
        for reason in signal_result['reasons']:
            st.write(f"• {reason}")

    # ==================== INDICATORS PANEL ====================
    st.header("📊 Indicators")

    ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
    with ind_col1:
        if rsi:
            icon = "🔴" if rsi > 70 else ("🟢" if rsi < 30 else "🟡")
            st.metric(f"{icon} RSI (1h)", f"{rsi:.1f}")
    with ind_col2:
        if macd:
            st.metric("MACD", f"{macd['macd']:.2f}", f"Hist: {macd['histogram']:.2f}")
    with ind_col3:
        if funding is not None:
            st.metric("Funding Rate", f"{funding:.4f}%")
    with ind_col4:
        if oi_change is not None:
            st.metric("OI Change (7d)", f"{oi_change:.2f}%")

    # ==================== VOLUME PROFILE ====================
    st.header("📊 Volume Profile (1h, 200 candles)")
    if vp:
        vp_col1, vp_col2, vp_col3 = st.columns(3)
        vp_col1.metric("POC", f"${vp['POC']:,.2f}")
        vp_col2.metric("VAH", f"${vp['VAH']:,.2f}")
        vp_col3.metric("VAL", f"${vp['VAL']:,.2f}")

    if absorption is not None:
        st.metric("Absorption Ratio", f"{absorption:.2f}")

    # ==================== LONG/SHORT RATIO ====================
    st.header("⚖️ Long/Short Ratios")

    ls_col1, ls_col2, ls_col3 = st.columns(3)

    with ls_col1:
        st.subheader("Global Accounts")
        if not ls_data.get('global', pd.DataFrame()).empty:
            df_g = ls_data['global'].tail(5)[['timestamp', 'longShortRatio']]
            df_g.columns = ['Time', 'L/S Ratio']
            st.dataframe(df_g, use_container_width=True, hide_index=True)

    with ls_col2:
        st.subheader("Top Accounts")
        if not ls_data.get('top_account', pd.DataFrame()).empty:
            df_t = ls_data['top_account'].tail(5)[['timestamp', 'longShortRatio']]
            df_t.columns = ['Time', 'L/S Ratio']
            st.dataframe(df_t, use_container_width=True, hide_index=True)

    with ls_col3:
        st.subheader("Top Positions")
        if not ls_data.get('top_position', pd.DataFrame()).empty:
            df_p = ls_data['top_position'].tail(5)[['timestamp', 'longShortRatio']]
            df_p.columns = ['Time', 'L/S Ratio']
            st.dataframe(df_p, use_container_width=True, hide_index=True)

    # ==================== CVD ====================
    st.header("📉 CVD (Taker Buy/Sell Volume)")

    cvd_col1, cvd_col2 = st.columns(2)

    with cvd_col1:
        st.subheader("OKX CVD")
        if not okx_cvd.empty:
            fig_okx = go.Figure()
            fig_okx.add_trace(go.Scatter(x=okx_cvd['timestamp'], y=okx_cvd['CVD'], mode='lines', name='CVD', line=dict(color='cyan')))
            fig_okx.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(fig_okx, use_container_width=True)

    with cvd_col2:
        st.subheader("Bitget CVD")
        if not bitget_cvd.empty:
            fig_bit = go.Figure()
            fig_bit.add_trace(go.Scatter(x=bitget_cvd.index, y=bitget_cvd['CVD'], mode='lines', name='CVD', line=dict(color='orange')))
            fig_bit.update_layout(height=300, template="plotly_dark")
            st.plotly_chart(fig_bit, use_container_width=True)

    # ==================== PRICE CHART ====================
    st.header("📈 BTC/USDT Price Chart (1h)")

    fig = go.Figure(data=[
        go.Candlestick(
            x=klines['open_time'],
            open=klines['open'],
            high=klines['high'],
            low=klines['low'],
            close=klines['close']
        )
    ])

    # Add Volume Profile levels
    if vp:
        fig.add_hline(y=vp['POC'], line_dash="dash", line_color="yellow", annotation_text="POC")
        fig.add_hline(y=vp['VAH'], line_dash="dot", line_color="green", annotation_text="VAH")
        fig.add_hline(y=vp['VAL'], line_dash="dot", line_color="red", annotation_text="VAL")

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== SIDEBAR ====================
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
**Binance se:**
- ✅ Price/Volume/RSI
- ✅ MACD
- ✅ Open Interest
- ✅ Funding Rate
- ✅ Long/Short Ratio (3 types)
- ✅ Taker Buy/Sell

**OKX se:**
- ✅ Taker Volume (CVD)

**Bitget se:**
- ✅ Taker Buy/Sell

**Calculated:**
- ✅ Volume Profile (POC/VAH/VAL)
- ✅ Absorption
- ✅ Signal (BUY/SELL/WAIT)
- ✅ Entry/SL/TP1/TP2/TP3
""")

st.sidebar.header("🚫 Abhi Nahi Hai")
st.sidebar.markdown("""
- ❌ GEX / Gamma Levels (paid)
- ❌ Liquidation Heatmap (paid)
- ❌ Chat Interface (phase 3)
""")

st.sidebar.header("⚠️ Disclaimer")
st.sidebar.caption("Educational only. Not financial advice.")

# ==================== AUTO REFRESH ====================
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.info("Data auto-refreshes every 5 minutes")
