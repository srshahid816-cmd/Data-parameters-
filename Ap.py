import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Pro Crypto Dashboard", layout="wide")
st.title("📊 Professional Crypto Futures Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: MEXC + OKX + Bitget")

# ==================== MEXC TICKERS (US Accessible) ====================
MEXC_BASE = "https://contract.mexc.com"

@st.cache_data(ttl=300)
def fetch_mexc_tickers():
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
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_mexc_klines(symbol, interval="Min60", limit=200):
    try:
        resp = requests.get(f"{MEXC_BASE}/api/v1/contract/kline/{symbol}", 
                          params={"interval": interval}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") and data.get("data"):
            k = data["data"]
            df = pd.DataFrame({
                'time': pd.to_datetime(k['time'], unit='s'),
                'open': k['open'], 'high': k['high'],
                'low': k['low'], 'close': k['close'],
                'volume': k['vol']
            })
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df.tail(limit)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_mexc_funding(symbol="BTC_USDT"):
    try:
        resp = requests.get(f"{MEXC_BASE}/api/v1/contract/funding_rate/{symbol}", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return float(data["data"]["fundingRate"]) * 100
        return None
    except:
        return None

# ==================== OKX DATA ====================
OKX_BASE = "https://www.okx.com"

@st.cache_data(ttl=300)
def fetch_okx_taker_volume(ccy="BTC"):
    try:
        resp = requests.get(
            f"{OKX_BASE}/api/v5/rubik/stat/taker-volume",
            params={"ccy": ccy, "instType": "CONTRACTS", "period": "1H"},
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
BITGET_BASE = "https://api.bitget.com"

@st.cache_data(ttl=300)
def fetch_bitget_taker(symbol="BTCUSDT"):
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
            return {'macd': macd.iloc[-1, 0], 'signal': macd.iloc[-1, 1], 'histogram': macd.iloc[-1, 2]}
        return None
    except:
        return None

def calculate_volume_profile(df, bins=50):
    try:
        if df.empty or len(df) < 10:
            return None
        price_min, price_max = df['low'].min(), df['high'].max()
        if price_min == price_max:
            return None
        bins_edges = np.linspace(price_min, price_max, bins + 1)
        df['price_bin'] = pd.cut(df['close'], bins=bins_edges, labels=False)
        volume_profile = df.groupby('price_bin')['volume'].sum()
        poc_bin = volume_profile.idxmax()
        poc = (bins_edges[poc_bin] + bins_edges[poc_bin + 1]) / 2
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
    try:
        if df.empty or len(df) < 20:
            return None
        df['range'] = df['high'] - df['low']
        df['vol_per_range'] = df['volume'] / (df['range'] + 1e-10)
        recent = df.tail(10)
        avg_vol = recent['volume'].mean()
        avg_range = recent['range'].mean()
        absorption_ratio = (recent['volume'] / avg_vol) / (recent['range'] / avg_range + 1e-10)
        return absorption_ratio.mean()
    except:
        return None

# ==================== SIGNAL ENGINE ====================
def generate_signal(price, rsi, macd, funding, cvd, vp, absorption):
    score = 0
    reasons = []

    if rsi:
        if rsi < 30: score += 2; reasons.append("RSI Oversold")
        elif rsi > 70: score -= 2; reasons.append("RSI Overbought")

    if macd:
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            score += 1; reasons.append("MACD Bullish")
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            score -= 1; reasons.append("MACD Bearish")

    if funding is not None:
        if funding > 0.05: score -= 1; reasons.append("High Funding (Longs crowded)")
        elif funding < -0.05: score += 1; reasons.append("Negative Funding (Shorts crowded)")

    if cvd is not None:
        if cvd > 0: score += 1; reasons.append("Buyers Dominant (CVD+)")
        else: score -= 1; reasons.append("Sellers Dominant (CVD-)")

    if vp and price:
        if price < vp['VAL']: score += 1; reasons.append("Below Value Area (Oversold)")
        elif price > vp['VAH']: score -= 1; reasons.append("Above Value Area (Overbought)")

    if absorption and absorption > 1.5:
        reasons.append("High Absorption (Institutional)")

    if score >= 3:
        signal, confidence = "BUY", min(95, 50 + score * 8)
    elif score <= -3:
        signal, confidence = "SELL", min(95, 50 + abs(score) * 8)
    else:
        signal, confidence = "WAIT / HOLD", 40 + abs(score) * 5

    entry = price
    if signal == "BUY":
        sl, tp1, tp2, tp3 = entry*0.98, entry*1.02, entry*1.04, entry*1.06
    elif signal == "SELL":
        sl, tp1, tp2, tp3 = entry*1.02, entry*0.98, entry*0.96, entry*0.94
    else:
        sl = tp1 = tp2 = tp3 = None

    return {'signal': signal, 'confidence': confidence, 'score': score,
            'reasons': reasons, 'entry': entry, 'sl': sl,
            'tp1': tp1, 'tp2': tp2, 'tp3': tp3}

# ==================== MAIN UI ====================
tickers_df = fetch_mexc_tickers()

if tickers_df.empty:
    st.error("MEXC data fetch nahi ho saka. Thodi der baad try karo.")
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

# BTC/ETH
st.header("₿ BTC & ETH")
btc = tickers_df[tickers_df['symbol'] == 'BTC_USDT']
eth = tickers_df[tickers_df['symbol'] == 'ETH_USDT']
if not btc.empty:
    b = btc.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BTC", f"${b['lastPrice']:,.2f}", f"{b['riseFallRate']:.2f}%")
    c2.metric("Volume", f"${b['volume24']:,.0f}")
    if not eth.empty:
        e = eth.iloc[0]
        c3.metric("ETH", f"${e['lastPrice']:,.2f}", f"{e['riseFallRate']:.2f}%")
        c4.metric("Volume", f"${e['volume24']:,.0f}")

# Signal Engine
st.header("🎯 Signal Engine (BTC_USDT)")
klines = fetch_mexc_klines("BTC_USDT", "Min60", 200)
if not klines.empty:
    price = klines['close'].iloc[-1]
    rsi = calculate_rsi(klines)
    macd = calculate_macd(klines)
    vp = calculate_volume_profile(klines)
    absorption = calculate_absorption(klines)
    funding = fetch_mexc_funding("BTC_USDT")
    okx_cvd = fetch_okx_taker_volume("BTC")
    cvd = okx_cvd['CVD'].iloc[-1] if not okx_cvd.empty else None

    sig = generate_signal(price, rsi, macd, funding, cvd, vp, absorption)

    c1, c2, c3 = st.columns(3)
    with c1:
        if sig['signal'] == 'BUY': st.success(f"🟢 {sig['signal']}")
        elif sig['signal'] == 'SELL': st.error(f"🔴 {sig['signal']}")
        else: st.warning(f"🟡 {sig['signal']}")
        st.metric("Confidence", f"{sig['confidence']}%")
    with c2:
        st.metric("Entry", f"${sig['entry']:,.2f}")
        if sig['sl']: st.metric("SL", f"${sig['sl']:,.2f}")
    with c3:
        if sig['tp1']:
            st.metric("TP1", f"${sig['tp1']:,.2f}")
            st.metric("TP2", f"${sig['tp2']:,.2f}")
            st.metric("TP3", f"${sig['tp3']:,.2f}")

    with st.expander("📋 Reasons"):
        for r in sig['reasons']: st.write(f"• {r}")

    # Indicators
    st.header("📊 Indicators")
    i1, i2, i3, i4 = st.columns(4)
    with i1:
        if rsi: st.metric("RSI", f"{rsi:.1f}")
    with i2:
        if macd: st.metric("MACD", f"{macd['macd']:.2f}", f"Hist: {macd['histogram']:.2f}")
    with i3:
        if funding is not None: st.metric("Funding", f"{funding:.4f}%")
    with i4:
        if cvd is not None: st.metric("CVD (OKX)", f"{cvd:,.0f}")

    # Volume Profile
    st.header("📊 Volume Profile")
    if vp:
        v1, v2, v3 = st.columns(3)
        v1.metric("POC", f"${vp['POC']:,.2f}")
        v2.metric("VAH", f"${vp['VAH']:,.2f}")
        v3.metric("VAL", f"${vp['VAL']:,.2f}")

    # Chart
    st.header("📈 BTC/USDT (1h)")
    fig = go.Figure(data=[go.Candlestick(
        x=klines['time'], open=klines['open'], high=klines['high'],
        low=klines['low'], close=klines['close']
    )])
    if vp:
        fig.add_hline(y=vp['POC'], line_dash="dash", line_color="yellow")
        fig.add_hline(y=vp['VAH'], line_dash="dot", line_color="green")
        fig.add_hline(y=vp['VAL'], line_dash="dot", line_color="red")
    fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
**MEXC se:**
- Tickers / Gainers / Losers
- Klines / Price
- Funding Rate

**OKX se:**
- Taker Volume (CVD)

**Calculated:**
- RSI, MACD
- Volume Profile
- Absorption
- Signal (BUY/SELL/WAIT)
- Entry / SL / TP1-3
""")
st.sidebar.header("🚫 Hata Diya")
st.sidebar.markdown("- ❌ Binance (US blocked)")
st.sidebar.button("🔄 Refresh", on_click=lambda: st.cache_data.clear())
