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

MEXC_BASE = "https://contract.mexc.com"
OKX_BASE = "https://www.okx.com"
BITGET_BASE = "https://api.bitget.com"

# ==================== MEXC DATA ====================
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

# Sidebar: Coin Selection
st.sidebar.header("🔍 Coin Select Karo")
all_symbols = sorted_df['symbol'].tolist()
default_index = all_symbols.index('BTC_USDT') if 'BTC_USDT' in all_symbols else 0

selected_symbol = st.sidebar.selectbox(
    "Symbol chuno:",
    options=all_symbols,
    index=default_index
)

search_query = st.sidebar.text_input("Ya search karo (e.g. ETH_USDT):", "")
if search_query and search_query.upper() in all_symbols:
    selected_symbol = search_query.upper()

st.sidebar.markdown(f"**Selected: `{selected_symbol}`**")
base_ccy = selected_symbol.replace("_USDT", "")

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

# Selected Coin Analysis
st.header(f"🎯 Analysis: {selected_symbol}")
coin_row = tickers_df[tickers_df['symbol'] == selected_symbol]

if not coin_row.empty:
    coin = coin_row.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${coin['lastPrice']:,.4f}", f"{coin['riseFallRate']:.2f}%")
    c2.metric("24h Volume", f"${coin['volume24']:,.0f}")

klines = fetch_mexc_klines(selected_symbol, "Min60", 200)

if klines.empty:
    st.warning(f"{selected_symbol} ka data nahi mila. Koi aur coin try karo.")
    st.stop()

price = klines['close'].iloc[-1]
rsi = calculate_rsi(klines)
macd = calculate_macd(klines)
vp = calculate_volume_profile(klines)
absorption = calculate_absorption(klines)
funding = fetch_mexc_funding(selected_symbol)

okx_cvd = fetch_okx_taker_volume(base_ccy)
cvd = None
if not okx_cvd.empty:
    cvd = okx_cvd['CVD'].iloc[-1]
else:
    bitget_cvd = fetch_bitget_taker(selected_symbol.replace("_", ""))
    if not bitget_cvd.empty:
        cvd = bitget_cvd['CVD'].iloc[-1]

sig = generate_signal(price, rsi, macd, funding, cvd, vp, absorption)

c1, c2, c3 = st.columns(3)
with c1:
    if sig['signal'] == 'BUY': st.success(f"🟢 **{sig['signal']}**")
    elif sig['signal'] == 'SELL': st.error(f"🔴 **{sig['signal']}**")
    else: st.warning(f"🟡 **{sig['signal']}**")
    st.metric("Confidence", f"{sig['confidence']}%")
with c2:
    st.metric("Entry", f"${sig['entry']:,.4f}")
    if sig['sl']: st.metric("Stop Loss", f"${sig['sl']:,.4f}")
with c3:
    if sig['tp1']:
        st.metric("TP1", f"${sig['tp1']:,.4f}")
        st.metric("TP2", f"${sig['tp2']:,.4f}")
        st.metric("TP3", f"${sig['tp3']:,.4f}")

with st.expander("📋 Signal Reasons"):
    for r in sig['reasons']: st.write(f"• {r}")

st.header(f"📊 Indicators ({selected_symbol})")
i1, i2, i3, i4 = st.columns(4)
with i1:
    if rsi: st.metric("RSI (1h)", f"{rsi:.1f}")
with i2:
    if macd: st.metric("MACD", f"{macd['macd']:.4f}", f"Hist: {macd['histogram']:.4f}")
with i3:
    if funding is not None: st.metric("Funding Rate", f"{funding:.4f}%")
with i4:
    if cvd is not None: st.metric("CVD (OKX)", f"{cvd:,.0f}")

st.header(f"📊 Volume Profile ({selected_symbol})")
if vp:
    v1, v2, v3 = st.columns(3)
    v1.metric("POC", f"${vp['POC']:,.4f}")
    v2.metric("VAH", f"${vp['VAH']:,.4f}")
    v3.metric("VAL", f"${vp['VAL']:,.4f}")

if absorption is not None:
    st.metric("Absorption Ratio", f"{absorption:.2f}")

st.header(f"📈 {selected_symbol} (1h)")
fig = go.Figure(data=[go.Candlestick(
    x=klines['time'], open=klines['open'], high=klines['high'],
    low=klines['low'], close=klines['close']
)])
if vp:
    fig.add_hline(y=vp['POC'], line_dash="dash", line_color="yellow", annotation_text="POC")
    fig.add_hline(y=vp['VAH'], line_dash="dot", line_color="green", annotation_text="VAH")
    fig.add_hline(y=vp['VAL'], line_dash="dot", line_color="red", annotation_text="VAL")
fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Sidebar Info
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
- Top Gainers/Losers
- Price / Klines
- RSI / MACD
- Funding Rate
- CVD (OKX)
- Volume Profile
- Absorption
- Signal (BUY/SELL/WAIT)
- Entry / SL / TP1-3
""")
st.sidebar.button("🔄 Refresh", on_click=lambda: st.cache_data.clear())
