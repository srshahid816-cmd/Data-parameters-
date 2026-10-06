import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime
from scipy.signal import argrelextrema

st.set_page_config(page_title="Pro Crypto Dashboard", layout="wide")
st.title("📊 Professional Crypto Futures Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data: MEXC + OKX + Bitget")

MEXC_BASE = "https://contract.mexc.com"
OKX_BASE = "https://www.okx.com"
BITGET_BASE = "https://api.bitget.com"

# ==================== DATA FETCH ====================
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

@st.cache_data(ttl=300)
def fetch_okx_open_interest(ccy="BTC"):
    try:
        resp = requests.get(
            f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-volume",
            params={"ccy": ccy, "period": "1H"},
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') == '0' and data.get('data'):
            df = pd.DataFrame(data['data'], columns=['timestamp', 'oi', 'vol'])
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df['oi'] = pd.to_numeric(df['oi'])
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

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

def calculate_atr(df, period=14):
    try:
        atr = ta.atr(df['high'], df['low'], df['close'], length=period)
        return atr.iloc[-1] if len(atr) > 0 else None
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

# ==================== TREND LINE BREAKOUT DETECTION ====================
def detect_trendline_breakout(df, window=5):
    """
    Detects trendline breakout.
    Returns: {'breakout': 1, -1, or 0, 'status': str, 'details': list}
    """
    try:
        df = df.copy()
        df['max'] = df.iloc[argrelextrema(df['high'].values, np.greater_equal, order=window)[0]]['high']
        df['min'] = df.iloc[argrelextrema(df['low'].values, np.less_equal, order=window)[0]]['low']

        recent_highs = df['max'].dropna().tail(3)
        recent_lows = df['min'].dropna().tail(3)

        current_idx = len(df) - 1
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]

        # Check Downtrend line (connect swing highs) -> Bullish Breakout
        if len(recent_highs) >= 2:
            idx1, idx2 = recent_highs.index[-2], recent_highs.index[-1]
            if idx1 < idx2:
                slope = (recent_highs.iloc[-1] - recent_highs.iloc[-2]) / (idx2 - idx1)
                intercept = recent_highs.iloc[-1] - slope * idx2
                tl_current = slope * current_idx + intercept
                tl_prev = slope * (current_idx - 1) + intercept
                
                if prev_close <= tl_prev and current_close > tl_current:
                    return {
                        'breakout': 1, 
                        'status': 'Bullish Trendline Breakout', 
                        'details': [f"Broke above: ${tl_current:,.4f}", "Wait 30-45 mins for confirmation"]
                    }

        # Check Uptrend line (connect swing lows) -> Bearish Breakout
        if len(recent_lows) >= 2:
            idx1, idx2 = recent_lows.index[-2], recent_lows.index[-1]
            if idx1 < idx2:
                slope = (recent_lows.iloc[-1] - recent_lows.iloc[-2]) / (idx2 - idx1)
                intercept = recent_lows.iloc[-1] - slope * idx2
                tl_current = slope * current_idx + intercept
                tl_prev = slope * (current_idx - 1) + intercept
                
                if prev_close >= tl_prev and current_close < tl_current:
                    return {
                        'breakout': -1, 
                        'status': 'Bearish Trendline Breakout', 
                        'details': [f"Broke below: ${tl_current:,.4f}", "Wait 30-45 mins for confirmation"]
                    }

        # No Trendline Breakout Found -> SKIP completely
        return {'breakout': 0, 'status': 'Skipped', 'details': []}
    except Exception as e:
        return {'breakout': 0, 'status': 'Skipped', 'details': []}

# ==================== SIGNAL ENGINE ====================
def generate_signal(price, rsi, macd, funding, cvd, vp, absorption, tlb, oi_change, atr):
    bull_points = 0
    bear_points = 0
    reasons = []
    details = []

    # 1. RSI
    if rsi:
        if rsi < 30:
            bull_points += 3
            reasons.append("RSI Oversold (Bullish)")
            details.append(f"RSI={rsi:.1f} → Strong Buy Signal")
        elif rsi < 40:
            bull_points += 1.5
            reasons.append("RSI Near Oversold")
            details.append(f"RSI={rsi:.1f} → Mild Bullish")
        elif rsi > 70:
            bear_points += 3
            reasons.append("RSI Overbought (Bearish)")
            details.append(f"RSI={rsi:.1f} → Strong Sell Signal")
        elif rsi > 60:
            bear_points += 1.5
            reasons.append("RSI Near Overbought")
            details.append(f"RSI={rsi:.1f} → Mild Bearish")
        else:
            details.append(f"RSI={rsi:.1f} → Neutral")

    # 2. MACD
    if macd:
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            bull_points += 2
            reasons.append("MACD Bullish Crossover")
            details.append("MACD: Bullish momentum")
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            bear_points += 2
            reasons.append("MACD Bearish Crossover")
            details.append("MACD: Bearish momentum")
        elif macd['histogram'] > 0:
            bull_points += 1
            details.append("MACD: Mild Bullish")
        elif macd['histogram'] < 0:
            bear_points += 1
            details.append("MACD: Mild Bearish")

    # 3. Funding Rate
    if funding is not None:
        if funding > 0.05:
            bear_points += 2
            reasons.append("High Funding (Longs Crowded)")
            details.append(f"Funding={funding:.4f}% → Crowded Longs (Bearish)")
        elif funding > 0.02:
            bear_points += 1
            details.append(f"Funding={funding:.4f}% → Mild Long Bias")
        elif funding < -0.05:
            bull_points += 2
            reasons.append("Negative Funding (Shorts Crowded)")
            details.append(f"Funding={funding:.4f}% → Crowded Shorts (Bullish)")
        elif funding < -0.02:
            bull_points += 1
            details.append(f"Funding={funding:.4f}% → Mild Short Bias")
        else:
            details.append(f"Funding={funding:.4f}% → Neutral")

    # 4. CVD (Delta)
    if cvd is not None:
        if cvd > 0:
            bull_points += 2
            reasons.append("Buyers Dominant (CVD+)")
            details.append(f"CVD={cvd:,.0f} → Buyers in control")
        else:
            bear_points += 2
            reasons.append("Sellers Dominant (CVD-)")
            details.append(f"CVD={cvd:,.0f} → Sellers in control")

    # 5. Volume Profile
    if vp and price:
        if price < vp['VAL']:
            bull_points += 2
            reasons.append("Below Value Area (Oversold)")
            details.append("Price below VAL → Mean reversion bullish")
        elif price > vp['VAH']:
            bear_points += 2
            reasons.append("Above Value Area (Overbought)")
            details.append("Price above VAH → Mean reversion bearish")
        elif price < vp['POC']:
            bull_points += 0.5
            details.append("Below POC → Slight bullish bias")
        else:
            bear_points += 0.5
            details.append("Above POC → Slight bearish bias")

    # 6. Absorption
    if absorption is not None and absorption > 1.5:
        reasons.append("High Absorption (Institutional Activity)")
        details.append(f"Absorption={absorption:.2f} → Watch for breakout")

    # 7. Trendline Breakout (SKIP IF 0)
    if tlb['breakout'] == 1:
        bull_points += 3
        reasons.append("Bullish Trendline Breakout")
        details.extend(tlb['details'])
    elif tlb['breakout'] == -1:
        bear_points += 3
        reasons.append("Bearish Trendline Breakout")
        details.extend(tlb['details'])
    # Agar 0 hai to yahan kuch nahi hoga, skip ho jayega

    # 8. Open Interest (OI) Change
    if oi_change is not None:
        if oi_change > 2:
            bull_points += 1.5
            reasons.append("Open Interest Increasing (Strong Momentum)")
            details.append(f"OI Change: +{oi_change:.2f}%")
        elif oi_change < -2:
            bear_points += 1.5
            reasons.append("Open Interest Decreasing (Weak Momentum)")
            details.append(f"OI Change: {oi_change:.2f}%")
        else:
            details.append(f"OI Change: {oi_change:.2f}% (Neutral)")

    # ==================== PROBABILITY ====================
    total = bull_points + bear_points
    if total == 0:
        buy_prob, sell_prob, hold_prob = 33.3, 33.3, 33.4
    else:
        buy_prob = (bull_points / total) * 100
        sell_prob = (bear_points / total) * 100
        hold_prob = max(0, 100 - buy_prob - sell_prob)

    if buy_prob > sell_prob and buy_prob > 50:
        buy_prob = min(95, buy_prob + 10)
        sell_prob = max(5, sell_prob - 5)
    elif sell_prob > buy_prob and sell_prob > 50:
        sell_prob = min(95, sell_prob + 10)
        buy_prob = max(5, buy_prob - 5)

    hold_prob = max(0, 100 - buy_prob - sell_prob)

    # ==================== FINAL SIGNAL ====================
    # If Trendline Just Broke, Force HOLD for confirmation
    if tlb['breakout'] != 0:
        signal = "WAIT / HOLD (Trendline Breakout)"
        confidence = 50
        reasons.append("⚠️ Trendline just broke. Wait 30-45 mins for confirmation before entry.")
    else:
        if buy_prob >= 55:
            signal, confidence = "BUY", buy_prob
        elif sell_prob >= 55:
            signal, confidence = "SELL", sell_prob
        else:
            signal, confidence = "WAIT / HOLD", hold_prob

    # ==================== SL / TP (ATR-Based for 3-12h Trades) ====================
    entry = price
    if atr is None:
        atr = entry * 0.02 # Fallback 2%
        
    if signal == "BUY":
        sl = entry - (1.5 * atr)
        tp1 = entry + (2.0 * atr)
        tp2 = entry + (3.5 * atr)
        tp3 = entry + (5.0 * atr)
    elif signal == "SELL":
        sl = entry + (1.5 * atr)
        tp1 = entry - (2.0 * atr)
        tp2 = entry - (3.5 * atr)
        tp3 = entry - (5.0 * atr)
    else:
        sl = tp1 = tp2 = tp3 = None

    return {
        'signal': signal, 'confidence': confidence,
        'buy_prob': round(buy_prob, 1), 'sell_prob': round(sell_prob, 1), 'hold_prob': round(hold_prob, 1),
        'bull_points': round(bull_points, 1), 'bear_points': round(bear_points, 1),
        'reasons': reasons, 'details': details,
        'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
        'tlb_status': tlb['status']
    }

# ==================== MAIN ====================
tickers_df = fetch_mexc_tickers()

if tickers_df.empty:
    st.error("MEXC data fetch nahi ho saka. Refresh karo.")
    st.stop()

sorted_df = tickers_df.sort_values('riseFallRate', ascending=False)
all_symbols = sorted_df['symbol'].tolist()

# Coin Selector
st.markdown("### 🔍 Coin Select Karo")
sel_col1, sel_col2 = st.columns([2, 2])

with sel_col1:
    default_index = all_symbols.index('BTC_USDT') if 'BTC_USDT' in all_symbols else 0
    selected_from_dropdown = st.selectbox("Dropdown se chuno:", options=all_symbols, index=default_index, key="dropdown_select")

with sel_col2:
    search_text = st.text_input("Ya manually likho (jaise ETH_USDT):", key="search_input")

if search_text.strip():
    user_input = search_text.strip().upper()
    if user_input in all_symbols:
        selected_symbol = user_input
    else:
        st.warning(f"'{user_input}' list mein nahi mila. Dropdown use karo.")
        selected_symbol = selected_from_dropdown
else:
    selected_symbol = selected_from_dropdown

st.markdown(f"### ✅ Selected: `{selected_symbol}`")
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

# Selected Coin Info
st.header(f"🎯 Analysis: {selected_symbol}")
coin_row = tickers_df[tickers_df['symbol'] == selected_symbol]
if not coin_row.empty:
    coin = coin_row.iloc[0]
    c1, c2 = st.columns(2)
    c1.metric("Price", f"${coin['lastPrice']:,.4f}", f"{coin['riseFallRate']:.2f}%")
    c2.metric("24h Volume", f"${coin['volume24']:,.0f}")

# Fetch data
klines = fetch_mexc_klines(selected_symbol, "Min60", 200)
if klines.empty:
    st.warning(f"{selected_symbol} ka data nahi mila.")
    st.stop()

price = klines['close'].iloc[-1]
rsi = calculate_rsi(klines)
macd = calculate_macd(klines)
vp = calculate_volume_profile(klines)
absorption = calculate_absorption(klines)
funding = fetch_mexc_funding(selected_symbol)
atr = calculate_atr(klines)

# Trendline Breakout Detection
tlb = detect_trendline_breakout(klines)

# CVD
okx_cvd = fetch_okx_taker_volume(base_ccy)
cvd = None
if not okx_cvd.empty:
    cvd = okx_cvd['CVD'].iloc[-1]
else:
    bitget_cvd = fetch_bitget_taker(selected_symbol.replace("_", ""))
    if not bitget_cvd.empty:
        cvd = bitget_cvd['CVD'].iloc[-1]

# Open Interest Change
oi_df = fetch_okx_open_interest(base_ccy)
oi_change = None
if not oi_df.empty and len(oi_df) > 1:
    oi_change = ((oi_df['oi'].iloc[-1] - oi_df['oi'].iloc[0]) / oi_df['oi'].iloc[0]) * 100

sig = generate_signal(price, rsi, macd, funding, cvd, vp, absorption, tlb, oi_change, atr)

# ==================== SIGNAL DISPLAY ====================
st.header(f"🎯 Signal for {selected_symbol}")

# Trendline Status Banner (Only show if breakout happened)
if tlb['breakout'] != 0:
    st.warning(f"⚠️ **{tlb['status']}** — 30-45 mins wait karo confirmation ke liye.")

prob_col1, prob_col2, prob_col3 = st.columns(3)
with prob_col1:
    st.metric("🟢 BUY", f"{sig['buy_prob']}%")
with prob_col2:
    st.metric("🔴 SELL", f"{sig['sell_prob']}%")
with prob_col3:
    st.metric("🟡 HOLD", f"{sig['hold_prob']}%")

prob_chart = go.Figure(go.Bar(
    x=['BUY', 'SELL', 'HOLD'],
    y=[sig['buy_prob'], sig['sell_prob'], sig['hold_prob']],
    marker_color=['#00ff88', '#ff4444', '#ffaa00'],
    text=[f"{sig['buy_prob']}%", f"{sig['sell_prob']}%", f"{sig['hold_prob']}%"],
    textposition='auto'
))
prob_chart.update_layout(height=250, template="plotly_dark", showlegend=False, yaxis_title="Probability %", yaxis_range=[0, 100])
st.plotly_chart(prob_chart, use_container_width=True)

final_col1, final_col2 = st.columns([1, 1])
with final_col1:
    if "BUY" in sig['signal']:
        st.success(f"## 🟢 **{sig['signal']}**")
    elif "SELL" in sig['signal']:
        st.error(f"## 🔴 **{sig['signal']}**")
    else:
        st.warning(f"## 🟡 **{sig['signal']}**")
    st.metric("Confidence", f"{sig['confidence']:.1f}%")
with final_col2:
    st.metric("Entry", f"${sig['entry']:,.4f}")
    if sig['sl']:
        st.metric("Stop Loss (ATR Based)", f"${sig['sl']:,.4f}")

if sig['tp1']:
    tp_col1, tp_col2, tp_col3 = st.columns(3)
    tp_col1.metric("TP1 (2x ATR)", f"${sig['tp1']:,.4f}")
    tp_col2.metric("TP2 (3.5x ATR)", f"${sig['tp2']:,.4f}")
    tp_col3.metric("TP3 (5x ATR)", f"${sig['tp3']:,.4f}")
    st.caption(f"ATR Value: ${atr:,.4f} — SL/TP ATR ke hisaab se dynamic hain (3-12h trades ke liye)")

with st.expander("📋 Signal Analysis (Parameter by Parameter)"):
    st.markdown("**Bullish Points:** " + str(sig['bull_points']))
    st.markdown("**Bearish Points:** " + str(sig['bear_points']))
    st.markdown("---")
    st.markdown("**Parameter Breakdown:**")
    for d in sig['details']:
        st.write(f"• {d}")
    st.markdown("---")
    st.markdown("**Key Reasons:**")
    if sig['reasons']:
        for r in sig['reasons']:
            st.write(f"✅ {r}")
    else:
        st.write("Koi strong reason nahi mila.")

# Indicators
st.header(f"📊 Indicators ({selected_symbol})")
i1, i2, i3, i4 = st.columns(4)
with i1:
    if rsi:
        icon = "🔴" if rsi > 70 else ("🟢" if rsi < 30 else "🟡")
        st.metric(f"{icon} RSI (1h)", f"{rsi:.1f}")
with i2:
    if macd:
        st.metric("MACD", f"{macd['macd']:.4f}", f"Hist: {macd['histogram']:.4f}")
with i3:
    if funding is not None:
        st.metric("Funding Rate", f"{funding:.4f}%")
with i4:
    if cvd is not None:
        st.metric("CVD (OKX)", f"{cvd:,.0f}")

if oi_change is not None:
    st.metric("Open Interest Change", f"{oi_change:.2f}%")

# Volume Profile
st.header(f"📊 Volume Profile ({selected_symbol})")
if vp:
    v1, v2, v3 = st.columns(3)
    v1.metric("POC", f"${vp['POC']:,.4f}")
    v2.metric("VAH", f"${vp['VAH']:,.4f}")
    v3.metric("VAL", f"${vp['VAL']:,.4f}")

if absorption is not None:
    st.metric("Absorption Ratio", f"{absorption:.2f}")

# Chart
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

# Sidebar
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
- Top Gainers/Losers
- Price / Klines
- RSI / MACD / ATR
- Funding Rate
- CVD (Delta)
- Open Interest (OI)
- Volume Profile
- Absorption
- Trendline Breakout (If Applicable)
- Signal (BUY/SELL/WAIT) with Probability
- Dynamic SL/TP (ATR Based for 3-12h)
""")
st.sidebar.button("🔄 Refresh", on_click=lambda: st.cache_data.clear())
