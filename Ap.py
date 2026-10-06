import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime
from scipy.signal import argrelextrema

st.set_page_config(page_title="Pro Crypto Dashboard v2", layout="wide")
st.title("📊 Professional Crypto Dashboard v2")
st.caption(f"Last: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | MEXC + OKX + Bitget | MTF + SMC")

MEXC_BASE = "https://contract.mexc.com"
OKX_BASE = "https://www.okx.com"
BITGET_BASE = "https://api.bitget.com"

# ==================== DATA FETCH ====================
@st.cache_data(ttl=300)
def fetch_tickers():
    try:
        r = requests.get(f"{MEXC_BASE}/api/v1/contract/ticker", timeout=15)
        r.raise_for_status()
        d = r.json()
        if d.get("success"):
            df = pd.DataFrame(d["data"])
            df = df[df['symbol'].str.endswith('_USDT')]
            for c in ['riseFallRate', 'lastPrice', 'volume24']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df['riseFallRate'] = df['riseFallRate'] * 100
            return df
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_klines(symbol, interval="Min60", limit=200):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v1/contract/kline/{symbol}",
                        params={"interval": interval}, timeout=15)
        r.raise_for_status()
        d = r.json()
        if d.get("success") and d.get("data"):
            k = d["data"]
            df = pd.DataFrame({
                'time': pd.to_datetime(k['time'], unit='s'),
                'open': pd.to_numeric(k['open'], errors='coerce'),
                'high': pd.to_numeric(k['high'], errors='coerce'),
                'low': pd.to_numeric(k['low'], errors='coerce'),
                'close': pd.to_numeric(k['close'], errors='coerce'),
                'volume': pd.to_numeric(k['vol'], errors='coerce'),
            })
            return df.tail(limit).reset_index(drop=True)
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_funding(symbol):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v1/contract/funding_rate/{symbol}", timeout=15)
        d = r.json()
        if d.get("success"):
            return float(d["data"]["fundingRate"]) * 100
    except:
        pass
    return None

@st.cache_data(ttl=300)
def fetch_cvd(ccy):
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/rubik/stat/taker-volume",
                        params={"ccy": ccy, "instType": "CONTRACTS", "period": "1H"}, timeout=15)
        d = r.json()
        if d.get('code') == '0' and d.get('data'):
            df = pd.DataFrame(d['data'], columns=['timestamp', 'sellVol', 'buyVol'])
            df['buyVol'] = pd.to_numeric(df['buyVol'])
            df['sellVol'] = pd.to_numeric(df['sellVol'])
            df['CVD'] = df['buyVol'] - df['sellVol']
            return df
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_oi(ccy):
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/rubik/stat/contracts/open-interest-volume",
                        params={"ccy": ccy, "period": "1H"}, timeout=15)
        d = r.json()
        if d.get('code') == '0' and d.get('data'):
            df = pd.DataFrame(d['data'], columns=['timestamp', 'oi', 'vol'])
            df['oi'] = pd.to_numeric(df['oi'])
            return df
    except:
        pass
    return pd.DataFrame()

# ==================== INDICATORS ====================
def calc_rsi(df, n=14):
    try:
        v = ta.rsi(df['close'], length=n)
        return float(v.iloc[-1]) if v is not None and len(v) > 0 else None
    except:
        return None

def calc_macd(df):
    try:
        m = ta.macd(df['close'])
        if m is not None and len(m) > 0:
            return {'macd': float(m.iloc[-1,0]), 'signal': float(m.iloc[-1,1]), 'hist': float(m.iloc[-1,2])}
    except:
        pass
    return None

def calc_atr(df, n=14):
    try:
        v = ta.atr(df['high'], df['low'], df['close'], length=n)
        return float(v.iloc[-1]) if v is not None and len(v) > 0 else None
    except:
        return None

def calc_vp(df, bins=50):
    try:
        if len(df) < 10: return None
        pmin, pmax = df['low'].min(), df['high'].max()
        if pmin == pmax: return None
        edges = np.linspace(pmin, pmax, bins+1)
        d = df.copy()
        d['bin'] = pd.cut(d['close'], bins=edges, labels=False)
        vp = d.groupby('bin')['volume'].sum()
        poc_bin = vp.idxmax()
        poc = (edges[poc_bin] + edges[poc_bin+1]) / 2
        total = vp.sum()
        sv = vp.sort_values(ascending=False)
        cum = 0; vb = []
        for b, v in sv.items():
            cum += v; vb.append(b)
            if cum >= total * 0.7: break
        vah = edges[max(vb)+1]
        val = edges[min(vb)]
        return {'POC': poc, 'VAH': vah, 'VAL': val}
    except:
        return None

def calc_absorption(df):
    try:
        if len(df) < 20: return None
        d = df.copy()
        d['range'] = d['high'] - d['low']
        r = d.tail(10)
        av = r['volume'].mean(); ar = r['range'].mean()
        return float(((r['volume']/av)/(r['range']/ar + 1e-10)).mean())
    except:
        return None

# ==================== MULTI-TIMEFRAME TREND ====================
def get_trend(df, k=3):
    if df is None or len(df) < 50: return 0, "N/A"
    try:
        highs = df['high'].values; lows = df['low'].values
        ph = argrelextrema(highs, np.greater_equal, order=k)[0]
        pl = argrelextrema(lows, np.less_equal, order=k)[0]
        if len(ph) < 2 or len(pl) < 2: return 0, "RANGE"
        lh, ph_prev = highs[ph[-1]], highs[ph[-2]]
        ll, pl_prev = lows[pl[-1]], lows[pl[-2]]
        if lh > ph_prev and ll > pl_prev: return 1, "UP"
        if lh < ph_prev and ll < pl_prev: return -1, "DOWN"
        return 0, "RANGE"
    except:
        return 0, "N/A"

# ==================== MARKET STRUCTURE (BoS / CHoCH) ====================
def detect_ms(df, k=3):
    res = {'bos': 0, 'choch': 0, 'status': 'No Structure'}
    if df is None or len(df) < 50: return res
    try:
        highs = df['high'].values; lows = df['low'].values; close = df['close'].values
        ph = argrelextrema(highs, np.greater_equal, order=k)[0]
        pl = argrelextrema(lows, np.less_equal, order=k)[0]
        if len(ph) < 2 or len(pl) < 2: return res
        last_ph = highs[ph[-1]]; prev_ph = highs[ph[-2]]
        last_pl = lows[pl[-1]]; prev_pl = lows[pl[-2]]
        cur = close[-1]
        up = last_ph > prev_ph and last_pl > prev_pl
        dn = last_ph < prev_ph and last_pl < prev_pl
        if up:
            if cur > last_ph:
                res['bos'] = 1; res['status'] = 'Bullish BoS (Continuation Up)'
            elif cur < last_pl:
                res['choch'] = -1; res['status'] = 'Bearish CHoCH (Reversal Down)'
        elif dn:
            if cur < last_pl:
                res['bos'] = -1; res['status'] = 'Bearish BoS (Continuation Down)'
            elif cur > last_ph:
                res['choch'] = 1; res['status'] = 'Bullish CHoCH (Reversal Up)'
        else:
            res['status'] = 'Ranging'
        return res
    except:
        return res

# ==================== ORDER BLOCKS ====================
def detect_obs(df, lookback=50):
    obs = []
    if df is None or len(df) < 20: return obs
    try:
        d = df.tail(lookback).reset_index(drop=True)
        body = (d['close'] - d['open']).abs()
        atr_v = body.rolling(14).mean().iloc[-1]
        if pd.isna(atr_v): atr_v = body.mean()
        for i in range(2, len(d)-2):
            if d['close'].iloc[i] < d['open'].iloc[i]:
                if (d['close'].iloc[i+1] > d['high'].iloc[i] and
                    (d['close'].iloc[i+2] - d['open'].iloc[i+2]) > 1.5*atr_v):
                    obs.append({'type': 'bullish', 'top': d['high'].iloc[i], 'bottom': d['low'].iloc[i]})
            if d['close'].iloc[i] > d['open'].iloc[i]:
                if (d['close'].iloc[i+1] < d['low'].iloc[i] and
                    (d['open'].iloc[i+2] - d['close'].iloc[i+2]) > 1.5*atr_v):
                    obs.append({'type': 'bearish', 'top': d['high'].iloc[i], 'bottom': d['low'].iloc[i]})
        return obs[-3:]
    except:
        return []

# ==================== FVG ====================
def detect_fvg(df, lookback=50):
    fvgs = []
    if df is None or len(df) < 5: return fvgs
    try:
        d = df.tail(lookback).reset_index(drop=True)
        for i in range(2, len(d)):
            if d['low'].iloc[i] > d['high'].iloc[i-2]:
                fvgs.append({'type': 'bullish', 'top': d['low'].iloc[i], 'bottom': d['high'].iloc[i-2]})
            if d['high'].iloc[i] < d['low'].iloc[i-2]:
                fvgs.append({'type': 'bearish', 'top': d['low'].iloc[i-2], 'bottom': d['high'].iloc[i]})
        return fvgs[-3:]
    except:
        return []

# ==================== LIQUIDATION ZONES (Estimated) ====================
def liq_zones(df, k=5):
    zones = []
    if df is None or len(df) < 30: return zones
    try:
        d = df.tail(100).reset_index(drop=True)
        highs = d['high'].values; lows = d['low'].values
        ph = argrelextrema(highs, np.greater_equal, order=k)[0]
        pl = argrelextrema(lows, np.less_equal, order=k)[0]
        for idx in ph[-3:]:
            zones.append({'type': 'buy_side', 'price': float(highs[idx])})
        for idx in pl[-3:]:
            zones.append({'type': 'sell_side', 'price': float(lows[idx])})
        return zones
    except:
        return []

# ==================== SIGNAL ENGINE ====================
def make_signal(price, rsi, macd, funding, cvd, vp, absorb, oi_ch, tf, ms, obs, fvgs, liqz, atr):
    bull = 0; bear = 0
    reasons = []; details = []

    # 1. RSI
    if rsi:
        if rsi < 30: bull += 3; reasons.append("RSI Oversold"); details.append(f"RSI={rsi:.1f} → Buy")
        elif rsi < 40: bull += 1.5; details.append(f"RSI={rsi:.1f} → Mild Bullish")
        elif rsi > 70: bear += 3; reasons.append("RSI Overbought"); details.append(f"RSI={rsi:.1f} → Sell")
        elif rsi > 60: bear += 1.5; details.append(f"RSI={rsi:.1f} → Mild Bearish")
        else: details.append(f"RSI={rsi:.1f} → Neutral")

    # 2. MACD
    if macd:
        if macd['hist'] > 0 and macd['macd'] > macd['signal']:
            bull += 2; reasons.append("MACD Bullish"); details.append("MACD: Bullish")
        elif macd['hist'] < 0 and macd['macd'] < macd['signal']:
            bear += 2; reasons.append("MACD Bearish"); details.append("MACD: Bearish")
        elif macd['hist'] > 0: bull += 1; details.append("MACD: Mild Bullish")
        elif macd['hist'] < 0: bear += 1; details.append("MACD: Mild Bearish")

    # 3. Funding
    if funding is not None:
        if funding > 0.05: bear += 2; reasons.append("High Funding"); details.append(f"Funding={funding:.4f}% → Bearish")
        elif funding < -0.05: bull += 2; reasons.append("Neg Funding"); details.append(f"Funding={funding:.4f}% → Bullish")
        else: details.append(f"Funding={funding:.4f}% → Neutral")

    # 4. CVD
    if cvd is not None:
        if cvd > 0: bull += 2; reasons.append("Buyers (CVD+)"); details.append(f"CVD={cvd:,.0f} → Buyers")
        else: bear += 2; reasons.append("Sellers (CVD-)"); details.append(f"CVD={cvd:,.0f} → Sellers")

    # 5. VP
    if vp and price:
        if price < vp['VAL']: bull += 2; reasons.append("Below VA"); details.append("Price<VAL → Bullish")
        elif price > vp['VAH']: bear += 2; reasons.append("Above VA"); details.append("Price>VAH → Bearish")
        elif price < vp['POC']: bull += 0.5; details.append("Price<POC → Slight Bullish")
        else: bear += 0.5; details.append("Price>POC → Slight Bearish")

    # 6. Absorption
    if absorb and absorb > 1.5:
        reasons.append("High Absorption"); details.append(f"Absorption={absorb:.2f}")

    # 7. OI
    if oi_ch is not None:
        if oi_ch > 2: bull += 1.5; reasons.append("OI Increasing"); details.append(f"OI +{oi_ch:.2f}%")
        elif oi_ch < -2: bear += 1.5; reasons.append("OI Decreasing"); details.append(f"OI {oi_ch:.2f}%")
        else: details.append(f"OI {oi_ch:.2f}% (Neutral)")

    # 8. MTF Alignment
    t15 = tf.get('15m', (0, 'N/A'))[0]
    t1h = tf.get('1H', (0, 'N/A'))[0]
    t4h = tf.get('4H', (0, 'N/A'))[0]
    if t15 == 1 and t1h == 1 and t4h == 1:
        bull += 3; reasons.append("MTF All Bullish"); details.append("MTF: 15m↑ 1H↑ 4H↑")
    elif t15 == -1 and t1h == -1 and t4h == -1:
        bear += 3; reasons.append("MTF All Bearish"); details.append("MTF: 15m↓ 1H↓ 4H↓")
    else:
        details.append(f"MTF: 15m={tf['15m'][1]} 1H={tf['1H'][1]} 4H={tf['4H'][1]}")

    # 9. Market Structure
    if ms['bos'] == 1: bull += 2.5; reasons.append("Bullish BoS"); details.append(ms['status'])
    elif ms['bos'] == -1: bear += 2.5; reasons.append("Bearish BoS"); details.append(ms['status'])
    elif ms['choch'] == 1: bull += 2; reasons.append("Bullish CHoCH"); details.append(ms['status'])
    elif ms['choch'] == -1: bear += 2; reasons.append("Bearish CHoCH"); details.append(ms['status'])
    else: details.append(f"Structure: {ms['status']}")

    # 10. Order Blocks
    if obs and price:
        in_bull_ob = any(o['type']=='bullish' and o['bottom'] <= price <= o['top'] for o in obs)
        in_bear_ob = any(o['type']=='bearish' and o['bottom'] <= price <= o['top'] for o in obs)
        if in_bull_ob: bull += 2; reasons.append("In Bullish OB"); details.append("Price inside Bullish OB")
        if in_bear_ob: bear += 2; reasons.append("In Bearish OB"); details.append("Price inside Bearish OB")
        if not in_bull_ob and not in_bear_ob: details.append("Not in OB")

    # 11. FVG
    if fvgs and price:
        in_bfvg = any(f['type']=='bullish' and f['bottom'] <= price <= f['top'] for f in fvgs)
        in_sfvg = any(f['type']=='bearish' and f['bottom'] <= price <= f['top'] for f in fvgs)
        if in_bfvg: bull += 1.5; reasons.append("In Bullish FVG"); details.append("Price in Bullish FVG")
        if in_sfvg: bear += 1.5; reasons.append("In Bearish FVG"); details.append("Price in Bearish FVG")

    # 12. Liquidation Zones
    if liqz and price:
        for z in liqz:
            if abs(price - z['price'])/price < 0.005:
                if z['type'] == 'sell_side':
                    bull += 1; reasons.append("Near Sell-Side Liq"); details.append(f"Sell-side liq @ ${z['price']:,.4f}")
                else:
                    bear += 1; reasons.append("Near Buy-Side Liq"); details.append(f"Buy-side liq @ ${z['price']:,.4f}")

    # Probability
    total = bull + bear
    if total == 0:
        bp, sp, hp = 33.3, 33.3, 33.4
    else:
        bp = (bull/total)*100; sp = (bear/total)*100; hp = max(0, 100-bp-sp)
    if bp > sp and bp > 50:
        bp = min(95, bp+10); sp = max(5, sp-5)
    elif sp > bp and sp > 50:
        sp = min(95, sp+10); bp = max(5, bp-5)
    hp = max(0, 100-bp-sp)

    if bp >= 55: signal, conf = "BUY", bp
    elif sp >= 55: signal, conf = "SELL", sp
    else: signal, conf = "WAIT / HOLD", hp

    entry = price
    if atr is None: atr = entry * 0.02
    if signal == "BUY":
        sl = entry - 1.5*atr; tp1 = entry + 2*atr; tp2 = entry + 3.5*atr; tp3 = entry + 5*atr
    elif signal == "SELL":
        sl = entry + 1.5*atr; tp1 = entry - 2*atr; tp2 = entry - 3.5*atr; tp3 = entry - 5*atr
    else:
        sl = tp1 = tp2 = tp3 = None

    return {'signal': signal, 'conf': conf, 'bp': round(bp,1), 'sp': round(sp,1), 'hp': round(hp,1),
            'bull': round(bull,1), 'bear': round(bear,1), 'reasons': reasons, 'details': details,
            'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3}

# ==================== MAIN ====================
tickers = fetch_tickers()
if tickers.empty:
    st.error("MEXC data fetch nahi hua. Refresh karo.")
    st.stop()

sorted_df = tickers.sort_values('riseFallRate', ascending=False)
all_sym = sorted_df['symbol'].tolist()

st.markdown("### 🔍 Coin Select Karo")
c1, c2 = st.columns([2,2])
with c1:
    di = all_sym.index('BTC_USDT') if 'BTC_USDT' in all_sym else 0
    sel_dd = st.selectbox("Dropdown:", all_sym, index=di, key="dd")
with c2:
    sel_txt = st.text_input("Ya likho (e.g. ETH_USDT):", key="txt")

if sel_txt.strip() and sel_txt.strip().upper() in all_sym:
    SYM = sel_txt.strip().upper()
else:
    SYM = sel_dd

st.markdown(f"### ✅ Selected: `{SYM}`")
ccy = SYM.replace("_USDT", "")

# Top movers
st.header("📈 Top 10 Gainers & Losers")
g1, g2 = st.columns(2)
with g1:
    st.subheader("🟢 Gainers")
    g = sorted_df.head(10)[['symbol','lastPrice','riseFallRate','volume24']].copy()
    g.columns = ['Symbol','Price','24h %','Volume']
    g['24h %'] = g['24h %'].apply(lambda x: f"+{x:.2f}%")
    st.dataframe(g, use_container_width=True, hide_index=True)
with g2:
    st.subheader("🔴 Losers")
    l = sorted_df.tail(10).sort_values('riseFallRate')[['symbol','lastPrice','riseFallRate','volume24']].copy()
    l.columns = ['Symbol','Price','24h %','Volume']
    l['24h %'] = l['24h %'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(l, use_container_width=True, hide_index=True)

# Coin info
st.header(f"🎯 Analysis: {SYM}")
crow = tickers[tickers['symbol'] == SYM]
if not crow.empty:
    co = crow.iloc[0]
    m1, m2 = st.columns(2)
    m1.metric("Price", f"${co['lastPrice']:,.4f}", f"{co['riseFallRate']:.2f}%")
    m2.metric("24h Volume", f"${co['volume24']:,.0f}")

# Fetch all data
klines_15m = fetch_klines(SYM, "Min15", 200)
klines_1h = fetch_klines(SYM, "Min60", 200)
klines_4h = fetch_klines(SYM, "Hour4", 200)

if klines_1h.empty:
    st.warning(f"{SYM} ka data nahi mila.")
    st.stop()

price = float(klines_1h['close'].iloc[-1])
rsi = calc_rsi(klines_1h)
macd = calc_macd(klines_1h)
vp = calc_vp(klines_1h)
absorb = calc_absorption(klines_1h)
atr = calc_atr(klines_1h)
funding = fetch_funding(SYM)
cvd_df = fetch_cvd(ccy)
cvd = float(cvd_df['CVD'].iloc[-1]) if not cvd_df.empty else None
oi_df = fetch_oi(ccy)
oi_ch = None
if not oi_df.empty and len(oi_df) > 1:
    oi_ch = ((oi_df['oi'].iloc[-1] - oi_df['oi'].iloc[0]) / oi_df['oi'].iloc[0]) * 100

# MTF trends
tf_trends = {
    '15m': get_trend(klines_15m),
    '1H': get_trend(klines_1h),
    '4H': get_trend(klines_4h),
}

# SMC
ms = detect_ms(klines_1h)
obs = detect_obs(klines_1h)
fvgs = detect_fvg(klines_1h)
liqz = liq_zones(klines_1h)

sig = make_signal(price, rsi, macd, funding, cvd, vp, absorb, oi_ch,
                  tf_trends, ms, obs, fvgs, liqz, atr)

# ==================== DISPLAY ====================
st.header(f"🎯 Signal: {SYM}")

p1, p2, p3 = st.columns(3)
p1.metric("🟢 BUY", f"{sig['bp']}%")
p2.metric("🔴 SELL", f"{sig['sp']}%")
p3.metric("🟡 HOLD", f"{sig['hp']}%")

chart = go.Figure(go.Bar(
    x=['BUY','SELL','HOLD'],
    y=[sig['bp'], sig['sp'], sig['hp']],
    marker_color=['#00ff88','#ff4444','#ffaa00'],
    text=[f"{sig['bp']}%", f"{sig['sp']}%", f"{sig['hp']}%"],
    textposition='auto'
))
chart.update_layout(height=220, template="plotly_dark", showlegend=False, yaxis_range=[0,100])
st.plotly_chart(chart, use_container_width=True)

f1, f2 = st.columns([1,1])
with f1:
    if "BUY" in sig['signal']: st.success(f"## 🟢 {sig['signal']}")
    elif "SELL" in sig['signal']: st.error(f"## 🔴 {sig['signal']}")
    else: st.warning(f"## 🟡 {sig['signal']}")
    st.metric("Confidence", f"{sig['conf']:.1f}%")
with f2:
    st.metric("Entry", f"${sig['entry']:,.4f}")
    if sig['sl']: st.metric("Stop Loss", f"${sig['sl']:,.4f}")

if sig['tp1']:
    t1, t2, t3 = st.columns(3)
    t1.metric("TP1 (2x ATR)", f"${sig['tp1']:,.4f}")
    t2.metric("TP2 (3.5x ATR)", f"${sig['tp2']:,.4f}")
    t3.metric("TP3 (5x ATR)", f"${sig['tp3']:,.4f}")
    st.caption(f"ATR: ${atr:,.4f} — SL/TP ATR-based (3-12h trades)")

with st.expander("📋 Full Analysis"):
    st.markdown(f"**Bull Points:** {sig['bull']}  |  **Bear Points:** {sig['bear']}")
    st.markdown("**Parameter Breakdown:**")
    for d in sig['details']:
        st.write(f"• {d}")
    st.markdown("**Key Reasons:**")
    for r in sig['reasons']:
        st.write(f"✅ {r}")

# MTF
st.header("⏱️ Multi-Timeframe Trends")
mt1, mt2, mt3 = st.columns(3)
mt1.metric("15m", tf_trends['15m'][1])
mt2.metric("1H", tf_trends['1H'][1])
mt3.metric("4H", tf_trends['4H'][1])

# SMC
st.header("🏗️ Market Structure (SMC)")
st.info(f"**{ms['status']}**")

if obs:
    st.markdown("**Order Blocks (latest 3):**")
    ob_rows = [{"Type": o['type'], "Top": f"${o['top']:,.4f}", "Bottom": f"${o['bottom']:,.4f}"} for o in obs]
    st.dataframe(pd.DataFrame(ob_rows), use_container_width=True, hide_index=True)
else:
    st.caption("Koi Order Block nahi mila.")

if fvgs:
    st.markdown("**Fair Value Gaps (latest 3):**")
    fvg_rows = [{"Type": f['type'], "Top": f"${f['top']:,.4f}", "Bottom": f"${f['bottom']:,.4f}"} for f in fvgs]
    st.dataframe(pd.DataFrame(fvg_rows), use_container_width=True, hide_index=True)
else:
    st.caption("Koi FVG nahi mila.")

if liqz:
    st.markdown("**Liquidation Zones (estimated):**")
    lz_rows = [{"Type": z['type'], "Price": f"${z['price']:,.4f}"} for z in liqz]
    st.dataframe(pd.DataFrame(lz_rows), use_container_width=True, hide_index=True)

# Indicators
st.header("📊 Indicators")
i1, i2, i3, i4 = st.columns(4)
with i1:
    if rsi: st.metric("RSI (1H)", f"{rsi:.1f}")
with i2:
    if macd: st.metric("MACD", f"{macd['macd']:.4f}", f"Hist: {macd['hist']:.4f}")
with i3:
    if funding is not None: st.metric("Funding", f"{funding:.4f}%")
with i4:
    if cvd is not None: st.metric("CVD", f"{cvd:,.0f}")
if oi_ch is not None:
    st.metric("OI Change", f"{oi_ch:.2f}%")

# VP
if vp:
    st.header("📊 Volume Profile")
    v1, v2, v3 = st.columns(3)
    v1.metric("POC", f"${vp['POC']:,.4f}")
    v2.metric("VAH", f"${vp['VAH']:,.4f}")
    v3.metric("VAL", f"${vp['VAL']:,.4f}")

# Chart
st.header(f"📈 {SYM} (1H)")
fig = go.Figure(data=[go.Candlestick(
    x=klines_1h['time'], open=klines_1h['open'], high=klines_1h['high'],
    low=klines_1h['low'], close=klines_1h['close']
)])
if vp:
    fig.add_hline(y=vp['POC'], line_dash="dash", line_color="yellow", annotation_text="POC")
    fig.add_hline(y=vp['VAH'], line_dash="dot", line_color="green", annotation_text="VAH")
    fig.add_hline(y=vp['VAL'], line_dash="dot", line_color="red", annotation_text="VAL")
for o in obs:
    fig.add_hrect(y0=o['bottom'], y1=o['top'],
                  fillcolor="rgba(0,255,136,0.15)" if o['type']=='bullish' else "rgba(255,68,68,0.15)",
                  line_width=0, annotation_text=f"{o['type'][:4]} OB")
for f in fvgs:
    fig.add_hrect(y0=f['bottom'], y1=f['top'],
                  fillcolor="rgba(0,150,255,0.10)",
                  line_width=0, annotation_text="FVG")
fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Sidebar
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
1. RSI
2. MACD
3. Funding Rate
4. CVD (OKX)
5. Volume Profile
6. Absorption
7. Open Interest
8. MTF Trend (15m/1H/4H)
9. Market Structure (BoS/CHoCH)
10. Order Blocks
11. Fair Value Gaps
12. Liquidation Zones (est.)
""")
st.sidebar.header("🚫 Not Added")
st.sidebar.markdown("- News (needs CryptoPanic API key)")
st.sidebar.button("🔄 Refresh", on_click=lambda: st.cache_data.clear())
