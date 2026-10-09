import streamlit as st
import requests
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import plotly.graph_objects as go
from datetime import datetime
from scipy.signal import argrelextrema

st.set_page_config(page_title="Pro Crypto Dashboard + AI", layout="wide")

# ==================== GROQ CHATBOT SETUP ====================
try:
    from groq import Groq
    _groq_ok = True
except Exception:
    _groq_ok = False

try:
    _groq_key = st.secrets.get("GROQ_API_KEY", None)
except Exception:
    _groq_key = None

# ==================== MODE SELECTOR ====================
st.sidebar.markdown("## 🧭 Mode")
MODE = st.sidebar.radio(
    "Kaunsa mode?",
    ["📊 Crypto Dashboard", "🤖 AI Chatbot"],
    index=0,
    key="mode_selector"
)

# ==================== CHATBOT MODE ====================
if MODE == "🤖 AI Chatbot":
    st.title("🤖 AI Chatbot")
    if not _groq_ok:
        st.error("`groq` library install nahi hai.")
        st.stop()
    if not _groq_key:
        st.error("GROQ_API_KEY nahi mila. Secrets mein daalo.")
        st.code('GROQ_API_KEY = "gsk_xxxxxxxxxxxxx"', language="toml")
        st.stop()

    client = Groq(api_key=_groq_key)
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Chatbot Settings")

    model = st.sidebar.selectbox("Model", [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct"
    ], index=0)

    temp = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tok = st.sidebar.slider("Max Response Length", 256, 4096, 1024, 256)
    sys_prompt = st.sidebar.text_area("System Prompt",
        "You are a helpful AI assistant. Answer clearly and concisely.", height=100)

    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Apna sawal likho..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": sys_prompt}]
                    + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    temperature=temp, max_tokens=max_tok, stream=True,
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"Error: {e}"
                placeholder.error(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.sidebar.markdown("---")
    st.sidebar.caption("Powered by Groq (free tier)")
    st.stop()

# ==================== DASHBOARD MODE ====================
st.title("📊 Professional Crypto Dashboard v6")
st.caption(f"Last: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | MEXC + OKX | Spot+Futures | Liquidity Sweep")

MEXC_BASE = "https://contract.mexc.com"
MEXC_SPOT = "https://api.mexc.com"
OKX_BASE = "https://www.okx.com"

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
def fetch_spot_klines(symbol_base, interval="60m", limit=200):
    """MEXC Spot API (symbol without _USDT)"""
    try:
        spot_sym = symbol_base + "USDT"
        r = requests.get(f"{MEXC_SPOT}/api/v3/klines",
                        params={"symbol": spot_sym, "interval": interval, "limit": limit}, timeout=15)
        r.raise_for_status()
        d = r.json()
        if isinstance(d, list) and len(d) > 0:
            df = pd.DataFrame(d, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'trades', 'tb_base', 'tb_quote', 'ignore'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            return df[['time', 'open', 'high', 'low', 'close', 'volume']]
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
def fetch_spot_cvd(ccy):
    """OKX Spot Taker Volume"""
    try:
        r = requests.get(f"{OKX_BASE}/api/v5/rubik/stat/taker-volume",
                        params={"ccy": ccy, "instType": "SPOT", "period": "1H"}, timeout=15)
        d = r.json()
        if d.get('code') == '0' and d.get('data'):
            df = pd.DataFrame(d['data'], columns=['timestamp', 'sellVol', 'buyVol'])
            df['buyVol'] = pd.to_numeric(df['buyVol'])
            df['sellVol'] = pd.to_numeric(df['sellVol'])
            df['SpotCVD'] = df['buyVol'] - df['sellVol']
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

# ==================== TREND ====================
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

# ==================== MARKET STRUCTURE ====================
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
                res['bos'] = 1; res['status'] = 'Bullish BoS'
            elif cur < last_pl:
                res['choch'] = -1; res['status'] = 'Bearish CHoCH'
        elif dn:
            if cur < last_pl:
                res['bos'] = -1; res['status'] = 'Bearish BoS'
            elif cur > last_ph:
                res['choch'] = 1; res['status'] = 'Bullish CHoCH'
        else:
            res['status'] = 'Ranging'
        return res
    except:
        return res

# ==================== TRENDLINE BREAKOUT ====================
def detect_trendline(df, window=5):
    res = {'breakout': 0, 'status': 'No Trendline', 'details': []}
    try:
        d = df.copy().reset_index(drop=True)
        highs = d['high'].values
        lows = d['low'].values
        close = d['close'].values

        ph = argrelextrema(highs, np.greater_equal, order=window)[0]
        pl = argrelextrema(lows, np.less_equal, order=window)[0]

        if len(ph) < 2 and len(pl) < 2:
            return res

        cur_idx = len(d) - 1
        cur_close = close[-1]
        prev_close = close[-2]

        if len(ph) >= 2:
            i1, i2 = ph[-2], ph[-1]
            if i2 > i1:
                slope = (highs[i2] - highs[i1]) / (i2 - i1)
                intercept = highs[i2] - slope * i2
                tl_cur = slope * cur_idx + intercept
                tl_prev = slope * (cur_idx - 1) + intercept
                if prev_close <= tl_prev and cur_close > tl_cur:
                    res['breakout'] = 1
                    res['status'] = 'Bullish Trendline Breakout'
                    res['details'] = [f"Broke above: ${tl_cur:,.4f}"]
                    return res

        if len(pl) >= 2:
            i1, i2 = pl[-2], pl[-1]
            if i2 > i1:
                slope = (lows[i2] - lows[i1]) / (i2 - i1)
                intercept = lows[i2] - slope * i2
                tl_cur = slope * cur_idx + intercept
                tl_prev = slope * (cur_idx - 1) + intercept
                if prev_close >= tl_prev and cur_close < tl_cur:
                    res['breakout'] = -1
                    res['status'] = 'Bearish Trendline Breakout'
                    res['details'] = [f"Broke below: ${tl_cur:,.4f}"]
                    return res

        res['status'] = 'No Trendline Breakout (Skipped)'
        return res
    except:
        return res

# ==================== LIQUIDITY SWEEP (NEW) ====================
def detect_liquidity_sweep(df, lookback=20):
    """
    Detect liquidity sweep (stop hunt):
    Bullish sweep: long lower wick below recent low, then close back above
    Bearish sweep: long upper wick above recent high, then close back below
    """
    res = {'sweep': 0, 'status': 'No Sweep', 'details': []}
    try:
        if df is None or len(df) < lookback + 5:
            return res
        d = df.copy().reset_index(drop=True)
        recent = d.tail(lookback)

        for i in range(len(d) - 5, len(d)):
            candle = d.iloc[i]
            body = abs(candle['close'] - candle['open'])
            rng = candle['high'] - candle['low']
            if rng == 0: continue
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            upper_wick = candle['high'] - max(candle['open'], candle['close'])

            # Bullish sweep: long lower wick + close above open, and swept a prior low
            if lower_wick > 2 * body and lower_wick / rng > 0.6:
                prev_low = recent['low'].iloc[:-5].min() if len(recent) > 5 else candle['low']
                if candle['low'] < prev_low:
                    res['sweep'] = 1
                    res['status'] = 'Bullish Liquidity Sweep (Stop Hunt)'
                    res['details'] = [f"Swept low @ ${candle['low']:,.4f}", "Reversal up likely"]
                    return res

            # Bearish sweep: long upper wick + close below open
            if upper_wick > 2 * body and upper_wick / rng > 0.6:
                prev_high = recent['high'].iloc[:-5].max() if len(recent) > 5 else candle['high']
                if candle['high'] > prev_high:
                    res['sweep'] = -1
                    res['status'] = 'Bearish Liquidity Sweep (Stop Hunt)'
                    res['details'] = [f"Swept high @ ${candle['high']:,.4f}", "Reversal down likely"]
                    return res

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

# ==================== LIQUIDATION ZONES ====================
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
def make_signal(price, rsi, macd, funding, cvd, spot_cvd, vp, absorb, oi_ch, tf, ms, obs, fvgs, liqz, tlb, sweep, atr):
    bull = 0; bear = 0
    reasons = []; details = []

    if rsi:
        if rsi < 30: bull += 3; reasons.append("RSI Oversold")
        elif rsi < 40: bull += 1.5
        elif rsi > 70: bear += 3; reasons.append("RSI Overbought")
        elif rsi > 60: bear += 1.5
        details.append(f"RSI={rsi:.1f}")

    if macd:
        if macd['hist'] > 0 and macd['macd'] > macd['signal']:
            bull += 2; reasons.append("MACD Bullish")
        elif macd['hist'] < 0 and macd['macd'] < macd['signal']:
            bear += 2; reasons.append("MACD Bearish")
        details.append("MACD analyzed")

    if funding is not None:
        if funding > 0.05: bear += 2; reasons.append("High Funding")
        elif funding < -0.05: bull += 2; reasons.append("Neg Funding")
        details.append(f"Funding={funding:.4f}%")

    if cvd is not None:
        if cvd > 0: bull += 2; reasons.append("Futures Buyers (CVD+)")
        else: bear += 2; reasons.append("Futures Sellers (CVD-)")
        details.append(f"Futures CVD={cvd:,.0f}")

    # ===== SPOT CVD (NEW) =====
    if spot_cvd is not None:
        if spot_cvd > 0:
            bull += 1.5; reasons.append("Spot Buyers Active")
            details.append(f"Spot CVD={spot_cvd:,.0f} → Spot buying pressure")
        else:
            bear += 1.5; reasons.append("Spot Sellers Active")
            details.append(f"Spot CVD={spot_cvd:,.0f} → Spot selling pressure")

        # Divergence check
        if cvd is not None:
            if cvd < 0 and spot_cvd > 0:
                bull += 1
                reasons.append("Spot/Futures Bullish Divergence")
                details.append("Spot buyers > Futures sellers → Bullish signal")
            elif cvd > 0 and spot_cvd < 0:
                bear += 1
                reasons.append("Spot/Futures Bearish Divergence")
                details.append("Spot sellers > Futures buyers → Bearish signal")

    if vp and price:
        if price < vp['VAL']: bull += 2; reasons.append("Below VA")
        elif price > vp['VAH']: bear += 2; reasons.append("Above VA")
        elif price < vp['POC']: bull += 0.5
        else: bear += 0.5
        details.append(f"VP: POC={vp['POC']:.4f}")

    if absorb and absorb > 1.5:
        reasons.append("High Absorption")

    if oi_ch is not None:
        if oi_ch > 2: bull += 1.5; reasons.append("OI Increasing")
        elif oi_ch < -2: bear += 1.5; reasons.append("OI Decreasing")
        details.append(f"OI {oi_ch:.2f}%")

    t15 = tf.get('15m', (0, 'N/A'))[0]
    t1h = tf.get('1H', (0, 'N/A'))[0]
    t4h = tf.get('4H', (0, 'N/A'))[0]
    if t15 == 1 and t1h == 1 and t4h == 1:
        bull += 3; reasons.append("MTF All Bullish")
    elif t15 == -1 and t1h == -1 and t4h == -1:
        bear += 3; reasons.append("MTF All Bearish")
    details.append(f"MTF: 15m={tf['15m'][1]} 1H={tf['1H'][1]} 4H={tf['4H'][1]}")

    if ms['bos'] == 1: bull += 2.5; reasons.append("Bullish BoS")
    elif ms['bos'] == -1: bear += 2.5; reasons.append("Bearish BoS")
    elif ms['choch'] == 1: bull += 2; reasons.append("Bullish CHoCH")
    elif ms['choch'] == -1: bear += 2; reasons.append("Bearish CHoCH")
    details.append(f"Structure: {ms['status']}")

    if obs and price:
        in_bull_ob = any(o['type']=='bullish' and o['bottom'] <= price <= o['top'] for o in obs)
        in_bear_ob = any(o['type']=='bearish' and o['bottom'] <= price <= o['top'] for o in obs)
        if in_bull_ob: bull += 2; reasons.append("In Bullish OB")
        if in_bear_ob: bear += 2; reasons.append("In Bearish OB")

    if fvgs and price:
        in_bfvg = any(f['type']=='bullish' and f['bottom'] <= price <= f['top'] for f in fvgs)
        in_sfvg = any(f['type']=='bearish' and f['bottom'] <= price <= f['top'] for f in fvgs)
        if in_bfvg: bull += 1.5; reasons.append("In Bullish FVG")
        if in_sfvg: bear += 1.5; reasons.append("In Bearish FVG")

    if liqz and price:
        for z in liqz:
            if abs(price - z['price'])/price < 0.005:
                if z['type'] == 'sell_side':
                    bull += 1; reasons.append("Near Sell-Side Liq")
                else:
                    bear += 1; reasons.append("Near Buy-Side Liq")

    if tlb['breakout'] == 1:
        bull += 3; reasons.append("Bullish Trendline Breakout")
    elif tlb['breakout'] == -1:
        bear += 3; reasons.append("Bearish Trendline Breakout")
    details.append(f"Trendline: {tlb['status']}")

    # ===== LIQUIDITY SWEEP (NEW) =====
    if sweep['sweep'] == 1:
        bull += 2.5
        reasons.append("Bullish Liquidity Sweep (Stop Hunt)")
        details.append("Sweep: Bullish reversal after stop hunt")
        for d in sweep['details']:
            details.append(f"  {d}")
    elif sweep['sweep'] == -1:
        bear += 2.5
        reasons.append("Bearish Liquidity Sweep (Stop Hunt)")
        details.append("Sweep: Bearish reversal after stop hunt")
        for d in sweep['details']:
            details.append(f"  {d}")
    else:
        details.append(f"Sweep: {sweep['status']}")

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

    strong_buy = bull >= 12 and bull >= bear * 2
    strong_sell = bear >= 12 and bear >= bull * 2

    if strong_buy:
        signal, conf = "STRONG BUY 🚀", bp
    elif strong_sell:
        signal, conf = "STRONG SELL 🔻", sp
    elif bp >= 55:
        signal, conf = "BUY", bp
    elif sp >= 55:
        signal, conf = "SELL", sp
    else:
        signal, conf = "WAIT / HOLD", hp

    entry = price
    if atr is None: atr = entry * 0.02
    if "BUY" in signal:
        sl = entry - 1.5*atr; tp1 = entry + 2*atr; tp2 = entry + 3.5*atr; tp3 = entry + 5*atr
    elif "SELL" in signal:
        sl = entry + 1.5*atr; tp1 = entry - 2*atr; tp2 = entry - 3.5*atr; tp3 = entry - 5*atr
    else:
        sl = tp1 = tp2 = tp3 = None

    return {'signal': signal, 'conf': conf, 'bp': round(bp,1), 'sp': round(sp,1), 'hp': round(hp,1),
            'bull': round(bull,1), 'bear': round(bear,1), 'reasons': reasons, 'details': details,
            'entry': entry, 'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'strong_buy': strong_buy, 'strong_sell': strong_sell}

# ==================== MAIN ====================
tickers = fetch_tickers()
if tickers.empty:
    st.error("MEXC data fetch nahi hua. Refresh karo.")
    st.stop()

sorted_df = tickers.sort_values('riseFallRate', ascending=False)
all_sym = sorted_df['symbol'].tolist()

# ==================== TRENDLINE SCANNER (UPDATED) ====================
st.header("📐 Trendline Scanner — Alag Filters Ke Saath")
st.caption("15m aur 1H ko alag-alag filter karo. Sirf woh coins dikhayega jahan selected timeframe par breakout ho.")

sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    scan_top_n = st.selectbox("Top coins scan", [50, 100, 200, 300], index=2)
with sc2:
    tf_filter = st.selectbox("Timeframe Filter", ["15m Only", "1H Only", "Both (15m + 1H)"], index=0)
with sc3:
    min_volume_m = st.number_input("Min 24h Volume ($M)", min_value=0.0, value=10.0, step=1.0)
with sc4:
    min_24h_change = st.number_input("Min 24h Change (%)", min_value=-100.0, value=-50.0, step=5.0)

if st.button("🚀 Scan Trendlines", type="primary"):
    scan_df = sorted_df.copy()
    scan_df['vol_m'] = scan_df['volume24'] / 1_000_000
    scan_df = scan_df[scan_df['vol_m'] >= min_volume_m]
    scan_df = scan_df[scan_df['riseFallRate'] >= min_24h_change]
    scan_df = scan_df.head(scan_top_n)
    scan_symbols = scan_df['symbol'].tolist()

    if not scan_symbols:
        st.warning("Koi coin filter pass nahi hua.")
    else:
        progress = st.progress(0)
        status = st.empty()
        bull_results = []
        bear_results = []

        for i, sym in enumerate(scan_symbols):
            status.text(f"Scanning {sym}... ({i+1}/{len(scan_symbols)})")
            try:
                df_15m = fetch_klines(sym, "Min15", 200)
                df_1h = fetch_klines(sym, "Min60", 200)

                if df_15m.empty and df_1h.empty:
                    progress.progress((i+1)/len(scan_symbols))
                    continue

                tlb_15m = detect_trendline(df_15m) if not df_15m.empty else {'breakout': 0, 'status': 'N/A', 'details': []}
                tlb_1h = detect_trendline(df_1h) if not df_1h.empty else {'breakout': 0, 'status': 'N/A', 'details': []}
                sweep_15m = detect_liquidity_sweep(df_15m) if not df_15m.empty else {'sweep': 0, 'status': 'N/A', 'details': []}
                sweep_1h = detect_liquidity_sweep(df_1h) if not df_1h.empty else {'sweep': 0, 'status': 'N/A', 'details': []}

                # Determine match based on filter
                if tf_filter == "15m Only":
                    bull_match = tlb_15m['breakout'] == 1 or sweep_15m['sweep'] == 1
                    bear_match = tlb_15m['breakout'] == -1 or sweep_15m['sweep'] == -1
                    tf_detail = "15m"
                elif tf_filter == "1H Only":
                    bull_match = tlb_1h['breakout'] == 1 or sweep_1h['sweep'] == 1
                    bear_match = tlb_1h['breakout'] == -1 or sweep_1h['sweep'] == -1
                    tf_detail = "1H"
                else:  # Both
                    bull_match = (tlb_15m['breakout'] == 1 and tlb_1h['breakout'] == 1)
                    bear_match = (tlb_15m['breakout'] == -1 and tlb_1h['breakout'] == -1)
                    tf_detail = "15m+1H"

                row = scan_df[scan_df['symbol'] == sym].iloc[0]

                if bull_match:
                    bull_results.append({
                        'Symbol': sym, 'Price': f"${row['lastPrice']:,.4f}",
                        '24h %': f"{row['riseFallRate']:+.2f}%",
                        'Volume ($M)': f"{row['vol_m']:.1f}",
                        'Timeframe': tf_detail,
                        '15m': tlb_15m['status'][:25] + (' (Sweep!)' if sweep_15m['sweep'] == 1 else ''),
                        '1H': tlb_1h['status'][:25] + (' (Sweep!)' if sweep_1h['sweep'] == 1 else '')
                    })
                elif bear_match:
                    bear_results.append({
                        'Symbol': sym, 'Price': f"${row['lastPrice']:,.4f}",
                        '24h %': f"{row['riseFallRate']:+.2f}%",
                        'Volume ($M)': f"{row['vol_m']:.1f}",
                        'Timeframe': tf_detail,
                        '15m': tlb_15m['status'][:25] + (' (Sweep!)' if sweep_15m['sweep'] == -1 else ''),
                        '1H': tlb_1h['status'][:25] + (' (Sweep!)' if sweep_1h['sweep'] == -1 else '')
                    })
            except Exception:
                pass
            progress.progress((i+1)/len(scan_symbols))

        status.text(f"✅ Scan Complete! {len(scan_symbols)} coins scanned.")
        st.markdown("---")
        st.markdown(f"### 📊 Results: {len(bull_results)} Bullish | {len(bear_results)} Bearish")

        if bull_results:
            st.subheader(f"🟢 Bullish ({len(bull_results)})")
            st.dataframe(pd.DataFrame(bull_results), use_container_width=True, hide_index=True)
        else:
            st.info("Koi Bullish nahi mila.")

        if bear_results:
            st.subheader(f"🔴 Bearish ({len(bear_results)})")
            st.dataframe(pd.DataFrame(bear_results), use_container_width=True, hide_index=True)
        else:
            st.info("Koi Bearish nahi mila.")

st.markdown("---")

# ==================== COIN SELECTOR ====================
st.markdown("### 🔍 Coin Select Karo (Detailed Analysis)")
c1, c2 = st.columns([2,2])
with c1:
    di = all_sym.index('BTC_USDT') if 'BTC_USDT' in all_sym else 0
    sel_dd = st.selectbox("Dropdown:", all_sym, index=di, key="dd")
with c2:
    sel_txt = st.text_input("Ya likho:", key="txt")

if sel_txt.strip() and sel_txt.strip().upper() in all_sym:
    SYM = sel_txt.strip().upper()
else:
    SYM = sel_dd

st.markdown(f"### ✅ Selected: `{SYM}`")
ccy = SYM.replace("_USDT", "")

# Volume Warning
crow = tickers[tickers['symbol'] == SYM]
if not crow.empty:
    co = crow.iloc[0]
    vol_m = co['volume24'] / 1_000_000
    m1, m2 = st.columns(2)
    m1.metric("Price", f"${co['lastPrice']:,.4f}", f"{co['riseFallRate']:.2f}%")
    m2.metric("24h Volume", f"${co['volume24']:,.0f}")
    if vol_m < 20:
        st.error(f"⚠️ **LOW VOLUME WARNING:** Sirf ${vol_m:.1f}M volume hai. $20M se kam volume wale coins mein bot ke signals reliable nahi hote. Trade mat karo.")

# Fetch all data
klines_15m = fetch_klines(SYM, "Min15", 200)
klines_1h = fetch_klines(SYM, "Min60", 200)
klines_4h = fetch_klines(SYM, "Hour4", 200)
spot_klines = fetch_spot_klines(ccy, "60m", 200)

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
spot_cvd_df = fetch_spot_cvd(ccy)
spot_cvd = float(spot_cvd_df['SpotCVD'].iloc[-1]) if not spot_cvd_df.empty else None
oi_df = fetch_oi(ccy)
oi_ch = None
if not oi_df.empty and len(oi_df) > 1:
    oi_ch = ((oi_df['oi'].iloc[-1] - oi_df['oi'].iloc[0]) / oi_df['oi'].iloc[0]) * 100

tf_trends = {'15m': get_trend(klines_15m), '1H': get_trend(klines_1h), '4H': get_trend(klines_4h)}
ms = detect_ms(klines_1h)
obs = detect_obs(klines_1h)
fvgs = detect_fvg(klines_1h)
liqz = liq_zones(klines_1h)
tlb = detect_trendline(klines_1h)
sweep = detect_liquidity_sweep(klines_1h)

sig = make_signal(price, rsi, macd, funding, cvd, spot_cvd, vp, absorb, oi_ch,
                  tf_trends, ms, obs, fvgs, liqz, tlb, sweep, atr)

# Signal display
st.header(f"🎯 Signal: {SYM}")
p1, p2, p3 = st.columns(3)
p1.metric("🟢 BUY", f"{sig['bp']}%")
p2.metric("🔴 SELL", f"{sig['sp']}%")
p3.metric("🟡 HOLD", f"{sig['hp']}%")

chart = go.Figure(go.Bar(x=['BUY','SELL','HOLD'], y=[sig['bp'], sig['sp'], sig['hp']],
    marker_color=['#00ff88','#ff4444','#ffaa00'],
    text=[f"{sig['bp']}%", f"{sig['sp']}%", f"{sig['hp']}%"], textposition='auto'))
chart.update_layout(height=220, template="plotly_dark", showlegend=False, yaxis_range=[0,100])
st.plotly_chart(chart, use_container_width=True)

if tlb['breakout'] != 0:
    st.warning(f"⚠️ **{tlb['status']}**")
if sweep['sweep'] != 0:
    st.info(f"🎯 **{sweep['status']}**")

f1, f2 = st.columns([1,1])
with f1:
    if "STRONG BUY" in sig['signal']: st.success(f"## 🚀 {sig['signal']}")
    elif "STRONG SELL" in sig['signal']: st.error(f"## 🔻 {sig['signal']}")
    elif "BUY" in sig['signal']: st.success(f"## 🟢 {sig['signal']}")
    elif "SELL" in sig['signal']: st.error(f"## 🔴 {sig['signal']}")
    else: st.warning(f"## 🟡 {sig['signal']}")
    st.metric("Confidence", f"{sig['conf']:.1f}%")
with f2:
    st.metric("Entry", f"${sig['entry']:,.4f}")
    if sig['sl']: st.metric("Stop Loss", f"${sig['sl']:,.4f}")

if sig['tp1']:
    t1, t2, t3 = st.columns(3)
    t1.metric("TP1", f"${sig['tp1']:,.4f}")
    t2.metric("TP2", f"${sig['tp2']:,.4f}")
    t3.metric("TP3", f"${sig['tp3']:,.4f}")

with st.expander("📋 Full Analysis"):
    st.markdown(f"**Bull:** {sig['bull']} | **Bear:** {sig['bear']}")
    for d in sig['details']:
        st.write(f"• {d}")
    st.markdown("**Reasons:**")
    for r in sig['reasons']:
        st.write(f"✅ {r}")

# Spot vs Futures
st.header("🔄 Spot vs Futures Analysis")
if spot_cvd is not None and cvd is not None:
    sf1, sf2 = st.columns(2)
    sf1.metric("Spot CVD", f"{spot_cvd:,.0f}")
    sf2.metric("Futures CVD", f"{cvd:,.0f}")
    if spot_cvd > 0 and cvd < 0:
        st.success("🟢 Spot buyers > Futures sellers → Bullish Divergence")
    elif spot_cvd < 0 and cvd > 0:
        st.error("🔴 Spot sellers > Futures buyers → Bearish Divergence")
    else:
        st.info("Spot aur Futures same direction mein hain.")
else:
    st.caption("Spot CVD data nahi mila is coin ke liye.")

# MTF
st.header("⏱️ MTF Trends")
mt1, mt2, mt3 = st.columns(3)
mt1.metric("15m", tf_trends['15m'][1])
mt2.metric("1H", tf_trends['1H'][1])
mt3.metric("4H", tf_trends['4H'][1])

# Trendline + Sweep
st.header("📐 Trendline & Sweep")
if tlb['breakout'] == 1: st.success(f"🟢 {tlb['status']}")
elif tlb['breakout'] == -1: st.error(f"🔴 {tlb['status']}")
else: st.info(f"⚪ {tlb['status']}")

if sweep['sweep'] == 1: st.success(f"🟢 {sweep['status']}")
elif sweep['sweep'] == -1: st.error(f"🔴 {sweep['status']}")
else: st.info(f"⚪ {sweep['status']}")

# SMC
st.header("🏗️ Market Structure")
st.info(f"**{ms['status']}**")

if obs:
    ob_rows = [{"Type": o['type'], "Top": f"${o['top']:,.4f}", "Bottom": f"${o['bottom']:,.4f}"} for o in obs]
    st.dataframe(pd.DataFrame(ob_rows), use_container_width=True, hide_index=True)

if fvgs:
    fvg_rows = [{"Type": f['type'], "Top": f"${f['top']:,.4f}", "Bottom": f"${f['bottom']:,.4f}"} for f in fvgs]
    st.dataframe(pd.DataFrame(fvg_rows), use_container_width=True, hide_index=True)

# Indicators
st.header("📊 Indicators")
i1, i2, i3, i4 = st.columns(4)
with i1:
    if rsi: st.metric("RSI", f"{rsi:.1f}")
with i2:
    if macd: st.metric("MACD", f"{macd['macd']:.4f}")
with i3:
    if funding is not None: st.metric("Funding", f"{funding:.4f}%")
with i4:
    if cvd is not None: st.metric("CVD", f"{cvd:,.0f}")

if vp:
    v1, v2, v3 = st.columns(3)
    v1.metric("POC", f"${vp['POC']:,.4f}")
    v2.metric("VAH", f"${vp['VAH']:,.4f}")
    v3.metric("VAL", f"${vp['VAL']:,.4f}")

# Chart
st.header(f"📈 {SYM} (1H)")
fig = go.Figure(data=[go.Candlestick(
    x=klines_1h['time'], open=klines_1h['open'], high=klines_1h['high'],
    low=klines_1h['low'], close=klines_1h['close'])])
if vp:
    fig.add_hline(y=vp['POC'], line_dash="dash", line_color="yellow", annotation_text="POC")
    fig.add_hline(y=vp['VAH'], line_dash="dot", line_color="green", annotation_text="VAH")
    fig.add_hline(y=vp['VAL'], line_dash="dot", line_color="red", annotation_text="VAL")
fig.update_layout(xaxis_rangeslider_visible=False, height=500, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.header("✅ Active Parameters")
st.sidebar.markdown("""
1. RSI
2. MACD
3. Funding Rate
4. Futures CVD
5. **Spot CVD** (NEW)
6. **Spot/Futures Divergence** (NEW)
7. Volume Profile
8. Absorption
9. Open Interest
10. MTF Trend
11. Market Structure
12. Order Blocks
13. Fair Value Gaps
14. Liquidation Zones
15. Trendline Breakout
16. **Liquidity Sweep** (NEW)
17. Strong Buy/Sell Labels
18. Scanner (15m/1H separate)
""")
st.sidebar.button("🔄 Refresh", on_click=lambda: st.cache_data.clear())
