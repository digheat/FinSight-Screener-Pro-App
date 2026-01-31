# (可省略) !pip install -q pandas requests python-dateutil tqdm

import os, time, warnings, requests, pandas as pd
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from functools import lru_cache
from typing import Dict, Tuple, Optional, List
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import streamlit as st

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # 沒裝就不顯示進度條

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ========= 基本設定 =========
API_KEY = os.getenv("POLYGON_API_KEY", "J_wZYB3rGZBaFv2tdyg21X1vmVXrMW21").strip()
assert API_KEY and all(ord(c) < 128 for c in API_KEY), "API_KEY 無效或含非 ASCII"
BASE = "https://api.polygon.io"

TARGET_YEAR = date.today().year - 1     # 年度對比：去年的 Q4
MAX_WORKERS = 16

# ========= HTTP Session（連線池 + 重試）=========
SESSION = requests.Session()
retries = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=retries)
SESSION.mount("https://", adapter); SESSION.mount("http://", adapter)

# 健檢
r = SESSION.get(f"{BASE}/v3/reference/tickers", params={"limit":1,"apiKey":API_KEY}, timeout=15)
r.raise_for_status()
print(f"✅ Polygon API OK | TARGET_YEAR={TARGET_YEAR}")

# ========= 通用工具 =========
def _get(url, params=None, retry=2, sleep=0.25):
    params = dict(params or {}); params["apiKey"] = API_KEY
    last = None
    for _ in range(retry+1):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, dict) and js.get("status") == "ERROR":
                raise RuntimeError(js.get("error") or "Polygon API error")
            return js
        except Exception as e:
            last = e; time.sleep(sleep)
    raise last

def _paged(url, params=None):
    params = dict(params or {}); params["apiKey"] = API_KEY
    out = []
    r = SESSION.get(url, params=params, timeout=30); r.raise_for_status()
    js = r.json()
    if isinstance(js, dict) and js.get("status") == "ERROR":
        raise RuntimeError(js.get("error") or "Polygon API error")
    out += js.get("results", []) or []
    while js.get("next_url"):
        nxt = js["next_url"]; base = nxt.split("?")[0]
        qs = nxt.split("?")[1] if "?" in nxt else ""; p2={}
        for kv in qs.split("&"):
            if not kv or kv.startswith("apiKey="): continue
            k, v = kv.split("=", 1); p2[k] = v
        p2["apiKey"] = API_KEY
        r = SESSION.get(base, params=p2, timeout=30); r.raise_for_status()
        js = r.json()
        if isinstance(js, dict) and js.get("status") == "ERROR":
            raise RuntimeError(js.get("error") or "Polygon API error")
        out += js.get("results", []) or []
    return out

def _to_date(s):
    try: return pd.to_datetime(s).date()
    except: return None

# ========= 參考與基本資料 =========
@lru_cache(maxsize=1_000)
def fetch_all_us_common():
    url = f"{BASE}/v3/reference/tickers"
    params={"market":"stocks","type":"CS","active":"true","locale":"us","limit":1000,"sort":"ticker"}
    out=[]
    while True:
        js=_get(url,params); out += [x["ticker"] for x in js.get("results",[]) if x.get("ticker")]
        nxt=js.get("next_url")
        if not nxt: break
        url=nxt.split("?")[0]
        params={kv.split("=")[0]:kv.split("=")[1] for kv in nxt.split("?")[1].split("&") if not kv.startswith("apiKey=")}
    return out

@lru_cache(maxsize=10_000)
def get_ref(ticker):
    return _get(f"{BASE}/v3/reference/tickers/{ticker}").get("results",{}) or {}

def classify_mc(market_cap):
    try:
        if market_cap is None: return None
        mc = float(market_cap)
        if not (mc > 0): return None
    except Exception:
        return None
    B = 1_000_000_000; M = 1_000_000
    if mc >= 200*B: return "Mega"
    if mc >= 10*B:  return "Large"
    if mc >= 2*B:   return "Mid"
    if mc >= 300*M: return "Small"
    if mc >= 50*M:  return "Micro"
    return "Nano"

# === 前一交易日日期 ===
def _most_recent_mkt_day():
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

# === 只取昨收（舊版相容保留）===
@lru_cache(maxsize=2)
def prev_close_map_for_all(group_day: str | None = None):
    day = group_day or _most_recent_mkt_day()
    url = f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{day}"
    js = _get(url, {"adjusted": "true"})
    mp = {}
    for r in js.get("results", []) or []:
        t = r.get("T") or r.get("ticker")
        c = r.get("c")
        if t and c is not None:
            mp[t] = float(c)
    return mp

# === 新增：昨收 + 成交量（同時快取）===
@lru_cache(maxsize=2)
def daily_agg_map_for_all(group_day: str | None = None):
    """
    回傳 {ticker: {"c": close, "v": volume}}，資料為上一交易日 (adjusted)
    """
    day = group_day or _most_recent_mkt_day()
    url = f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{day}"
    js = _get(url, {"adjusted": "true"})
    mp = {}
    for r in js.get("results", []) or []:
        t = r.get("T") or r.get("ticker")
        if not t:
            continue
        c = r.get("c"); v = r.get("v")
        mp[t] = {
            "c": (float(c) if c is not None else None),
            "v": (float(v) if v is not None else None),
        }
    return mp

def last_close_fast(ticker, mp=None):
    if mp and ticker in mp:
        val = mp[ticker]
        if isinstance(val, dict):
            c = val.get("c")
            return (float(c) if c is not None else None)
        return val  # 兼容舊版 map: 直接是 close 值
    js = _get(f"{BASE}/v2/aggs/ticker/{ticker}/prev")
    arr = js.get("results", [])
    return (arr[0]["c"] if arr else None)

@lru_cache(maxsize=50_000)
def polygon_avg_volume_last_n_days(ticker: str, ndays: int = 10) -> Optional[float]:
    """
    回傳過去 ndays 內（最多 ndays 個交易日）的平均日成交量。
    會多抓幾天以避開週末 / 假日。
    """
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=ndays * 2)  # 多抓避免週末
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/day/{start_str}/{end_str}"
    js = _get(url, {"adjusted": "true", "sort": "desc", "limit": ndays})
    arr = js.get("results", []) or []
    vols = [r.get("v") for r in arr if r.get("v") is not None]

    if not vols:
        return None
    return float(sum(vols) / len(vols))

# ========= Balance Sheets（季報）=========
def _pick_fin_value(obj, path_flat, path_nested):
    if path_flat in obj and obj[path_flat] is not None:
        try: return float(obj[path_flat])
        except: pass
    cur = obj
    try:
        for k in path_nested: cur = cur.get(k, {})
        if cur is not None: return float(cur)
    except: pass
    return None

def _pick_date(obj):
    for k in ("period_end","period_of_report_date","end_date","fiscal_period_end_date","reporting_date"):
        if obj.get(k):
            d = _to_date(obj[k]);
            if d: return d
    return None

@lru_cache(maxsize=50_000)
def polygon_bs_quarterly_map(ticker: str) -> Dict[Tuple[int,int], Dict[str, Optional[float]]]:
    url  = f"{BASE}/stocks/financials/v1/balance-sheets"
    fy_gte = date.today().year - 3
    rows = _paged(url, {
        "tickers": ticker, "timeframe": "quarterly",
        "fiscal_year.gte": fy_gte, "limit": 2000, "sort": "period_end.asc"
    })
    out = {}
    for r in rows:
        tk_list = r.get("tickers") or []
        if tk_list and ticker not in tk_list: continue
        y = r.get("fiscal_year"); q = r.get("fiscal_quarter")
        if not y or not q: continue
        dt = _pick_date(r)
        eq = _pick_fin_value(r, "total_equity", ["financials","balance_sheet","total_equity","value"])
        db = _pick_fin_value(r, "long_term_debt_and_capital_lease_obligations",
                                ["financials","balance_sheet","long_term_debt_and_capital_lease_obligations","value"])
        out[(int(y), int(q))] = {"total_equity": eq, "long_term_debt_and_capital_lease_obligations": db, "label_date": dt}
    return out

def polygon_bs_annual_q4(ticker: str, years: Tuple[int,int]) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    mp  = polygon_bs_quarterly_map(ticker)
    out = {}
    for y in years:
        v = mp.get((y,4)); out[y] = (v.get("total_equity"), v.get("long_term_debt_and_capital_lease_obligations")) if v else (None, None)
    return out

def polygon_bs_latest_quarter(ticker: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    mp = polygon_bs_quarterly_map(ticker)
    if not mp: return (None, None, None)
    y,q = sorted(mp.keys())[-1]; v = mp[(y,q)]
    lab = f"{y}Q{q}"
    return (lab, v.get("total_equity"), v.get("long_term_debt_and_capital_lease_obligations"))

# ========= Ratios（取 P/E 與 EPS_TTM）=========
@lru_cache(maxsize=50_000)
def polygon_ratios_latest_pe_and_eps(ticker: str) -> tuple[Optional[float], Optional[float]]:
    """
    回傳 (PE, EPS_TTM)。若 PE 缺，就用 Price/EPS_TTM 回推。
    """
    url  = f"{BASE}/stocks/financials/v1/ratios"
    rows = _paged(url, {"tickers": ticker, "order": "desc", "limit": 20})
    pe = eps_ttm = None
    for r in rows:
        ok = False
        if isinstance(r.get("tickers"), list): ok = (ticker in r["tickers"])
        elif isinstance(r.get("ticker"), str): ok = (r["ticker"] == ticker)
        else: ok = True
        if not ok:
            continue
        eps_ttm = _pick_fin_value(r, "earnings_per_share", ["financials","ratios","earnings_per_share","value"])
        pe      = _pick_fin_value(r, "price_to_earnings", ["financials","ratios","price_to_earnings","value"])
        break

    # 若 PE 缺且 EPS_TTM 有，用昨收價回推
    if (pe is None or pe <= 0) and eps_ttm not in (None, 0):
        px = last_close_fast(ticker)
        if px:
            try: pe = float(px) / float(eps_ttm)
            except: pass

    try: pe = float(pe) if pe is not None else None
    except: pe = None
    try: eps_ttm = float(eps_ttm) if eps_ttm is not None else None
    except: eps_ttm = None
    return pe, eps_ttm

# ========= 多口徑「淨利」抽取（供 YoY 與 TTM 計算）=========
def _pick_net_income_any(r) -> Optional[float]:
    CANDS = [
        ("net_income", ["financials","income_statement","net_income","value"]),
        ("net_income_loss", ["financials","income_statement","net_income_loss","value"]),
        ("net_income_loss_attributable_common_shareholders",
         ["financials","income_statement","net_income_loss_attributable_common_shareholders","value"]),
        ("net_income_loss_available_to_common_stockholders_basic",
         ["financials","income_statement","net_income_loss_available_to_common_stockholders_basic","value"]),
        ("consolidated_net_income_loss",
         ["financials","income_statement","consolidated_net_income_loss","value"]),
        ("profit_loss", ["financials","income_statement","profit_loss","value"]),
    ]
    for flat, nested in CANDS:
        try:
            if r.get(flat) is not None:
                return float(r.get(flat))
        except:
            pass
        v = _pick_fin_value(r, flat, nested)
        if v is not None:
            try: return float(v)
            except: pass
    return None

@lru_cache(maxsize=50_000)
def polygon_net_income_last_annual_and_quarterly(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """
    回傳 (net_income_lastyear, net_income_lastQ)
    - lastyear: 最新一個年度（annual）的淨利
    - lastQ   : 最新一個季度（quarterly）的淨利
    """
    url  = f"{BASE}/stocks/financials/v1/income-statements"

    # ---- 年度：找最新一年的淨利 ----
    rows_y = _paged(url, {
        "tickers": ticker, "timeframe": "annual",
        "fiscal_year.gte": date.today().year - 6,
        "limit": 2000, "sort": "period_end.asc"
    })
    annual = []
    for r in rows_y:
        tks = r.get("tickers") or []
        if tks and ticker not in tks:
            continue
        pe = _to_date(r.get("period_end")) or date(r.get("fiscal_year") or 1900, 12, 31)
        ni = _pick_net_income_any(r)
        if ni is not None:
            annual.append((pe, float(ni)))
    annual.sort(key=lambda x: x[0])
    net_income_lastyear = annual[-1][1] if annual else None

    # ---- 季度：找最新一季的淨利 ----
    rows_q = _paged(url, {
        "tickers": ticker, "timeframe": "quarterly",
        "fiscal_year.gte": date.today().year - 3,
        "limit": 2000, "sort": "period_end.asc"
    })
    quarterly = []
    for r in rows_q:
        tks = r.get("tickers") or []
        if tks and ticker not in tks:
            continue
        pe = _to_date(r.get("period_end")) or date(r.get("fiscal_year") or 1900, 12, 31)
        ni = _pick_net_income_any(r)
        if ni is not None:
            quarterly.append((pe, float(ni)))
    quarterly.sort(key=lambda x: x[0])
    net_income_lastQ = quarterly[-1][1] if quarterly else None

    return net_income_lastyear, net_income_lastQ

@lru_cache(maxsize=50_000)
def polygon_net_income_growth_yoy_pct(ticker: str) -> Optional[float]:
    """
    回傳百分比（如 5% -> 5.0）。
    先嘗試年度 YoY；若不行再用 TTM YoY（最近4季 vs 前4季）。
    """
    url  = f"{BASE}/stocks/financials/v1/income-statements"

    # ---- A) 年度 YoY ----
    rows_y = _paged(url, {
        "tickers": ticker, "timeframe": "annual",
        "fiscal_year.gte": date.today().year - 6,
        "limit": 2000, "sort": "period_end.asc"
    })
    annual = []
    for r in rows_y:
        tks = r.get("tickers") or []
        if tks and ticker not in tks:
            continue
        pe = _to_date(r.get("period_end")) or date(r.get("fiscal_year") or 1900, 12, 31)
        ni = _pick_net_income_any(r)
        if ni is not None:
            annual.append((pe, float(ni)))
    annual.sort(key=lambda x: x[0])

    if len(annual) >= 2:
        prev_ni = annual[-2][1]
        last_ni = annual[-1][1]
        if prev_ni is not None and prev_ni > 0:
            try:
                return float(((last_ni - prev_ni) / abs(prev_ni)) * 100.0)
            except:
                pass  # 落到 TTM 嘗試

    # ---- B) TTM YoY（近8季）----
    rows_q = _paged(url, {
        "tickers": ticker, "timeframe": "quarterly",
        "fiscal_year.gte": date.today().year - 3,
        "limit": 2000, "sort": "period_end.asc"
    })
    qvals = []
    for r in rows_q:
        tks = r.get("tickers") or []
        if tks and ticker not in tks:
            continue
        ni = _pick_net_income_any(r)
        if ni is not None:
            qvals.append(float(ni))
    if len(qvals) >= 8:
        last4 = sum(qvals[-4:])
        prev4 = sum(qvals[-8:-4])
        if prev4 > 0:
            try:
                return float(((last4 - prev4) / abs(prev4)) * 100.0)
            except:
                return None
    return None

# ========= 技術指標：MACD（依文件） & EMA =========
@lru_cache(maxsize=50_000)
def polygon_macd_latest_n(
    ticker: str,
    n: int = 2,
    *,
    timespan: str = "day",
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
    series_type: str = "close",
    order: str = "desc",
    expand_underlying: bool | str = False,
    timestamp: str | None = None,
) -> List[Dict]:
    params = {
        "timespan": timespan,
        "adjusted": "true",
        "short_window": short_window,
        "long_window": long_window,
        "signal_window": signal_window,
        "series_type": series_type,
        "order": order,
        "limit": max(1, min(5000, int(n))),
    }
    if expand_underlying:
        params["expand_underlying"] = "true"
    if timestamp:
        params["timestamp"] = timestamp

    js = _get(f"{BASE}/v1/indicators/macd/{ticker}", params)
    vals = (js.get("results", {}) or {}).get("values", []) or []

    out: List[Dict] = []
    for v in vals:
        try:
            out.append({
                "timestamp": int(v.get("timestamp")),
                "value": float(v.get("value")),
                "signal": float(v.get("signal")),
                "histogram": float(v.get("histogram")),
            })
        except Exception:
            continue
    return out

def macd_cross_flags_from_latest2(items: List[Dict]) -> Dict[str, bool]:
    EPS = 1e-12
    if len(items) < 2:
        return {k: False for k in [
            "above_zero","below_zero","golden_cross","death_cross","zero_cross_up","zero_cross_down"
        ]}
    cur, prev = items[0], items[1]
    m_now, s_now = cur["value"], cur["signal"]
    m_pre, s_pre = prev["value"], prev["signal"]
    h_now, h_pre = (m_now - s_now), (m_pre - s_pre)

    return {
        "above_zero": (m_now > 0 + EPS),
        "below_zero": (m_now < 0 - EPS),
        "golden_cross": (h_pre <= 0 + EPS) and (h_now > 0 + EPS),
        "death_cross":  (h_pre >= 0 - EPS) and (h_now < 0 - EPS),
        "zero_cross_up":   (m_pre <= 0 + EPS) and (m_now > 0 + EPS),
        "zero_cross_down": (m_pre >= 0 - EPS) and (m_now < 0 - EPS),
    }

@lru_cache(maxsize=50_000)
def polygon_ema_latest_value(ticker: str, window=200, timespan="day"):
    js = _get(f"{BASE}/v1/indicators/ema/{ticker}", {
        "timespan": timespan, "window": window, "series_type": "close",
        "adjusted": "true", "order": "desc", "limit": 1
    })
    vals = (js.get("results", {}) or {}).get("values", []) or []
    try:
        return float(vals[0]["value"])
    except:
        return None

# ========= Equity / Debt 雙旗標 =========
def equity_debt_flags_dual_polygon(ticker: str):
    y1, y2 = TARGET_YEAR, TARGET_YEAR - 1
    ann = polygon_bs_annual_q4(ticker, years=(y2, y1))
    eq_y2, db_y2 = ann.get(y2, (None, None))
    eq_y1, db_y1 = ann.get(y1, (None, None))
    _, eq_q, db_q = polygon_bs_latest_quarter(ticker)

    eq_up = (
        "Y" if (eq_y1 is not None and eq_y2 is not None and eq_q is not None
                and (eq_y1 > eq_y2) and (eq_q > eq_y1)) else "N"
    )
    db_down = (
        "Y" if (db_y1 is not None and db_y2 is not None and db_q is not None
                and (db_y1 < db_y2) and (db_q < db_y1)) else "N"
    )
    return eq_up, db_down

# ========= 單檔蒐集 =========
def collect_all_inputs_for_debug(ticker, PREV=None):
    row = OrderedDict()
    row["Symbol"] = ticker

    ref = get_ref(ticker) or {}
    row["Company"]  = ref.get("name")
    row["Sector"]   = ref.get("sic_sector")
    row["Industry"] = ref.get("sic_description")
    row["Market Cap"] = ref.get("market_cap")
    row["Market Cap Class"] = classify_mc(ref.get("market_cap"))

    # Price（昨收）
    px = last_close_fast(ticker, PREV)
    row["Price"] = round(px, 4) if px else None

    # 昨日成交量（與 Price 同一天）
    vol = None
    if PREV and ticker in PREV and isinstance(PREV[ticker], dict):
        vol = PREV[ticker].get("v")
    row["Volume (Prev Day)"] = (int(vol) if isinstance(vol, (int, float)) and vol == int(vol)
                                else (float(vol) if vol is not None else None))

    # 新增：過去 10 天平均成交量
    avg_vol_10d = polygon_avg_volume_last_n_days(ticker, ndays=10)
    row["AvgVolume_10D"] = (round(avg_vol_10d, 2) if avg_vol_10d is not None else None)

    # 年報/季報（balance sheets）
    y1, y2 = TARGET_YEAR, TARGET_YEAR-1
    ann = polygon_bs_annual_q4(ticker, years=(y2, y1))
    eq_y2 = db_y2 = eq_y1 = db_y1 = None
    if y2 in ann: eq_y2, db_y2 = ann[y2]
    if y1 in ann: eq_y1, db_y1 = ann[y1]
    row[f"Equity_{y2}"] = eq_y2; row[f"Debt_{y2}"] = db_y2
    row[f"Equity_{y1}"] = eq_y1; row[f"Debt_{y1}"] = db_y1

    qlab, eq_q, db_q = polygon_bs_latest_quarter(ticker)
    row["Latest Quarter Label"] = qlab
    row["Equity_LatestQ"] = eq_q; row["Debt_LatestQ"] = db_q

    row["Annual_Equity_Up(y1>y2)"] = (eq_y1 is not None and eq_y2 is not None and eq_y1 > eq_y2)
    row["Annual_Debt_Down(y1<y2)"] = (db_y1 is not None and db_y2 is not None and db_y1 < db_y2)
    row["Q_vs_Y1_Equity_Up"]       = (eq_q is not None and eq_y1 is not None and eq_q > eq_y1)
    row["Q_vs_Y1_Debt_Down"]       = (db_q is not None and db_y1 is not None and db_q < db_y1)

    # Equity / Debt 獨立旗標
    eq_up, db_down = equity_debt_flags_dual_polygon(ticker)
    row["Equity Up"] = eq_up
    row["Debt Down"] = db_down

    # 估值：P/E、EPS (TTM)、Net Income YoY Growth（%）、PE/G
    pe, eps_ttm = polygon_ratios_latest_pe_and_eps(ticker)
    row["P/E"] = pe
    row["EPS (TTM)"] = (round(eps_ttm, 6) if eps_ttm is not None else None)

    growth_pct = polygon_net_income_growth_yoy_pct(ticker)
    row["NetIncome YoY Growth (%)"] = (round(growth_pct, 4) if growth_pct is not None else None)
    row["PE/G"] = (round(pe / growth_pct, 4) if (pe is not None and growth_pct not in (None, 0)) else None)

    # 新增：淨利水準（年度 / 季度）＋是否 > 0
    ni_lastyear, ni_lastQ = polygon_net_income_last_annual_and_quarterly(ticker)
    row["net_income_lastyear"] = (round(ni_lastyear, 2) if ni_lastyear is not None else None)
    row["net_income_lastQ"] = (round(ni_lastQ, 2) if ni_lastQ is not None else None)
    row["net_income_lastyear>0?"] = (ni_lastyear is not None and ni_lastyear > 0)
    row["net_income_lastQ>0?"] = (ni_lastQ is not None and ni_lastQ > 0)

    # MACD（依文件參數）
    macd_items = polygon_macd_latest_n(
        ticker, n=2,
        timespan="day", short_window=12, long_window=26, signal_window=9,
        series_type="close", order="desc", expand_underlying=False
    )
    if macd_items:
        row["MACD (12,26,9)"] = macd_items[0]["value"]
        row["MACD_Signal"]    = macd_items[0]["signal"]
        row["MACD_Hist"]      = macd_items[0]["histogram"]
    else:
        row["MACD (12,26,9)"] = None
        row["MACD_Signal"]    = None
        row["MACD_Hist"]      = None

    flags = macd_cross_flags_from_latest2(macd_items)
    row["MACD_AboveZero"] = flags["above_zero"]
    row["MACD_BelowZero"] = flags["below_zero"]
    row["ZeroCross_Up"]   = flags["zero_cross_up"]
    row["ZeroCross_Down"] = flags["zero_cross_down"]
    row["GoldenCross"]    = flags["golden_cross"]
    row["DeathCross"]     = flags["death_cross"]

    # EMA200 與價格位置
    ema200 = polygon_ema_latest_value(ticker, window=200, timespan="day")
    row["EMA200"]         = ema200
    row["Price>EMA200"]   = (row.get("Price") is not None and ema200 is not None and row["Price"] > ema200)
    row["Price<EMA200"]   = (row.get("Price") is not None and ema200 is not None and row["Price"] < ema200)
    row["Trend_EMA200"]   = ("Up" if row["Price>EMA200"] else ("Down" if row["Price<EMA200"] else None))

    # 交易狀態（你的規則）
    buy  = (row["GoldenCross"] and row["MACD_BelowZero"] and row["Price>EMA200"])
    sell = (row["DeathCross"]  and row["MACD_AboveZero"] and row["Price<EMA200"])
    row["Status"] = ("BUY" if buy else ("SELL" if sell else "HOLD"))

    return row

def process_one_with_reason_and_inputs_v2(ticker, PREV=None):
    return collect_all_inputs_for_debug(ticker, PREV=PREV)

# ========= 全市場掃描（含進度條）=========
def run_full_market_inputs_with_reason(outfile="full_market_inputs_with_reason.csv",
                                       part_every=1500, max_workers=MAX_WORKERS,
                                       skip_share_classes=True, 
                                       limit: int | None = None): # <-- 新增 limit 參數

    # 整個流程使用 st.status 包裝，取代原有的頂部 print()
    with st.status("🚀 Starting full market scan...", expanded=True) as status:
        st.write("Fetching stock list...") # <-- 取代 print()

        syms = fetch_all_us_common()
        if skip_share_classes:
            syms = [s for s in syms if not any(s.endswith(f".{c}") for c in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))]
        
        # 應用 limit 參數 (由 11251.py 傳入，例如 10)
        if limit is not None and limit > 0:
            syms = syms[:limit]

        PREV = daily_agg_map_for_all()  # 批次昨收 + 成交量
        total = len(syms)
        st.write(f"✅ Got {total} tickers to process | MAX_WORKERS = {max_workers}") # <-- 取代 print()

        rows = []
        t0 = time.time()
        
        # 替換 tqdm：建立 st.progress
        progress = st.progress(0, text="Starting scan...")
        processed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(process_one_with_reason_and_inputs_v2, s, PREV): s for s in syms}
            for i, f in enumerate(as_completed(futs), 1):
                s = futs[f]
                try:
                    r = f.result()
                    if r: rows.append(r)
                except Exception as e:
                    st.write(f"[WARN] {s} error: {e}") # <-- 取代 print()

                # 替換 tqdm.update(1) 的邏輯
                processed = i
                pct = int((processed / total) * 100)
                progress.progress(pct, text=f"Processed {processed}/{total} tickers ({pct}%)")

                if part_every and i % part_every == 0:
                    df_part = pd.DataFrame(rows)
                    path = f"{outfile}.part_{i}.csv"
                    df_part.to_csv(path, index=False, encoding="utf-8")
                    st.write(f"💾 partial saved: {path} (rows={len(df_part)})") # <-- 取代 print()

        # 移除原有的 if bar: bar.close()

        st.write("Saving full CSV...")
        df = pd.DataFrame(rows)
        try:
            df.to_csv(outfile, index=False, encoding="utf-8")
            st.write(f"✅ saved full CSV: {outfile} | rows={len(df)}") # <-- 取代 print()
        except Exception as e:
            st.write(f"[ERROR] save full CSV: {e}") # <-- 取代 print()

        elapsed_min = (time.time() - t0) / 60
        st.write(f"⏱️ Total elapsed time: {elapsed_min:.1f} minutes")
        
        # 最終更新狀態
        progress.progress(100, text="✅ Completed all tasks")
        status.update(label="✅ Full market scan completed successfully!", state="complete")

    return df
# ----------------------------------------------------------------------
# 已移除底部所有自動執行（測試和全市場執行）的程式碼，確保作為模組載入時不會自動執行。
# ----------------------------------------------------------------------
