"""
多信号组合评分系统（V4 — 全面优化版）
- 动态权重：大盘趋势/震荡/下跌自动调整
- 共振加分：多指标一致时加分
- 量价配合：上涨带量加分，上涨缩量扣分
- 板块强度：热点板块加分
- 大盘共振：大盘强时加分，大盘弱时扣分
"""
import logging
from typing import Optional
import numpy as np
from src.signals import _sma, _ema, _calc_rsi

logger = logging.getLogger(__name__)

# 基础权重（可优化调整）
BASE_WEIGHTS = {"均线": 25, "RSI": 20, "MACD": 20, "成交量": 15, "资金流向": 20}

# 大盘模式权重
MODE_WEIGHTS = {
    "trending": {"均线": 30, "RSI": 15, "MACD": 20, "成交量": 15, "资金流向": 20},
    "declining": {"均线": 15, "RSI": 15, "MACD": 15, "成交量": 20, "资金流向": 30},
}


_market_cache = {"mode": "ranging", "desc": "震荡市", "chg": 0, "time": 0}

def _get_market_mode() -> tuple[str, str, float]:
    import time as _time
    now = _time.time()
    if now - _market_cache["time"] < 60:
        return _market_cache["mode"], _market_cache["desc"], _market_cache["chg"]
    try:
        from src.fetcher import fetch_market_index
        idx = fetch_market_index("000001")
        if not idx:
            _market_cache["time"] = now
            return "ranging", "震荡市", 0
        chg = idx.get("change_pct", 0)
        mode = "trending" if chg > 0.5 else ("declining" if chg < -0.5 else "ranging")
        desc = {"trending": "趋势市", "ranging": "震荡市", "declining": "跌势市"}[mode]
        _market_cache.update({"mode": mode, "desc": desc, "chg": chg, "time": now})
        return mode, desc, chg
    except:
        return "ranging", "震荡市", 0


def _kdj(closes, highs, lows, n=9):
    """计算KDJ指标"""
    if len(closes) < n + 1: return 50, 50, 50
    recent_low = np.min(lows[-n:])
    recent_high = np.max(highs[-n:])
    if recent_high == recent_low: return 50, 50, 50
    rsv = (closes[-1] - recent_low) / (recent_high - recent_low) * 100
    k = rsv * 2/3 + 50 * 1/3
    d = k * 2/3 + 50 * 1/3
    j = 3 * k - 2 * d
    return k, d, j


_sector_cache = {"data": None, "time": 0}

def _get_sector_hot_score(code: str) -> tuple[int, str]:
    try:
        from src.sectors import get_sector_tag
        tag = get_sector_tag(code)
        if not tag:
            return 0, ""
        # 缓存板块数据，每10分钟刷新一次
        import time as _time
        now = _time.time()
        if _sector_cache["data"] is None or now - _sector_cache["time"] > 600:
            from src.fetcher import fetch_sector_performance
            _sector_cache["data"] = fetch_sector_performance()
            _sector_cache["time"] = now
        sectors = _sector_cache["data"]
        if not sectors:
            return 0, ""
        for s in sectors[:5]:
            if tag.replace("[", "").replace("]", "") in s.get("name", ""):
                return 5, f"热点板块+5"
        return 0, ""
    except:
        return 0, ""


def compute_score(closes: np.ndarray, volumes: np.ndarray,
                  price: float, fund_flow: Optional[dict] = None,
                  market_mode: str = "", code: str = "",
                  change_pct: float = 0) -> dict:
    if not market_mode:
        market_mode, mode_desc, market_chg = _get_market_mode()
    else:
        md = {"trending": "趋势市", "ranging": "震荡市", "declining": "跌势市"}
        mode_desc = md.get(market_mode, "震荡市")
        market_chg = 0

    weights = dict(MODE_WEIGHTS.get(market_mode, BASE_WEIGHTS))
    details = {}
    total = 0
    bullish_count = 0
    bearish_count = 0

    # ── 1. 均线趋势 ──
    ma_score = 0; ma_desc = "数据不足"
    if len(closes) >= 25:
        ma5 = _sma(closes, 5); ma10 = _sma(closes, 10); ma20 = _sma(closes, 20)
        valid = ~np.isnan(ma5) & ~np.isnan(ma10) & ~np.isnan(ma20)
        s5, s10, s20 = ma5[valid], ma10[valid], ma20[valid]
        if len(s5) > 0:
            c5, c10, c20 = s5[-1], s10[-1], s20[-1]
            if c5 > c10 > c20:
                ma_score = weights["均线"]; ma_desc = "多头排列"; bullish_count += 1
            elif c5 > c20 and c10 > c20:
                ma_score = int(weights["均线"] * 0.6); ma_desc = "均线向上"
            elif c5 < c10 < c20:
                ma_score = 0; ma_desc = "空头排列"; bearish_count += 1
            else:
                ma_score = int(weights["均线"] * 0.3); ma_desc = "均线交错"
            dev_ma5 = (price - c5) / c5 * 100
            if -1.5 < dev_ma5 < 0.5:
                ma_score += 5; ma_desc += "·回踩MA5"
    total += ma_score
    details["均线"] = {"score": ma_score, "desc": ma_desc, "weight": weights["均线"]}

    # ── 2. RSI ──
    rsi_score = 0; rsi_desc = "数据不足"
    if len(closes) >= 15:
        rsi = _calc_rsi(closes, 14)
        if rsi is not None:
            w = weights["RSI"]
            if 40 <= rsi <= 60:
                rsi_score = w; rsi_desc = f"RSI{rsi:.0f}适中"; bullish_count += 1
            elif 30 <= rsi < 40:
                rsi_score = int(w * 0.75); rsi_desc = f"RSI{rsi:.0f}偏低(反弹潜力)"
            elif 60 < rsi <= 70:
                rsi_score = int(w * 0.5); rsi_desc = f"RSI{rsi:.0f}偏强(注意回调)"
            elif rsi < 30:
                rsi_score = int(w * 0.25); rsi_desc = f"RSI{rsi:.0f}偏低(可能跌过头了)"
            else:
                rsi_score = 0; rsi_desc = f"RSI{rsi:.0f}偏高(短期涨太猛，别追高)"; bearish_count += 1
    total += rsi_score
    details["RSI"] = {"score": rsi_score, "desc": rsi_desc, "weight": weights["RSI"]}

    # ── 3. MACD ──
    macd_score = 0; macd_desc = "数据不足"
    if len(closes) >= 35:
        ema_f = _ema(closes, 12); ema_s = _ema(closes, 26)
        diff_arr = ema_f - ema_s; dea_arr = _ema(diff_arr, 9)
        valid = ~np.isnan(diff_arr) & ~np.isnan(dea_arr)
        dv, deav = diff_arr[valid], dea_arr[valid]
        if len(dv) > 1:
            cd, cdea = dv[-1], deav[-1]
            w = weights["MACD"]
            if cd > cdea and cd > 0:
                macd_score = w; macd_desc = "MACD多头"; bullish_count += 1
                if dv[-2] <= deav[-2]:
                    macd_desc = "MACD金叉"
            elif cd > cdea:
                macd_score = int(w * 0.5); macd_desc = "MACD转好"
            elif cd < cdea and cd < 0:
                macd_score = 0; macd_desc = "MACD死叉·空头"; bearish_count += 1
            else:
                macd_score = int(w * 0.25); macd_desc = "MACD偏弱"
    total += macd_score
    details["MACD"] = {"score": macd_score, "desc": macd_desc, "weight": weights["MACD"]}

    # ── 4. 成交量（含量价配合校验） ──
    vol_score = 0; vol_desc = "数据不足"
    if volumes is not None and len(volumes) >= 6:
        avg_v = np.mean(volumes[-5:])
        if avg_v > 0:
            vol_ratio = volumes[-1] / avg_v
            w = weights["成交量"]
            # 量价配合：检查价格方向
            price_chg = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
            if vol_ratio >= 1.5:
                if price_chg > 0:
                    vol_score = w; vol_desc = f"放量上涨{vol_ratio:.1f}倍✅"; bullish_count += 1
                else:
                    vol_score = int(w * 0.5); vol_desc = f"放量下跌{vol_ratio:.1f}倍⚠️"; bearish_count += 1
            elif vol_ratio >= 1.2:
                vol_score = int(w * 0.7); vol_desc = f"略放量{vol_ratio:.1f}倍"
            elif vol_ratio >= 0.8:
                vol_score = int(w * 0.3); vol_desc = "量能正常"
                if price_chg > 0: vol_score -= 3; vol_desc += "·缩量上涨❌"  # 无量上涨扣分
            else:
                vol_score = 0; vol_desc = "缩量"; bearish_count += 1
    total += vol_score
    details["成交量"] = {"score": vol_score, "desc": vol_desc, "weight": weights["成交量"]}

    # ── 5. 资金流向 ──
    ff_score = 0; ff_desc = "无数据"
    if fund_flow:
        main_net = fund_flow.get("main_force_net", 0)
        w = weights["资金流向"]
        if main_net > 50_000_000:
            ff_score = w; ff_desc = f"主力大额流入{main_net/1e8:.1f}亿💰"; bullish_count += 1
        elif main_net > 10_000_000:
            ff_score = int(w * 0.6); ff_desc = f"主力流入{main_net/1e8:.1f}亿"
        elif main_net > -10_000_000:
            ff_score = int(w * 0.25); ff_desc = "资金平衡"
        elif main_net > -50_000_000:
            ff_score = 0; ff_desc = f"主力流出{abs(main_net)/1e8:.1f}亿"; bearish_count += 1
        else:
            ff_score = int(w * -0.25); ff_desc = f"主力大幅流出{abs(main_net)/1e8:.1f}亿💸"; bearish_count += 1
    total += ff_score
    details["资金流向"] = {"score": ff_score, "desc": ff_desc, "weight": weights["资金流向"]}

    # ── 6. KDJ指标（新增） ──
    kdj_score = 0; kdj_desc = "数据不足"
    if len(closes) >= 10 and "high" in dir() or "highs" in dir():
        try:
            # 需要从外部传入highs/lows，这里如果没有就跳过
            pass
        except:
            pass
    # 简化版：用收盘价近似计算KDJ
    if len(closes) >= 14:
        lows_kdj = closes * 0.98  # 近似
        highs_kdj = closes * 1.02
        # 取最后N天的真实高低点
        n = 9
        recent_high = float(np.max(closes[-n:])) * 1.01
        recent_low = float(np.min(closes[-n:])) * 0.99
        if recent_high != recent_low:
            rsv = (closes[-1] - recent_low) / (recent_high - recent_low) * 100
            k = rsv * 2/3 + 50 * 1/3
            d = k * 2/3 + 50 * 1/3
            j = 3 * k - 2 * d
            if j < 20:
                kdj_score = 8; kdj_desc = f"KDJ偏低(J={j:.0f})🟢(可能跌过头了)"; bullish_count += 1
            elif j > 100:
                kdj_score = -5; kdj_desc = f"KDJ偏高(J={j:.0f})🔴(短期涨太猛了)"; bearish_count += 1
            elif k > d and k > 50:
                kdj_score = 5; kdj_desc = f"KDJ金叉(J={j:.0f})📈"; bullish_count += 1
            elif k < d and k < 50:
                kdj_score = -3; kdj_desc = f"KDJ死叉(J={j:.0f})📉"; bearish_count += 1
            else:
                kdj_score = 0; kdj_desc = f"KDJ中性(J={j:.0f})"
    total += kdj_score
    if kdj_desc != "数据不足":
        details["KDJ"] = {"score": kdj_score, "desc": kdj_desc}

    # ── 7.5 日内涨幅调整（涨太多扣分，跌是机会加分） ──
    if change_pct != 0:
        if change_pct > 4:
            intraday_adj = -15
            intraday_desc = f"今日涨{change_pct:+.1f}%过高❌-15"
        elif change_pct > 2:
            intraday_adj = -8
            intraday_desc = f"今日涨{change_pct:+.1f}%偏高⚠️-8"
        elif change_pct < -3:
            intraday_adj = 8
            intraday_desc = f"今日跌{change_pct:+.1f}%低吸机会🟢+8"
        elif change_pct < -1.5:
            intraday_adj = 4
            intraday_desc = f"今日跌{change_pct:+.1f}%可关注+4"
        else:
            intraday_adj = 0
            intraday_desc = ""
        if intraday_adj:
            total += intraday_adj
            details["日内调整"] = {"score": intraday_adj, "desc": intraday_desc}

    # ── 7. 板块强度加成 ──
    if code:
        sector_bonus, sector_desc = _get_sector_hot_score(code)
        if sector_bonus:
            total += sector_bonus
            details["板块】"] = {"score": sector_bonus, "desc": sector_desc}

    # ── 8. 共振加分 ──
    resonance = 0
    if bullish_count >= 3:
        resonance = 8
        details["共振"] = {"score": 8, "desc": f"{bullish_count}个指标共振看多✅"}
    elif bearish_count >= 3:
        resonance = -8
        details["共振"] = {"score": -8, "desc": f"{bearish_count}个指标共振看空❌"}
    total += resonance

    # ── 9. 大盘共振调整 ──
    market_adj = 0
    if market_chg > 1:
        market_adj = 5; total += 5
        details["大盘共振"] = {"score": 5, "desc": f"大盘上涨{market_chg:+.1f}%+5"}
    elif market_chg < -1:
        market_adj = -5; total += -5
        details["大盘共振"] = {"score": -5, "desc": f"大盘下跌{market_chg:+.1f}%-5"}

    # ── 综合判断 ──
    total = max(0, min(100, total))
    if total >= 70:
        action = "强烈买入"; suggestion = "适合买入"
    elif total >= 55:
        action = "可买入"; suggestion = "可考虑买入"
    elif total >= 40:
        action = "观望"; suggestion = "继续观望"
    else:
        action = "回避"; suggestion = "不建议买入"

    return {
        "score": total,
        "action": action,
        "suggestion": suggestion,
        "market_mode": mode_desc,
        "details": details,
    }
