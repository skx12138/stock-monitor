"""
多信号组合评分系统（V5 — 大升级版）
参考 daily_stock_analysis 策略体系

核心改进：
  1. 7级趋势判断（强势多头→强势空头）+ 趋势强度 0-100
  2. 乖离率评分：回踩MA5加分，过度偏离扣分，强趋势补偿
  3. 量能5形态分类：缩量回调最佳，放量下跌最差
  4. MACD精细评分：零轴上金叉→金叉→上穿零轴→多头→空头→死叉
  5. RSI三周期(6/12/24)综合判断
  6. 箱体震荡检测：箱底买箱顶卖
  7. 回踩均线支撑加分
  8. 情绪周期参考（换手率热度）
"""
import logging
import re
from typing import Optional
import numpy as np
import requests
from src.signals import _sma, _ema, _calc_rsi

logger = logging.getLogger(__name__)

# ── 权重配置（可从策略优化器覆盖） ──
BASE_WEIGHTS = {"均线": 25, "RSI": 20, "MACD": 20, "成交量": 15, "资金流向": 20}
MODE_WEIGHTS = {
    "trending": {"均线": 30, "RSI": 15, "MACD": 20, "成交量": 15, "资金流向": 20},
    "declining": {"均线": 15, "RSI": 15, "MACD": 15, "成交量": 20, "资金流向": 30},
}

try:
    import json as _json, os as _os
    _opt_path = "strategy_optimizer.json"
    if _os.path.exists(_opt_path):
        with open(_opt_path, "r", encoding="utf-8") as _f:
            _opt = _json.load(_f)
        _ow = _opt.get("current_params", {}).get("weights", {})
        if _ow:
            if "base" in _ow and all(k in _ow["base"] for k in BASE_WEIGHTS):
                BASE_WEIGHTS.update(_ow["base"])
            if "trending" in _ow:
                MODE_WEIGHTS["trending"].update(_ow["trending"])
            if "declining" in _ow:
                MODE_WEIGHTS["declining"].update(_ow["declining"])
            logger.info("已加载优化器权重配置")
except Exception:
    pass

# ── 趋势状态枚举（7级） ──
TREND_LEVELS = {
    "strong_bull": "强势多头",
    "bull": "多头排列",
    "weak_bull": "弱势多头",
    "consolidation": "盘整",
    "weak_bear": "弱势空头",
    "bear": "空头排列",
    "strong_bear": "强势空头",
}

# ── 量能状态枚举（5种） ──
VOLUME_STATUS = {
    "shrink_down": "缩量回调",     # 最佳：洗盘特征
    "heavy_up": "放量上涨",        # 次之：多头强劲
    "normal": "量能正常",
    "shrink_up": "缩量上涨",       # 较差：无量上涨
    "heavy_down": "放量下跌",      # 最差：资金出逃
}

# ── MACD信号枚举（按强度排序） ──
MACD_SIGNAL = {
    "golden_cross_zero": "零轴上金叉",
    "golden_cross": "金叉",
    "cross_zero_up": "上穿零轴",
    "bullish": "多头",
    "neutral": "中性",
    "bearish": "空头",
    "cross_zero_down": "下穿零轴",
    "death_cross": "死叉",
}

# ── 买入信号枚举 ──
BUY_SIGNAL = {
    "strong_buy": "强烈买入",
    "buy": "可买入",
    "hold": "观望",
    "wait": "等待",
    "sell": "建议卖出",
    "strong_sell": "强烈卖出",
}

# ── 缓存 ──
_market_cache = {"mode": "ranging", "desc": "震荡市", "chg": 0, "time": 0}
_trend_cache = {"trend": "震荡", "yesterday_close": 0, "open": 0, "high": 0, "low": 0, "time": 0}
_sentiment_cache = {"level": 0, "label": "正常", "time": 0}
_sector_cache = {"data": None, "time": 0}


# ══════════════════════════════════════════════
#  大盘模式 & 日内趋势
# ══════════════════════════════════════════════

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


def get_intraday_trend() -> tuple[str, float]:
    """大盘日内趋势：高开低走/低开高走/单边涨/单边跌/冲高回落/探底回升/震荡"""
    import time as _time
    now = _time.time()
    if now - _trend_cache["time"] < 120:
        return _trend_cache["trend"], 0
    try:
        from src.fetcher import fetch_market_index
        url = "https://hq.sinajs.cn/list=s_sh000001"
        resp = requests.get(url, headers={
            "Referer": "https://finance.sina.com.cn",
            "User-Agent": "Mozilla/5.0",
        }, timeout=10)
        resp.encoding = "gbk"
        match = re.search(r'"(.*?)"', resp.text.strip())
        if not match:
            return "震荡", 0
        parts = match.group(1).split(",")
        if len(parts) < 6:
            return "震荡", 0
        open_price = float(parts[1]) if parts[1] else 0
        yesterday_close = float(parts[2]) if parts[2] else 0
        current = float(parts[3]) if parts[3] else 0
        high = float(parts[4]) if parts[4] else 0
        low = float(parts[5]) if parts[5] else 0
        if yesterday_close == 0 or open_price == 0:
            return "震荡", 0

        open_chg = (open_price / yesterday_close - 1) * 100
        now_chg = (current / yesterday_close - 1) * 100
        chg_from_open = (current / open_price - 1) * 100
        high_chg = (high / yesterday_close - 1) * 100
        drop_from_high = (current - high) / high * 100
        low_chg = (low / yesterday_close - 1) * 100
        rally_from_low = (current - low) / low * 100
        day_range = (high - low) / yesterday_close * 100

        if high_chg > 1.0 and drop_from_high < -1.5 and now_chg < 0:
            trend = "冲高回落💀"
            intensity = min(abs(drop_from_high) / 3, 1.0)
        elif low_chg < -1.0 and rally_from_low > 1.5 and now_chg >= -0.2:
            trend = "探底回升📈"
            intensity = min(rally_from_low / 3, 1.0)
        elif open_chg > 0.5 and chg_from_open < -1:
            trend = "高开低走📉"
            intensity = min(abs(chg_from_open) / 3, 1.0)
        elif open_chg < -0.5 and chg_from_open > 1:
            trend = "低开高走📈"
            intensity = min(chg_from_open / 3, 1.0)
        elif now_chg > 2 and chg_from_open > 1:
            trend = "单边上涨🚀"
            intensity = min(now_chg / 4, 1.0)
        elif now_chg < -2 and chg_from_open < -1:
            trend = "单边下跌💀"
            intensity = min(abs(now_chg) / 4, 1.0)
        # 剧烈震荡：日内振幅大(>2.5%)但最终涨跌幅小(<0.5%)，说明多空激烈博弈
        elif day_range > 2.5 and abs(now_chg) < 0.5:
            trend = "剧烈震荡⚡"
            intensity = min(day_range / 5, 1.0)
        else:
            trend = "震荡📊"
            intensity = min(day_range / 3, 0.5)
        _trend_cache.update({"trend": trend, "yesterday_close": yesterday_close,
                             "open": open_price, "high": high, "low": low, "time": now})
        return trend, intensity
    except:
        return "震荡", 0


def get_market_sentiment() -> tuple[int, str]:
    """市场情绪：-2恐慌 -1恐惧 0正常 +1贪婪 +2狂热"""
    import time as _time
    now = _time.time()
    if now - _sentiment_cache["time"] < 120:
        return _sentiment_cache["level"], _sentiment_cache["label"]
    try:
        from src.fetcher import fetch_market_breadth, fetch_market_index
        breadth = fetch_market_breadth()
        idx = fetch_market_index("000001")
        idx_chg = idx.get("change_pct", 0) if idx else 0
        limit_up = breadth.get("limit_up", 0)
        limit_down = breadth.get("limit_down", 0)
        extreme = limit_up - limit_down
        if extreme < -50 or (limit_down > 30 and idx_chg < -2):
            level, label = -2, "恐慌😱"
        elif extreme < -20 or (limit_down > 15 and idx_chg < -1.5):
            level, label = -1, "恐惧😰"
        elif extreme > 50 or (limit_up > 30 and idx_chg > 3):
            level, label = 2, "狂热🔥"
        elif extreme > 20 or (limit_up > 15 and idx_chg > 1.5):
            level, label = 1, "贪婪😏"
        else:
            level, label = 0, "正常😐"
        logger.info("市场情绪[%s] 涨停%d 跌停%d 大盘%.1f%%", label, limit_up, limit_down, idx_chg)
        _sentiment_cache.update({"level": level, "label": label, "time": now})
        return level, label
    except Exception as e:
        logger.debug("情绪判断失败: %s", e)
        return 0, "正常"


def _get_sector_hot_score(code: str) -> tuple[int, str]:
    try:
        from src.sectors import get_sector_tag
        tag = get_sector_tag(code)
        if not tag:
            return 0, ""
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
                return 5, "热点板块+5"
        return 0, ""
    except:
        return 0, ""


# ══════════════════════════════════════════════
#  箱体震荡检测
# ══════════════════════════════════════════════

def _detect_box(closes: np.ndarray, price: float) -> dict:
    """识别箱体震荡区间，返回 {in_box, top, bottom, box_width, position}

    逻辑：取近60日高点和低点，找至少触碰2~3次的价位作为箱体边界
    """
    result = {"in_box": False, "top": 0.0, "bottom": 0.0, "width_pct": 0.0, "position": ""}
    if len(closes) < 20:
        return result

    lookback = min(60, len(closes))
    segment = closes[-lookback:]

    # 用百分位数近似找箱体（20分位=支撑，80分位=阻力）
    low_pct = np.percentile(segment, 15)
    high_pct = np.percentile(segment, 85)

    # 检查价格是否在箱体内
    width = (high_pct - low_pct) / low_pct * 100 if low_pct > 0 else 0

    if width < 3 or width > 30:
        # 太窄(无操作空间)或太宽(趋势非箱体)
        return result

    result["top"] = round(high_pct, 2)
    result["bottom"] = round(low_pct, 2)
    result["width_pct"] = round(width, 1)

    if low_pct <= price <= high_pct:
        result["in_box"] = True
        # 判断在箱体的位置
        pos_pct = (price - low_pct) / (high_pct - low_pct) * 100
        if pos_pct <= 30:
            result["position"] = "箱底区域"
        elif pos_pct >= 70:
            result["position"] = "箱顶区域"
        else:
            result["position"] = "箱中区域"
    elif price > high_pct:
        result["position"] = "突破箱顶"
    else:
        result["position"] = "跌破箱底"

    return result


# ══════════════════════════════════════════════
#  V5 综合评分
# ══════════════════════════════════════════════

def compute_score(closes: np.ndarray, volumes: np.ndarray,
                  price: float, fund_flow: Optional[dict] = None,
                  market_mode: str = "", code: str = "",
                  change_pct: float = 0) -> dict:
    """V5综合评分系统（参考daily_stock_analysis策略框架）

    评分维度：
      1. 趋势（30分）：7级趋势 + 趋势强度
      2. 乖离率（20分）：回踩加分，偏离扣分，强趋势补偿
      3. 量能（15分）：缩量回调最佳，放量下跌最差
      4. MACD（15分）：零轴上金叉最强，死叉最弱
      5. RSI（10分）：三周期综合判断
      6. 箱体/支撑（10分）：箱底企稳加分，MA支撑加分
      7. 日内调整：涨太高扣分，跌是机会加分
      8. 板块强度 + 共振 + 大盘联动
    """
    if not market_mode:
        market_mode, mode_desc, market_chg = _get_market_mode()
    else:
        md = {"trending": "趋势市", "ranging": "震荡市", "declining": "跌势市"}
        mode_desc = md.get(market_mode, "震荡市")
        market_chg = 0

    weights = dict(MODE_WEIGHTS.get(market_mode, BASE_WEIGHTS))
    details = {}
    reasons = []
    risks = []
    score = 0

    # ── 0. 数据完整性检查 ──
    if closes is None or len(closes) < 20:
        return {
            "score": 0, "action": "回避", "suggestion": "数据不足",
            "market_mode": mode_desc,
            "details": {"数据": {"score": 0, "desc": "数据不足，无法分析"}},
        }

    # ── 1. 趋势评分（30分）—— 7级趋势 ──
    trend_info = _score_trend(closes, weights["均线"])
    score += trend_info["score"]
    details["均线"] = trend_info
    if trend_info["signal"] == "bullish":
        reasons.append(trend_info["desc"])
    elif trend_info["signal"] == "bearish":
        risks.append(trend_info["desc"])

    # ── 2. 乖离率评分（20分）—— 回踩MA5加分，追高扣分，强趋势补偿 ──
    bias_info = _score_bias(closes, price, trend_info["trend_status"])
    score += bias_info["score"]
    details["乖离率"] = bias_info
    if "回踩" in bias_info.get("desc", ""):
        reasons.append(bias_info["desc"])
    elif "追高" in bias_info.get("desc", ""):
        risks.append(bias_info["desc"])

    # ── 3. 量能评分（15分）—— 5形态 ──
    vol_info = _score_volume(closes, volumes, weights["成交量"])
    score += vol_info["score"]
    details["成交量"] = vol_info
    if vol_info["signal"] == "bullish":
        reasons.append(vol_info["desc"])
    elif vol_info["signal"] == "bearish":
        risks.append(vol_info["desc"])

    # ── 4. MACD评分（15分）—— 精细信号 ──
    macd_info = _score_macd(closes, weights["MACD"])
    score += macd_info["score"]
    details["MACD"] = macd_info
    if macd_info["signal"] == "bullish":
        reasons.append(macd_info["desc"])
    elif macd_info["signal"] == "bearish":
        risks.append(macd_info["desc"])

    # ── 5. RSI评分（10分）—— 三周期综合 ──
    rsi_info = _score_rsi(closes, weights["RSI"])
    score += rsi_info["score"]
    details["RSI"] = rsi_info
    if rsi_info["signal"] == "bullish":
        reasons.append(rsi_info["desc"])
    elif rsi_info["signal"] == "bearish":
        risks.append(rsi_info["desc"])

    # ── 6. 箱体/支撑评分（10分）—— 箱底买箱顶卖 ──
    box_info = _score_box_support(closes, price)
    score += box_info["score"]
    details["箱体支撑"] = box_info
    if box_info["signal"] == "bullish":
        reasons.append(box_info["desc"])
    elif box_info["signal"] == "bearish":
        risks.append(box_info["desc"])

    # ── 7. 资金流向（额外加减） ──
    ff_info = _score_fund_flow(fund_flow, weights.get("资金流向", 20))
    score += ff_info["score"]
    details["资金流向"] = ff_info
    if ff_info["signal"] == "bullish":
        reasons.append(ff_info["desc"])
    elif ff_info["signal"] == "bearish":
        risks.append(ff_info["desc"])

    # ── 日内涨幅调整 ──
    intra_info = _score_intraday(change_pct)
    score += intra_info["score"]
    if intra_info["score"] != 0:
        details["日内调整"] = intra_info

    # ── 板块强度 ──
    if code:
        bonus, b_desc = _get_sector_hot_score(code)
        if bonus:
            score += bonus
            details["板块"] = {"score": bonus, "desc": b_desc}

    # ── 大盘共振 ──
    if market_chg > 1:
        score += 5
        details["大盘共振"] = {"score": 5, "desc": f"大盘上涨{market_chg:+.1f}%+5"}
    elif market_chg < -1:
        score -= 5
        details["大盘共振"] = {"score": -5, "desc": f"大盘下跌{market_chg:+.1f}%-5"}

    # ── 综合判断 ──
    score = max(0, min(100, score))
    if score >= 75:
        action = "强烈买入"
        suggestion = "多指标共振看多，适合买入"
    elif score >= 60:
        action = "可买入"
        suggestion = "趋势偏好，可考虑建仓"
    elif score >= 45:
        action = "观望"
        suggestion = "信号不明确，继续观望"
    elif score >= 30:
        action = "回避"
        suggestion = "指标偏弱，不建议买入"
    else:
        action = "强烈回避"
        suggestion = "多个指标看空，规避风险"

    return {
        "score": score,
        "action": action,
        "suggestion": suggestion,
        "market_mode": mode_desc,
        "reasons": reasons,
        "risks": risks,
        "details": details,
    }


# ══════════════════════════════════════════════
#  各维度评分函数
# ══════════════════════════════════════════════

def _score_trend(closes: np.ndarray, weight: int) -> dict:
    """趋势评分：7级趋势 + 趋势强度

    强趋势(多头排列+发散) → 满分
    空头排列 → 0分
    """
    if len(closes) < 25:
        return {"score": 0, "desc": "数据不足(需≥25)", "trend_status": "consolidation", "signal": "neutral"}

    ma5 = _sma(closes, 5)
    ma10 = _sma(closes, 10)
    ma20 = _sma(closes, 20)
    valid = ~np.isnan(ma5) & ~np.isnan(ma10) & ~np.isnan(ma20)
    s5, s10, s20 = ma5[valid], ma10[valid], ma20[valid]
    if len(s5) < 2:
        return {"score": 0, "desc": "均线数据不足", "trend_status": "consolidation", "signal": "neutral"}

    c5, c10, c20 = s5[-1], s10[-1], s20[-1]

    # 判断趋势状态
    if c5 > c10 > c20:
        # 多头排列，检查是否发散（强势）
        prev_spread = (s5[-2] - s20[-2]) / s20[-2] * 100 if s20[-2] > 0 else 0
        curr_spread = (c5 - c20) / c20 * 100 if c20 > 0 else 0
        if curr_spread > prev_spread and curr_spread > 5:
            trend_status = "strong_bull"
            score = weight
            desc = "强势多头排列，均线发散上行 ✅"
        else:
            trend_status = "bull"
            score = int(weight * 0.87)
            desc = "多头排列 MA5>MA10>MA20 ✅"
    elif c5 > c10 and c10 <= c20:
        trend_status = "weak_bull"
        score = int(weight * 0.6)
        desc = "弱势多头，短期站上但中期承压"
    elif c5 < c10 < c20:
        prev_spread = (s20[-2] - s5[-2]) / s5[-2] * 100 if s5[-2] > 0 else 0
        curr_spread = (c20 - c5) / c5 * 100 if c5 > 0 else 0
        if curr_spread > prev_spread and curr_spread > 5:
            trend_status = "strong_bear"
            score = 0
            desc = "强势空头排列，均线发散下行 ❌"
        else:
            trend_status = "bear"
            score = int(weight * 0.13)
            desc = "空头排列 MA5<MA10<MA20 ❌"
    elif c5 < c10 and c10 >= c20:
        trend_status = "weak_bear"
        score = int(weight * 0.27)
        desc = "弱势空头，短期破位但中期尚可"
    else:
        trend_status = "consolidation"
        score = int(weight * 0.4)
        desc = "均线缠绕，趋势不明"

    signal = "bullish" if trend_status in ("strong_bull", "bull") else \
             "bearish" if trend_status in ("strong_bear", "bear") else "neutral"

    return {"score": score, "desc": desc, "trend_status": trend_status, "signal": signal}


def _score_bias(closes: np.ndarray, price: float, trend_status: str) -> dict:
    """乖离率评分（20分）：回踩加分，追高扣分，强趋势补偿

    核心哲学（来自daily_stock_analysis）：
    - 不追高，追求每笔交易成功率
    - 回踩MA5/MA10附近买入
    - 强势趋势可适当放宽乖离率上限
    """
    if len(closes) < 5:
        return {"score": 0, "desc": "数据不足", "signal": "neutral"}

    ma5 = _sma(closes, 5)
    valid = ~np.isnan(ma5)
    c5 = ma5[valid][-1] if len(ma5[valid]) > 0 else 0

    if c5 <= 0:
        return {"score": 10, "desc": "均线数据异常", "signal": "neutral"}

    bias = (price - c5) / c5 * 100  # 乖离率（与MA5的偏离度）

    # 强势趋势补偿：strong_bull 时乖离率上限从5%放宽到7.5%
    is_strong_trend = trend_status == "strong_bull"
    effective_threshold = 7.5 if is_strong_trend else 5.0

    # 评分
    if bias < 0:
        # 价格低于MA5（回踩）
        if bias > -3:
            score = 20
            desc = f"价格略低于MA5({bias:+.1f}%)，回踩买点 ✅"
        elif bias > -5:
            score = 16
            desc = f"价格回踩MA5({bias:+.1f}%)，观察支撑 👀"
        else:
            score = 8
            desc = f"乖离率过大({bias:+.1f}%)，可能破位 ⚠️"
    elif bias < 2:
        score = 18
        desc = f"价格贴近MA5({bias:+.1f}%)，介入好时机 ✅"
    elif bias < effective_threshold:
        score = 12
        desc = f"价格略高于MA5({bias:+.1f}%)，可小仓介入 ⚡"
    elif is_strong_trend:
        score = 10
        desc = f"强势趋势中乖离率偏高({bias:+.1f}%)，可轻仓追踪"
    else:
        score = 4
        desc = f"乖离率过高({bias:+.1f}%>{effective_threshold:.0f}%)，严禁追高！❌"

    signal = "bullish" if score >= 16 else ("bearish" if score < 10 else "neutral")
    # 回踩MA5加分
    if -1.5 < bias < 0.5:
        desc += "·精准回踩MA5"

    return {"score": score, "desc": desc, "bias_ma5": round(bias, 2), "signal": signal}


def _score_volume(closes: np.ndarray, volumes: np.ndarray, weight: int) -> dict:
    """量能评分（15分）：5种形态

    偏好排序（来自daily_stock_analysis）：
    缩量回调(最佳) > 放量上涨(次之) > 量能正常 > 缩量上涨(差) > 放量下跌(最差)
    """
    if volumes is None or len(volumes) < 6:
        return {"score": int(weight * 0.33), "desc": "量能数据不足", "signal": "neutral"}

    avg_v = np.mean(volumes[-5:])
    if avg_v <= 0:
        return {"score": int(weight * 0.33), "desc": "量能异常", "signal": "neutral"}

    vol_ratio = volumes[-1] / avg_v
    price_chg = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0

    if vol_ratio >= 1.5:
        if price_chg > 0:
            score = weight
            desc = f"放量上涨{vol_ratio:.1f}倍，多头力量强劲 ✅"
            signal = "bullish"
        else:
            score = 0
            desc = f"放量下跌{vol_ratio:.1f}倍，资金出逃 ❌"
            signal = "bearish"
    elif vol_ratio >= 1.2:
        if price_chg > 0:
            score = int(weight * 0.73)
            desc = f"略放量{vol_ratio:.1f}倍，温和上涨"
            signal = "neutral"
        else:
            score = int(weight * 0.27)
            desc = f"略放量下跌{vol_ratio:.1f}倍，注意压力"
            signal = "bearish"
    elif vol_ratio >= 0.8:
        if price_chg > 0:
            score = int(weight * 0.4)
            desc = f"缩量上涨{vol_ratio:.1f}倍，动力不足 ⚠️"
            signal = "bearish"
        elif price_chg < -1:
            score = weight
            desc = f"缩量回调{vol_ratio:.1f}倍，洗盘特征 ✅"
            signal = "bullish"
        else:
            score = int(weight * 0.6)
            desc = "量能正常"
            signal = "neutral"
    else:
        # 极度缩量
        if price_chg < 0:
            score = int(weight * 0.73)
            desc = f"极度缩量下跌{vol_ratio:.1f}倍，惜售明显 ✅"
            signal = "bullish"
        else:
            score = int(weight * 0.2)
            desc = f"极度缩量{vol_ratio:.1f}倍，无人参与 ❌"
            signal = "bearish"

    return {"score": score, "desc": desc, "vol_ratio": round(vol_ratio, 2), "signal": signal}


def _score_macd(closes: np.ndarray, weight: int) -> dict:
    """MACD评分（15分）：精细信号判定

    强度排序：零轴上金叉 > 金叉 > 上穿零轴 > 多头 > 中性 > 空头 > 下穿零轴 > 死叉
    """
    if len(closes) < 35:
        return {"score": int(weight * 0.33), "desc": "数据不足(需≥35)", "signal": "neutral"}

    ema_f = _ema(closes, 12)
    ema_s = _ema(closes, 26)
    diff = ema_f - ema_s
    dea = _ema(diff, 9)
    valid = ~np.isnan(diff) & ~np.isnan(dea)
    dv, deav = diff[valid], dea[valid]
    if len(dv) < 2:
        return {"score": int(weight * 0.33), "desc": "MACD数据不足", "signal": "neutral"}

    cd, cdea = dv[-1], deav[-1]
    pd, pdea = dv[-2], deav[-2]

    # 金叉：DIF上穿DEA
    is_golden_cross = pd <= pdea and cd > cdea
    # 死叉：DIF下穿DEA
    is_death_cross = pd >= pdea and cd < cdea
    # 上穿零轴
    is_cross_zero_up = pd <= 0 and cd > 0
    is_cross_zero_down = pd >= 0 and cd < 0

    if is_golden_cross and cd > 0:
        score = weight
        desc = "⭐ 零轴上金叉，强烈买入信号！"
        signal = "bullish"
    elif is_golden_cross:
        score = int(weight * 0.8)
        desc = "✅ MACD金叉，趋势向上"
        signal = "bullish"
    elif is_cross_zero_up:
        score = int(weight * 0.67)
        desc = "⚡ DIF上穿零轴，趋势转强"
        signal = "bullish"
    elif cd > cdea and cd > 0:
        score = int(weight * 0.53)
        desc = "✓ 多头排列，持续上涨"
        signal = "bullish"
    elif cd < cdea and cd < 0:
        score = int(weight * 0.13)
        desc = "⚠ 空头排列，持续下跌"
        signal = "bearish"
    elif is_cross_zero_down:
        score = int(weight * 0.07)
        desc = "⚠️ DIF下穿零轴，趋势转弱"
        signal = "bearish"
    elif is_death_cross:
        score = 0
        desc = "❌ MACD死叉，趋势向下"
        signal = "bearish"
    else:
        score = int(weight * 0.33)
        desc = " MACD中性区域"
        signal = "neutral"

    return {"score": score, "desc": desc, "diff": round(cd, 4), "dea": round(cdea, 4), "signal": signal}


def _score_rsi(closes: np.ndarray, weight: int) -> dict:
    """RSI评分（10分）：三周期(6/12/24)综合判断

    规则：
    - RSI(12)超卖<30：反弹机会大 → 高分
    - RSI(12)强势60-70：多头力量充足 → 高分
    - RSI(12)中性40-60：正常
    - RSI(12)超买>70：谨慎追高 → 低分
    - RSI(12)弱势30-40：关注反弹
    - RSI(6) < RSI(12) < RSI(24) 多头排列加分
    """
    if len(closes) < 25:
        return {"score": int(weight * 0.5), "desc": "数据不足", "signal": "neutral"}

    rsi6 = _calc_rsi(closes, 6)
    rsi12 = _calc_rsi(closes, 12)
    rsi24 = _calc_rsi(closes, 24)

    if rsi12 is None:
        return {"score": int(weight * 0.5), "desc": "RSI数据不足", "signal": "neutral"}

    # 以中期RSI(12)为主判断
    score = 0
    signal = "neutral"

    if rsi12 > 70:
        score = 0
        desc = f"⚠️ RSI超买({rsi12:.0f}>70)，短期回调风险高"
        signal = "bearish"
    elif rsi12 > 60:
        score = weight
        desc = f"✅ RSI强势({rsi12:.0f})，多头力量充足"
        signal = "bullish"
    elif rsi12 >= 40:
        score = int(weight * 0.5)
        desc = f" RSI中性({rsi12:.0f})，震荡整理中"
    elif rsi12 >= 30:
        score = int(weight * 0.6)
        desc = f"⚡ RSI弱势({rsi12:.0f})，关注反弹"
        signal = "bullish"
    else:
        score = weight
        desc = f"⭐ RSI超卖({rsi12:.0f}<30)，反弹机会大"
        signal = "bullish"

    # RSI多头排列加分：RSI6 > RSI12 > RSI24
    if rsi6 is not None and rsi24 is not None:
        if rsi6 > rsi12 > rsi24:
            extra = "+RSI多头排列"
            if signal == "bullish":
                score = min(score + 3, weight)
            desc += f"·{extra}"

    return {"score": score, "desc": desc, "rsi6": round(rsi6, 1) if rsi6 else 0,
            "rsi12": round(rsi12, 1), "rsi24": round(rsi24, 1) if rsi24 else 0, "signal": signal}


def _score_box_support(closes: np.ndarray, price: float) -> dict:
    """箱体/支撑评分（10分）

    箱体策略（来自daily_stock_analysis）：
    - 箱底企稳 + 缩量：最佳买点
    - 箱顶区域：不追高
    - 有效突破箱顶：转趋势策略
    """
    score = 0
    signal = "neutral"

    # 1. 箱体检测
    box = _detect_box(closes, price)
    desc = ""

    if box["in_box"]:
        if box["position"] == "箱底区域":
            score = 10
            desc = f"箱底企稳(箱体{box['width_pct']:.0f}%)，低吸机会 ✅"
            signal = "bullish"
        elif box["position"] == "箱顶区域":
            score = 0
            desc = f"箱顶附近({box['top']:.2f})，不宜追高 ❌"
            signal = "bearish"
        else:
            score = 5
            desc = f"箱中区域({box['width_pct']:.0f}%)，观望等待"
            signal = "neutral"
    elif box["position"] == "突破箱顶":
        score = 5
        desc = f"突破箱顶({box['top']:.2f})，趋势延续可关注 ✅"
        signal = "bullish"
    elif box["position"] == "跌破箱底":
        score = 0
        desc = f"跌破箱底({box['bottom']:.2f})，离场观望 ❌"
        signal = "bearish"
    else:
        desc = "无明显箱体结构"

    # 2. 均线支撑加分（无箱体时也有用）
    if not box["in_box"] and len(closes) >= 20:
        ma20 = _sma(closes, 20)
        valid = ~np.isnan(ma20)
        if len(ma20[valid]) > 0:
            c20 = ma20[valid][-1]
            dev_ma20 = (price - c20) / c20 * 100
            if -1 < dev_ma20 < 1 and c20 > 0:
                score = max(score, 8)
                desc += "·MA20支撑有效 ✅"
                signal = "bullish"

    return {"score": score, "desc": desc.strip("·"), "signal": signal,
            "box_top": box["top"], "box_bottom": box["bottom"]}


def _score_fund_flow(fund_flow: Optional[dict], weight: int) -> dict:
    """资金流向评分（额外加减）"""
    if not fund_flow:
        return {"score": 0, "desc": "无数据", "signal": "neutral"}

    main_net = fund_flow.get("main_force_net", 0)

    if main_net > 50_000_000:
        score = int(weight * 0.5)
        desc = f"主力大额流入{main_net/1e8:.1f}亿💰"
        signal = "bullish"
    elif main_net > 10_000_000:
        score = int(weight * 0.3)
        desc = f"主力流入{main_net/1e8:.1f}亿"
        signal = "bullish"
    elif main_net > -10_000_000:
        score = 0
        desc = "资金平衡"
        signal = "neutral"
    elif main_net > -50_000_000:
        score = -int(weight * 0.15)
        desc = f"主力流出{abs(main_net)/1e8:.1f}亿💸"
        signal = "bearish"
    else:
        score = -int(weight * 0.25)
        desc = f"主力大幅流出{abs(main_net)/1e8:.1f}亿💸"
        signal = "bearish"

    return {"score": score, "desc": desc, "signal": signal}


def _score_intraday(change_pct: float) -> dict:
    """日内涨幅调整：涨太多扣分(追高惩罚)，跌是机会加分(低吸奖励)"""
    if change_pct == 0:
        return {"score": 0, "desc": ""}

    if change_pct > 4:
        return {"score": -15, "desc": f"今日涨{change_pct:+.1f}%过高❌追高惩罚"}
    elif change_pct > 2:
        return {"score": -8, "desc": f"今日涨{change_pct:+.1f}%偏高⚠️追高警惕"}
    elif change_pct < -3:
        return {"score": 8, "desc": f"今日跌{change_pct:+.1f}%低吸机会🟢"}
    elif change_pct < -1.5:
        return {"score": 4, "desc": f"今日跌{change_pct:+.1f}%可关注+4"}
    return {"score": 0, "desc": ""}
