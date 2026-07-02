"""
龙回头策略 — 涨停后回调到支撑位，二次上车机会

检测逻辑:
  1. 前期涨停检测: 近3-10交易日内出现过涨停(>=9.8%)
  2. 回调阶段: 涨停后出现回调(-3%~-15%)
  3. 缩量确认: 回调期间量比<0.8(洗盘特征)
  4. 关键支撑: 价格不破MA10/MA20
  5. 再次启动: 当日放量上涨,站上MA5
"""
import numpy as np
from typing import Optional
from src.signals import _sma


def _is_limit_up(chg: float, code: str = "") -> bool:
    """判断是否涨停 (主板±10%, 创业板/科创板±20%)"""
    if code.startswith(("3", "68")):
        return chg >= 19.8
    return chg >= 9.8


def detect_dragon_back(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    price: float,
    code: str = "",
    lookback: int = 15,
) -> Optional[dict]:
    """检测龙回头信号

    Returns:
        {is_dragon, confidence, reasons, signal_type} or None
    """
    n = len(closes)
    if n < lookback + 5:
        return None

    # 1. 找最近的涨停日
    limit_up_idx = -1
    for i in range(n - 3, max(n - lookback, 0), -1):
        if i < 1:
            continue
        chg = (closes[i] / closes[i - 1] - 1) * 100
        if _is_limit_up(chg, code):
            limit_up_idx = i
            break

    if limit_up_idx < 0:
        return None

    # 涨停后至少3个交易日（充分回调）
    if n - 1 - limit_up_idx < 3 or n - 1 - limit_up_idx > 10:
        return None

    # 2. 计算回调幅度
    post_high = float(np.max(highs[limit_up_idx:]))
    post_low = float(np.min(lows[limit_up_idx:]))
    drawdown = (price - post_high) / post_high * 100
    if drawdown > -3 or drawdown < -15:
        return None

    # 3. 成交量萎缩
    pre_vol = float(np.mean(volumes[max(0, limit_up_idx - 5): limit_up_idx])) if limit_up_idx >= 5 else float(np.mean(volumes[:limit_up_idx]))
    post_vol = float(np.mean(volumes[limit_up_idx: -1])) if limit_up_idx < n - 2 else volumes[-1]
    vol_ratio = post_vol / pre_vol if pre_vol > 0 else 1
    if vol_ratio > 0.8:
        return None

    # 4. 支撑位确认
    ma10 = _sma(closes, 10)
    ma20 = _sma(closes, 20)
    valid10 = ~np.isnan(ma10)
    valid20 = ~np.isnan(ma20)
    m10_val = ma10[valid10][-1] if valid10.any() else 0
    m20_val = ma20[valid20][-1] if valid20.any() else 0

    has_ma_support = False
    support_desc = ""
    if m10_val > 0:
        dev10 = (price / m10_val - 1) * 100
        if abs(dev10) <= 3:
            has_ma_support = True
            support_desc = f"MA10({m10_val:.2f})"
    if not has_ma_support and m20_val > 0:
        dev20 = (price / m20_val - 1) * 100
        if abs(dev20) <= 3:
            has_ma_support = True
            support_desc = f"MA20({m20_val:.2f})"

    if not has_ma_support:
        return None

    # 5. 再次启动信号
    if len(volumes) >= 5:
        today_vol = volumes[-1]
        avg_last5 = float(np.mean(volumes[-5:-1]))
        restart_vol_ok = today_vol > avg_last5 * 1.2 if avg_last5 > 0 else False
    else:
        restart_vol_ok = False

    price_above_ma5 = price > _sma(closes, 5)[-1] if not np.isnan(_sma(closes, 5)[-1]) else False

    # 综合评分
    signals = 0
    reasons = []
    if abs(drawdown) >= 5:
        signals += 2; reasons.append(f"回调{abs(drawdown):.0f}%充分")
    else:
        signals += 1; reasons.append(f"浅回调{abs(drawdown):.0f}%")
    if vol_ratio < 0.5:
        signals += 2; reasons.append("极度缩量洗盘")
    else:
        signals += 1; reasons.append("缩量回调")
    if has_ma_support:
        signals += 1; reasons.append(f"回踩{support_desc}")
    if restart_vol_ok and price_above_ma5:
        signals += 2; reasons.append("放量重启站上MA5")
    elif price_above_ma5:
        signals += 1; reasons.append("站上MA5")

    confidence = min(50 + signals * 10, 95)
    is_dragon = signals >= 3

    return {
        "is_dragon": is_dragon,
        "confidence": confidence,
        "reasons": "，".join(reasons),
        "signal_type": "dragon_back",
        "limit_up_day": limit_up_idx,
        "drawdown": round(drawdown, 1),
        "vol_ratio": round(vol_ratio, 2),
        "days_since_limit": n - 1 - limit_up_idx,
    }
