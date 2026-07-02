"""
突破追涨策略 — 检测价格放量突破整理区间前高
"""
import numpy as np
from typing import Optional
from src.signals import _sma


def detect_consolidation(closes: np.ndarray, lookback: int = 20) -> dict:
    """检测是否处于整理区间"""
    segment = closes[-lookback:]
    high = float(np.max(segment))
    low = float(np.min(segment))
    range_pct = (high - low) / low * 100 if low > 0 else 0
    is_consolidation = 3 < range_pct < 15
    current_pos = (float(closes[-1]) - low) / (high - low) * 100 if high > low else 50
    return {
        "is_consolidation": is_consolidation,
        "high": high,
        "low": low,
        "range_pct": round(range_pct, 1),
        "avg_price": (high + low) / 2,
        "current_pos": round(current_pos, 0),
    }


def detect_breakout(
    closes: np.ndarray,
    highs: np.ndarray,
    volumes: np.ndarray,
    price: float,
) -> Optional[dict]:
    """检测是否出现突破追涨信号"""
    if len(closes) < 25:
        return None

    consol = detect_consolidation(closes, 20)
    if not consol["is_consolidation"] and consol["range_pct"] > 0:
        consol = detect_consolidation(closes, 10)

    consol_high = consol["high"]
    if price <= consol_high * 1.005:
        return None

    # 成交量确认
    if len(volumes) < 5:
        return None
    avg_vol_5 = float(np.mean(volumes[-6:-1])) if len(volumes) >= 6 else float(np.mean(volumes[:-1]))
    vol_ratio = float(volumes[-1]) / avg_vol_5 if avg_vol_5 > 0 else 1
    if vol_ratio < 1.15:
        return None

    # 趋势确认
    ma5 = _sma(closes, 5)
    ma20 = _sma(closes, 20)
    valid = ~np.isnan(ma5) & ~np.isnan(ma20)
    trend_ok = ma5[valid][-1] > ma20[valid][-1] if len(ma5[valid]) > 0 else False

    pct_above = (price / consol_high - 1) * 100

    signals = 0
    reasons = []
    if vol_ratio >= 1.5:
        signals += 2; reasons.append(f"放量{vol_ratio:.1f}倍")
    elif vol_ratio >= 1.2:
        signals += 1; reasons.append("温和放量")
    if trend_ok:
        signals += 2; reasons.append("均线多头")
    if pct_above >= 2:
        signals += 1; reasons.append(f"强突破+{pct_above:.1f}%")
    if consol["range_pct"] < 10:
        signals += 1; reasons.append("窄幅整理")

    confidence = min(50 + signals * 10, 95)
    is_breakout = signals >= 2

    return {
        "is_breakout": is_breakout,
        "breakout_level": consol_high,
        "price": price,
        "pct_above_high": round(pct_above, 2),
        "vol_ratio": round(vol_ratio, 2),
        "trend_ok": trend_ok,
        "consolidation": consol,
        "confidence": confidence,
        "reasons": "，".join(reasons),
        "signal_type": "breakout",
    }
