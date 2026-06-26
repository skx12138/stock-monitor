"""
次日涨势预判 — 根据今日数据推算下一交易日走势（周五时自动预测下周一）
"""
import logging
from datetime import datetime

import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def _target_label() -> str:
    """返回预测目标标签：周五返回"下周一"，其余返回"明日" """
    return "下周一" if datetime.today().weekday() == 4 else "明日"


def predict_tomorrow(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     volumes: np.ndarray, price: float) -> dict:
    """根据今日数据预判下一交易日涨跌（周五自动预测下周一）

    Args:
        closes: 收盘价数组
        highs: 最高价数组
        lows: 最低价数组
        volumes: 成交量数组
        price: 当前价格

    Returns:
        {direction, confidence, reason, score_adj, target_label}
        direction: "看涨"/"看跌"/"震荡"
        confidence: 0-100 信心指数
        score_adj: 评分调整值（-10 到 +10）
        target_label: "下周一"(周五) 或 "明日"
    """
    if len(closes) < 20:
        return {"direction": "震荡", "confidence": 50, "reason": "数据不足", "score_adj": 0, "target_label": _target_label()}

    from src.signals import _sma, _calc_rsi

    reasons = []
    bullish_signals = 0
    bearish_signals = 0
    score_adj = 0

    # 1. 今日涨幅
    today_chg = (price / closes[-2] - 1) * 100 if len(closes) >= 2 else 0

    # 2. K线形态：收盘在当日高位（阳线实体）
    today_open = closes[-2]  # 近似
    if highs[-1] > lows[-1]:
        body_range = abs(price - closes[-2])
        total_range = highs[-1] - lows[-1]
        if total_range > 0:
            upper_shadow = highs[-1] - max(price, closes[-2])
            lower_shadow = min(price, closes[-2]) - lows[-1]
            # 光头阳线：收盘=最高（强势）
            if upper_shadow < total_range * 0.05 and price > closes[-2]:
                bullish_signals += 2
                reasons.append("光头阳线强势收盘")
            # 锤子线：长下影（探底回升）
            elif lower_shadow > total_range * 0.6 and upper_shadow < total_range * 0.3:
                bullish_signals += 2
                reasons.append("锤子线探底回升")
            # 倒锤子：长上影（高位受阻）
            elif upper_shadow > total_range * 0.6 and lower_shadow < total_range * 0.3:
                bearish_signals += 2
                reasons.append("倒锤线高位受阻")

    # 3. 均线支撑
    ma5 = _sma(closes, 5)
    ma10 = _sma(closes, 10)
    ma20 = _sma(closes, 20)
    valid_ma = ~np.isnan(ma5) & ~np.isnan(ma10) & ~np.isnan(ma20)
    if len(ma5[valid_ma]) > 0:
        c5, c10, c20 = ma5[valid_ma][-1], ma10[valid_ma][-1], ma20[valid_ma][-1]
        # 价格在MA5上方获得支撑
        dev_ma5 = (price / c5 - 1) * 100
        if 0 < dev_ma5 < 2:
            bullish_signals += 1
            reasons.append(f"站稳MA5({c5:.2f})")
        elif dev_ma5 < -2 and today_chg > 0:
            # 跌破MA5但收回来，假跌破
            bullish_signals += 1
            reasons.append("假跌破MA5收回")
        # 均线多头排列
        if c5 > c10 > c20:
            bullish_signals += 1
            reasons.append("均线多头排列")
        elif c5 < c10 < c20:
            bearish_signals += 1
            reasons.append("均线空头排列")

    # 4. RSI动量
    rsi = _calc_rsi(closes, 14)
    if rsi is not None:
        rsi_prev = _calc_rsi(closes[:-1], 14) if len(closes) > 15 else None
        if rsi < 30:
            bullish_signals += 2
            reasons.append(f"RSI{rsi:.0f}超卖反弹预期")
        elif rsi < 40 and rsi_prev and rsi > rsi_prev:
            bullish_signals += 1
            reasons.append(f"RSI从低位回升({rsi_prev:.0f}→{rsi:.0f})")
        elif rsi > 70:
            bearish_signals += 2
            reasons.append(f"RSI{rsi:.0f}超买回调预期")
        elif rsi > 60 and rsi_prev and rsi < rsi_prev:
            bearish_signals += 1
            reasons.append(f"RSI从高位回落({rsi_prev:.0f}→{rsi:.0f})")

    # 5. 成交量确认
    if len(volumes) >= 5:
        avg_vol = np.mean(volumes[-5:-1])
        if avg_vol > 0:
            vol_ratio = volumes[-1] / avg_vol
            if today_chg > 0 and vol_ratio > 1.3:
                bullish_signals += 1
                reasons.append(f"放量上涨{vol_ratio:.1f}倍")
            elif today_chg < 0 and vol_ratio > 1.3:
                bearish_signals += 1
                reasons.append(f"放量下跌{vol_ratio:.1f}倍")
            elif today_chg > 0 and vol_ratio < 0.7:
                bearish_signals += 1
                reasons.append("缩量上涨动力不足")

    # 6. 综合判断
    net = bullish_signals - bearish_signals
    if net >= 3:
        direction = "看涨"
        confidence = min(50 + net * 10, 95)
        score_adj = min(net * 2, 10)
    elif net <= -3:
        direction = "看跌"
        confidence = min(50 + abs(net) * 10, 95)
        score_adj = max(net * 2, -10)
    else:
        direction = "震荡"
        confidence = 50
        score_adj = net

    return {
        "direction": direction,
        "confidence": int(confidence),
        "reason": "，".join(reasons[:5]),
        "score_adj": score_adj,
        "target_label": _target_label(),
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
    }
