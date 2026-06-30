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
    升级版V2：加入ATR趋势强度、MACD动量、KDJ超买超卖、量价配合

    Returns:
        {direction, confidence, reason, score_adj, target_label}
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
    today_open = closes[-2]
    if highs[-1] > lows[-1]:
        body_range = abs(price - closes[-2])
        total_range = highs[-1] - lows[-1]
        if total_range > 0:
            upper_shadow = highs[-1] - max(price, closes[-2])
            lower_shadow = min(price, closes[-2]) - lows[-1]
            if upper_shadow < total_range * 0.05 and price > closes[-2]:
                bullish_signals += 2
                reasons.append("强势收涨")
            elif lower_shadow > total_range * 0.6 and upper_shadow < total_range * 0.3:
                bullish_signals += 2
                reasons.append("跌后反弹")
            elif upper_shadow > total_range * 0.6 and lower_shadow < total_range * 0.3:
                bearish_signals += 2
                reasons.append("冲高回落")

    # 3. 均线支撑
    ma5 = _sma(closes, 5)
    ma10 = _sma(closes, 10)
    ma20 = _sma(closes, 20)
    ma60 = _sma(closes, 60) if len(closes) >= 60 else np.array([])
    valid_ma = ~np.isnan(ma5) & ~np.isnan(ma10) & ~np.isnan(ma20)
    if len(ma5[valid_ma]) > 0:
        c5, c10, c20 = ma5[valid_ma][-1], ma10[valid_ma][-1], ma20[valid_ma][-1]
        dev_ma5 = (price / c5 - 1) * 100
        if 0 < dev_ma5 < 2:
            bullish_signals += 1
            reasons.append(f"站稳MA5({c5:.2f})")
        elif dev_ma5 < -2 and today_chg > 0:
            bullish_signals += 1
            reasons.append("假跌破MA5收回")
        # 均线多头排列
        if c5 > c10 > c20:
            bullish_signals += 2
            reasons.append("多头排列")
        elif c5 < c10 < c20:
            bearish_signals += 2
            reasons.append("空头排列")
        # MA60趋势确认（中长期方向）
        if len(ma60) > 0 and not np.isnan(ma60[-1]):
            if c20 > ma60[-1]:
                bullish_signals += 1
                reasons.append("中期向上")
            else:
                bearish_signals += 1
                reasons.append("中期向下")

    # 4. RSI动量 + 趋势
    rsi = _calc_rsi(closes, 14)
    if rsi is not None:
        rsi_prev = _calc_rsi(closes[:-1], 14) if len(closes) > 15 else None
        if rsi < 30:
            bullish_signals += 2
            reasons.append(f"RSI{rsi:.0f}超卖")
        elif rsi < 40 and rsi_prev and rsi > rsi_prev:
            bullish_signals += 1
            reasons.append(f"RSI回升({rsi_prev:.0f}→{rsi:.0f})")
        elif rsi > 70:
            bearish_signals += 2
            reasons.append(f"RSI{rsi:.0f}超买")
        elif rsi > 60 and rsi_prev and rsi < rsi_prev:
            bearish_signals += 1
            reasons.append(f"RSI回落({rsi_prev:.0f}→{rsi:.0f})")
        # RSI斜率判断（最近3天）
        if len(closes) >= 16:
            rsi_3d_ago = _calc_rsi(closes[:-3], 14) if len(closes) > 17 else None
            if rsi_3d_ago is not None and rsi > rsi_3d_ago + 5:
                bullish_signals += 1
                reasons.append("RSI趋势向上")
            elif rsi_3d_ago is not None and rsi < rsi_3d_ago - 5:
                bearish_signals += 1
                reasons.append("RSI趋势向下")

    # 5. 成交量确认 + 量价背离
    if len(volumes) >= 5:
        avg_vol = np.mean(volumes[-5:-1])
        if avg_vol > 0:
            vol_ratio = volumes[-1] / avg_vol
            vol_ma5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else avg_vol
            vol_trend = "放量" if vol_ma5 > np.mean(volumes[-10:-5]) else "缩量"
            if today_chg > 0 and vol_ratio > 1.3:
                bullish_signals += 1
                reasons.append(f"放量涨{vol_ratio:.1f}倍")
            elif today_chg > 0 and vol_ratio < 0.7:
                bearish_signals += 1
                reasons.append("缩量上涨动力不足")
            elif today_chg < 0 and vol_ratio > 1.3:
                bearish_signals += 1
                reasons.append(f"放量跌{vol_ratio:.1f}倍")
            elif today_chg < 0 and vol_ratio < 0.7:
                bullish_signals += 1
                reasons.append("缩量下跌惜售")

    # 6. MACD动量
    if len(closes) >= 26:
        from src.signals import _ema
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        if not np.isnan(ema12[-1]) and not np.isnan(ema26[-1]):
            macd = ema12[-1] - ema26[-1]
            macd_prev = ema12[-2] - ema26[-2] if len(ema12) > 1 and len(ema26) > 1 else macd
            if macd > macd_prev and macd > 0:
                bullish_signals += 2
                reasons.append("MACD金叉")
            elif macd < macd_prev and macd < 0:
                bearish_signals += 2
                reasons.append("MACD死叉")
            elif macd > macd_prev:
                bullish_signals += 1
                reasons.append("MACD向上")
            elif macd < macd_prev:
                bearish_signals += 1
                reasons.append("MACD向下")

    # 7. ATR波动率过滤（大幅波动后容易回归）
    try:
        from src.signals import calc_atr
        atr_val = calc_atr(closes, highs, lows, 14)
        if atr_val > 0 and price > 0:
            atr_pct = atr_val / price * 100
            if atr_pct > 5 and today_chg > 0:
                bearish_signals += 1
                reasons.append(f"高波动{atr_pct:.1f}%防回调")
            elif atr_pct > 5 and today_chg < 0:
                bullish_signals += 1
                reasons.append(f"高波动{atr_pct:.1f}%防反弹")
    except:
        pass

    # 8. 综合判断
    net = bullish_signals - bearish_signals
    if net >= 3:
        direction = "看涨"
        confidence = min(50 + net * 8, 95)
        score_adj = min(net * 2, 10)
    elif net <= -3:
        direction = "看跌"
        confidence = min(50 + abs(net) * 8, 95)
        score_adj = max(net * 2, -10)
    else:
        direction = "震荡"
        confidence = 50 + net * 5
        score_adj = net

    return {
        "direction": direction,
        "confidence": int(confidence),
        "reason": "，".join(reasons[:6]),
        "score_adj": score_adj,
        "target_label": _target_label(),
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
    }
