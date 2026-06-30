"""
信号引擎 — 多指标检测 + 去重

支持的信号:
  - ma_crossover: 均线金叉/死叉
  - rsi: RSI 高/低位区信号
  - macd: MACD 金叉/死叉
  - bollinger: 布林带突破
  - volume_breakout: 放量突破
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

import numpy as np
import pandas as pd

from src.sectors import get_sector_tag

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  数据结构
# ──────────────────────────────────────────────

@dataclass
class Signal:
    """一条信号记录"""
    stock_code: str
    stock_name: str
    signal_type: str
    signal_label: str
    direction: str            # "bullish" / "bearish" / "neutral"
    price: float
    message: str              # 推送消息正文
    change_pct: float = 0.0   # 当前涨跌幅
    suggestion: str = ""      # 操作建议：如"适合买入"、"建议卖出"、"短线机会"等
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
#  信号去重
# ──────────────────────────────────────────────

class SignalDedup:
    """同一天内同一只股票的同一类信号只推送一次"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._sent: set[tuple[str, str, str]] = set()  # (code, type, date)

    def is_duplicate(self, code: str, signal_type: str) -> bool:
        if not self.enabled:
            return False
        today = date.today().isoformat()
        key = (code, signal_type, today)
        return key in self._sent

    def mark_sent(self, code: str, signal_type: str):
        if not self.enabled:
            return
        today = date.today().isoformat()
        self._sent.add((code, signal_type, today))

    def reset(self):
        """新的一天重置"""
        self._sent.clear()


# ──────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────

def _sma(values: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均"""
    return pd.Series(values).rolling(window=period).mean().values


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均"""
    return pd.Series(values).ewm(span=period, adjust=False).mean().values


def _std(values: np.ndarray, period: int) -> np.ndarray:
    """滚动标准差"""
    return pd.Series(values).rolling(window=period).std(ddof=0).values


# ──────────────────────────────────────────────
#  各信号检测函数
# ──────────────────────────────────────────────

def _check_ma_crossover(
    closes: np.ndarray, code: str, name: str, price: float,
    params: dict,
) -> Optional[Signal]:
    """均线金叉/死叉"""
    ma_short = params.get("ma_short", 5)
    ma_long = params.get("ma_long", 20)

    if len(closes) < ma_long + 1:
        return None

    short_ma = _sma(closes, ma_short)
    long_ma = _sma(closes, ma_long)

    valid = ~np.isnan(short_ma) & ~np.isnan(long_ma)
    short_vals = short_ma[valid]
    long_vals = long_ma[valid]

    if len(short_vals) < 2:
        return None

    prev_s, prev_l = short_vals[-2], long_vals[-2]
    curr_s, curr_l = short_vals[-1], long_vals[-1]

    # 金叉：MA短线从下往上穿越MA长线
    if prev_s <= prev_l and curr_s > curr_l:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="ma_crossover", signal_label="均线金叉 📈",
            direction="bullish", price=price,
            message=(
                f"📈 **{name}({code}) 均线金叉**\n"
                f"当前价: {price:.2f}\n"
                f"MA{ma_short}: {curr_s:.2f}  |  MA{ma_long}: {curr_l:.2f}\n"
                f"短线已上穿长线，多头信号"
            ),
            suggestion="适合买入（短线）",
            extra={"ma_short_val": round(curr_s, 2), "ma_long_val": round(curr_l, 2)},
        )

    # 死叉：MA短线从上往下穿越MA长线
    if prev_s >= prev_l and curr_s < curr_l:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="ma_crossover", signal_label="均线死叉 📉",
            direction="bearish", price=price,
            suggestion="建议卖出/减仓",
            message=(
                f"📉 **{name}({code}) 均线死叉**\n"
                f"当前价: {price:.2f}\n"
                f"MA{ma_short}: {curr_s:.2f}  |  MA{ma_long}: {curr_l:.2f}\n"
                f"短线已下穿长线，建议考虑减仓"
            ),
            extra={"ma_short_val": round(curr_s, 2), "ma_long_val": round(curr_l, 2)},
        )

    return None


def _check_rsi(
    closes: np.ndarray, code: str, name: str, price: float,
    params: dict,
) -> Optional[Signal]:
    """RSI 超买超卖"""
    period = params.get("period", 14)
    overbought = params.get("overbought", 70)
    oversold = params.get("oversold", 30)

    if len(closes) < period + 1:
        return None

    # 计算 RSI
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = _sma(gains, period)
    avg_loss = _sma(losses, period)

    valid = (avg_loss > 0) & ~np.isnan(avg_gain) & ~np.isnan(avg_loss)
    if not valid.any():
        return None

    rs = avg_gain[valid] / avg_loss[valid]
    rsi_values = 100 - (100 / (1 + rs))

    if len(rsi_values) < 2:
        return None

    curr_rsi = rsi_values[-1]
    prev_rsi = rsi_values[-2]

    # 从低位反弹（从低位区上穿30）
    if prev_rsi <= oversold and curr_rsi > oversold:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="rsi", signal_label="RSI 从低位反弹 🟢",
            direction="bullish", price=price,
            message=(
                f"🟢 **{name}({code}) RSI 从低位反弹**\n"
                f"当前价: {price:.2f}\n"
                f"RSI({period}): {curr_rsi:.1f}（前值 {prev_rsi:.1f}）\n"
                f"已从低位反弹，短线可能走强"
            ),
            suggestion="短线反弹机会",
            extra={"rsi": round(curr_rsi, 1)},
        )

    # 从高位回落（从高位区下穿70）
    if prev_rsi >= overbought and curr_rsi < overbought:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="rsi", signal_label="RSI 从高位回落 🔴",
            direction="bearish", price=price,
            message=(
                f"🔴 **{name}({code}) RSI 从高位回落**\n"
                f"当前价: {price:.2f}\n"
                f"RSI({period}): {curr_rsi:.1f}（前值 {prev_rsi:.1f}）\n"
                f"已从高位回落，短线可能走弱"
            ),
            suggestion="注意回调风险，不宜追高",
            extra={"rsi": round(curr_rsi, 1)},
        )

    # 进入低位区（可能跌过头了）
    if curr_rsi <= oversold:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="rsi", signal_label="RSI 进入低位区 ⚠️",
            direction="bullish", price=price,
            message=(
                f"⚠️ **{name}({code}) RSI 进入低位区**\n"
                f"当前价: {price:.2f}\n"
                f"RSI({period}): {curr_rsi:.1f}（低于 {oversold}）\n"
                f"可能跌过头了，短期内可能有反弹机会"
            ),
            suggestion="低位区，观望等待机会",
            extra={"rsi": round(curr_rsi, 1)},
        )

    # 进入高位区（短期涨太猛了）
    if curr_rsi >= overbought:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="rsi", signal_label="RSI 进入高位区 ⚠️",
            direction="bearish", price=price,
            message=(
                f"⚠️ **{name}({code}) RSI 进入高位区**\n"
                f"当前价: {price:.2f}\n"
                f"RSI({period}): {curr_rsi:.1f}（高于 {overbought}）\n"
                f"短期涨太猛了，注意回调风险"
            ),
            suggestion="高位区，不建议追高",
            extra={"rsi": round(curr_rsi, 1)},
        )

    return None


def _check_macd(
    closes: np.ndarray, code: str, name: str, price: float,
    params: dict,
) -> Optional[Signal]:
    """MACD 金叉/死叉"""
    fast = params.get("fast", 12)
    slow = params.get("slow", 26)
    signal_period = params.get("signal", 9)

    if len(closes) < slow + signal_period:
        return None

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    diff = ema_fast - ema_slow
    dea = _ema(diff, signal_period)

    valid = ~np.isnan(diff) & ~np.isnan(dea)
    diff_v = diff[valid]
    dea_v = dea[valid]

    if len(diff_v) < 2:
        return None

    prev_diff, prev_dea = diff_v[-2], dea_v[-2]
    curr_diff, curr_dea = diff_v[-1], dea_v[-1]
    histogram = curr_diff - curr_dea

    # 金叉：DIFF上穿DEA
    if prev_diff <= prev_dea and curr_diff > curr_dea:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="macd", signal_label="MACD 金叉 📈",
            direction="bullish", price=price,
            message=(
                f"📈 **{name}({code}) MACD 金叉**\n"
                f"当前价: {price:.2f}\n"
                f"DIFF: {curr_diff:.3f}  |  DEA: {curr_dea:.3f}\n"
                f"DIFF上穿DEA，多头信号"
            ),
            suggestion="中短线看多",
            extra={"diff": round(curr_diff, 3), "dea": round(curr_dea, 3), "histogram": round(histogram, 3)},
        )

    # 死叉：DIFF下穿DEA
    if prev_diff >= prev_dea and curr_diff < curr_dea:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="macd", signal_label="MACD 死叉 📉",
            direction="bearish", price=price,
            message=(
                f"📉 **{name}({code}) MACD 死叉**\n"
                f"当前价: {price:.2f}\n"
                f"DIFF: {curr_diff:.3f}  |  DEA: {curr_dea:.3f}\n"
                f"DIFF下穿DEA，空头信号"
            ),
            suggestion="中短线看空",
            extra={"diff": round(curr_diff, 3), "dea": round(curr_dea, 3), "histogram": round(histogram, 3)},
        )

    return None


def _check_bollinger(
    closes: np.ndarray, code: str, name: str, price: float,
    params: dict,
) -> Optional[Signal]:
    """布林带突破"""
    period = params.get("period", 20)
    k = params.get("std_dev", 2.0)

    if len(closes) < period + 1:
        return None

    middle = _sma(closes, period)
    sd = _std(closes, period)
    upper = middle + k * sd
    lower = middle - k * sd

    valid = ~np.isnan(middle) & ~np.isnan(upper) & ~np.isnan(lower)
    upper_v, lower_v, mid_v = upper[valid], lower[valid], middle[valid]

    if len(upper_v) < 1:
        return None

    curr_upper, curr_lower, curr_mid = upper_v[-1], lower_v[-1], mid_v[-1]

    # 突破上轨
    if price >= curr_upper:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="bollinger", signal_label="布林带上轨突破 🔴",
            direction="bearish", price=price,
            message=(
                f"🔴 **{name}({code}) 突破布林带上轨**\n"
                f"当前价: {price:.2f}\n"
                f"上轨: {curr_upper:.2f}  |  中轨: {curr_mid:.2f}\n"
                f"股价已突破上轨，短期涨太猛了，注意回调风险"
            ),
            suggestion="注意回调风险",
            extra={"upper": round(curr_upper, 2), "middle": round(curr_mid, 2), "lower": round(curr_lower, 2)},
        )

    # 跌破下轨
    if price <= curr_lower:
        return Signal(
            stock_code=code, stock_name=name,
            signal_type="bollinger", signal_label="布林带下轨突破 🟢",
            direction="bullish", price=price,
            suggestion="短线反弹机会",
            message=(
                f"🟢 **{name}({code}) 跌破布林带下轨**\n"
                f"当前价: {price:.2f}\n"
                f"下轨: {curr_lower:.2f}  |  中轨: {curr_mid:.2f}\n"
                f"股价已跌破下轨，可能跌过头了，可能有反弹机会"
            ),
            extra={"upper": round(curr_upper, 2), "middle": round(curr_mid, 2), "lower": round(curr_lower, 2)},
        )

    return None


def _check_volume_breakout(
    volumes: np.ndarray, code: str, name: str, price: float,
    params: dict,
) -> Optional[Signal]:
    """放量突破"""
    period = params.get("ma_period", 5)
    ratio = params.get("ratio", 1.5)

    if len(volumes) < period + 1:
        return None

    avg_volume = _sma(volumes, period)
    valid = ~np.isnan(avg_volume)
    avg_v = avg_volume[valid]

    if len(avg_v) < 1:
        return None

    curr_vol = volumes[-1]
    curr_avg = avg_v[-1]

    if curr_avg <= 0:
        return None

    vol_ratio = curr_vol / curr_avg

    if vol_ratio >= ratio:
        # 判断涨跌方向（需要对应的价格数据来判断）
        direction = "neutral"
        label = f"放量突破 {vol_ratio:.1f}倍"
        emoji = "🔥"

        return Signal(
            stock_code=code, stock_name=name,
            signal_type="volume_breakout", signal_label=label,
            direction=direction, price=price,
            message=(
                f"{emoji} **{name}({code}) 放量突破**\n"
                f"当前价: {price:.2f}\n"
                f"今日量: {curr_vol:,.0f}  |  均量({period}日): {curr_avg:,.0f}\n"
                f"今日成交量是 {period}日均量的 **{vol_ratio:.1f} 倍**\n"
                f"资金活动明显，注意股价方向选择"
            ),
            suggestion="关注方向选择",
            extra={"volume": int(curr_vol), "avg_volume": int(curr_avg), "ratio": round(vol_ratio, 2)},
        )

    return None


# ──────────────────────────────────────────────
#  统一检查入口
# ──────────────────────────────────────────────

def check_signals(
    stock_code: str,
    stock_name: str,
    kline_df: pd.DataFrame,
    latest_price: float,
    config: dict,
    dedup: Optional["SignalDedup"] = None,
    change_pct: float = 0.0,
) -> list[Signal]:
    """对一只股票运行所有已启用的信号检查

    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        kline_df: 历史日K线 DataFrame
        latest_price: 最新价
        config: 完整配置字典
        dedup: 可选的去重器
        change_pct: 当前涨跌幅

    Returns:
        信号列表（可能为空）
    """
    if kline_df is None or kline_df.empty:
        return []

    closes = kline_df["close"].values.astype(float)
    volumes = kline_df["volume"].values.astype(float) if "volume" in kline_df.columns else None

    signals_cfg = config.get("signals", {})
    all_signals = []

    # 检查顺序
    checks = [
        ("ma_crossover", _check_ma_crossover, closes),
        ("rsi", _check_rsi, closes),
        ("macd", _check_macd, closes),
        ("bollinger", _check_bollinger, closes),
        ("volume_breakout", _check_volume_breakout, volumes if volumes is not None else np.array([])),
    ]

    for sig_name, check_func, data in checks:
        params = signals_cfg.get(sig_name, {})
        if not params.get("enabled", False):
            continue

        # 去重检查
        if dedup and dedup.is_duplicate(stock_code, sig_name):
            logger.debug("股票 %s 信号 %s 已推送过，跳过", stock_code, sig_name)
            continue

        sig = check_func(data, stock_code, stock_name, latest_price, params)
        if sig:
            sig.change_pct = change_pct
            all_signals.append(sig)
            if dedup:
                dedup.mark_sent(stock_code, sig_name)

    return all_signals


def check_price_alerts(
    stock_code: str,
    stock_name: str,
    latest_price: float,
    config: dict,
    dedup: Optional["SignalDedup"] = None,
) -> list[Signal]:
    """检查价格预警

    config.yaml 中定义:
      price_alerts:
        "600487":
          above: 20.0
          below: 15.0
    """
    alerts = config.get("price_alerts") or {}
    rule = alerts.get(stock_code)
    if not rule:
        return []

    signals = []
    above = rule.get("above")
    below = rule.get("below")

    # 涨破目标价
    if above is not None and latest_price >= above:
        sig_type = "price_alert_above"
        if not dedup or not dedup.is_duplicate(stock_code, sig_type):
            signals.append(Signal(
                stock_code=stock_code, stock_name=stock_name,
                signal_type=sig_type, signal_label=f"突破{above}元 🎯",
                direction="bullish", price=latest_price,
                suggestion="可考虑止盈",
                message=(
                    f"🎯 **{stock_name}({stock_code}) 突破目标价**\n"
                    f"当前价: {latest_price:.2f}\n"
                    f"已突破目标价 **{above}元**，达到预期！\n"
                    f"到了目标价位，可考虑分批止盈落袋为安"
                ),
            ))
            if dedup:
                dedup.mark_sent(stock_code, sig_type)

    # 跌破目标价
    if below is not None and latest_price <= below:
        sig_type = "price_alert_below"
        if not dedup or not dedup.is_duplicate(stock_code, sig_type):
            signals.append(Signal(
                stock_code=stock_code, stock_name=stock_name,
                signal_type=sig_type, signal_label=f"跌破{below}元 ⚠️",
                direction="bearish", price=latest_price,
                suggestion="建议止损",
                message=(
                    f"⚠️ **{stock_name}({stock_code}) 跌破目标价**\n"
                    f"当前价: {latest_price:.2f}\n"
                    f"已跌破止损价 **{below}元**，注意风险！\n"
                    f"到了止损位，建议按计划执行，控制亏损"
                ),
            ))
            if dedup:
                dedup.mark_sent(stock_code, sig_type)

    return signals


def generate_fund_flow_signal(
    stock_code: str, stock_name: str,
    latest_price: float, fund_flow: Optional[dict],
    config: dict, dedup: Optional["SignalDedup"] = None,
) -> Optional[Signal]:
    """资金流向信号 — 主力大额流入/流出时推送"""
    if not fund_flow:
        return None

    ff_cfg = config.get("fund_flow", {})
    if not ff_cfg.get("enabled", False):
        return None

    threshold = ff_cfg.get("alert_threshold", 30_000_000)
    main_net = fund_flow.get("main_force_net", 0)

    sig_type = "fund_flow"
    if dedup and dedup.is_duplicate(stock_code, sig_type):
        return None

    if abs(main_net) < threshold:
        return None

    direction = "bullish" if main_net > 0 else "bearish"
    label = f"主力{'流入' if main_net>0 else '流出'} {abs(main_net)/1e8:.1f}亿"
    suggestion_text = "中长线可关注" if main_net > 0 else "注意风险"

    return Signal(
        stock_code=stock_code, stock_name=stock_name,
        signal_type=sig_type, signal_label=label,
        direction=direction, price=latest_price,
        suggestion=suggestion_text,
        message=(
            f"{'💰' if main_net>0 else '💸'} **{stock_name}({stock_code}) "
            f"主力资金{'净流入' if main_net>0 else '净流出'}**\n"
            f"金额: **{abs(main_net)/1e8:.2f}亿**\n"
            f"占比: {fund_flow.get('main_force_ratio', 0):.2f}%\n"
            f"状态: {fund_flow.get('flow_status', '')}"
        ),
        extra={"main_force_net": main_net},
    )


# ──────────────────────────────────────────────
#  行情简报（当前所有指标状态一览）
# ──────────────────────────────────────────────

def generate_status_report(
    stock_code: str,
    stock_name: str,
    kline_df: pd.DataFrame,
    latest_price: float,
    realtime: dict,
    config: dict,
    fund_flow: Optional[dict] = None,
) -> str:
    """生成一只股票的当前指标状态简报（通俗易懂版）"""
    if kline_df is None or kline_df.empty:
        return f"⚠️ **{stock_name}({stock_code})** 数据不足，无法生成简报"

    closes = kline_df["close"].values.astype(float)
    volumes = kline_df["volume"].values.astype(float) if "volume" in kline_df.columns else None
    signals_cfg = config.get("signals", {})
    today = datetime.now().strftime("%m/%d")

    lines = []
    lines.append(f"📊 **{stock_name}({stock_code})**{get_sector_tag(stock_code)} · {today} 盘中情况")
    lines.append("")

    # ── 基本信息 ──
    chg = realtime.get("change_pct", 0)
    chg_icon = "📈" if chg > 0 else ("📉" if chg < 0 else "➖")
    vol_raw = realtime.get("volume", 0)
    lines.append(f"{chg_icon} **{latest_price:.2f}元**  ({chg:+.2f}%)  成交量 {_fmt_volume(vol_raw)}")
    lines.append("")

    # ── 趋势一览（新增） ──
    ma_cfg_t = signals_cfg.get("ma_crossover", {})
    if ma_cfg_t.get("enabled", False) and len(closes) >= 25:
        s5 = _sma(closes, 5)
        s10 = _sma(closes, 10)
        s20 = _sma(closes, 20)
        valid = ~np.isnan(s5) & ~np.isnan(s10) & ~np.isnan(s20)
        s5v, s10v, s20v = s5[valid], s10[valid], s20[valid]
        if len(s5v) > 1:
            # 短期趋势（MA5方向）
            short_dir = "上涨 ↗" if s5v[-1] > s5v[-2] else "下跌 ↘"
            # 中期趋势（MA20方向）
            mid_dir = "上涨 ↗" if s20v[-1] > s20v[-2] else "下跌 ↘"
            # 均线排列
            if s5v[-1] > s10v[-1] > s20v[-1]:
                arrange = "多头排列 ✅"
            elif s5v[-1] < s10v[-1] < s20v[-1]:
                arrange = "空头排列 ❌"
            else:
                arrange = "均线交错 ⚠️"

            lines.append(f"📌 **趋势**  短期{short_dir}  |  中期{mid_dir}  |  {arrange}")
            lines.append("")

    summary_items = []   # 收集判断，最后生成总体评价
    score = 50           # 基础分50，各项加减

    # ── 均线（趋势判断） ──
    ma_cfg = signals_cfg.get("ma_crossover", {})
    if ma_cfg.get("enabled", False):
        sp = ma_cfg.get("ma_short", 5)
        lp = ma_cfg.get("ma_long", 20)
        if len(closes) >= lp + 1:
            s_ma = _sma(closes, sp)
            l_ma = _sma(closes, lp)
            valid = ~np.isnan(s_ma) & ~np.isnan(l_ma)
            sv, lv = s_ma[valid], l_ma[valid]
            if len(sv) > 0:
                cs, cl = sv[-1], lv[-1]
                diff_ma = ((cs - cl) / cl) * 100 if cl > 0 else 0
                if cs > cl:
                    summary_items.append(("trend", "up"))
                    score += 15
                    lines.append(f"📈 **趋势：上涨**")
                    lines.append(f"   短期均线({cs:.2f}) 在 长期均线({cl:.2f}) 之上")
                    lines.append(f"   说明最近买入的人都在赚钱，走势偏强 ✓")
                elif cs < cl:
                    summary_items.append(("trend", "down"))
                    score -= 15
                    lines.append(f"📉 **趋势：下跌**")
                    lines.append(f"   短期均线({cs:.2f}) 在 长期均线({cl:.2f}) 之下")
                    lines.append(f"   说明最近买入的人被套了，走势偏弱 ⚠️")
                else:
                    summary_items.append(("trend", "flat"))
                    lines.append(f"➖ **趋势：横盘**")
                    lines.append(f"   短期均线和长期均线差不多，方向不明")
                lines.append("")

    # ── RSI（买卖力道） ──
    rsi_cfg = signals_cfg.get("rsi", {})
    if rsi_cfg.get("enabled", False):
        period = rsi_cfg.get("period", 14)
        ob = rsi_cfg.get("overbought", 70)
        os_ = rsi_cfg.get("oversold", 30)
        if len(closes) >= period + 1:
            rsi_val = _calc_rsi(closes, period)
            if rsi_val is not None:
                if rsi_val >= ob:
                    summary_items.append(("rsi", "hot"))
                    score -= 10
                    lines.append(f"🔥 **热度：过热了**（{rsi_val:.0f}分，超过{ob}）")
                    lines.append(f"   最近涨得太快，短线可能要回调，注意风险")
                elif rsi_val >= 60:
                    summary_items.append(("rsi", "strong"))
                    score += 10
                    lines.append(f"⚡ **热度：偏强**（{rsi_val:.0f}分）")
                    lines.append(f"   上涨动力不错，但还没到过热")
                elif rsi_val >= 40:
                    summary_items.append(("rsi", "mid"))
                    lines.append(f"➖ **热度：不温不火**（{rsi_val:.0f}分）")
                    lines.append(f"   没涨没跌，方向需要再观察")
                elif rsi_val >= os_:
                    summary_items.append(("rsi", "weak"))
                    score -= 5
                    lines.append(f"💧 **热度：偏弱**（{rsi_val:.0f}分）")
                    lines.append(f"   跌得有点多，但还没到极端")
                else:
                    summary_items.append(("rsi", "cold"))
                    score += 10  # 超卖是机会
                    lines.append(f"🧊 **热度：跌过头了**（{rsi_val:.0f}分，低于{os_}）")
                    lines.append(f"   短期跌得太多，可能会有反弹机会")
                lines.append("")

    # ── MACD（多空力量） ──
    macd_cfg = signals_cfg.get("macd", {})
    if macd_cfg.get("enabled", False):
        fast = macd_cfg.get("fast", 12)
        slow = macd_cfg.get("slow", 26)
        sig_p = macd_cfg.get("signal", 9)
        if len(closes) >= slow + sig_p:
            ema_f = _ema(closes, fast)
            ema_s = _ema(closes, slow)
            diff_arr = ema_f - ema_s
            dea_arr = _ema(diff_arr, sig_p)
            valid = ~np.isnan(diff_arr) & ~np.isnan(dea_arr)
            dv, deav = diff_arr[valid], dea_arr[valid]
            if len(dv) > 0:
                cd, cdea = dv[-1], deav[-1]
                if cd > cdea and cd > 0:
                    score += 10
                    lines.append(f"✅ **多空力量：多头占优**")
                    lines.append(f"   上涨的力量比下跌的力量强，整体偏多")
                elif cd < cdea and cd < 0:
                    score -= 10
                    lines.append(f"❌ **多空力量：空头占优**")
                    lines.append(f"   下跌的力量比上涨的力量强，整体偏空")
                elif cd > cdea and cd < 0:
                    score += 5
                    lines.append(f"↗️ **多空力量：空头减弱**")
                    lines.append(f"   虽然还在跌，但下跌力量在变小")
                elif cd < cdea and cd > 0:
                    score -= 5
                    lines.append(f"↘️ **多空力量：涨不动了**")
                    lines.append(f"   虽然还在涨，但上涨力量在变小")
                else:
                    lines.append(f"➖ **多空力量：势均力敌**")
                    lines.append(f"   多空双方差不多，方向不明朗")
                lines.append("")

    # ── 布林带（价格位置） ──
    bol_cfg = signals_cfg.get("bollinger", {})
    if bol_cfg.get("enabled", False):
        period = bol_cfg.get("period", 20)
        k = bol_cfg.get("std_dev", 2.0)
        if len(closes) >= period + 1:
            mid = _sma(closes, period)
            sd = _std(closes, period)
            up = mid + k * sd
            lo = mid - k * sd
            valid = ~np.isnan(mid) & ~np.isnan(up) & ~np.isnan(lo)
            uv, lov, mv = up[valid], lo[valid], mid[valid]
            if len(uv) > 0:
                cu, cl, cm = uv[-1], lov[-1], mv[-1]
                if latest_price >= cu:
                    score -= 8
                    lines.append(f"🔴 **价格位置：涨得太高了**")
                    lines.append(f"   已经突破上轨({cu:.2f})，短线可能有回调")
                elif latest_price <= cl:
                    score += 8
                    lines.append(f"🟢 **价格位置：跌得够低了**")
                    lines.append(f"   已经跌破下轨({cl:.2f})，短线可能有反弹")
                elif latest_price >= cm:
                    score += 3
                    lines.append(f"🟠 **价格位置：中等偏高**")
                    lines.append(f"   在中间偏上位置，还有上涨空间")
                else:
                    score -= 3
                    lines.append(f"🔵 **价格位置：中等偏低**")
                    lines.append(f"   在中间偏下位置，还算便宜")
                lines.append("")

    # ── 成交量 ──
    vol_cfg = signals_cfg.get("volume_breakout", {})
    if vol_cfg.get("enabled", False) and volumes is not None:
        period = vol_cfg.get("ma_period", 5)
        ratio = vol_cfg.get("ratio", 1.5)
        if len(volumes) >= period + 1:
            avg_v = _sma(volumes, period)
            valid = ~np.isnan(avg_v)
            av = avg_v[valid]
            if len(av) > 0:
                cv = volumes[-1]
                ca = av[-1]
                if ca > 0:
                    vr = cv / ca
                    if vr >= ratio:
                        score += 8
                        lines.append(f"🔥 **成交量：放量！** 是平时的 **{vr:.1f}倍**")
                        lines.append(f"   说明今天资金很活跃，有大动静")
                    elif vr >= 1.2:
                        score += 3
                        lines.append(f"💡 **成交量：比平时活跃** 是平时的{vr:.1f}倍")
                        lines.append(f"   资金在增加关注")
                    else:
                        lines.append(f"💤 **成交量：正常水平** 是平时的{vr:.1f}倍")
                        lines.append(f"   没有异常资金进出")
                    lines.append("")

    # ── 资金流向 ──
    if fund_flow:
        main_net = fund_flow.get("main_force_net", 0)
        main_ratio = fund_flow.get("main_force_ratio", 0)
        flow_status = fund_flow.get("flow_status", "")
        if abs(main_net) >= 50_000_000:
            score += 15 if main_net > 0 else -15
        elif abs(main_net) >= 10_000_000:
            score += 8 if main_net > 0 else -8
        lines.append(f"💰 **资金流向：{flow_status}**")
        if main_net > 0:
            lines.append(f"   主力买了 **{_fmt_money(main_net)}**，占比 {main_ratio:.1f}%")
            lines.append(f"   机构资金在进场，跟着大资金走")
        else:
            lines.append(f"   主力卖了 **{_fmt_money(abs(main_net))}**，占比 {abs(main_ratio):.1f}%")
            lines.append(f"   机构资金在撤退，小心跟随")
        lines.append("")

    # ── 总体判断（底部汇总） ──
    score = max(0, min(100, score))

    # 给出操作建议
    if score >= 70:
        suggestion_line = "💡 **建议：适合持有，趋势向好**"
        lines.append(f"⭐ **总体判断：偏多**（{score}分）")
        lines.append(f"   多个指标显示走势不错，但涨多了也要注意回调")
    elif score >= 45:
        suggestion_line = "💡 **建议：可继续持有观察**"
        lines.append(f"⭐ **总体判断：中性偏多**（{score}分）")
        lines.append(f"   整体走势还行，可以继续持有观察")
    elif score >= 30:
        suggestion_line = "💡 **建议：适当控制仓位，注意风险**"
        lines.append(f"⭐ **总体判断：中性偏弱**（{score}分）")
        lines.append(f"   走势有点弱，注意控制风险")
    else:
        suggestion_line = "💡 **建议：多看少动，等待机会**"
        lines.append(f"⭐ **总体判断：偏弱**（{score}分）")
        lines.append(f"   多个指标走弱，多看少动，等机会")

    lines.append("")
    lines.append(suggestion_line)

    return "\n".join(lines).strip()


def _calc_rsi(closes: np.ndarray, period: int = 14) -> Optional[float]:
    """计算最近一个 RSI 值"""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = _sma(gains, period)
    avg_loss = _sma(losses, period)
    valid = (avg_loss > 0) & ~np.isnan(avg_gain) & ~np.isnan(avg_loss)
    if not valid.any():
        return None
    rs = avg_gain[valid] / avg_loss[valid]
    rsi_vals = 100 - (100 / (1 + rs))
    return float(rsi_vals[-1]) if len(rsi_vals) > 0 else None


def _fmt_volume(v: float) -> str:
    """格式化成交量显示"""
    if v >= 100_000_000:
        return f"{v / 100_000_000:.2f}亿"
    if v >= 10_000:
        return f"{v / 10_000:.1f}万"
    return f"{v:.0f}"


def _fmt_money(v: float) -> str:
    """格式化金额显示"""
    if abs(v) >= 100_000_000:
        return f"{v / 100_000_000:.2f}亿"
    if abs(v) >= 10_000:
        return f"{v / 10_000:.1f}万"
    return f"{v:.0f}元"


def calc_atr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> float:
    """计算 ATR（平均真实波幅）

    用于自适应止损：止损位 = 买入价 - ATR × 2.5
    """
    if len(closes) < period + 1:
        return 0.0
    tr_values = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1]),
        )
        tr_values.append(tr)
    if len(tr_values) < period:
        return 0.0
    return float(np.mean(tr_values[-period:]))
