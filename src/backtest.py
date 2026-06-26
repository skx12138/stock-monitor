"""
回测系统 — 用历史数据验证策略收益

支持的策略:
  - ma_crossover: 均线金叉买入 / 死叉卖出
  - rsi_reversal: RSI超卖买入 / 超买卖出
  - macd_crossover: MACD金叉买入 / 死叉卖出
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.fetcher import fetch_kline
from src.signals import _sma, _ema, _calc_rsi, _fmt_volume, _fmt_money

logger = logging.getLogger(__name__)


# ── 交易记录 ──

@dataclass
class Trade:
    """一次交易记录"""
    date: str
    action: str          # "buy" / "sell"
    price: float
    reason: str = ""
    profit_pct: float = 0.0      # 本次交易盈亏
    profit_amount: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    stock_code: str
    stock_name: str
    strategy: str
    start_date: str
    end_date: str
    total_trades: int = 0
    win_trades: int = 0
    loss_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0    # 总收益率
    max_drawdown: float = 0.0    # 最大回撤
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)


# ── 策略实现 ──

def backtest_ma_crossover(
    code: str, name: str,
    df: pd.DataFrame,
    ma_short: int = 10,
    ma_long: int = 40,
    initial_cash: float = 100000,
    rsi_filter: bool = True,       # 金叉时RSI需>50才买入
    stop_loss_pct: float = 8,       # 止损百分比，0=不设止损
) -> BacktestResult:
    """均线金叉/死叉回测（带可选的RSI过滤和止损）

    规则:
      - 金叉买入（MA5上穿MA20），全仓
      - 死叉卖出（MA5下穿MA20），全仓
      - 可选：金叉时RSI>50才买入，避免假突破
      - 可选：设置止损线，亏到X%强制卖出
    """
    closes = df["close"].values.astype(float)
    dates = df["date"].values

    short_ma = _sma(closes, ma_short)
    long_ma = _sma(closes, ma_long)

    valid = ~np.isnan(short_ma) & ~np.isnan(long_ma)
    s_v, l_v = short_ma[valid], long_ma[valid]
    d_v = dates[valid]

    if len(s_v) < 2:
        return BacktestResult(code, name, f"MA{ma_short}/MA{ma_long}", "", "")

    strat_name = f"均线{ma_short}/{ma_long}"
    if rsi_filter:
        strat_name += "+RSI过滤"
    if stop_loss_pct > 0:
        strat_name += f"+止损{stop_loss_pct}%"

    result = BacktestResult(
        stock_code=code, stock_name=name,
        strategy=strat_name,
        start_date=str(d_v[0]),
        end_date=str(d_v[-1]),
    )

    cash = initial_cash
    hold = 0.0
    buy_price = 0.0       # 记录买入价，用于计算止损
    prev_s, prev_l = s_v[0], l_v[0]
    in_position = False
    peak = initial_cash

    result.equity_curve.append(initial_cash)

    for i in range(1, len(s_v)):
        curr_s, curr_l = s_v[i], l_v[i]
        curr_price = float(closes[np.where(dates == d_v[i])[0][0]])
        curr_date = str(d_v[i])

        # 止损检查
        if in_position and stop_loss_pct > 0 and buy_price > 0:
            loss = (curr_price - buy_price) / buy_price * 100
            if loss <= -stop_loss_pct:
                cash = hold * curr_price
                hold = 0
                in_position = False
                result.trades.append(Trade(
                    date=curr_date, action="sell(止损)",
                    price=round(curr_price, 2),
                    profit_pct=round(loss, 2),
                    profit_amount=round(cash - initial_cash, 2),
                    reason=f"触发止损{stop_loss_pct}%",
                ))
                if loss > 0:
                    result.win_trades += 1
                else:
                    result.loss_trades += 1
                total = cash
                result.equity_curve.append(total)
                if total > peak:
                    peak = total
                continue

        # 金叉买入
        if prev_s <= prev_l and curr_s > curr_l and not in_position:
            # RSI过滤：金叉时RSI需>50
            if rsi_filter:
                rsi_val = _calc_rsi(closes, 14)
                if rsi_val is None or rsi_val <= 50:
                    prev_s, prev_l = curr_s, curr_l
                    continue
            hold = cash / curr_price
            cash = 0
            buy_price = curr_price
            in_position = True
            result.trades.append(Trade(
                date=curr_date, action="buy",
                price=round(curr_price, 2),
                reason=f"MA{ma_short}({curr_s:.2f})上穿MA{ma_long}({curr_l:.2f})",
            ))

        # 死叉卖出
        elif prev_s >= prev_l and curr_s < curr_l and in_position:
            cash = hold * curr_price
            profit_pct = (curr_price / buy_price - 1) * 100 if buy_price > 0 else 0
            hold = 0
            buy_price = 0
            in_position = False
            result.trades.append(Trade(
                date=curr_date, action="sell",
                price=round(curr_price, 2),
                profit_pct=round(profit_pct, 2),
                profit_amount=round(cash - initial_cash, 2),
                reason=f"MA{ma_short}({curr_s:.2f})下穿MA{ma_long}({curr_l:.2f})",
            ))
            if profit_pct > 0:
                result.win_trades += 1
            else:
                result.loss_trades += 1

        # 计算当前总资产
        total = cash + hold * curr_price
        result.equity_curve.append(total)
        if total > peak:
            peak = total

        prev_s, prev_l = curr_s, curr_l

    # 平仓
    if in_position:
        cash = hold * float(closes[-1])
        if result.trades:
            profit_pct = (float(closes[-1]) / buy_price - 1) * 100 if buy_price > 0 else 0
            result.trades.append(Trade(
                date=str(dates[-1]), action="sell(收盘)",
                price=round(float(closes[-1]), 2),
                profit_pct=round(profit_pct, 2),
            ))

    result.total_trades = result.win_trades + result.loss_trades
    result.total_return = round((cash - initial_cash) / initial_cash * 100, 2)
    result.win_rate = round(result.win_trades / result.total_trades * 100, 2) if result.total_trades > 0 else 0

    peak_val = initial_cash
    for v in result.equity_curve:
        if v > peak_val:
            peak_val = v
        dd = (peak_val - v) / peak_val * 100
        if dd > result.max_drawdown:
            result.max_drawdown = round(dd, 2)

    return result


def print_backtest_result(result: BacktestResult) -> str:
    """打印回测结果"""
    lines = []
    lines.append(f"📊 **{result.stock_name}({result.stock_code}) · {result.strategy} 回测报告**")
    lines.append(f"回测区间: {result.start_date} → {result.end_date}")
    lines.append("")

    # 收益概况
    ret_icon = "📈" if result.total_return > 0 else "📉"
    lines.append(f"**收益概况**")
    lines.append(f"  {ret_icon} 总收益率: **{result.total_return:+.2f}%**")
    lines.append(f"  📉 最大回撤: **-{result.max_drawdown:.2f}%**")
    lines.append("")

    # 交易统计
    lines.append(f"**交易统计**")
    lines.append(f"  总交易次数: {result.total_trades} 次")
    if result.total_trades > 0:
        lines.append(f"  胜率: **{result.win_rate:.1f}%**")
        lines.append(f"  盈利: {result.win_trades} 次  |  亏损: {result.loss_trades} 次")
    lines.append("")

    # 交易明细（最多展示10笔）
    if result.trades:
        lines.append(f"**交易明细**")
        for t in result.trades[:10]:
            if t.action == "buy":
                lines.append(f"  🟢 {t.date} 买入 {t.price:.2f}元  {t.reason}")
            else:
                p_icon = "🟢" if t.profit_pct > 0 else "🔴"
                lines.append(f"  {p_icon} {t.date} 卖出 {t.price:.2f}元  {t.profit_pct:+.2f}%")

    return "\n".join(lines).strip()
