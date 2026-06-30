"""
策略对比回测 — 旧策略(MA金叉死叉) vs 新策略(V5评分系统)
结果推送到企业微信
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import numpy as np
from datetime import datetime

from src.fetcher import fetch_kline
from src.backtest import backtest_ma_crossover, backtest_scoring_strategy
from src.notifier import notify

# 加载配置
config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
stocks = config.get("stocks", {})

# 回测参数
OLD_STRAT_PARAMS = {
    "ma_short": 10,
    "ma_long": 40,
    "rsi_filter": True,
    "stop_loss_pct": 8,
}

results = []

print(f"策略对比回测 · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"监控股票: {len(stocks)} 只")
print("=" * 60)

for code, name in stocks.items():
    print(f"\n正在回测 {name}({code})...", end=" ")

    kline = fetch_kline(code, days=365)
    if kline is None or len(kline) < 40:
        print("数据不足，跳过")
        continue

    # 旧策略: MA金叉死叉 (MA10/MA40 + RSI过滤 + 止损8%)
    old_result = backtest_ma_crossover(
        code, name, kline,
        ma_short=OLD_STRAT_PARAMS["ma_short"],
        ma_long=OLD_STRAT_PARAMS["ma_long"],
        rsi_filter=OLD_STRAT_PARAMS["rsi_filter"],
        stop_loss_pct=OLD_STRAT_PARAMS["stop_loss_pct"],
    )

    # 新策略: V5评分系统
    new_result = backtest_scoring_strategy(
        code, name, kline,
        stop_loss_pct=8,
    )

    print(f"旧: {old_result.total_return:+.1f}% | 新: {new_result.total_return:+.1f}%")

    results.append({
        "code": code,
        "name": name,
        "old_return": old_result.total_return,
        "old_win_rate": old_result.win_rate,
        "old_trades": old_result.total_trades,
        "old_drawdown": old_result.max_drawdown,
        "new_return": new_result.total_return,
        "new_win_rate": new_result.win_rate,
        "new_trades": new_result.total_trades,
        "new_drawdown": new_result.max_drawdown,
    })

if not results:
    print("没有足够的回测数据")
    sys.exit(1)

# 汇总统计
old_avg = np.mean([r["old_return"] for r in results])
new_avg = np.mean([r["new_return"] for r in results])
old_median = np.median([r["old_return"] for r in results])
new_median = np.median([r["new_return"] for r in results])
old_win_count = sum(1 for r in results if r["old_return"] > 0)
new_win_count = sum(1 for r in results if r["new_return"] > 0)
old_avg_dd = np.mean([r["old_drawdown"] for r in results])
new_avg_dd = np.mean([r["new_drawdown"] for r in results])
better_count = sum(1 for r in results if r["new_return"] > r["old_return"])
worse_count = sum(1 for r in results if r["new_return"] < r["old_return"])

# 构建推送消息
today_str = datetime.now().strftime("%m/%d")
lines = []
lines.append(f"策略对比回测报告 {today_str}")
lines.append("")
lines.append(f"监控股票: {len(results)} 只 | 数据区间: 近1年")
lines.append("")
lines.append(f"总体对比")
old_icon = "上涨" if old_avg > new_avg else "下跌"
new_icon = "上涨" if new_avg > old_avg else "下跌"
lines.append(f"旧策略(MA10/40+RSI+止损)：平均 {old_avg:+.1f}% / 中位数 {old_median:+.1f}%")
lines.append(f"  盈利 {old_win_count}/{len(results)} 只 / 回撤 {old_avg_dd:.1f}%")
lines.append(f"新策略(评分系统V5)：平均 {new_avg:+.1f}% / 中位数 {new_median:+.1f}%")
lines.append(f"  盈利 {new_win_count}/{len(results)} 只 / 回撤 {new_avg_dd:.1f}%")
lines.append("")

# 每只股对比
lines.append(f"各股回测明细")
for r in sorted(results, key=lambda x: x["new_return"] - x["old_return"], reverse=True):
    diff = r["new_return"] - r["old_return"]
    icon = "△" if diff > 5 else ("▽" if diff < -5 else "→")
    old_s = f"{r['old_return']:+.1f}%" if r["old_trades"] > 0 else "无"
    lines.append(f"{icon} {r['name']}({r['code']})")
    lines.append(f"  旧:{old_s} → 新:{r['new_return']:+.1f}%  胜率:{r['old_win_rate']:.0f}%→{r['new_win_rate']:.0f}%")

lines.append("")

# 结论
lines.append(f"结论")
improvement = new_avg - old_avg
if better_count > worse_count:
    lines.append(f"新策略胜出: 跑赢{better_count}只 跑输{worse_count}只 平均提升{improvement:+.1f}%")
elif worse_count > better_count:
    lines.append(f"旧策略胜出: 跑赢{worse_count}只 跑输{better_count}只")
else:
    lines.append(f"持平: 各{better_count}只")

lines.append("")
lines.append(f"-- {datetime.now().strftime('%Y-%m-%d %H:%M')}")

content = "\n".join(lines)

print("\n推送到企业微信...")
success = notify(config, "策略对比回测", content)
if success:
    print("推送成功！")
else:
    print("推送失败，终端输出：")
    print(content)
