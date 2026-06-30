"""策略自动优化 — 根据回测数据自动调整策略参数"""
import json
import os
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

OPTIMIZER_FILE = "strategy_optimizer.json"

# 可调参数及范围（匹配V5评分体系）
TUNABLE_PARAMS = {
    "buy_threshold": {"min": 50, "max": 75, "default": 60, "step": 2},
    "sell_threshold": {"min": 35, "max": 55, "default": 45, "step": 2},
    "stop_loss": {"min": 5, "max": 15, "default": 8, "step": 1},
    "trail_activate": {"min": 3, "max": 10, "default": 6, "step": 1},
    "trail_pullback": {"min": 3, "max": 8, "default": 5, "step": 1},
}

# 评分权重及范围
WEIGHT_DEFAULTS = {
    "base": {"均线": 25, "RSI": 20, "MACD": 20, "成交量": 15, "资金流向": 20},
    "trending": {"均线": 30, "RSI": 15, "MACD": 20, "成交量": 15, "资金流向": 20},
    "declining": {"均线": 15, "RSI": 15, "MACD": 15, "成交量": 20, "资金流向": 30},
}
WEIGHT_RANGES = {"min": 5, "max": 40, "step": 5}

# 多策略配置模板（匹配V5评分）
STRATEGY_TEMPLATES = {
    "激进": {"buy_threshold": 55, "sell_threshold": 40, "stop_loss": 10, "trail_activate": 8, "trail_pullback": 6,
             "desc": "较低门槛买入+宽止损，适合ETF和强势股"},
    "稳健": {"buy_threshold": 60, "sell_threshold": 45, "stop_loss": 8, "trail_activate": 6, "trail_pullback": 5,
             "desc": "默认参数，适合大多数股票"},
    "保守": {"buy_threshold": 65, "sell_threshold": 50, "stop_loss": 6, "trail_activate": 4, "trail_pullback": 3,
             "desc": "高门槛买入+紧止损，适合弱势股"},
}

def classify_stock(avg_return: float, win_rate: float, max_drawdown: float) -> str:
    """根据回测结果给股票分类"""
    score = avg_return * 0.4 + win_rate * 0.4 - max_drawdown * 0.2
    if score > 50:
        return "激进"
    elif score > 10:
        return "稳健"
    return "保守"

def assign_stock_strategies() -> dict:
    """为每只股票分配最优策略"""
    from src.fetcher import fetch_kline
    from src.backtest import backtest_scoring_strategy
    import yaml
    
    config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    stocks = config.get("stocks", {})
    
    result = {}
    for code, name in stocks.items():
        kline = fetch_kline(code, 365)
        if kline is not None:
            r = backtest_scoring_strategy(code, name, kline)
            strategy = classify_stock(r.total_return, r.win_rate, r.max_drawdown)
            result[code] = {
                "strategy": strategy,
                "params": STRATEGY_TEMPLATES[strategy],
                "return": round(r.total_return, 1),
                "win_rate": round(r.win_rate, 1),
                "drawdown": round(r.max_drawdown, 1),
            }
            logger.info("股票%s(%s) 回测%.1f%% 胜率%.0f%% → %s策略", name, code, r.total_return, r.win_rate, strategy)
    return result

def get_stock_params(code: str) -> dict:
    """获取某只股票的个性化参数（回退到默认）"""
    state = load_state()
    per_stock = state.get("per_stock_strategies", {})
    if code in per_stock:
        return per_stock[code]["params"]
    return {k: v["default"] for k, v in TUNABLE_PARAMS.items()}

def refresh_strategies():
    """刷新所有股票的策略分类并保存"""
    state = load_state()
    state["per_stock_strategies"] = assign_stock_strategies()
    save_state(state)
    return state["per_stock_strategies"]

def load_state() -> dict:
    if os.path.exists(OPTIMIZER_FILE):
        with open(OPTIMIZER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "current_params": {k: v["default"] for k, v in TUNABLE_PARAMS.items()},
        "current_weights": {mode: dict(v) for mode, v in WEIGHT_DEFAULTS.items()},
        "history": [],
        "performance_log": [],
    }

def save_state(state: dict):
    with open(OPTIMIZER_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def auto_optimize(avg_return: float, win_rate: float) -> dict:
    """根据回测结果自动优化策略参数，返回调整建议"""
    state = load_state()
    
    # 记录本次性能
    state["performance_log"].append({
        "avg_return": round(avg_return, 2),
        "win_rate": round(win_rate, 2),
    })
    state["performance_log"] = state["performance_log"][-10:]  # 保留最近10次
    
    result = {"adjusted": False, "changes": {}, "reason": ""}
    
    # 需要至少5次记录才能判断趋势
    if len(state["performance_log"]) < 5:
        result["reason"] = f"数据不足(仅{len(state['performance_log'])}次)，继续观察"
        save_state(state)
        return result
    
    recent = state["performance_log"][-5:]
    returns = [r["avg_return"] for r in recent]
    
    # 判断趋势
    declining = all(returns[i] >= returns[i+1] for i in range(len(returns)-1))
    very_bad = avg_return < -10
    
    if not declining and not very_bad:
        result["reason"] = "策略表现稳定，无需调整"
        save_state(state)
        return result
    
    # 需要优化
    old_params = deepcopy(state["current_params"])
    old_weights = deepcopy(state.get("current_weights", {}))
    
    if very_bad:
        state["current_params"]["buy_threshold"] = max(
            TUNABLE_PARAMS["buy_threshold"]["min"],
            state["current_params"]["buy_threshold"] - TUNABLE_PARAMS["buy_threshold"]["step"]
        )
        state["current_params"]["sell_threshold"] = max(
            TUNABLE_PARAMS["sell_threshold"]["min"],
            state["current_params"]["sell_threshold"] - TUNABLE_PARAMS["sell_threshold"]["step"]
        )
        # 大幅亏损时：提高资金流向权重（更注重主力资金），降低均线权重（均线滞后）
        if "current_weights" in state:
            w = state["current_weights"]["base"]
            w["资金流向"] = min(WEIGHT_RANGES["max"], w.get("资金流向", 20) + WEIGHT_RANGES["step"])
            w["均线"] = max(WEIGHT_RANGES["min"], w.get("均线", 25) - WEIGHT_RANGES["step"])
        result["reason"] = f"连续亏损({avg_return:.1f}%)，降低买卖阈值+提高资金流向权重"
    elif declining:
        state["current_params"]["stop_loss"] = min(
            TUNABLE_PARAMS["stop_loss"]["max"],
            state["current_params"]["stop_loss"] + TUNABLE_PARAMS["stop_loss"]["step"]
        )
        # 持续下降时：提高成交量权重（量先行），降低RSI权重（RSI震荡市中容易误判）
        if "current_weights" in state:
            w = state["current_weights"]["base"]
            w["成交量"] = min(WEIGHT_RANGES["max"], w.get("成交量", 15) + WEIGHT_RANGES["step"])
            w["RSI"] = max(WEIGHT_RANGES["min"], w.get("RSI", 20) - WEIGHT_RANGES["step"])
        result["reason"] = f"收益持续下降，放宽止损+提高成交量权重"
    
    changes = {}
    for k in TUNABLE_PARAMS:
        if state["current_params"][k] != old_params[k]:
            changes[k] = {"from": old_params[k], "to": state["current_params"][k]}
    
    # 检查权重变化
    if old_weights and state.get("current_weights", {}):
        nw = state["current_weights"]["base"]
        ow = old_weights.get("base", {})
        for wk in WEIGHT_DEFAULTS["base"]:
            if nw.get(wk) != ow.get(wk):
                changes[f"权重_{wk}"] = {"from": ow.get(wk), "to": nw.get(wk)}
    
    if changes:
        result["adjusted"] = True
        result["changes"] = changes
        state["history"].append({
            "date": __import__("datetime").date.today().isoformat(),
            "old_params": old_params,
            "new_params": deepcopy(state["current_params"]),
            "new_weights": deepcopy(state.get("current_weights", {})),
            "reason": result["reason"],
        })
    
    save_state(state)
    return result

def get_param_summary() -> str:
    """获取当前参数摘要"""
    state = load_state()
    params = state["current_params"]
    lines = ["📋 **当前策略参数**", ""]
    for k, v in params.items():
        label = {"buy_threshold": "买入阈值", "sell_threshold": "卖出阈值",
                 "stop_loss": "止损%", "trail_activate": "移动止盈启动",
                 "trail_pullback": "移动止盈回撤"}.get(k, k)
        default = TUNABLE_PARAMS[k]["default"]
        marker = " ⚡" if v != default else ""
        lines.append(f"  {label}: {v}{marker}")
    
    recent = state.get("performance_log", [])
    if recent:
        lines.append("")
        lines.append(f"  最近{len(recent)}次平均收益: {recent[-1]['avg_return']:+.1f}%")
    
    history = state.get("history", [])
    if history:
        lines.append("")
        lines.append(f"  调整记录: {len(history)}次")
        for h in history[-3:]:
            lines.append(f"    {h['date']}: {h['reason']}")
    
    return "\n".join(lines)
