"""策略性能追踪 — 每日记录回测结果，用于长期优化"""
import json
import os
from datetime import date

TRACK_FILE = "strategy_performance.json"

def load_performance() -> dict:
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"records": [], "best_params": {}}

def save_performance(data: dict):
    with open(TRACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def record_daily_result(avg_return: float, win_rate: float, stock_count: int):
    data = load_performance()
    today = date.today().isoformat()
    data["records"].append({
        "date": today,
        "avg_return": round(avg_return, 2),
        "win_rate": round(win_rate, 2),
        "stock_count": stock_count,
    })
    # 只保留最近30天
    data["records"] = data["records"][-30:]
    save_performance(data)
    return get_trend(data)

def get_trend(data: dict) -> str:
    records = data.get("records", [])
    if len(records) < 2:
        return "数据不足，继续观察"
    recent = records[-5:] if len(records) >= 5 else records
    avg_returns = [r["avg_return"] for r in recent]
    if avg_returns[-1] > avg_returns[0]:
        return "策略收益呈上升趋势 ↗️"
    elif avg_returns[-1] < avg_returns[0]:
        return "策略收益呈下降趋势 ↘️ 建议调整参数"
    return "策略收益稳定 ➡️"
