"""
A股量化监控 Web 看板
在浏览器中查看实时行情、信号、模拟交易和回测数据
"""
import json
import sys
import os
import logging

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
from datetime import datetime, date
from flask import Flask, render_template, jsonify, request

logger = logging.getLogger(__name__)

from src.fetcher import (
    fetch_realtime, fetch_kline, fetch_fund_flow,
    fetch_all_indices, fetch_financial, fetch_sector_performance,
)
from src.signals import (
    check_signals, generate_status_report, SignalDedup,
    check_price_alerts, generate_fund_flow_signal,
    _sma, _calc_rsi,
)
from src.backtest import backtest_ma_crossover
from src.dip_buy import scan_dip_buy_candidates, generate_dip_buy_report, QUALITY_POOL
from src.sectors import get_sector_tag, STOCK_SECTOR
from src.papertrade import PaperTrading
from src.scoring import compute_score
import numpy as np

_response_cache: dict[str, tuple] = {}  # key -> (time, data)

app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

# K线缓存：每天只拉一次
_kline_cache: dict[str, tuple] = {}  # code -> (date_str, dataframe)
_fund_flow_cache: dict[str, tuple] = {}  # code -> (date_str, data)
_financial_cache: dict[str, tuple] = {}  # code -> (date_str, data)

def _get_kline_cached(code: str, days: int = 60):
    today = date.today().isoformat()
    if code in _kline_cache:
        cached_date, cached_df = _kline_cache[code]
        if cached_date == today and cached_df is not None:
            return cached_df
    df = fetch_kline(code, days)
    _kline_cache[code] = (today, df)
    return df

def _get_fund_flow_cached(code: str):
    today = date.today().isoformat()
    if code in _fund_flow_cache:
        cached_date, cached_data = _fund_flow_cache[code]
        if cached_date == today and cached_data is not None:
            return cached_data
    data = fetch_fund_flow(code)
    _fund_flow_cache[code] = (today, data)
    return data

def _get_financial_cached(code: str):
    today = date.today().isoformat()
    if code in _financial_cache:
        cached_date, cached_data = _financial_cache[code]
        if cached_date == today and cached_data is not None:
            return cached_data
    data = fetch_financial(code)
    _financial_cache[code] = (today, data)
    return data

# 加载配置
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config and "stocks" in config:
        config["stocks"] = {str(k): v for k, v in config["stocks"].items()}
    return config


@app.route("/")
def dashboard():
    """看板主页"""
    config = load_config()
    stocks = config.get("stocks", {})
    stocks_info = []
    for code, name in stocks.items():
        realtime = fetch_realtime(code)
        if realtime:
            stocks_info.append({
                "code": code,
                "name": realtime.get("name") or name,
                "sector": get_sector_tag(code),
                "price": realtime["price"],
                "change_pct": realtime.get("change_pct", 0),
            })
        else:
            stocks_info.append({"code": code, "name": name, "sector": "", "price": 0, "change_pct": 0})
    return render_template("dashboard.html", stocks=stocks_info, now=datetime.now().strftime("%Y-%m-%d %H:%M"))


@app.route("/api/stocks")
def api_stocks():
    """返回所有监控股票的实时数据（并行优化版，缓存5秒）"""
    now = datetime.now()
    ck = "stocks"
    if ck in _response_cache:
        ct, cd = _response_cache[ck]
        if (now - ct).total_seconds() < 5:
            return jsonify(cd)

    config = load_config()
    stocks = config.get("stocks", {})
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def fetch_stock_data(code, name):
        realtime = fetch_realtime(code)
        if not realtime:
            return None
        kline = _get_kline_cached(code, 60)
        fund_flow = _get_fund_flow_cached(code)
        price = realtime["price"]
        display_name = realtime.get("name") or name
        
        signals = []
        if kline is not None:
            sigs = check_signals(code, display_name, kline, price, config)
            signals = [{"type": s.signal_label, "suggestion": s.suggestion} for s in sigs]
        
        trend = ""
        if kline is not None and len(kline) > 25:
            closes = kline["close"].values.astype(float)
            s5 = _sma(closes, 5)
            s20 = _sma(closes, 20)
            if len(s5[~np.isnan(s5)]) > 0 and len(s20[~np.isnan(s20)]) > 0:
                cs = s5[~np.isnan(s5)][-1]
                cl = s20[~np.isnan(s20)][-1]
                trend = "上涨 ↑" if cs > cl else ("下跌 ↓" if cs < cl else "横盘 →")
        
        score_info = {}
        if kline is not None:
            closes = kline["close"].values.astype(float)
            volumes = kline["volume"].values.astype(float) if "volume" in kline.columns else np.array([])
            score_info = compute_score(closes, volumes, price, fund_flow, code=code, change_pct=realtime.get("change_pct", 0))
        
        fin = _get_financial_cached(code)
        
        return {
            "code": code, "name": display_name, "sector": get_sector_tag(code),
            "price": price, "change_pct": realtime.get("change_pct", 0),
            "volume": realtime.get("volume", 0), "trend": trend,
            "signals": signals, "fund_flow": fund_flow["flow_status"] if fund_flow else "",
            "score": score_info.get("score", 0), "score_action": score_info.get("action", ""),
            "score_details": score_info.get("details", {}), "financial": fin,
        }
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_stock_data, code, name): code for code, name in stocks.items()}
        for future in as_completed(futures):
            try:
                data = future.result(timeout=20)
                if data:
                    results.append(data)
            except Exception as e:
                logger.warning("获取股票数据超时/失败: %s", e)
                continue
    
    _response_cache[ck] = (datetime.now(), results)
    return jsonify(results)


@app.route("/api/account")
def api_account():
    """返回模拟账户数据（缓存5秒）"""
    now = datetime.now()
    ck = "account"
    if ck in _response_cache:
        ct, cd = _response_cache[ck]
        if (now - ct).total_seconds() < 5:
            return jsonify(cd)

    paper = PaperTrading()
    report = paper.generate_report()
    portfolio = paper.portfolio

    today_str = date.today().isoformat()

    # 收集今日卖出记录（用于计算各股实际今日盈亏）
    today_sells_by_code: dict[str, list] = {}
    for t in portfolio.trades:
        if t.date == today_str and "卖出" in t.action:
            today_sells_by_code.setdefault(t.stock_code, []).append(t)

    # 持仓详情
    positions = []
    for code, pos in portfolio.positions.items():
        rt = fetch_realtime(code)
        fresh_price = rt["price"] if rt else pos.current_price
        market_value = pos.shares * fresh_price
        profit_pct = round((fresh_price / pos.buy_price - 1) * 100, 2)
        profit_amount = round(market_value - pos.total_cost, 2)
        # 今日盈亏（未实现部分）
        if pos.buy_date == today_str:
            # 今天买的：今日盈亏 = 总盈亏（买入后到现在的变化）
            today_profit = round(market_value - pos.total_cost, 2)
            today_chg = profit_pct
        else:
            # 之前买的：今日盈亏 = 持股数 × (现价 - 昨收)
            yesterday_close = rt.get("yesterday_close", 0) if rt else 0
            if yesterday_close > 0:
                today_profit = round(pos.shares * (fresh_price - yesterday_close), 2)
                today_chg = round((fresh_price / yesterday_close - 1) * 100, 2)
            else:
                today_profit = 0
                today_chg = rt.get("change_pct", 0) if rt else 0
        # 加上今日已卖出部分的实际盈亏（按昨收计算）
        sells = today_sells_by_code.pop(code, [])
        if sells:
            yesterday_close = rt.get("yesterday_close", 0) if rt else 0
            for s in sells:
                if yesterday_close > 0:
                    realized = s.shares * (s.price - yesterday_close)
                else:
                    realized = s.profit_amount  # 无昨收时用交易盈亏兜底
                today_profit = round(today_profit + realized, 2)
        # 查找买入理由（最近一次买入/加仓记录）
        buy_reason = ""
        for t in reversed(portfolio.trades):
            if t.stock_code == code and t.action in ("买入", "加仓"):
                buy_reason = t.reason
                break
        positions.append({
            "name": pos.stock_name,
            "code": code,
            "buy_price": pos.buy_price,
            "current_price": fresh_price,
            "shares": pos.shares,
            "profit_pct": profit_pct,
            "profit_amount": profit_amount,
            "market_value": market_value,
            "today_profit": today_profit,
            "today_chg": today_chg,
            "buy_reason": buy_reason,
        })

    # 交易统计
    sells = [t for t in portfolio.trades if t.action == "卖出"]
    wins = [t for t in sells if t.profit_pct > 0]

    # 今日盈亏：各股今日盈亏之和（含已实现卖出利润）
    leftover_realized = sum(s.profit_amount for sells in today_sells_by_code.values() for s in sells)
    today_total_profit = round(sum(p["today_profit"] for p in positions) + leftover_realized, 2)
    today_total_chg = 0

    # 今日交易记录
    today_trades = []
    for t in reversed(portfolio.trades[-50:]):
        if t.date == today_str:
            today_trades.append({
                "time": t.date, "action": t.action, "name": t.stock_name,
                "code": t.stock_code, "price": t.price, "shares": t.shares,
                "profit_pct": t.profit_pct, "profit_amount": t.profit_amount,
                "reason": t.reason,
                "buy_price": round(t.price - t.profit_amount / t.shares, 2) if t.action in ["卖出", "卖出(部分)"] and t.shares > 0 else 0,
            })

    result = {
        "total_value": round(portfolio.total_value, 2),
        "cash": round(portfolio.cash, 2),
        "stock_value": round(portfolio.total_value - portfolio.cash, 2),
        "total_return": round((portfolio.total_value - 100000) / 100000 * 100, 2),
        "today_profit": today_total_profit,
        "today_chg": today_total_chg,
        "today_trades": today_trades,
        "trade_count": len(sells),
        "win_count": len(wins),
        "win_rate": round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "positions": positions,
        "equity_curve": paper.portfolio.daily_values[-60:],
    }
    _response_cache["account"] = (datetime.now(), result)
    return jsonify(result)


@app.route("/api/signals")
def api_signal_history():
    """返回回测数据（缓存5分钟）"""
    now = datetime.now()
    cache_key = "signals"
    if cache_key in _response_cache:
        ct, cd = _response_cache[cache_key]
        if (now - ct).total_seconds() < 300:
            return jsonify(cd)
    config = load_config()
    stocks = config.get("stocks", {})
    results = []
    for code, name in stocks.items():
        kline = fetch_kline(code, 365)
        if kline is not None:
            r = backtest_ma_crossover(code, name, kline, ma_short=10, ma_long=30, rsi_filter=True, stop_loss_pct=8)
            results.append({
                "name": name,
                "code": code,
                "return": r.total_return,
                "win_rate": r.win_rate,
                "max_drawdown": r.max_drawdown,
                "trades": r.total_trades,
            })
    _response_cache["signals"] = (datetime.now(), results)
    return jsonify(results)


@app.route("/api/dipbuy")
def api_dipbuy():
    """返回尾盘低吸扫描数据（缓存10分钟）"""
    now = datetime.now()
    ck = "dipbuy"
    if ck in _response_cache:
        ct, cd = _response_cache[ck]
        if (now - ct).total_seconds() < 600:
            return jsonify(cd)
    candidates = scan_dip_buy_candidates(max_price=150, tech_only=True)
    results = [{
        "name": c["name"],
        "code": c["code"],
        "price": c["price"],
        "score": c["score"],
        "reason": c["reason"],
    } for c in candidates[:10]]
    _response_cache["dipbuy"] = (datetime.now(), results)
    return jsonify(results)


@app.route("/api/search")
def api_search():
    """搜索股票（按代码精确查询）"""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify([])
    results = []
    seen = set()
    # 先查已监控的
    config = load_config()
    for code, name in config.get("stocks", {}).items():
        if q in code or q in name:
            results.append({"code": code, "name": name, "price": 0, "change_pct": 0})
            seen.add(code)
    # 查优质股票池
    try:
        from src.dip_buy import QUALITY_POOL
        for code, name in QUALITY_POOL.items():
            if code in seen: continue
            if q in code or q in name:
                rt = fetch_realtime(code)
                results.append({
                    "code": code, "name": rt.get("name", name) if rt else name,
                    "price": rt["price"] if rt else 0,
                    "change_pct": rt.get("change_pct", 0) if rt else 0,
                })
                seen.add(code)
            if len(results) >= 8:
                break
    except: pass
    # 按板块搜索
    try:
        from src.sectors import STOCK_SECTOR
        for code, sector in STOCK_SECTOR.items():
            if code in seen: continue
            if q in sector or q in code:
                rt = fetch_realtime(code)
                results.append({
                    "code": code, "name": rt.get("name", code) if rt else code,
                    "price": rt["price"] if rt else 0,
                    "change_pct": rt.get("change_pct", 0) if rt else 0,
                })
                seen.add(code)
            if len(results) >= 8:
                break
    except: pass
    # 精确代码搜索
    if q.isdigit() and 5 <= len(q) <= 6 and q not in seen:
        rt = fetch_realtime(q)
        if rt:
            results.append({
                "code": q, "name": rt.get("name", ""),
                "price": rt["price"], "change_pct": rt.get("change_pct", 0),
            })
    # 如果还没搜到，模糊名称搜索（ETF/基金等）
    if len(results) < 3 and len(q) >= 2:
        etfs = {"515880": "通信ETF", "159889": "通信ETF",
                "512880": "证券ETF", "510050": "50ETF",
                "518880": "黄金ETF", "513100": "纳指ETF"}
        for code, name in etfs.items():
            if code in seen: continue
            if q.lower() in name.lower() or q in code:
                rt = fetch_realtime(code)
                if rt:
                    results.append({
                        "code": code, "name": rt.get("name", name),
                        "price": rt["price"], "change_pct": rt.get("change_pct", 0),
                    })
                    seen.add(code)
    return jsonify(results[:8])


@app.route("/api/add_stock", methods=["POST"])
def api_add_stock():
    """添加股票到监控"""
    data = request.get_json()
    code = data.get("code", "").strip()
    name = data.get("name", "").strip()
    if not code:
        return jsonify({"ok": False, "msg": "代码不能为空"})
    try:
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        stocks = config.get("stocks", {})
        if code in stocks:
            return jsonify({"ok": False, "msg": f"{name or code} 已在监控中"})
        stocks[code] = name or code
        config["stocks"] = stocks
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        # 保存板块信息
        from src.sectors import get_sector_tag, lookup_sector_by_name, _save_dynamic_sector
        if not get_sector_tag(code):
            sector = lookup_sector_by_name(name)
            if sector:
                _save_dynamic_sector(code, sector)
        return jsonify({"ok": True, "msg": f"✅ 已添加 {name or code}"})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"添加失败: {e}"})


@app.route("/api/remove_stock", methods=["POST"])
def api_remove_stock():
    """移除股票"""
    data = request.get_json()
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"ok": False, "msg": "代码不能为空"})
    try:
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        stocks = config.get("stocks", {})
        if code not in stocks:
            return jsonify({"ok": False, "msg": f"{code} 不在监控中"})
        name = stocks.pop(code)
        config["stocks"] = stocks
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        return jsonify({"ok": True, "msg": f"✅ 已移除 {name or code}"})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"移除失败: {e}"})


@app.route("/api/indices")
def api_indices():
    """返回大盘指数（缓存5秒）"""
    now = datetime.now()
    ck = "indices"
    if ck in _response_cache:
        ct, cd = _response_cache[ck]
        if (now - ct).total_seconds() < 5:
            return jsonify(cd)
    result = fetch_all_indices()
    _response_cache[ck] = (datetime.now(), result)
    return jsonify(result)


@app.route("/api/market")
def api_market():
    from src.market import get_market_condition
    return jsonify(get_market_condition())


@app.route("/api/sectors")
def api_sectors():
    """返回板块行情（科技类）"""
    return jsonify(fetch_sector_performance())


# ── 企业微信机器人消息接收 ──
# 需要配合 ngrok 使用，将公网 URL 配置到企业微信群机器人的回调地址
# ngrok http 5000

@app.route("/bot", methods=["POST"])
def bot_webhook():
    """接收企业微信机器人的@消息"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"msgtype": "text", "text": {"content": "无法解析消息"}})

        content = data.get("text", {}).get("content", "").strip()
        sender = data.get("sender", {}).get("userid", "未知")

        # 去除@机器人的部分
        for part in content.split():
            if "@" in part:
                content = content.replace(part, "").strip()

        reply = handle_command(content)
        return jsonify({"msgtype": "text", "text": {"content": reply}})

    except Exception as e:
        return jsonify({"msgtype": "text", "text": {"content": f"处理异常: {str(e)}"}})


def handle_command(cmd: str) -> str:
    """解析并执行命令"""
    cmd = cmd.strip()
    parts = cmd.split()
    if not parts:
        return help_text()

    action = parts[0]

    if action in ["加股票", "加", "添加"]:
        if len(parts) < 2:
            return "格式: 加股票 600519"
        code = parts[1]
        name = parts[2] if len(parts) > 2 else ""
        return add_stock(code, name)

    elif action in ["删股票", "删", "删除", "移除"]:
        if len(parts) < 2:
            return "格式: 删股票 600519"
        return remove_stock(parts[1])

    elif action == "持仓":
        try:
            paper = PaperTrading()
            r = paper.generate_report()
            lines = [l for l in r.split("\n") if any(k in l for k in ["总资产", "持仓", "盈亏", "现金", "胜率"])]
            return "\n".join(lines) if lines else "暂无持仓"
        except Exception as e:
            return f"查询失败: {e}"

    elif action in ["评分", "分数"]:
        return get_scores_text()

    elif action in ["信号", "行情"]:
        return get_market_text()

    elif action == "分析":
        code = parts[1] if len(parts) > 1 else ""
        return analyze_stock(code)

    elif action == "大盘":
        return get_indices_text()

    elif action == "推荐":
        return get_recommend_text()

    elif action == "今日":
        return get_today_signals_text()

    elif action == "记录":
        return get_trade_history_text()

    elif action in ["帮助", "help", "h"]:
        return help_text()

    return f"未知命令: {action}\n{help_text()}"


def help_text() -> str:
    return (
        "🤖 **可用命令**\n"
        "加股票 600519 - 添加自选\n"
        "删股票 600519 - 删除自选\n"
        "持仓 - 模拟账户\n"
        "评分 - 各股评分\n"
        "信号 - 实时行情\n"
        "分析 600519 - 分析某股\n"
        "大盘 - 大盘指数\n"
        "推荐 - 尾盘机会\n"
        "今日 - 今日信号\n"
        "记录 - 交易记录\n"
        "帮助 - 此菜单"
    )


def add_stock(code: str, name: str = "") -> str:
    """添加股票到监控列表"""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    stocks = config.get("stocks", {})

    if code in stocks:
        return f"{code} 已在监控列表中"

    # 如果没有名称，尝试获取
    if not name:
        realtime = fetch_realtime(code)
        if realtime:
            name = realtime.get("name", "")
    if not name:
        name = code

    stocks[code] = name
    config["stocks"] = stocks
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # 同步到GitHub
    try:
        import subprocess, os
        subprocess.run(["git", "config", "user.name", "webapp"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "config", "user.email", "webapp@local"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "add", "config.yaml"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "commit", "-m", f"web: add {code} {name}"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "push"], capture_output=True, cwd=os.path.dirname(__file__))
    except:
        pass

    return f"✅ 已添加 {name}({code}) 到监控列表\n下次启动后生效"


def remove_stock(code: str) -> str:
    """从监控列表移除股票"""
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    stocks = config.get("stocks", {})

    if code not in stocks:
        return f"{code} 不在监控列表中"

    name = stocks.pop(code)
    config["stocks"] = stocks
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # 同步到GitHub
    try:
        import subprocess, os
        subprocess.run(["git", "config", "user.name", "webapp"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "config", "user.email", "webapp@local"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "add", "config.yaml"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "commit", "-m", f"web: remove {code}"], capture_output=True, cwd=os.path.dirname(__file__))
        subprocess.run(["git", "push"], capture_output=True, cwd=os.path.dirname(__file__))
    except:
        pass

    return f"✅ 已移除 {name}({code})"


def get_scores_text() -> str:
    """获取所有股票评分"""
    config = load_config()
    stocks = config.get("stocks", {})
    lines = ["📊 当前评分"]
    for code, name in stocks.items():
        rt = fetch_realtime(code)
        kline = fetch_kline(code, 60)
        ff = fetch_fund_flow(code)
        if kline is None or rt is None:
            continue
        closes = kline["close"].values.astype(float)
        volumes = kline["volume"].values.astype(float) if "volume" in kline.columns else np.array([])
        price = rt["price"]
        r = compute_score(closes, volumes, price, ff, code=code, change_pct=rt.get("change_pct", 0))
        lines.append(f"{name}({code}): {r['score']}分 [{r['action']}]")
    return "\n".join(lines)


def get_market_text() -> str:
    """获取实时行情"""
    config = load_config()
    stocks = config.get("stocks", {})
    lines = ["📈 实时行情"]
    for code, name in stocks.items():
        rt = fetch_realtime(code)
        if rt:
            chg = rt.get("change_pct", 0)
            icon = "📈" if chg > 0 else ("📉" if chg < 0 else "➖")
            lines.append(f"{icon} {rt.get('name',name)}({code}): {rt['price']:.2f}元 ({chg:+.2f}%)")
    return "\n".join(lines)


def analyze_stock(code: str) -> str:
    """分析某只股票"""
    if not code:
        return "格式: 分析 600519"
    rt = fetch_realtime(code)
    if not rt:
        return f"未找到股票 {code}"
    name = rt.get("name", code)
    price = rt["price"]
    chg = rt.get("change_pct", 0)
    kline = fetch_kline(code, 60)
    ff = fetch_fund_flow(code)
    fin = fetch_financial(code)

    lines = [f"📊 **{name}({code}) 分析**"]
    lines.append(f"现价: {price:.2f}元  ({chg:+.2f}%)")
    lines.append("")

    # 评分
    if kline is not None:
        from src.scoring import compute_score
        closes = kline["close"].values.astype(float)
        volumes = kline["volume"].values.astype(float) if "volume" in kline.columns else np.array([])
        r = compute_score(closes, volumes, price, ff, code=code, change_pct=rt.get("change_pct", 0))
        lines.append(f"综合评分: **{r['score']}分** [{r['action']}]")
        for k, d in r["details"].items():
            lines.append(f"  {k}: {d['score']}分 {d['desc']}")
        lines.append("")

    # 资金流向
    if ff:
        lines.append(f"资金流向: {ff.get('flow_status', '')}")

    # 财务
    if fin:
        lines.append(f"PE: {fin.get('pe', '--')}  PB: {fin.get('pb', '--')}  ROE: {fin.get('roe', '--')}%")

    return "\n".join(lines)


def get_indices_text() -> str:
    """获取大盘指数"""
    indices = fetch_all_indices()
    if not indices:
        return "大盘数据获取失败"
    lines = ["📊 **大盘指数**"]
    for idx in indices:
        icon = "📈" if idx["change_pct"] > 0 else "📉"
        lines.append(f"{icon} {idx['name']}: {idx['price']:.1f} ({idx['change_pct']:+.2f}%)")
    return "\n".join(lines)


def get_recommend_text() -> str:
    """获取尾盘推荐"""
    from src.dip_buy import scan_close_buy_candidates, scan_dip_buy_candidates
    candidates = scan_close_buy_candidates(max_price=150, tech_only=True)
    if candidates:
        lines = ["📋 **尾盘买入推荐**"]
        for c in candidates[:5]:
            lines.append(f"{c['name']}({c['code']}): {c['price']:.2f}元  评分{c['score']}")
        return "\n".join(lines)
    dip = scan_dip_buy_candidates(max_price=150, tech_only=True)
    if dip:
        lines = ["📋 **尾盘低吸关注**"]
        for c in dip[:5]:
            lines.append(f"{c['name']}({c['code']}): {c['price']:.2f}元  评分{c['score']}")
        return "\n".join(lines)
    return "当前没有推荐标的"


def get_today_signals_text() -> str:
    """获取今日信号"""
    config = load_config()
    stocks = config.get("stocks", {})
    lines = ["🚨 **今日信号**"]
    has_signal = False
    for code, name in stocks.items():
        rt = fetch_realtime(code)
        kline = fetch_kline(code, 60)
        if kline is None or rt is None:
            continue
        price = rt["price"]
        dn = rt.get("name", name)
        from src.signals import check_signals, SignalDedup
        sigs = check_signals(code, dn, kline, price, config)
        if sigs:
            has_signal = True
            for s in sigs:
                lines.append(f"  {s.signal_label}: {dn}")
    if not has_signal:
        lines.append("  暂无触发信号")
    return "\n".join(lines)


def get_trade_history_text() -> str:
    """获取模拟交易记录"""
    paper = PaperTrading()
    if not paper.portfolio.trades:
        return "暂无交易记录"
    lines = ["📋 **模拟交易记录**"]
    for t in paper.portfolio.trades[-10:]:
        icon = "🟢" if t.action == "买入" else ("🟢" if t.profit_pct >= 0 else "🔴")
        profit = f"  {t.profit_pct:+.2f}%" if t.profit_pct else ""
        lines.append(f"{icon} {t.date} {t.action} {t.stock_name} {t.price:.2f}元{profit}")
    return "\n".join(lines)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
