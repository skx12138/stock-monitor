"""
A股行情监控主程序
支持多信号检测 + 资金流向 + 价格预警 + 行情简报 + 多通道通知 + 去重
"""
import logging
import os
import sys
import time
from datetime import datetime, date, time as dt_time
from typing import Optional

import yaml

import numpy as np

from src.fetcher import fetch_realtime, fetch_kline, fetch_fund_flow
from src.signals import (
    check_signals,
    check_price_alerts,
    generate_fund_flow_signal,
    generate_status_report,
    SignalDedup,
)
from src.dip_buy import (
    scan_dip_buy_candidates, scan_close_buy_candidates,
)
from src.backtest import backtest_ma_crossover
from src.papertrade import PaperTrading
from src.pricerange import OrderManager
from src.scoring import compute_score
from src.sectors import get_sector_tag
from src.notifier import notify, notify_signal, notify_startup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # 确保股票代码都是字符串
        if config and "stocks" in config:
            config["stocks"] = {str(k): v for k, v in config["stocks"].items()}
        return config
    except Exception as e:
        logger.error("加载配置文件失败: %s", e)
        return None


def is_trading_time() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return (
        (dt_time(9, 30) <= t <= dt_time(11, 30))
        or (dt_time(13, 0) <= t <= dt_time(15, 0))
    )


def run_once(
    config: dict,
    dedup: SignalDedup,
    briefed_today: set,
    today: date,
) -> list:
    """执行一轮监控

    流程:
      1. 每只股票首检 → 推送「行情简报」（含资金流向）
      2. 实时检测 → 推送「触发信号」（金叉/死叉/突破等）
      3. 检查价格预警 + 资金流向异动
    """
    stocks = config.get("stocks", {})
    lookback_days = config.get("monitor", {}).get("lookback_days", 60)
    all_signals = []
    briefing_parts = []   # 收集所有股票的行情简报

    for code, name in stocks.items():
        logger.info("检查 %s (%s)...", name or code, code)

        # 1. 实时行情
        realtime = fetch_realtime(code)
        if realtime is None:
            continue

        price = realtime["price"]
        display_name = realtime.get("name") or name or code

        # 2. 历史K线
        kline = fetch_kline(code, days=lookback_days)
        if kline is None:
            continue

        # 3. 资金流向
        fund_flow = fetch_fund_flow(code)

        # 4. 收集行情简报（合并推送）
        key = (code, today)
        if key not in briefed_today:
            briefed_today.add(key)
            report = generate_status_report(
                stock_code=code,
                stock_name=display_name,
                kline_df=kline,
                latest_price=price,
                realtime=realtime,
                config=config,
                fund_flow=fund_flow,
            )
            if report:
                rlines = report.split("\n")
                price_line = [l for l in rlines if "元" in l]
                trend_line = [l for l in rlines if "趋势" in l and "短期" in l]
                score_line = [l for l in rlines if "总体" in l or "建议" in l]
                brief_parts = []
                if price_line: brief_parts.append(price_line[0])
                if trend_line: brief_parts.append(trend_line[0])
                if score_line: brief_parts.append(score_line[0])
                briefing_parts.append(f"📌 {display_name}({code})")
                for bp in brief_parts:
                    briefing_parts.append(f"   {bp.strip()}")
                briefing_parts.append("")

        # 5. 技术指标信号检测
        signals = check_signals(
            stock_code=code,
            stock_name=display_name,
            kline_df=kline,
            latest_price=price,
            config=config,
            dedup=dedup,
            change_pct=realtime.get("change_pct", 0.0),
        )

        for sig in signals:
            logger.info("触发信号: %s %s", display_name, sig.signal_label)
            all_signals.append(sig)

        # 6. 价格预警
        price_signals = check_price_alerts(
            stock_code=code,
            stock_name=display_name,
            latest_price=price,
            config=config,
            dedup=dedup,
        )
        for sig in price_signals:
            logger.info("价格预警: %s", sig.signal_label)
            all_signals.append(sig)

        # 7. 资金流向异动
        ff_signal = generate_fund_flow_signal(
            stock_code=code, stock_name=display_name,
            latest_price=price, fund_flow=fund_flow,
            config=config, dedup=dedup,
        )
        if ff_signal:
            logger.info("资金流向信号: %s", ff_signal.signal_label)
            all_signals.append(ff_signal)
            if dedup:
                dedup.mark_sent(code, "fund_flow")

    return all_signals, briefing_parts


def _generate_summary(config: dict, signals: list, paper: "PaperTrading") -> str:
    """生成盘后总结报告"""
    from datetime import date
    today_str = date.today().strftime("%m/%d")
    stocks = config.get("stocks", {})

    lines = []
    lines.append(f"📋 **今日盘后总结** · {today_str}")
    lines.append("")

    # 股票今日表现
    lines.append(f"**📈 持仓表现**")
    for code, name in stocks.items():
        realtime = fetch_realtime(code)
        if realtime:
            chg = realtime.get("change_pct", 0)
            price = realtime["price"]
            icon = "📈" if chg > 0 else ("📉" if chg < 0 else "➖")
            display = realtime.get("name") or name
            lines.append(f"  {icon} {display}({code})  {price:.2f}元  {chg:+.2f}%")
    lines.append("")

    # 今日信号
    today_sigs = [s for s in signals if hasattr(s, 'signal_label')]
    if today_sigs:
        lines.append(f"**🚨 今日信号（{len(today_sigs)}个）**")
        for s in today_sigs[-10:]:
            icon = "🔴" if "买入" in s.suggestion or "多头" in s.direction else "🟢"
            lines.append(f"  {icon} {s.stock_name} {s.signal_label}  {s.suggestion}")
        lines.append("")

    # 模拟账户
    paper_report = paper.generate_report()
    if paper_report:
        # 提取关键数据
        for line in paper_report.split("\n"):
            if "总资产" in line or "总收益" in line or "现金" in line or "历史统计" in line or "胜率" in line:
                lines.append(f"  {line.strip()}")

    lines.append("")
    lines.append(f"💡 明日 {date.today().strftime('%m/%d')} 开盘后自动恢复监控")

    # ── 今日交易总结（合并到盘后总结） ──
    try:
        today_iso = date.today().isoformat()
        today_trades = [t for t in paper.portfolio.trades if t.date == today_iso]
        buys = [t for t in today_trades if "买入" in t.action or t.action == "加仓"]
        sells = [t for t in today_trades if "卖出" in t.action]
        lines.append("")
        lines.append(f"**📊 今日交易**")
        if buys:
            lines.append(f"  🟢 买入 {len(buys)} 次")
            for t in buys[:5]:
                lines.append(f"     {t.stock_name}({t.stock_code}) {t.price:.2f}元×{t.shares}股")
        if sells:
            lines.append(f"  🔴 卖出 {len(sells)} 次")
            for t in sells[:5]:
                p_icon = "🟢" if t.profit_pct >= 0 else "🔴"
                lines.append(f"     {p_icon} {t.stock_name}({t.stock_code}) {t.price:.2f}元 {t.profit_pct:+.2f}%")
        if not buys and not sells:
            lines.append("  今日无交易")
        pos_cnt = len(paper.portfolio.positions)
        if pos_cnt > 0:
            lines.append(f"  📦 持仓 {pos_cnt} 只")
            for pc, pp in paper.portfolio.positions.items():
                p_i = "🟢" if pp.profit_pct >= 0 else "🔴"
                from src.sectors import get_sector_tag
                tag = get_sector_tag(pc)
                tag_str = f" [{tag}]" if tag else ""
                lines.append(f"     {p_i} {pp.stock_name}({pc}){tag_str} {pp.current_price:.2f}元 {pp.profit_pct:+.2f}%")
        pnl = (paper.portfolio.total_value - 500000) / 500000 * 100
        lines.append(f"  💰 账户: {paper.portfolio.total_value:,.2f}元 ({pnl:+.2f}%)")
    except Exception as e:
        logger.debug("交易总结合并失败: %s", e)

    # ── 市场行情分析（盘后总结） ──
    try:
        from src.scoring import get_market_sentiment, get_intraday_trend
        from src.fetcher import fetch_sector_performance, fetch_market_index
        
        idx = fetch_market_index("000001")
        if idx:
            idx_chg = idx.get("change_pct", 0)
            idx_icon = "📈" if idx_chg > 0 else "📉"
            lines.append("")
            lines.append(f"**📊 市场概况**")
            lines.append(f"  上证指数: {idx.get('price', 0):.0f} {idx_icon} {idx_chg:+.2f}%")
        
        s_level, s_label = get_market_sentiment()
        t_desc, _ = get_intraday_trend()
        lines.append(f"  市场情绪: {s_label}")
        lines.append(f"  日内走势: {t_desc}")
        
        # 板块TOP3
        sectors = fetch_sector_performance()
        if sectors:
            lines.append(f"  **板块TOP3:**")
            for s in sectors[:3]:
                s_icon = "📈" if s["change_pct"] > 0 else "📉"
                lines.append(f"    {s_icon} {s['name']} {s['change_pct']:+.2f}%")
            lines.append(f"  **板块BOTTOM3:**")
            for s in sectors[-3:]:
                s_icon = "📈" if s["change_pct"] > 0 else "📉"
                lines.append(f"    {s_icon} {s['name']} {s['change_pct']:+.2f}%")
    except Exception as e:
        logger.debug("市场行情分析失败: %s", e)

    return "\n".join(lines).strip()


def main():
    config = load_config()
    if not config:
        sys.exit(1)

    stocks = config.get("stocks", {})
    if not stocks:
        logger.warning("股票池为空！请在 config.yaml 中配置自选股")
        sys.exit(0)

    dedup = SignalDedup(enabled=config.get("dedup", {}).get("enabled", True))
    briefed_today: set[tuple[str, date]] = set()
    dip_buy_done_today = False
    close_buy_done_today = False  # 尾盘买入是否已完成
    pred_done_today = False      # 明日预测是否已推送
    premarket_done = False       # 开盘前分析是否已推送
    startup_done = False         # 启动通知是否已推送
    summary_done_today = False   # 盘后总结是否已发
    today_signals = []           # 今日信号记录
    cycle_count = 0
    last_snapshot: dict = {}     # 上一轮各股状态 {code: (price, score)}
    paper = PaperTrading()
    order_mgr = OrderManager()

    # 盘中价格追踪（用于检测跳水/深V）
    price_history: dict[str, list] = {}  # code -> [(time, price)]
    intraday_alerts: set = set()         # 已推送的异动预警

    interval = config.get("monitor", {}).get("interval_seconds", 60)
    last_date = datetime.now().date()

    signals_cfg = config.get("signals", {})
    enabled_signals = [k for k, v in signals_cfg.items() if isinstance(v, dict) and v.get("enabled")]
    extras = []
    if config.get("price_alerts"):
        extras.append("价格预警")
    if config.get("fund_flow", {}).get("enabled"):
        extras.append("资金流向")
    extras.append("尾盘低吸(科技≤150元)")
    extras.append("模拟交易")
    extras.append("尾盘买入")
    notifier_type = config.get("notifier", {}).get("type", "无")

    logger.info("=" * 50)
    logger.info("A股行情监控")
    logger.info("监控股票: %d 只", len(stocks))
    for code, name in stocks.items():
        tag = get_sector_tag(code)
        logger.info("  %s %s (%s)", name or code, tag, code)
    logger.info("技术信号: %s", ", ".join(enabled_signals))
    if extras:
        logger.info("扩展功能: %s", ", ".join(extras))
    logger.info("通知通道: %s", notifier_type)
    logger.info("轮询间隔: %d 秒", interval)
    logger.info("=" * 50)

    # 启动通知加上板块和回测（只在交易时段推送）
    startup_stocks = []
    backtest_lines = []
    for code, name in stocks.items():
        tag = get_sector_tag(code)
        startup_stocks.append(f"{name or code}({code}) {tag}")
        # 跑回测
        kline = fetch_kline(code, 365)
        if kline is not None:
            r = backtest_ma_crossover(code, name, kline, ma_short=10, ma_long=40, rsi_filter=True, stop_loss_pct=8)
            ret_icon = "📈" if r.total_return > 0 else "📉"
            backtest_lines.append(f"  {ret_icon} {name}({code}) {r.strategy}: {r.total_return:+.1f}% 胜率{r.win_rate:.0f}% 回撤{r.max_drawdown:.0f}% 交易{r.total_trades}次")

    config["_stock_list_display"] = startup_stocks
    config["_backtest_summary"] = backtest_lines
    config["_paper_report"] = paper.generate_report()
    # 启动通知改为9:30在循环中触发

    # 配置文件监控（支持热加载）
    config_mtime = os.path.getmtime("config.yaml")

    while True:
        now = datetime.now()

        # 检查配置是否被Webhook修改
        try:
            new_mtime = os.path.getmtime("config.yaml")
            if new_mtime != config_mtime:
                config_mtime = new_mtime
                new_config = load_config()
                if new_config and new_config.get("stocks"):
                    old_stocks = len(config.get("stocks", {}))
                    config = new_config
                    stocks = config.get("stocks", {})
                    logger.info("检测到配置变更，已热加载 (原%d只->现%d只)", old_stocks, len(stocks))
        except:
            pass

        if now.date() != last_date:
            last_date = now.date()
            dedup.reset()
            briefed_today.clear()
            dip_buy_done_today = False
            close_buy_done_today = False
            summary_done_today = False
            today_signals.clear()
            cycle_count = 0
            last_snapshot.clear()
            intraday_alerts.clear()
            logger.info("新交易日，已重置记录")

        now_time = now.time()

        if is_trading_time():

            # ── 启动通知（9:30，每天一次） ──
            if not startup_done and now_time >= dt_time(9, 30) and now_time <= dt_time(9, 35):
                startup_done = True
                logger.info("推送启动通知...")
                notify_startup(config)

            # ── 开盘前分析（9:20-9:30，每天一次） ──
            if not premarket_done and now_time >= dt_time(9, 20) and now_time <= dt_time(9, 30):
                premarket_done = True
                logger.info("生成开盘前分析...")
                try:
                    from src.fetcher import fetch_market_index
                    from src.predictor import predict_tomorrow
                    pre_lines = ["\U0001f305 **开盘前分析**", ""]
                    # 大盘
                    try:
                        sh = fetch_market_index("000001")
                        if sh:
                            icon = "\U0001f4c8" if sh["change_pct"] >= 0 else "\U0001f4c9"
                            pre_lines.append(f"{icon} 大盘: {sh['name']} {sh['price']} {sh['change_pct']:+.2f}%")
                            pre_lines.append("")
                    except: pass
                    # 个股明日预测+评分
                    bullish = bearish = neutral = 0
                    for p_code, p_name in stocks.items():
                        pk = fetch_kline(p_code, 60)
                        if pk is not None and len(pk) > 20:
                            pc = pk["close"].values.astype(float)
                            pv = pk["volume"].values.astype(float) if "volume" in pk.columns else np.array([])
                            ph = pk["high"].values.astype(float) if "high" in pk.columns else pc
                            pl = pk["low"].values.astype(float) if "low" in pk.columns else pc
                            p_pred = predict_tomorrow(pc, ph, pl, pv, pc[-1])
                            icon = {"看涨": "\U0001f4c8", "看跌": "\U0001f4c9", "震荡": "\u2796"}.get(p_pred["direction"], "")
                            # K线形态分析
                            kline_reason = p_pred.get("reason", "")
                            # 近5日趋势
                            if len(pc) >= 5:
                                chg_5d = (pc[-1] / pc[-5] - 1) * 100
                                trend_5d = f"5日{chg_5d:+.1f}%"
                            else:
                                trend_5d = ""
                            # 均线位置
                            from src.signals import _sma
                            ma5_v = _sma(pc, 5); ma20_v = _sma(pc, 20)
                            ma_valid = ~np.isnan(ma5_v) & ~np.isnan(ma20_v)
                            ma_pos = ""
                            if len(ma5_v[ma_valid]) > 0:
                                m5 = ma5_v[ma_valid][-1]; m20 = ma20_v[ma_valid][-1]
                                ma_pos = "趋势向上" if m5 > m20 else "趋势向下"
                            # 成交量
                            vol_trend = ""
                            if len(pv) >= 5:
                                avg_v = np.mean(pv[-5:])
                                if avg_v > 0:
                                    vr = pv[-1] / avg_v
                                    vol_trend = f"量{vr:.1f}"
                            # 精简显示
                            extra = " | ".join(filter(None, [kline_reason[:15], trend_5d, ma_pos, vol_trend]))
                            pre_lines.append(f"  {icon} {p_name}({p_code}): {p_pred['direction']}({p_pred['confidence']}%)")
                            if extra:
                                pre_lines.append(f"     {extra}")
                            if p_pred["direction"] == "看涨": bullish += 1
                            elif p_pred["direction"] == "看跌": bearish += 1
                            else: neutral += 1
                    if bullish + bearish + neutral > 0:
                        t = bullish + bearish + neutral
                        ti = "\U0001f4c8" if bullish >= bearish else "\U0001f4c9"
                        pre_lines.insert(1, f"{ti} 整体趋势: 看涨{bullish}只 / 看跌{bearish}只 / 震荡{neutral}只")
                    # ── 持仓风险预警（开盘前危险信号） ──
                    danger_lines = []
                    for d_code, d_pos in paper.portfolio.positions.items():
                        d_name = d_pos.stock_name
                        d_kline = fetch_kline(d_code, 60)
                        if d_kline is None or len(d_kline) < 20: continue
                        d_closes = d_kline["close"].values.astype(float)
                        d_volumes = d_kline["volume"].values.astype(float) if "volume" in d_kline.columns else np.array([])
                        d_highs = d_kline["high"].values.astype(float) if "high" in d_kline.columns else d_closes
                        d_lows = d_kline["low"].values.astype(float) if "low" in d_kline.columns else d_closes
                        d_pred = predict_tomorrow(d_closes, d_highs, d_lows, d_volumes, d_pos.current_price)
                        # 检查昨日是否接近跌停
                        prev_chg_d = (d_closes[-1] / d_closes[-2] - 1) * 100 if len(d_closes) >= 2 else 0
                        near_limit = prev_chg_d <= -8
                        pred_bearish = d_pred["direction"] == "看跌" and d_pred["confidence"] >= 60
                        if near_limit or pred_bearish:
                            warn = ""
                            if near_limit:
                                warn = f"昨日大跌{prev_chg_d:.0f}%"
                            if pred_bearish:
                                if warn: warn += " | "
                                warn += f"预测看跌({d_pred['confidence']}%)"
                            icon = "\U0001f6a8"
                            danger_lines.append(f"  {icon} {d_name}({d_code}): ⚠️ {warn}")
                    if danger_lines:
                        pre_lines.append("")
                        pre_lines.append(f"\U0001f6a8 **持仓风险预警**")
                        pre_lines.extend(danger_lines)
                    # 先推送开盘前分析
                    notify(config, "\U0001f305 开盘前分析", "\n".join(pre_lines))
                    # ── 开盘前自动交易（基于昨日收盘数据） ──
                    trade_lines = []
                    for t_code, t_name in stocks.items():
                        rt = fetch_realtime(t_code)
                        if not rt: continue
                        t_price = rt.get("price", 0)
                        t_kline = fetch_kline(t_code, 60)
                        if t_kline is None or len(t_kline) < 25: continue
                        t_closes = t_kline["close"].values.astype(float)
                        t_volumes = t_kline["volume"].values.astype(float) if "volume" in t_kline.columns else np.array([])
                        t_ff = fetch_fund_flow(t_code)
                        t_si = compute_score(t_closes, t_volumes, t_price, t_ff, code=t_code)
                        t_score = t_si.get("score", 0)
                        # 预测（使用K线中的高/低数据）
                        t_highs = t_kline["high"].values.astype(float) if "high" in t_kline.columns else t_closes
                        t_lows = t_kline["low"].values.astype(float) if "low" in t_kline.columns else t_closes
                        t_pred = predict_tomorrow(t_closes, t_highs, t_lows, t_volumes, t_price)
                        t_has = t_code in paper.portfolio.positions
                        if not t_has and t_score >= 45 and t_pred["direction"] == "看涨":
                            buy_t = paper._buy_position(t_code, t_name, t_price, 0.20,
                                f"开盘买入·评分{t_score}·预测{t_pred['direction']}", add_count=0)
                            if buy_t:
                                trade_lines.append(f"  \U0001f7e2 买入 {t_name}({t_code}) {t_price:.2f}元×{buy_t.shares}股 评分{t_score}")
                        elif t_has and (t_score < 35 or t_pred["direction"] == "看跌"):
                            pos = paper.portfolio.positions.get(t_code)
                            if pos and pos.buy_date != date.today().isoformat():
                                sell_t = paper._sell_position(t_code, t_price,
                                    f"开盘卖出·评分{t_score}·预测{t_pred['direction']}")
                                if sell_t:
                                    sell_profit = f" 盈亏{sell_t.profit_pct:+.2f}%" if sell_t.profit_pct else ""
                                    trade_lines.append(f"  \U0001f534 卖出 {t_name}({t_code}) {t_price:.2f}元×{sell_t.shares}股{sell_profit}")
                    if trade_lines:
                        trade_notify = ["\U0001f504 **开盘自动交易**", ""]
                        trade_notify.extend(trade_lines)
                        # 账户概况
                        total_pos = len(paper.portfolio.positions)
                        pnl = (paper.portfolio.total_value - 500000) / 500000 * 100
                        trade_notify.append("")
                        trade_notify.append(f"  📊 持仓{total_pos}只 总资产{paper.portfolio.total_value:,.0f}元 ({pnl:+.2f}%)")
                        notify(config, "\U0001f504 开盘自动交易", "\n".join(trade_notify))
                except Exception as e:
                    logger.debug("开盘前分析失败: %s", e)

            # ── 尾盘低吸扫描（14:30-15:00，每天一次） ──
            if (
                not dip_buy_done_today
                and dt_time(14, 30) <= now_time <= dt_time(15, 0)
            ):
                dip_buy_done_today = True
                logger.info("开始尾盘低吸扫描...")
                candidates = scan_dip_buy_candidates(max_price=150, tech_only=True)
                # 尾盘低吸推送已取消
                # 尾盘低吸推送已取消
                # 模拟账户日报改到尾盘推送

            # ── 尾盘买入扫描（14:50-15:00，每天一次） ──
            if (
                not close_buy_done_today
                and dt_time(14, 50) <= now_time <= dt_time(15, 0)
            ):
                close_buy_done_today = True
                logger.info("开始尾盘买入扫描...")
                close_candidates = scan_close_buy_candidates(max_price=150, tech_only=True, monitored_stocks=stocks)
                # 尾盘买入推荐已取消
                # 尾盘推送模拟账户日报
                acc_report = paper.generate_report()
                if acc_report:
                    notify(config, "📋 模拟账户日报", acc_report)
                # 尾盘自动交易：推荐股票买入（需次日看涨），ETF优先
                for c in close_candidates[:3]:
                    if c["code"] not in paper.portfolio.positions and c["score"] >= 55:
                        rt_trade = fetch_realtime(c["code"])
                        price = rt_trade["price"] if rt_trade else c["price"]
                        k_pred = fetch_kline(c["code"], 60)
                        pred_adj = 0
                        pred = {"direction": "未知", "target_label": "明日", "reason": "数据不足", "confidence": 50}
                        if k_pred is not None and "high" in k_pred.columns:
                            from src.predictor import predict_tomorrow
                            pred = predict_tomorrow(
                                k_pred["close"].values.astype(float),
                                k_pred["high"].values.astype(float),
                                k_pred["low"].values.astype(float),
                                k_pred["volume"].values.astype(float) if "volume" in k_pred.columns else np.array([]),
                                price,
                            )
                            if pred["direction"] == "看涨":
                                pred_adj = 5
                                logger.info("%s预判看涨: %s %s", pred.get("target_label", "明日"), c["name"], pred["reason"])
                            elif pred["direction"] == "看跌":
                                pred_adj = -10
                                logger.info("%s预判看跌，跳过买入: %s", pred.get("target_label", "明日"), c["name"])
                        if pred_adj >= 0:
                            # ETF优先：ETF仓位更高
                            buy_ratio = 0.25 if c["code"].startswith(("5", "1")) else 0.20
                            # 接入情绪+日内趋势调节
                            try:
                                from src.scoring import get_market_sentiment, get_intraday_trend
                                s_level, s_label = get_market_sentiment()
                                t_desc, t_intensity = get_intraday_trend()
                                sentiment_map = {-2: 0.3, -1: 0.6, 0: 1.0, 1: 0.7, 2: 0.4}
                                trend_map = {"高开低走": 0.5, "单边下跌": 0.4, "低开高走": 1.2, "单边上涨": 0.8, "震荡": 1.0}
                                s_adj = sentiment_map.get(s_level, 1.0)
                                t_adj = next((v for k, v in trend_map.items() if t_desc.startswith(k)), 1.0)
                                buy_ratio = round(buy_ratio * s_adj * t_adj, 2)
                                logger.info("尾盘情绪[%s]趋势[%s] 仓位%.0f%%→%.0f%%", s_label, t_desc, 0.25 if c["code"].startswith(("5", "1")) else 0.20, buy_ratio*100)
                            except: pass
                            if buy_ratio >= 0.03:
                                buy_trade = paper._buy_position(c["code"], c["name"], price, buy_ratio,
                                f"尾盘买入·{pred.get('target_label','明日')}{pred['direction'] if k_pred is not None else ''}评分{c['score']}", add_count=0)
                                if buy_trade:
                                    pos_info = paper.portfolio.positions.get(c["code"])
                                    pos_str = f""
                                    if pos_info:
                                        pos_str = f"\n📦 持仓: {pos_info.shares}股 均价{pos_info.buy_price:.2f} 盈亏{pos_info.profit_pct:+.2f}%"
                                    pred_str = pred['direction'] if k_pred is not None else ''
                                    notify(config, "🔄 尾盘买入", 
                                        f"尾盘买入 {c['name']}({c['code']})\n"
                                        f"价格: {price:.2f}元×{buy_trade.shares}股\n"
                                        f"预测: {pred_str} 评分: {c['score']}\n"
                                        f"仓位: {buy_ratio*100:.0f}%{pos_str}")
                # 尾盘加仓：大跌日允许补仓摊低成本
                for c in close_candidates[:6]:
                    if c["code"] in paper.portfolio.positions and c["score"] >= 45:
                        pos = paper.portfolio.positions.get(c["code"])
                        if pos and pos.add_count < 3:
                            # 正常情况：盈利中才加仓；大跌日(亏损>3%)允许补仓
                            allow_add = pos.profit_pct > 0
                            if pos.profit_pct <= -3 and pos.add_count < 2:
                                allow_add = True  # 大跌摊低成本，但最多补2次
                            if allow_add:
                                rt_trade = fetch_realtime(c["code"])
                                price = rt_trade["price"] if rt_trade else c["price"]
                                k_pred = fetch_kline(c["code"], 60)
                                if k_pred is not None and "high" in k_pred.columns:
                                    from src.predictor import predict_tomorrow
                                    pred = predict_tomorrow(
                                        k_pred["close"].values.astype(float),
                                        k_pred["high"].values.astype(float),
                                        k_pred["low"].values.astype(float),
                                        k_pred["volume"].values.astype(float) if "volume" in k_pred.columns else np.array([]),
                                        price,
                                    )
                                    if pred["direction"] != "看跌":
                                        add_ratio_w = 0.10
                                        try:
                                            from src.scoring import get_market_sentiment, get_intraday_trend
                                            s_level, s_label = get_market_sentiment()
                                            t_desc, _ = get_intraday_trend()
                                            s_map = {-2: 0.3, -1: 0.6, 0: 1.0, 1: 0.7, 2: 0.4}
                                            t_map = {"高开低走": 0.5, "单边下跌": 0.4, "低开高走": 1.2, "单边上涨": 0.8}
                                            add_ratio_w = round(0.10 * s_map.get(s_level, 1.0) * next((v for k, v in t_map.items() if t_desc.startswith(k)), 1.0), 2)
                                        except: pass
                                        if add_ratio_w < 0.03: add_ratio_w = 0.03
                                        add_trade = paper._buy_position(c["code"], c["name"], price, add_ratio_w,
                                            f"尾盘加仓·评分{c['score']}", add_count=pos.add_count + 1)
                                        if add_trade:
                                            pos_info = paper.portfolio.positions.get(c["code"])
                                            pos_str2 = f""
                                            if pos_info:
                                                pos_str2 = f" 📦 {pos_info.shares}股 均价{pos_info.buy_price:.2f} 盈亏{pos_info.profit_pct:+.2f}%"
                                            pred_str2 = pred['direction'] if k_pred is not None else ''
                                            notify(config, "🔄 尾盘加仓", 
                                                f"尾盘加仓 {c['name']}({c['code']})\n"
                                                f"价格: {price:.2f}元×{add_trade.shares}股\n"
                                                f"预测: {pred_str2} 评分: {c['score']}{pos_str2}")
                # 尾盘卖出：大跌日不止损（避免割在最低点）
                # 检查今天是否普跌日
                today_is_bloody = False
                try:
                    sh_check = fetch_market_index("000001")
                    if sh_check and sh_check.get("change_pct", 0) <= -1.5:
                        today_is_bloody = True
                except: pass
                for pcode in list(paper.portfolio.positions.keys()):
                    pos = paper.portfolio.positions.get(pcode)
                    if not pos: continue
                    rt_sell = fetch_realtime(pcode)
                    sell_price = rt_sell["price"] if rt_sell else pos.current_price
                    sell_action = None
                    if pos.profit_pct >= 8:
                        sell_action = paper._sell_position(pcode, sell_price, f"尾盘止盈{pos.profit_pct:.1f}%")
                    elif pos.profit_pct <= -5 and not today_is_bloody:
                        # 大跌日不触发止损，等反弹再说
                        sell_action = paper._sell_position(pcode, sell_price, f"尾盘止损{pos.profit_pct:.1f}%")
                    elif pos.profit_pct <= -8 and today_is_bloody:
                        # 即使大跌日，亏损超过8%也止损
                        sell_action = paper._sell_position(pcode, sell_price, f"尾盘止损{pos.profit_pct:.1f}%（普跌日放宽至8%）")
                    if sell_action:
                        profit_icon = "🟢" if sell_action.profit_pct >= 0 else "🔴"
                        notify(config, "🔄 尾盘卖出", 
                            f"尾盘卖出 {pos.stock_name}({pcode})\n"
                            f"价格: {sell_price:.2f}元×{sell_action.shares}股\n"
                            f"盈亏: {profit_icon} {sell_action.profit_pct:+.2f}%\n"
                            f"原因: {sell_action.reason}")

                # ── 明日预测汇总（14:50-15:00，仅推送一次） ──
                if not pred_done_today and dt_time(14, 55) <= now_time <= dt_time(15, 0):
                    pred_done_today = True
                    logger.info("生成明日预测汇总...")
                    try:
                        from src.fetcher import fetch_market_index
                        from src.predictor import predict_tomorrow
                        pred_lines = ["🔮 **明日预测汇总**", ""]
                        bullish = bearish = neutral = 0
                        for p_code, p_name in stocks.items():
                            pk = fetch_kline(p_code, 60)
                            if pk is not None and len(pk) > 20:
                                pc = pk["close"].values.astype(float)
                                pv = pk["volume"].values.astype(float) if "volume" in pk.columns else np.array([])
                                ph = pk["high"].values.astype(float) if "high" in pk.columns else pc
                                pl = pk["low"].values.astype(float) if "low" in pk.columns else pc
                                p_pred = predict_tomorrow(pc, ph, pl, pv, pc[-1])
                                icon = {"看涨": "📈", "看跌": "📉", "震荡": "➖"}.get(p_pred["direction"], "❓")
                                pred_lines.append(f"  {icon} {p_name}({p_code}): {p_pred['direction']}({p_pred['confidence']}%)")
                                if p_pred["direction"] == "看涨": bullish += 1
                                elif p_pred["direction"] == "看跌": bearish += 1
                                else: neutral += 1
                        if bullish + bearish + neutral > 0:
                            total = bullish + bearish + neutral
                            trend_icon = "📈" if bullish >= bearish else "📉"
                            trend_text = f"{trend_icon} 整体趋势: 看涨{bullish}只 / 看跌{bearish}只 / 震荡{neutral}只"
                            pred_lines.insert(1, trend_text)
                        if len(pred_lines) > 2:
                            notify(config, "🔮 明日预测", "\n".join(pred_lines))
                    except Exception as e:
                        logger.debug("明日预测推送失败: %s", e)

            logger.info("--- 轮询 %s ---", now.strftime("%H:%M:%S"))
            
            # 收集本轮所有消息，合并推送
            batch_messages = []
            paper._messages = batch_messages  # 做T消息钩子
            
            signals, briefing_parts = run_once(config, dedup, briefed_today, last_date)
            # 行情简报已取消
            for sig in signals:
                today_signals.append(sig)
                # 收集所有信号，每条带股票名
                chg_str = f" {sig.change_pct:+.2f}%" if sig.change_pct else ""
                batch_messages.append(f"  · {sig.stock_name}: {sig.signal_label}{chg_str}")
                # ── 突破追涨 → 自动生成挂单 ──
                if sig.signal_type == "breakout" and hasattr(sig, 'extra') and sig.extra:
                    try:
                        kline_sig = fetch_kline(sig.stock_code, 60)
                        if kline_sig is not None and "high" in kline_sig.columns:
                            from src.signals import calc_atr
                            c_sig = kline_sig["close"].values.astype(float)
                            h_sig = kline_sig["high"].values.astype(float)
                            l_sig = kline_sig["low"].values.astype(float)
                            atr_sig = calc_atr(c_sig, h_sig, l_sig, 14)
                            from src.pricerange import calc_buy_range_on_breakout
                            pr = calc_buy_range_on_breakout(sig.price, atr_sig, sig.extra["consolidation"]["high"])
                            order_mgr.place_order(sig.stock_code, sig.stock_name, "buy", pr, sig.extra["reasons"])
                    except Exception as e:
                        logger.debug("生成突破挂单失败: %s", e)
                # ── 龙回头 → 发送信号通知 ──
                if sig.signal_type == "dragon_back":
                    notify(config, "🐉 龙回头信号",
                        f"🐉 **龙回头 {sig.stock_name}({sig.stock_code})**\n"
                        f"信号: {sig.signal_label}\n"
                        f"建议: {sig.suggestion}\n"
                        f"\n{sig.message}")

            # 评分驱动模拟交易
            stock_snapshots = []  # 用于定期快报
            current_state = {}    # 用于检测变动  code -> (price, score)
            current_prices = {}
            buy_analysis = []     # AI买入条件分析
            for code, name in stocks.items():
                kline = fetch_kline(code, 60)
                fund_flow = fetch_fund_flow(code)
                realtime = fetch_realtime(code)
                if kline is None or realtime is None:
                    continue
                price = realtime["price"]
                current_prices[code] = price
                closes = kline["close"].values.astype(float)
                volumes = kline["volume"].values.astype(float) if "volume" in kline.columns else np.array([])
                if "high" in kline.columns and "low" in kline.columns:
                    from src.signals import calc_atr
                    highs = kline["high"].values.astype(float)
                    lows = kline["low"].values.astype(float)
                    atr_val = calc_atr(closes, highs, lows, 14)
                else:
                    atr_val = 0
                score_info = compute_score(closes, volumes, price, fund_flow, code=code, 
                    change_pct=realtime.get("change_pct", 0),
                    highs=kline["high"].values.astype(float) if "high" in kline.columns else None,
                    lows=kline["low"].values.astype(float) if "low" in kline.columns else None)
                score_info["atr"] = atr_val
                score_info["change_pct"] = realtime.get("change_pct", 0)
                # 传入均线值用于趋势保护
                from src.signals import _sma
                ma5_v = _sma(closes, 5)
                ma20_v = _sma(closes, 20)
                valid_ma = ~np.isnan(ma5_v) & ~np.isnan(ma20_v)
                if len(ma5_v[valid_ma]) > 0:
                    score_info["ma5"] = ma5_v[valid_ma][-1]
                    score_info["ma20"] = ma20_v[valid_ma][-1]
                trade = paper.process_score(code, name, price, score_info, kline)
                # 收集快照（用于定期推送）
                chg = realtime.get("change_pct", 0)
                chg_icon = "📈" if chg >= 0 else "📉"
                score_str = f"评分{score_info.get('score',0)}"
                action_str = score_info.get("action", "")
                if action_str:
                    score_str += f"·{action_str}"
                stock_snapshots.append(f"  {chg_icon} {name}({code}) {price:.2f} {chg:+.2f}% {score_str}")
                current_state[code] = (price, score_info.get("score", 0))
                # 收集AI买入条件分析
                score_val = score_info.get("score", 0)
                has_pos = code in paper.portfolio.positions
                if score_val >= 40 and not has_pos:
                    details = score_info.get("details", {})
                    ma_d = details.get("均线", {}).get("desc", "")
                    rsi_d = details.get("RSI", {}).get("desc", "")
                    macd_d = details.get("MACD", {}).get("desc", "")
                    vol_d = details.get("成交量", {}).get("desc", "")
                    buy_analysis.append(f"  {chg_icon} {name}({code}) 评分{score_val} {ma_d} {rsi_d}")
                if trade:
                    logger.info("评分交易: %s %s", trade.action, name)
                    profit_str = f" 盈亏{trade.profit_pct:+.2f}%" if trade.profit_pct else ""
                    # 持仓信息
                    pos_info = paper.portfolio.positions.get(trade.stock_code)
                    pos_str = ""
                    if pos_info:
                        pos_str = f"\n📦 持仓: {pos_info.shares}股 均价{pos_info.buy_price:.2f} 市值{pos_info.shares*pos_info.current_price:.0f}元 总盈亏{pos_info.profit_pct:+.2f}%"
                    # 买入区间参考（基于ATR）
                    range_str = ""
                    try:
                        atr_v = score_info.get("atr", 0)
                        if atr_v > 0 and trade.price > 0:
                            atr_p = atr_v / trade.price * 100
                            r_low = trade.price - atr_v * 0.5
                            r_high = trade.price + atr_v * 0.5
                            range_str = f"\n📏 参考区间: [{r_low:.2f} ~ {r_high:.2f}] ATR({atr_p:.1f}%)"
                    except: pass
                    # 重点提醒交易（带时间+详细原因）
                    trade_icon = "🟢" if "买入" in trade.action or trade.action == "加仓" else ("🔴" if "卖出" in trade.action else "🔄")
                    is_t = " 做T" if trade.reason and "做T" in trade.reason else ""
                    profit_extra = f" {trade.profit_pct:+.2f}%" if trade.profit_pct else ""
                    notify(config, f"{trade_icon} 交易提醒",
                        f"{trade_icon} **{trade.action}{is_t} {name}({trade.stock_code})**\n"
                        f"⏰ {now.strftime('%H:%M:%S')}\n"
                        f"板块: [{get_sector_tag(code)}]\n"
                        f"价格: {trade.price:.2f}元\n"
                        f"数量: {trade.shares}股\n"
                        f"金额: {trade.price*trade.shares:.0f}元{profit_extra}\n"
                        f"原因: {trade.reason}"
                        f"{range_str}"
                        f"{pos_str}")

            paper.update_prices(current_prices)

            # ── 检测单股异动，有变动时单独推送 ──
            if stock_snapshots and last_snapshot:
                changed_snaps = []
                for code, (price, score) in current_state.items():
                    prev = last_snapshot.get(code)
                    if prev:
                        prev_price, _ = prev
                        price_chg_pct = abs(price - prev_price) / max(prev_price, 0.01) * 100
                        if price_chg_pct > 2.5:
                            # 找到这只股票对应的快照行
                            for snap in stock_snapshots:
                                if f"({code})" in snap:
                                    changed_snaps.append(snap)
                                    break
                last_snapshot = current_state

                for snap in changed_snaps:
                    # 从快照行提取股票名
                    name_part = snap.split("(", 1)[0].strip().lstrip("📈📉")
                    msg = f"⚡ **{name_part}异动**\n{snap}"
                    notify(config, "⚡ 异动提醒", msg)
                if not changed_snaps:
                    logger.debug("无显著变动，跳过本次推送")

            elif not last_snapshot:
                # 首次运行，记录状态但不推送
                last_snapshot = current_state
                logger.info("首次监测，已记录初始状态")

            # ── 盘中异动检测（跳水/深V） ──
            intraday_events = []
            for code, name in stocks.items():
                realtime = fetch_realtime(code)
                if not realtime:
                    continue
                price = realtime["price"]
                chg = realtime.get("change_pct", 0)
                
                # 记录价格历史
                now_ts = now.timestamp()
                if code not in price_history:
                    price_history[code] = []
                price_history[code].append((now_ts, price))
                # 只保留最近30分钟的数据
                cutoff = now_ts - 1800
                price_history[code] = [(t, p) for t, p in price_history[code] if t > cutoff]

                hist = price_history[code]
                if len(hist) < 3:
                    continue

                prices_only = [p for _, p in hist]
                recent = prices_only[-5:] if len(prices_only) >= 5 else prices_only
                first_price = recent[0]
                last_price = recent[-1]
                min_price = min(recent)
                max_price = max(recent)
                drop_pct = (min_price - first_price) / first_price * 100
                rebound_pct = (last_price - min_price) / min_price * 100
                total_chg = (last_price - first_price) / first_price * 100

                alert_key = f"{code}_{now.strftime('%H')}"

                # ── 涨停/跌停提醒 + 跌停智能分析 ──
                if chg <= -9.5 and alert_key + "_limitdown" not in intraday_alerts:
                    intraday_alerts.add(alert_key + "_limitdown")
                    display = realtime.get("name", name)
                    logger.info("跌停预警: %s %.1f%%", display, chg)
                    intraday_events.append(f"  🔴 {display} 跌停 {chg:.1f}%")
                    # ── 跌停智能分析：看是否值得抄底 ──
                    limitdown_code = code
                    limitdown_name = display
                    try:
                        k_ld = fetch_kline(code, 60)
                        if k_ld is not None and len(k_ld) > 20:
                            closes_ld = k_ld["close"].values.astype(float)
                            volumes_ld = k_ld["volume"].values.astype(float) if "volume" in k_ld.columns else np.array([])
                            # 1. 看今日成交量是否异常放大（放量跌停=恐慌，缩量跌停=惜售）
                            vol_ratio_ld = 1.0
                            if len(volumes_ld) >= 10:
                                avg_v = np.mean(volumes_ld[-10:-1])
                                cur_v = realtime.get("volume", 0) if realtime else 0
                                vol_ratio_ld = cur_v / avg_v if avg_v > 0 else 1.0
                            # 2. 看是否连续跌停（检查昨日的涨跌幅）
                            prev_close = closes_ld[-2] if len(closes_ld) >= 2 else 0
                            prev_chg = (closes_ld[-1] / prev_close - 1) * 100 if prev_close > 0 else 0
                            consecutive = prev_chg <= -9  # 昨日也接近跌停
                            # 3. 从预测看明日方向
                            from src.predictor import predict_tomorrow
                            pred_ld = predict_tomorrow(
                                closes_ld,
                                k_ld["high"].values.astype(float) if "high" in k_ld.columns else closes_ld,
                                k_ld["low"].values.astype(float) if "low" in k_ld.columns else closes_ld,
                                volumes_ld, price
                            )
                            # 分析结论
                            analysis_parts = [f"  📊 {display}({code}) 跌停分析:"]
                            # 成交量
                            if vol_ratio_ld > 2:
                                analysis_parts.append(f"     放量{vol_ratio_ld:.1f}倍跌停😱 — 恐慌盘涌出，暂不抄底")
                            elif vol_ratio_ld < 0.5:
                                analysis_parts.append(f"     缩量{vol_ratio_ld:.1f}倍跌停🤔 — 惜售明显，关注反弹机会")
                            else:
                                analysis_parts.append(f"     量能正常跌停 — 观望")
                            # 连续跌停
                            if consecutive:
                                analysis_parts.append(f"     连续跌停⚠️ — 趋势极弱，不接飞刀")
                            else:
                                analysis_parts.append(f"     首次跌停 — 可能是恐慌过度")
                            # 预测
                            if pred_ld["direction"] == "看涨":
                                analysis_parts.append(f"     明日预测看涨({pred_ld['confidence']}%)✅ — 反弹概率大")
                                # 条件满足：缩量+首次+预测看涨 → 自动买入
                                if vol_ratio_ld < 1.5 and not consecutive and pred_ld["confidence"] >= 55:
                                    buy_ld = paper._buy_position(code, display, price, 0.05,
                                        f"跌停抄底·{pred_ld['direction']}{pred_ld['confidence']}%·缩量{vol_ratio_ld:.1f}倍")
                                    if buy_ld:
                                        analysis_parts.append(f"     🟢 自动买入5%仓位抄底")
                                        batch_messages.append(f"  🔄 跌停抄底 {display}({code}) {price:.2f}元×{buy_ld.shares}股")
                            else:
                                analysis_parts.append(f"     明日预测{pred_ld['direction']} — 等企稳再考虑")
                            for ap in analysis_parts:
                                intraday_events.append(ap)
                    except Exception as e:
                        logger.debug("跌停分析失败: %s", e)
                elif chg >= 9.5 and alert_key + "_limitup" not in intraday_alerts:
                    intraday_alerts.add(alert_key + "_limitup")
                    display = realtime.get("name", name)
                    logger.info("涨停预警: %s +%.1f%%", display, chg)
                    intraday_events.append(f"  🟢 {display} 涨停 +{chg:.1f}%")

                # 跳水检测：使用ATR动态阈值（1.5倍ATR，高波动股阈值更宽）
                if drop_pct <= -3.0 and total_chg <= -2.0 and alert_key + "_drop" not in intraday_alerts:
                    # 计算ATR动态阈值
                    dynamic_threshold = -3.0
                    kline_atr = fetch_kline(code, 60)
                    if kline_atr is not None and len(kline_atr) > 20 and "high" in kline_atr.columns and "low" in kline_atr.columns:
                        from src.signals import calc_atr
                        c_atr = kline_atr["close"].values.astype(float)
                        h_atr = kline_atr["high"].values.astype(float)
                        l_atr = kline_atr["low"].values.astype(float)
                        atr_v = calc_atr(c_atr, h_atr, l_atr, 14)
                        if atr_v > 0 and price > 0:
                            atr_pct = atr_v / price * 100
                            dynamic_threshold = max(-5.0, -atr_pct * 1.5)  # 1.5倍ATR，最多放宽到-5%
                    if drop_pct > dynamic_threshold:
                        continue  # 在ATR正常范围内，不算真实跳水
                    intraday_alerts.add(alert_key + "_drop")
                    display = realtime.get("name", name)
                    logger.info("跳水预警: %s %.1f%%", display, abs(drop_pct))
                    intraday_events.append(f"  ⚠️ {display} 跳水 {abs(drop_pct):.1f}%")
                    # 数据分析后再决定是否卖出
                    should_sell = True  # 默认卖出
                    analysis_reason = f"跳水{abs(drop_pct):.1f}%"
                    # 检查是否有持仓
                    if code in paper.portfolio.positions:
                        pos = paper.portfolio.positions[code]
                        # 获取K线数据计算ATR
                        kline_drop = fetch_kline(code, 60)
                        if kline_drop is not None and len(kline_drop) > 20:
                            from src.signals import calc_atr, _sma
                            closes_drop = kline_drop["close"].values.astype(float)
                            if "high" in kline_drop.columns and "low" in kline_drop.columns:
                                highs_drop = kline_drop["high"].values.astype(float)
                                lows_drop = kline_drop["low"].values.astype(float)
                                atr_drop = calc_atr(closes_drop, highs_drop, lows_drop, 14)
                                # 如果跳水幅度小于ATR的2倍，可能是正常波动
                                if atr_drop > 0:
                                    atr_pct = atr_drop / price * 100
                                    if abs(drop_pct) < atr_pct * 2:
                                        should_sell = False
                                        analysis_reason = f"跳水{abs(drop_pct):.1f}%但在ATR范围内"
                        # 如果价格仍在MA20上方，说明趋势未破，不卖
                        kline_temp = fetch_kline(code, 60)
                        if kline_temp is not None and len(kline_temp) > 25:
                            from src.signals import _sma
                            closes_temp = kline_temp["close"].values.astype(float)
                            ma20 = _sma(closes_temp, 20)
                            valid = ~np.isnan(ma20)
                            if len(ma20[valid]) > 0:
                                ma20_val = ma20[valid][-1]
                                if price > ma20_val:
                                    should_sell = False
                                    analysis_reason = f"跳水但仍在MA20({ma20_val:.2f})上方"
                    if should_sell:
                        # 跳水改为卖一半，保留底仓
                        pos2 = paper.portfolio.positions.get(code)
                        if pos2 and pos2.shares >= 200:
                            half_shares = max(int(pos2.shares / 2 / 100) * 100, 100)
                            sell_trade = paper._sell_partial(code, price, half_shares, analysis_reason)
                        else:
                            sell_trade = paper._sell_position(code, price, analysis_reason)
                        if sell_trade:
                            profit_str = f" 盈亏{sell_trade.profit_pct:+.2f}%" if sell_trade.profit_pct else ""
                            profit_extra = f" {sell_trade.profit_pct:+.2f}%" if sell_trade.profit_pct else ""
                            notify(config, "🔴 交易提醒",
                                f"🔴 **卖出 {display}({code})**\n"
                                f"⏰ {now.strftime('%H:%M:%S')}\n"
                                f"价格: {price:.2f}元\n"
                                f"数量: {sell_trade.shares}股\n"
                                f"金额: {price*sell_trade.shares:.0f}元{profit_extra}\n"
                                f"原因: {sell_trade.reason}")
                    else:
                        logger.info("跳水不卖出: %s", analysis_reason)
                        intraday_events.append(f"  ℹ️ {display} {analysis_reason}，暂不操作")

                # 深V检测：先跌超2%再反弹超0.5%，成交量萎缩更可靠
                if drop_pct <= -2.0 and rebound_pct >= 0.5 and abs(total_chg) < 1.0 and alert_key + "_vv" not in intraday_alerts:
                    intraday_alerts.add(alert_key + "_vv")
                    display = realtime.get("name", name)
                    logger.info("深V检测: %s 跌%.1f%%后反弹%.1f%%", display, abs(drop_pct), rebound_pct)
                    intraday_events.append(f"  🆘 {display} 深V反弹 {rebound_pct:.1f}%")
                    should_buy = True
                    analysis_reason = f"深V反弹{rebound_pct:.1f}%"
                    # ── 当日累计跌幅检查：跌超5%不接飞刀 ──
                    if chg <= -5:
                        should_buy = False
                        analysis_reason = f"深V但当日累跌{chg:.1f}%，趋势已坏不接"
                    # 成交量萎缩（说明抛压衰竭）加分
                    vol_ok = True
                    if kline is not None and "volume" in kline.columns and len(kline) >= 10:
                        vols = kline["volume"].values.astype(float)
                        avg_v = np.mean(vols[-10:-1])
                        cur_v = realtime.get("volume", 0) if realtime else 0
                        vol_ratio = cur_v / avg_v if avg_v > 0 else 1
                        if vol_ratio > 2:
                            vol_ok = False
                            analysis_reason = f"深V但放量{vol_ratio:.1f}倍，抛压仍在"
                    # 检查趋势 - 如果均线空头排列则不买
                    kline_vv = fetch_kline(code, 60)
                    if kline_vv is not None and len(kline_vv) > 25:
                        from src.signals import _sma
                        closes_vv = kline_vv["close"].values.astype(float)
                        ma5_vv = _sma(closes_vv, 5)
                        ma20_vv = _sma(closes_vv, 20)
                        valid_vv = ~np.isnan(ma5_vv) & ~np.isnan(ma20_vv)
                        if len(ma5_vv[valid_vv]) > 0 and len(ma20_vv[valid_vv]) > 0:
                            if ma5_vv[valid_vv][-1] < ma20_vv[valid_vv][-1]:
                                should_buy = False
                                analysis_reason = "均线空头排列，反弹可能只是昙花一现"
                    if should_buy and vol_ok and code not in paper.portfolio.positions:
                        # ETF优先：ETF给更高仓位
                        vv_ratio = 0.15 if code.startswith(("5", "1")) else 0.10
                        buy_trade = paper._buy_position(code, display, price, vv_ratio, analysis_reason)
                        if buy_trade:
                            notify(config, "🟢 交易提醒",
                                f"🟢 **买入 {display}({code})**\n"
                                f"⏰ {now.strftime('%H:%M:%S')}\n"
                                f"价格: {price:.2f}元\n"
                                f"数量: {buy_trade.shares}股\n"
                                f"金额: {price*buy_trade.shares:.0f}元\n"
                                f"原因: {analysis_reason}")

                # 急跌抄底：分级抄底+量能判断+大盘联动
                # 跌3%/5%/7%三档，逐级加仓
                panic_tier = None
                panic_ratio = 0
                if drop_pct <= -7.0:
                    panic_tier = 7; panic_ratio = 0.15
                elif drop_pct <= -5.0:
                    panic_tier = 5; panic_ratio = 0.10
                elif drop_pct <= -3.0:
                    panic_tier = 3; panic_ratio = 0.05
                if panic_tier and alert_key + "_panic" not in intraday_alerts:
                    intraday_alerts.add(alert_key + "_panic")
                    display = realtime.get("name", name)
                    should_buy_panic = False
                    kline_panic = fetch_kline(code, 60)
                    if kline_panic is not None and len(kline_panic) > 20:
                        closes_panic = kline_panic["close"].values.astype(float)
                        from src.signals import _calc_rsi, _sma
                        rsi_val = _calc_rsi(closes_panic, 14)
                        ma20_panic = _sma(closes_panic, 20)
                        ma60_panic = _sma(closes_panic, 60)
                        valid_p = ~np.isnan(ma20_panic) & ~np.isnan(ma60_panic)
                        above_ma60 = ma60_panic[valid_p][-1] < price if len(ma60_panic[valid_p]) > 0 else False
                        # 超卖 + 趋势未破
                        rsi_ok = rsi_val is not None and rsi_val < 30
                        trend_ok = above_ma60
                        # 成交量判断：放量下跌不抄（恐慌未出清），缩量下跌才抄
                        vol_ok_panic = True
                        if "volume" in kline_panic.columns:
                            vols_p = kline_panic["volume"].values.astype(float)
                            avg_v_p = np.mean(vols_p[-10:-1]) if len(vols_p) >= 10 else np.mean(vols_p)
                            cur_v_p = realtime.get("volume", 0) if realtime else vols_p[-1] if len(vols_p) > 0 else 0
                            v_ratio_p = cur_v_p / avg_v_p if avg_v_p > 0 else 1
                            if v_ratio_p > 2:
                                vol_ok_panic = False  # 放量下跌，不抄
                        # 大盘环境：跌太猛时减半仓而不是禁止
                        market_penalty = 1.0
                        try:
                            from src.fetcher import fetch_market_index
                            sh = fetch_market_index("000001")
                            if sh:
                                sh_chg = sh.get("change_pct", 0)
                                if sh_chg <= -3:
                                    market_penalty = 0.3  # 大盘暴跌，极轻仓
                                elif sh_chg <= -2:
                                    market_penalty = 0.5  # 大盘大跌，减半
                                elif sh_chg <= -1:
                                    market_penalty = 0.7  # 大盘小跌，7折
                        except: pass
                        if (rsi_ok or trend_ok) and vol_ok_panic:
                            should_buy_panic = True
                    if should_buy_panic and code not in paper.portfolio.positions:
                        final_ratio = panic_ratio * market_penalty
                        # ETF优先：ETF给更高仓位
                        if code.startswith(("5", "1")):
                            final_ratio = min(final_ratio * 1.5, 0.20)
                        if final_ratio >= 0.03:
                            buy_trade = paper._buy_position(code, display, price, final_ratio,
                                f"急跌{panic_tier}%·RSI{rsi_val:.0f}·仓位{final_ratio:.0%}")
                            if buy_trade:
                                logger.info("急跌抄底: %s 跌%.1f%% RSI%.0f 仓位%.0f%%", display, abs(drop_pct), rsi_val, final_ratio*100)
                                notify(config, "🟢 交易提醒",
                                    f"🟢 **急跌抄底 {display}({code})**\n"
                                    f"⏰ {now.strftime('%H:%M:%S')}\n"
                                    f"价格: {price:.2f}元\n"
                                    f"数量: {buy_trade.shares}股\n"
                                    f"金额: {price*buy_trade.shares:.0f}元\n"
                                    f"原因: 急跌{abs(drop_pct):.0f}%抄底 RSI{rsi_val:.0f}")

                # ── 今日整体大跌抄底（全天累计跌幅深，不看几分钟窗口） ──
                daily_drop_key = f"daily_{code}_{now.strftime('%Y%m%d')}"
                if chg <= -4.0 and daily_drop_key not in intraday_alerts:
                    intraday_alerts.add(daily_drop_key)
                    d_display = realtime.get("name", name)
                    should_buy_daily = False
                    buy_reason_daily = ""
                    # 跌4~6%：需要缩量+MA60上方
                    if -6 < chg <= -4:
                        kline_daily = fetch_kline(code, 60)
                        if kline_daily is not None and len(kline_daily) > 25:
                            from src.signals import _sma
                            closes_d = kline_daily["close"].values.astype(float)
                            ma60_d = _sma(closes_d, 60)
                            valid_d = ~np.isnan(ma60_d)
                            above_ma60 = ma60_d[valid_d][-1] < price if len(ma60_d[valid_d]) > 0 else False
                            vol_daily = kline_daily["volume"].values.astype(float) if "volume" in kline_daily.columns else []
                            vol_ok_daily = True
                            if len(vol_daily) >= 5:
                                vr = vol_daily[-1] / max(np.mean(vol_daily[-5:-1]), 1)
                                if vr > 2: vol_ok_daily = False  # 放量不抄
                            if above_ma60 and vol_ok_daily:
                                should_buy_daily = True
                                buy_reason_daily = f"今日跌{abs(chg):.0f}%·缩量·MA60上方"
                    # 跌超6%：RSI超卖就直接抄
                    elif chg <= -6:
                        kline_daily = fetch_kline(code, 60)
                        if kline_daily is not None and len(kline_daily) > 20:
                            from src.signals import _calc_rsi
                            closes_daily = kline_daily["close"].values.astype(float)
                            rsi_daily = _calc_rsi(closes_daily, 14)
                            if rsi_daily is not None and rsi_daily < 35:
                                should_buy_daily = True
                                buy_reason_daily = f"今日暴跌{abs(chg):.0f}%·RSI{rsi_daily:.0f}超卖"
                    if should_buy_daily and code not in paper.portfolio.positions:
                        daily_ratio = 0.08 if code.startswith(("5", "1")) else 0.05
                        buy_daily = paper._buy_position(code, d_display, price, daily_ratio,
                            f"今日大跌{abs(chg):.0f}%抄底·{buy_reason_daily}")
                        if buy_daily:
                            logger.info("今日大跌抄底: %s 跌%.1f%% %s", d_display, abs(chg), buy_reason_daily)
                            batch_messages.append(f"  🔄 今日大跌抄底 {d_display}({code}) {price:.2f}元×{buy_daily.shares}股")

                # ── 持仓低吸加仓：持仓股大跌时，检查支撑位是否值得加仓 ──
                dip_key = f"dip_{code}_{now.strftime('%Y%m%d')}"
                if code in paper.portfolio.positions and chg <= -3 and chg >= -8 and dip_key not in intraday_alerts:
                    kline_dip = fetch_kline(code, 60)
                    if kline_dip is not None and len(kline_dip) > 20:
                        dip_display = realtime.get("name", name)
                        dip_trade = paper.dip_add_position(code, dip_display, price, chg, kline_dip)
                        if dip_trade:
                            intraday_alerts.add(dip_key)
                            logger.info("低吸加仓: %s 跌%.1f%% %d股", dip_display, abs(chg), dip_trade.shares)
                            batch_messages.append(f"  🔄 低吸加仓 {dip_display}({code}) {price:.2f}元×{dip_trade.shares}股")

            # ── 大盘风险预警（上证涨跌超1.5%/1%，每小时一次） ──
            try:
                sh = fetch_market_index("000001")
                if sh:
                    sh_chg = sh.get("change_pct", 0)
                    sh_key = f"sh_{now.strftime('%Y%m%d_%H')}"
                    if sh_chg <= -1.0 and sh_key + "_risk" not in intraday_alerts:
                        intraday_alerts.add(sh_key + "_risk")
                        level = "🔴 风险" if sh_chg <= -1.5 else "🟡 警告"
                        intraday_events.append(f"  {level} 大盘跌{sh_chg:.1f}%({sh.get('name','上证')})")
                    elif sh_chg >= 1.5 and sh_key + "_rally" not in intraday_alerts:
                        intraday_alerts.add(sh_key + "_rally")
                        intraday_events.append(f"  🟢 大盘涨{sh_chg:+.1f}%({sh.get('name','上证')}) 强势")
            except Exception as e:
                logger.debug("大盘风险检测失败: %s", e)

            # ── 合并推送本轮消息（技术信号+交易+异动）—— 每5分钟一次，有交易立即推 ──
            important_msgs = [m for m in batch_messages if "🔄" in m]
            signal_msgs = [m for m in batch_messages if m.startswith("  · ")]
            
            # 异动/预警立即推送（每轮都推）
            if intraday_events:
                ev_lines = [f"🚨 **盘中异动** · {now.strftime('%H:%M')}"]
                for m in intraday_events:
                    ev_lines.append(m)
                if important_msgs:
                    ev_lines.append("")
                    for m in important_msgs:
                        ev_lines.append(m)
                notify(config, "🚨 盘中异动", "\n".join(ev_lines))
                intraday_events = []  # 清空已推送的异动
            
            # 信号播报每5分钟推一次
            flash_key = f"flash_{now.strftime('%Y%m%d_%H')}_{now.minute // 5}"
            if important_msgs or (flash_key not in intraday_alerts and signal_msgs):
                if not important_msgs:
                    intraday_alerts.add(flash_key)
                merged_lines = [f"📊 **盘中快报** · {now.strftime('%H:%M')}"]
                if signal_msgs:
                    for m in signal_msgs:
                        merged_lines.append(m)
                if important_msgs:
                    if signal_msgs:
                        merged_lines.append("")
                    for m in important_msgs:
                        merged_lines.append(m)
                notify(config, "📊 盘中快报", "\n".join(merged_lines))

            # ── 资金不足推荐买入 ──
            if paper.buy_recommendations:
                rec_lines = ["💡 **资金不足·推荐买入**", ""]
                for rec in paper.buy_recommendations:
                    rec_lines.append(f"  📌 {rec}")
                rec_lines.append("")
                rec_lines.append("  💰 建议: 卖出部分持仓或追加资金")
                paper.buy_recommendations.clear()
                notify(config, "💡 资金不足推荐", "\n".join(rec_lines))

            # ── 板块集中风险检测（每小时一次） ──
            sector_key = f"sector_{now.strftime('%Y%m%d_%H')}"
            if sector_key not in intraday_alerts:
                sector_map = {}
                for pc, pp in paper.portfolio.positions.items():
                    tag = get_sector_tag(pc)
                    if tag:
                        if tag not in sector_map: sector_map[tag] = []
                        sector_map[tag].append(f"{pp.stock_name}({pc})")
                risk_sectors = {k: v for k, v in sector_map.items() if len(v) >= 3}
                if risk_sectors:
                    intraday_alerts.add(sector_key)
                    sec_lines = ["⚠️ **板块集中风险**", ""]
                    for s, stocks_list in risk_sectors.items():
                        sec_lines.append(f"  📉 {s}: {len(stocks_list)}只股票")
                        for s_name in stocks_list:
                            sec_lines.append(f"     {s_name}")
                    sec_lines.append("")
                    sec_lines.append("  💡 建议: 关注分散风险，同板块持仓不超过3只")
                    notify(config, "⚠️ 板块集中风险", "\n".join(sec_lines))

            # ── 盘后总结（15:00，每天一次） ──
            if not summary_done_today and now_time >= dt_time(15, 0):
                summary_done_today = True
                logger.info("生成盘后总结...")
                summary = _generate_summary(config, today_signals, paper)

                # ── 策略回测合并到盘后总结 ──
                try:
                    from src.backtest import backtest_scoring_strategy
                    from src.performance import record_daily_result
                    from src.optimizer import auto_optimize, get_param_summary, refresh_strategies, get_stock_params, STRATEGY_TEMPLATES
                    bt_lines = []
                    bt_total = 0.0
                    bt_count = 0
                    bt_wins = 0
                    # 获取策略分类
                    try:
                        strat_map = refresh_strategies()
                    except:
                        strat_map = {}
                    for bt_code, bt_name in stocks.items():
                        bt_kline = fetch_kline(bt_code, 365)
                        if bt_kline is not None:
                            bt_r = backtest_scoring_strategy(bt_code, bt_name, bt_kline)
                            bt_icon = "+" if bt_r.total_return > 0 else ""
                            s_info = strat_map.get(bt_code, {})
                            s_type = s_info.get("strategy", "稳健")
                            s_mark = {"激进": "🚀", "稳健": "⚖️", "保守": "🛡️"}.get(s_type, "❓")
                            bt_lines.append(f"  {s_mark} {bt_name}({bt_code}): {bt_r.total_return:+.1f}% 胜率{bt_r.win_rate:.0f}%  {s_type}")
                            bt_total += bt_r.total_return
                            bt_count += 1
                            if bt_r.total_return > 0:
                                bt_wins += 1
                    if bt_count > 0:
                        avg_bt = bt_total / bt_count
                        win_rate_bt = bt_wins / bt_count * 100
                        summary += "\n\n📈 **策略回测**\n"
                        summary += f"  📊 平均收益率: {avg_bt:+.1f}% | 胜率: {win_rate_bt:.0f}%\n"
                        trend = record_daily_result(avg_bt, win_rate_bt, bt_count)
                        summary += f"  {trend}\n"
                        opt_result = auto_optimize(avg_bt, win_rate_bt)
                        if opt_result["adjusted"]:
                            summary += f"  🔧 {opt_result['reason']}\n"
                            for k, v in opt_result["changes"].items():
                                label_map = {"buy_threshold": "买入阈值", "sell_threshold": "卖出阈值",
                                         "stop_loss": "止损", "trail_activate": "止盈启动",
                                         "trail_pullback": "止盈回撤"}
                                summary += f"    {label_map.get(k,k)}: {v['from']} → {v['to']}\n"
                        try:
                            strategies = refresh_strategies()
                            strat_counts = {"激进": 0, "稳健": 0, "保守": 0}
                            for s_info in strategies.values():
                                strat_counts[s_info["strategy"]] = strat_counts.get(s_info["strategy"], 0) + 1
                            summary += f"  📋 策略: 激进{strat_counts['激进']}只/稳健{strat_counts['稳健']}只/保守{strat_counts['保守']}只\n"
                        except: pass
                except Exception as e:
                    logger.debug("策略回测失败: %s", e)

                if summary:
                    notify(config, "📋 收盘总览", summary)

                # ── 板块热点推送（每日一次） ──
                try:
                    from src.fetcher import fetch_sector_performance
                    sectors = fetch_sector_performance()
                    if sectors:
                        sec_lines = ["🏆 **今日板块热点 TOP5**", ""]
                        for i, s in enumerate(sectors[:5], 1):
                            icon = "📈" if s["change_pct"] >= 0 else "📉"
                            sec_lines.append(f"  {i}. {icon} {s['name']} {s['change_pct']:+.2f}%")
                        notify(config, "🏆 板块热点", "\n".join(sec_lines))
                except Exception as e:
                    logger.debug("板块热点推送失败: %s", e)

                # ── 每周交易总结（周五盘后） ──
                if now.weekday() == 4:
                    try:
                        sells = [t for t in paper.portfolio.trades if t.action == "卖出"]
                        wins = [t for t in sells if t.profit_pct > 0]
                        total_sells = len(sells)
                        week_trades = [t for t in paper.portfolio.trades if (now - datetime.strptime(t.date, "%Y-%m-%d")).days < 7]
                        week_sells = [t for t in week_trades if t.action == "卖出"]
                        week_wins = [t for t in week_sells if t.profit_pct > 0]
                        week_lines = ["📊 **本周交易总结**", ""]
                        week_lines.append(f"  交易次数: {len(week_trades)} 次")
                        week_lines.append(f"  卖出: {len(week_sells)} 次 | 盈利: {len(week_wins)} 次")
                        if week_sells:
                            win_rate = len(week_wins) / len(week_sells) * 100
                            total_profit = sum(t.profit_amount for t in week_sells if t.profit_amount)
                            week_lines.append(f"  胜率: {win_rate:.1f}% | 总盈亏: {total_profit:+.2f}元")
                        if sells:
                            all_win_rate = len(wins) / total_sells * 100 if total_sells > 0 else 0
                            all_profit = sum(t.profit_amount for t in sells if t.profit_amount)
                            week_lines.append("")
                            week_lines.append(f"  📈 历史累计: 胜率{all_win_rate:.1f}% 盈亏{all_profit:+.2f}元")
                        week_lines.append("")
                        ret = (paper.portfolio.total_value - 500000) / 500000 * 100
                        week_lines.append(f"  💰 账户收益: {ret:+.2f}%")
                        notify(config, "📊 每周交易总结", "\n".join(week_lines))
                    except Exception as e:
                        logger.debug("每周总结推送失败: %s", e)
        else:
            logger.debug("非交易时间，跳过")

        # 15:05后自动退出
        if now_time >= dt_time(15, 5):
            if not summary_done_today:
                summary_done_today = True
            logger.info("15:05 收盘，监控停止")
            break

        # ── 挂单检查：突破信号触发后生成的挂单，等待价格进入区间 ──
        try:
            filled = order_mgr.check_orders(current_state)
            for oid, f_ord in filled:
                if f_ord.direction == "buy":
                    buy_t = paper.execute_range_buy(oid, f_ord, current_state.get(f_ord.stock_code, 0))
                    if buy_t:
                        pr = f_ord.price_range
                        notify(config, "🟢 突破买入",
                            f"🟢 **突破买入 {f_ord.stock_name}({f_ord.stock_code})**\n"
                            f"⏰ {now.strftime('%H:%M:%S')}\n"
                            f"买入价: {buy_t.price:.2f}元\n"
                            f"数量: {buy_t.shares}股\n"
                            f"金额: {buy_t.price*buy_t.shares:.0f}元\n"
                            f"区间: [{pr.buy_lower:.2f}~{pr.buy_upper:.2f}]\n"
                            f"止损: {pr.stop_loss:.2f}\n"
                            f"原因: {f_ord.reason}")
        except Exception as e:
            logger.debug("挂单检查失败: %s", e)

        time.sleep(interval)


if __name__ == "__main__":
    main()
