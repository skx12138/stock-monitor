"""
通知模块 — 支持多通道消息推送

通道:
  - serverchan: Server酱 → 个人微信（主推，免费）
  - wecom: 企业微信机器人 Webhook
"""
import json
import logging
import os
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Server酱 推送（推送到个人微信）
# ──────────────────────────────────────────────

def send_serverchan(key: str, title: str, content: str) -> bool:
    """通过 Server酱 推送消息到个人微信

    Args:
        key: Server酱 SendKey
        title: 消息标题
        content: 消息内容（支持 Markdown）

    Returns:
        成功返回 True
    """
    if not key:
        logger.warning("Server酱 SendKey 为空，跳过推送")
        return False

    url = f"https://sct.ftqq.com/{key}.send"

    try:
        resp = requests.post(
            url,
            data={"title": title, "desp": content},
            timeout=15,
        )
        result = resp.json()
        if result.get("code") == 0:
            logger.info("Server酱 推送成功: %s", result.get("data", {}).get("pushid", ""))
            return True
        else:
            logger.error("Server酱 推送失败: %s", result.get("message", resp.text))
            return False
    except Exception as e:
        logger.error("Server酱 推送异常: %s", e)
        return False


# ──────────────────────────────────────────────
#  企业微信机器人 推送
# ──────────────────────────────────────────────

def send_wecom(webhook_url: str, content: str) -> bool:
    """通过企业微信机器人 Webhook 推送 Markdown 消息"""
    if not webhook_url:
        return False

    payload = {
        "msgtype": "markdown",
        "markdown": {"content": content},
    }

    try:
        resp = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        result = resp.json()
        if result.get("errcode") == 0:
            logger.info("企业微信推送成功")
            return True
        else:
            logger.error("企业微信推送失败: %s", result.get("errmsg", resp.text))
            return False
    except Exception as e:
        logger.error("企业微信推送异常: %s", e)
        return False


# ──────────────────────────────────────────────
#  统一推送接口
# ──────────────────────────────────────────────

def notify(config: dict, title: str, content: str) -> bool:
    """根据配置，通过指定通道推送消息

    支持同时推送到多个通道

    Args:
        config: 完整配置字典
        title: 消息标题（Server酱 专用）
        content: 消息正文

    Returns:
        至少一个通道成功返回 True
    """
    notifier_cfg = config.get("notifier", {})
    notify_type = notifier_cfg.get("type", "wecom")
    success = False

    # Server酱 通道
    serverchan_key = notifier_cfg.get("serverchan_key", "")
    if serverchan_key:
        ok = send_serverchan(serverchan_key, title, content)
        if ok:
            success = True

    # Qmsg酱 通道（QQ推送）
    qmsg_key = notifier_cfg.get("qmsg_key", "")
    if qmsg_key:
        from src.qmsg_bot import send_qmsg
        ok = send_qmsg(qmsg_key, content)
        if ok:
            success = True

    # 企业微信 通道
    wecom_webhook = os.environ.get("WECOM_WEBHOOK", "") or notifier_cfg.get("wecom_webhook", "")
    if wecom_webhook:
        # 企业微信不支持 title/desp 分离，直接发 content
        ok = send_wecom(wecom_webhook, content)
        if ok:
            success = True

    if not success:
        logger.warning("所有通知通道均未成功发送")
        # 无通道配置时打印到终端
        if not serverchan_key and not wecom_webhook:
            print("\n" + "=" * 40)
            print(content)
            print("=" * 40 + "\n")

    return success


# ──────────────────────────────────────────────
#  语义化包装
# ──────────────────────────────────────────────

def notify_signal(config: dict, signal) -> bool:
    """推送信号通知（含操作建议和涨跌幅）"""
    content = signal.message
    # 添加涨跌幅
    if signal.change_pct:
        chg_icon = "📈" if signal.change_pct > 0 else "📉"
        content += f"\n{chg_icon} 当前涨跌: {signal.change_pct:+.2f}%"
    if signal.suggestion:
        content += f"\n💡 **建议：{signal.suggestion}**"
    title = f"💰 {signal.stock_name} {signal.signal_label}"
    return notify(config, title, content)


SIGNAL_NAMES_CN = {
    "ma_crossover": "均线金叉/死叉（MA5与MA20交叉）",
    "rsi": "RSI超买超卖（涨太猛或跌太多提醒）",
    "macd": "MACD金叉/死叉（多空力量对比）",
    "bollinger": "布林带突破（涨太高或跌太低提醒）",
    "volume_breakout": "放量突破（资金异动提醒）",
    "breakout": "突破追涨（放量突破整理区间）",
    "dragon_back": "🐉龙回头（涨停后回调到支撑位）",
}


def notify_startup(config: dict) -> bool:
    """推送启动通知（含所有监控股票实时行情）"""
    stocks = config.get("stocks", {})
    signals_enabled = []
    signals_cfg = config.get("signals", {})
    for name, cfg in signals_cfg.items():
        if isinstance(cfg, dict) and cfg.get("enabled"):
            signals_enabled.append(SIGNAL_NAMES_CN.get(name, name))

    # 获取所有监控股票实时行情
    stock_lines = []
    for code, name in stocks.items():
        try:
            from src.fetcher import fetch_realtime
            rt = fetch_realtime(code)
            if rt:
                price = rt.get("price", 0)
                chg = rt.get("change_pct", 0)
                icon = "📈" if chg >= 0 else "📉"
                stock_lines.append(f"  {icon} {name or code}({code}) {price:.2f} {chg:+.2f}%")
            else:
                stock_lines.append(f"  · {name or code}({code})")
        except:
            stock_lines.append(f"  · {name or code}({code})")

    # 回测表现
    backtest_summary = config.get("_backtest_summary", [])
    paper_report = config.get("_paper_report", "")

    title = "🚀 A股监控已启动"
    signal_lines = "\n".join(f"  · {s}" for s in signals_enabled)
    backtest_lines = "\n".join(backtest_summary) if backtest_summary else ""
    content_parts = [
        f"🚀 **A股行情监控已启动**",
        f"监控股票: {len(stocks)} 只",
        *(["\n".join(stock_lines)] if stock_lines else []),
        "",
        f"信号策略:",
        signal_lines,
    ]
    if backtest_lines:
        content_parts += ["", f"**📊 策略回测（MA10/40+RSI过滤+止损12%）**", backtest_lines]
    if paper_report:
        # 提取前几行作为摘要
        paper_lines = paper_report.split("\n")[:6]
        content_parts += ["", "**💼 模拟账户**", *paper_lines]
    content_parts += ["", f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    content = "\n".join(content_parts)
    return notify(config, title, content)
