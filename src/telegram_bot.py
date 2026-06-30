"""
Telegram 机器人模块 — 双向通信，支持推送信号和接收命令
"""
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── 先注册一个空的bot，等用户提供token后启用 ──
_bot = None
_token = ""
_chat_id = 0


def init_bot(token: str, chat_id: int):
    """初始化Telegram机器人"""
    global _bot, _token, _chat_id
    _token = token
    _chat_id = chat_id
    try:
        from telegram import Bot
        from telegram.ext import Application, CommandHandler
        _bot = Bot(token)
        logger.info("Telegram机器人初始化成功")
        return True
    except ImportError:
        logger.warning("请先安装: pip install python-telegram-bot")
        return False
    except Exception as e:
        logger.error("Telegram初始化失败: %s", e)
        return False


def send_message(text: str) -> bool:
    """发送消息到Telegram"""
    if not _bot or not _chat_id:
        return False
    try:
        _bot.send_message(chat_id=_chat_id, text=text, parse_mode="Markdown")
        return True
    except Exception as e:
        logger.error("Telegram发送失败: %s", e)
        return False


def notify(content: str) -> bool:
    """统一通知接口"""
    if send_message(content):
        logger.info("Telegram推送成功")
        return True
    return False
