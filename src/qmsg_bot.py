"""
Qmsg酱 通知模块 — 推送消息到个人QQ
"""
import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def send_qmsg(key: str, content: str, msg_type: str = "text") -> bool:
    """通过 Qmsg酱 发送消息到QQ

    Args:
        key: Qmsg酱的key
        content: 消息内容
        msg_type: "text" 或 "markdown"

    Returns:
        成功返回 True
    """
    if not key:
        logger.warning("Qmsg Key 为空")
        return False

    url = f"https://qmsg.zendee.cn/send/{key}"
    payload = {"msg": content, "msg_type": msg_type}

    try:
        resp = requests.post(url, data=payload, timeout=10)
        result = resp.json()
        if result.get("code") == 0:
            logger.info("Qmsg推送成功: %s", result.get("msg", ""))
            return True
        else:
            logger.warning("Qmsg推送失败: %s", result.get("msg", resp.text))
            return False
    except Exception as e:
        logger.error("Qmsg推送异常: %s", e)
        return False
