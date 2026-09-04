from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Any
from urllib.parse import quote_plus, urlsplit

import httpx
from fastapi import WebSocket
from loguru import logger


def _build_dingtalk_payload(alert: dict) -> dict:
    emoji = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "green": "🟢"}.get(alert.get("level", ""), "⚪")
    title = str(alert.get("title", "预警通知"))
    text = (
        f"## {emoji} 智护安康 预警通知\n\n"
        f"**告警内容：**{title}\n\n"
        f"**设备：**{alert.get('device_name', '')}（{alert.get('scene', '')}）\n\n"
        f"**老人：**{alert.get('resident_name', '') or '未绑定'}\n\n"
        f"**时间：**{alert.get('created_at', '')}\n\n"
        f"**风险评分：**{alert.get('score', 0)}\n\n---\n请及时查看并处置。"
    )
    return {"msgtype": "markdown", "markdown": {"title": title, "text": text}}


def _build_feishu_payload(alert: dict) -> dict:
    color = {"red": "red", "orange": "orange", "yellow": "yellow", "green": "green"}.get(alert.get("level", ""), "grey")
    text = (
        f"**设备：**{alert.get('device_name', '')}（{alert.get('scene', '')}）\n"
        f"**老人：**{alert.get('resident_name', '') or '未绑定'}\n"
        f"**时间：**{alert.get('created_at', '')}\n"
        f"**风险评分：**{alert.get('score', 0)}"
    )
    return {"msg_type": "interactive", "card": {"header": {"title": {"tag": "plain_text", "content": str(alert.get("title", "预警通知"))}, "template": color}, "elements": [{"tag": "div", "text": {"tag": "lark_md", "content": text}}, {"tag": "hr"}, {"tag": "note", "elements": [{"tag": "plain_text", "content": "请及时查看并处置"}]}]}}


def _build_wechat_payload(alert: dict) -> dict:
    emoji = {"red": "🔴", "orange": "🟠", "yellow": "🟡", "green": "🟢"}.get(alert.get("level", ""), "⚪")
    content = (
        f"## {emoji} 智护安康 预警通知\n"
        f"> 告警：<font color=\"warning\">{alert.get('title', '')}</font>\n"
        f"> 设备：{alert.get('device_name', '')}（{alert.get('scene', '')}）\n"
        f"> 老人：{alert.get('resident_name', '') or '未绑定'}\n"
        f"> 时间：{alert.get('created_at', '')}\n"
        f"> 风险评分：{alert.get('score', 0)}"
    )
    return {"msgtype": "markdown", "markdown": {"content": content}}


def _sign_dingtalk(secret: str) -> str:
    timestamp = str(round(time.time() * 1000))
    digest = hmac.new(secret.encode(), f"{timestamp}\n{secret}".encode(), hashlib.sha256).digest()
    return f"timestamp={timestamp}&sign={quote_plus(base64.b64encode(digest).decode())}"


async def send_webhook(webhook_url: str, platform: str, secret: str | None, alert: dict) -> bool:
    """推送告警；日志只记录主机，不泄露 URL query/token。"""
    url = webhook_url
    if platform == "dingtalk" and secret:
        url += ("&" if "?" in url else "?") + _sign_dingtalk(secret)
    payload = _build_dingtalk_payload(alert) if platform == "dingtalk" else _build_feishu_payload(alert) if platform == "feishu" else _build_wechat_payload(alert) if platform == "wechat" else {"msgtype": "alert", "alert": alert}
    host = urlsplit(webhook_url).hostname or "invalid-host"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
            if response.status_code >= 400:
                logger.warning(f"Webhook push failed [{platform}] host={host}: HTTP {response.status_code}")
                return False
            logger.info(f"Webhook push ok [{platform}] host={host}")
            return True
    except Exception as exc:
        logger.warning(f"Webhook push error [{platform}] host={host}: {type(exc).__name__}")
        return False


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WebSocket 客户端接入, total={len(self.active)}")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
