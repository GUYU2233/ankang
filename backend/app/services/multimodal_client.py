"""多模态大模型统一客户端：Qwen / GLM / GPT / Gemini（OpenAI 兼容协议）。

仅依赖 httpx + opencv，通过 /chat/completions 以 base64 图片调用视觉模型，
返回结构化风险结果。
"""
from __future__ import annotations

import base64
import time
from typing import Any

import cv2
import httpx

from app.core.multimodal_prompt import (
    SYSTEM_PROMPT,
    USER_INSTRUCTION,
    normalize_result,
    parse_json_response,
)

# 各提供商 OpenAI 兼容接入点与默认模型。api_key 优先从运行时配置取，其次对应环境变量。
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "qwen": {
        "label": "通义千问 Qwen3-VL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen3-vl-235b-a22b-instruct", "qwen3-vl-235b-a22b-thinking", "qwen3-vl-plus", "qwen3-vl-flash", "qwen2.5-vl-72b-instruct"],
        "default_model": "qwen3-vl-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
    },
    "glm": {
        "label": "智谱 GLM-5V",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-5v-turbo", "glm-4.6v", "glm-4v-plus", "glm-4v-flash"],
        "default_model": "glm-5v-turbo",
        "api_key_env": "ZHIPUAI_API_KEY",
    },
    "gpt": {
        "label": "OpenAI GPT-5.6",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-5.6-sol", "gpt-5.5", "gpt-5.4", "gpt-4o"],
        "default_model": "gpt-5.6-sol",
        "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "label": "Google Gemini 3",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": ["gemini-3.5-flash", "gemini-3-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
        "default_model": "gemini-3.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "doubao": {
        "label": "豆包 Seed Vision",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "models": ["doubao-seed-2.0-pro", "doubao-seed-1.6-vision", "doubao-seed-1.6-vision-250815"],
        "default_model": "doubao-seed-1.6-vision",
        "api_key_env": "ARK_API_KEY",
    },
    "deepseek": {
        "label": "DeepSeek-VL2",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-vl2", "deepseek-chat"],
        "default_model": "deepseek-vl2",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "custom": {
        "label": "自定义 OpenAI 兼容接口",
        "base_url": "",
        "models": [],
        "default_model": "",
        "api_key_env": "",
    },
}


class MultimodalError(RuntimeError):
    """多模态调用失败。"""


def _resolve_endpoint(provider: str, base_url: str, model: str, api_key: str) -> tuple[str, str, str]:
    import os

    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])
    resolved_base = base_url or preset["base_url"]
    resolved_model = model or preset["default_model"]
    resolved_key = api_key or os.getenv(preset["api_key_env"] or "", "")
    if not resolved_base:
        raise MultimodalError("未配置 base_url，且所选提供商没有默认接入点")
    if not resolved_model:
        raise MultimodalError("未配置模型名称")
    if not resolved_key:
        raise MultimodalError("未配置 API Key")
    return resolved_base.rstrip("/"), resolved_model, resolved_key


def _encode_image(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise MultimodalError("截图编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def analyze_image(
    frame,
    *,
    provider: str = "qwen",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    temperature: float = 0.2,
    max_tokens: int = 800,
    timeout: float = 30.0,
    system_prompt: str | None = None,
) -> tuple[dict, str, int]:
    """分析一张监控截图，返回 (规范化结果, 原始文本, 耗时毫秒)。"""
    base, model, key = _resolve_endpoint(provider, base_url, model, api_key)
    data_url = _encode_image(frame)

    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        raise MultimodalError(f"多模态服务返回 {exc.response.status_code}: {exc.response.text[:300]}") from exc
    except httpx.HTTPError as exc:
        raise MultimodalError(f"多模态服务调用失败: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    text = ""
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise MultimodalError(f"多模态响应结构异常: {str(payload)[:300]}") from None

    raw = parse_json_response(text)
    if raw is None:
        raise MultimodalError(f"无法从模型输出中解析 JSON: {text[:300]}")
    return normalize_result(raw), text, elapsed_ms
