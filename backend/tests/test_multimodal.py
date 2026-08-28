"""多模态巡检模块单元测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from app.core.multimodal_prompt import (
    SYSTEM_PROMPT,
    level_for,
    normalize_result,
    parse_json_response,
    should_alert,
)
from app.services.multimodal_client import (
    PROVIDER_PRESETS,
    MultimodalError,
    _encode_image,
    _resolve_endpoint,
)
from app.services.multimodal_config import RuntimeConfig


def test_parse_json():
    fence = chr(96) * 3
    clean = '{"has_issue": true, "event_type": "fall", "severity": "critical", "confidence": 0.95}'
    assert parse_json_response(clean)["event_type"] == "fall"
    fenced = fence + "json\n" + clean + "\n" + fence
    assert parse_json_response(fenced)["has_issue"] is True
    noisy = "分析结果如下：" + clean + " 这是补充说明。"
    assert parse_json_response(noisy)["severity"] == "critical"
    assert parse_json_response("") is None
    assert parse_json_response("无法解析的内容") is None
    print("  parse_json OK")


def test_normalize():
    out = normalize_result({"event_type": "fall", "severity": "critical", "confidence": 3.5, "person_count": -2})
    assert out["event_type"] == "fall"
    assert out["severity"] == "critical"
    assert out["confidence"] == 1.0
    assert out["person_count"] == 0
    assert out["has_issue"] is True
    assert out["alert"] is True
    bad = normalize_result({"event_type": "fall", "severity": "ooo"})
    assert bad["severity"] == "low"
    assert bad["alert"] is False
    norm = normalize_result({"event_type": "normal", "confidence": 0.9})
    assert norm["alert"] is False
    assert norm["level"] == "green"
    print("  normalize OK")


def test_level_alert():
    assert level_for("fall", "critical") == "red"
    assert level_for("fall", "high") == "orange"
    assert level_for("floor_clutter", "medium") == "yellow"
    assert level_for("posture_abnormal", "low") == "green"
    assert should_alert("fall", "medium") is True
    assert should_alert("fall", "low") is False
    assert should_alert("posture_abnormal", "high") is True
    assert should_alert("posture_abnormal", "medium") is False
    assert should_alert("floor_clutter", "medium") is True
    assert should_alert("normal", "critical") is False
    print("  level/alert OK")


def test_prompt_contract():
    assert "智护安康" in SYSTEM_PROMPT
    assert "fall" in SYSTEM_PROMPT and "floor_clutter" in SYSTEM_PROMPT
    assert '"has_issue"' in SYSTEM_PROMPT and '"event_type"' in SYSTEM_PROMPT
    assert "不构成医疗诊断" in SYSTEM_PROMPT
    print("  prompt contract OK")


def test_provider_presets():
    for name in ("qwen", "glm", "gpt", "gemini", "custom"):
        assert name in PROVIDER_PRESETS
        assert "label" in PROVIDER_PRESETS[name]
        assert "default_model" in PROVIDER_PRESETS[name]
    assert PROVIDER_PRESETS["qwen"]["base_url"].startswith("https://dashscope")
    assert PROVIDER_PRESETS["gemini"]["base_url"].endswith("/openai")
    print("  provider presets OK")


def test_resolve_endpoint():
    base, model, key = _resolve_endpoint("qwen", "", "", "sk-test")
    assert base == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert model == "qwen-vl-max"
    assert key == "sk-test"
    try:
        _resolve_endpoint("custom", "", "", "")
        raise AssertionError("custom 未配置 base_url 应报错")
    except MultimodalError:
        pass
    print("  resolve endpoint OK")


def test_encode_image():
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    url = _encode_image(frame)
    assert url.startswith("data:image/jpeg;base64,")
    assert len(url) > 30
    print("  encode image OK")


def test_runtime_config():
    key = "sk-abcdefgh12345678"
    cfg = RuntimeConfig(api_key=key)
    assert cfg.mask_key() == key[:4] + "*" * (len(key) - 8) + key[-4:]
    assert cfg.mask_key().count("*") == len(key) - 8
    assert RuntimeConfig(api_key="").mask_key() == ""
    assert RuntimeConfig().is_ready() is False
    assert RuntimeConfig(enabled=True, provider="qwen").is_ready() is True
    assert RuntimeConfig(enabled=True, provider="custom").is_ready() is False
    print("  runtime config OK")


def main():
    test_parse_json()
    test_normalize()
    test_level_alert()
    test_prompt_contract()
    test_provider_presets()
    test_resolve_endpoint()
    test_encode_image()
    test_runtime_config()
    print("MULTIMODAL TESTS OK")


if __name__ == "__main__":
    main()
