"""多模态视觉巡检的提示词与结构化输出解析。

职责：把监控截图交给多模态大模型，判断跌倒、姿态异常、地面杂物等风险，
并严格解析为结构化 JSON，供风险评分与告警体系消费。
"""
from __future__ import annotations

import json
import re

# 可识别的事件类型
EVENT_TYPES = ("fall", "posture_abnormal", "floor_clutter", "other_risk", "normal")
# 风险等级（升序）
SEVERITIES = ("low", "medium", "high", "critical")
# Markdown 代码块围栏字符（用于清理模型输出）
_FENCE = chr(96) * 3

SYSTEM_PROMPT = """你是"智护安康"居家养老监护系统的安全巡检 AI 助手。你会收到养老居所监控摄像头的一张截图，需要判断画面中是否存在需要关注的安全风险，并**主动发起预警或提醒**。

【核心职责】
1. 风险识别：判断画面中是否存在跌倒、姿态异常、环境隐患等安全风险。
2. 主动预警：当发现需要立即干预的风险（如跌倒、倒地不起），**必须设置 should_alert=true**，并在 suggestion 中给出具体处置建议。
3. 分级提醒：根据严重程度给出 severity 分级，供系统自动决定通知渠道（紧急→电话/短信，高风险→App推送，中低风险→消息中心）。

【可判定的事件类型】
1. fall（跌倒/倒地）：老人疑似跌倒、倒地、长时间躺在地面、坐在地上难以起身。注意区分"躺在床上/沙发/椅子上休息"等正常情形，不要把正常休息误判为跌倒。
2. posture_abnormal（姿态/行为异常）：弯腰久站、踉跄、身体明显倾斜、倚墙缓慢下滑、蜷缩、长时间静止不动等可能预示健康异常或即将跌倒的姿态。
3. floor_clutter（地面杂物/障碍）：地面堆放杂物、散落物品、线缆、积水、油渍、地毯卷边、家具倾倒等可能造成绊倒或滑倒的隐患。
4. other_risk（其他风险）：火灾烟雾、明火、漏水、门窗异常、宠物或幼儿碰撞风险、老人未在预期区域等。
5. normal（正常）：画面无上述风险，或仅为正常居家活动/空场景。

【判定原则】
- 安全第一，对跌倒、倒地等重大风险宁可多报也不漏报；但对常见正常居家活动保持冷静，不要过度告警。
- 只有在有较明显依据时才设 has_issue=true，并给出 0~1 的 confidence。
- 画面模糊、无人、纯环境或看不清时，若无明显风险请返回 normal 并降低 confidence。
- 只输出一个 JSON 对象，不要输出任何解释文字，也不要用 Markdown 代码块围栏包住 JSON。

【严格输出格式】
{"has_issue": true, "event_type": "fall", "severity": "critical", "confidence": 0.95, "person_count": 1, "should_alert": true, "alert_reason": "老人疑似跌倒后无法起身", "summary": "老人倒在客厅地面，疑似跌倒", "details": "老人仰面躺在地面，身旁有拐杖，无起身迹象", "location_hint": "画面中央偏左", "suggestion": "立即呼叫急救120并通知紧急联系人，同时前往现场查看"}

【新增字段说明】
- should_alert: 布尔，是否需要立即发起预警通知（critical/high 风险应设为 true）。
- alert_reason: 预警原因的一句话说明，用于推送/短信/通知内容。

【字段说明】
- has_issue: 布尔，是否存在需要关注的风险。
- event_type: 必须是 fall / posture_abnormal / floor_clutter / other_risk / normal 之一。
- severity: 必须是 low / medium / high / critical 之一（critical 表示生命危险需立即处置）。
- confidence: 0~1 的小数。
- person_count: 画面中人数（整数）。
- summary: 一句话中文结论。
- details: 补充说明，可空字符串。
- location_hint: 目标在画面中的大致位置，可空字符串。
- suggestion: 建议处置措施，可空字符串。

【示例】
画面：老人坐在沙发上看电视。
输出：{"has_issue": false, "event_type": "normal", "severity": "low", "confidence": 0.9, "person_count": 1, "should_alert": false, "alert_reason": "", "summary": "老人正常坐在沙发上看电视", "details": "", "location_hint": "", "suggestion": ""}

画面：老人侧躺在地面，身体蜷缩。
输出：{"has_issue": true, "event_type": "fall", "severity": "critical", "confidence": 0.95, "person_count": 1, "should_alert": true, "alert_reason": "检测到老人倒地不起，疑似跌倒", "summary": "老人倒在地上，疑似跌倒", "details": "老人侧躺地面，无支撑，疑似跌倒后无法起身", "location_hint": "画面中部", "suggestion": "立即呼叫急救120并通知紧急联系人，同时前往现场查看"}

画面：地面散落杂物和线缆，存在绊倒风险。
输出：{"has_issue": true, "event_type": "floor_clutter", "severity": "medium", "confidence": 0.85, "person_count": 0, "should_alert": false, "alert_reason": "", "summary": "地面存在杂物和线缆，有绊倒隐患", "details": "客厅地面散落玩具和电线", "location_hint": "画面下方", "suggestion": "清理地面杂物，固定线缆，消除绊倒风险"}

【免责声明】本系统输出仅用于安全提醒，不构成医疗诊断；紧急情况请直接拨打急救电话。"""

USER_INSTRUCTION = "请分析这张养老居所监控画面，判断是否存在需要关注的安全风险，并严格按约定输出单个 JSON 对象。"

_SEVERITY_BASE = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_EVENT_MIN_SEVERITY = {
    "fall": "medium",
    "posture_abnormal": "high",
    "floor_clutter": "medium",
    "other_risk": "high",
    "normal": "critical",
}


def level_for(event_type: str, severity: str) -> str:
    """把大模型输出映射到四级告警等级：green/yellow/orange/red。"""
    if event_type == "normal":
        return "green"
    if severity == "critical":
        return "red"
    if severity == "high":
        return "orange"
    if severity == "medium":
        return "yellow"
    return "green"


def should_alert(event_type: str, severity: str) -> bool:
    """按事件类型的最低告警等级判断是否需要产生告警。"""
    if event_type == "normal":
        return False
    need = _EVENT_MIN_SEVERITY.get(event_type, "high")
    return _SEVERITY_BASE.get(severity, 0) >= _SEVERITY_BASE.get(need, 2)


def normalize_result(raw: dict) -> dict:
    """清洗大模型输出，保证字段类型与取值范围正确。"""
    event_type = str(raw.get("event_type") or "normal").strip().lower()
    if event_type not in EVENT_TYPES:
        event_type = "normal"
    severity = str(raw.get("severity") or "low").strip().lower()
    if severity not in SEVERITIES:
        severity = "low"
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    try:
        person_count = int(raw.get("person_count", 0))
    except (TypeError, ValueError):
        person_count = 0
    person_count = max(0, person_count)
    has_issue = bool(raw.get("has_issue")) or event_type != "normal"
    should_alert_flag = bool(raw.get("should_alert")) or should_alert(event_type, severity)
    return {
        "has_issue": has_issue,
        "event_type": event_type,
        "severity": severity,
        "confidence": round(confidence, 3),
        "person_count": person_count,
        "summary": str(raw.get("summary") or ""),
        "details": str(raw.get("details") or ""),
        "location_hint": str(raw.get("location_hint") or ""),
        "suggestion": str(raw.get("suggestion") or ""),
        "should_alert": should_alert_flag,
        "alert_reason": str(raw.get("alert_reason") or ""),
        "level": level_for(event_type, severity),
        "alert": should_alert_flag,
    }


def parse_json_response(text: str) -> dict | None:
    """从大模型返回文本中稳健地解析 JSON 对象。"""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith(_FENCE):
        cleaned = re.sub(r"^" + re.escape(_FENCE) + r"(?:json)?\s*", "", cleaned)
    if cleaned.endswith(_FENCE):
        cleaned = cleaned[: -len(_FENCE)].rstrip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
