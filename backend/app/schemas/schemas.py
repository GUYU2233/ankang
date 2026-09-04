from datetime import datetime
from pydantic import BaseModel, Field


class ResidentCreate(BaseModel):
    name: str
    gender: str = "未知"
    age: int | None = None
    address: str | None = None
    medical_history: str | None = None
    guardian_name: str | None = None
    guardian_phone: str | None = None


class ResidentOut(ResidentCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class DeviceCreate(BaseModel):
    device_name: str
    device_serial: str
    vendor: str = "sim"  # ezviz / hikvision / onvif / rtsp / sim
    scene: str = "客厅"
    model: str | None = None
    access_url: str | None = None
    channel_no: int = 1
    enabled: bool = True
    resident_id: int | None = None


class DeviceOut(BaseModel):
    id: int
    device_name: str
    device_serial: str
    vendor: str
    scene: str
    status: str
    model: str | None = None
    access_url: str | None = None
    channel_no: int
    enabled: bool
    resident_id: int | None = None
    resident_name: str = ""
    created_at: datetime

    class Config:
        from_attributes = True


class AlertOut(BaseModel):
    id: int
    alert_no: str
    device_id: int
    resident_id: int | None = None
    level: str
    event_type: str
    title: str
    detail_json: str | None = None
    replay_clip_id: str | None = None
    confirmed: bool
    handled: bool
    status: str = "pending"
    resident_name: str = ""
    guardian_phone: str = ""
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None
    confirm_note: str | None = None
    handled_by: str | None = None
    handled_at: datetime | None = None
    handle_note: str | None = None
    closed_at: datetime | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class AlertActionIn(BaseModel):
    operator: str = ""  # 处理人，空则后端记为“值班员”
    note: str | None = None  # confirm 写入 confirm_note，handle 写入 handle_note


class AlertVerifyIn(BaseModel):
    operator: str = "值班员"
    decision: bool
    note: str | None = None
    target: str | None = None  # fall / risk；缺省按 alert.event_type 推断


class AlertAIVerifyOut(BaseModel):
    alert_id: int
    target: str
    decision: bool
    confidence: float
    weight: float
    summary: str = ""
    suggestion: str = ""
    fused_score: float
    fused_decision: bool


class RiskFactor(BaseModel):
    key: str
    label: str
    value: float = 0.0
    unit: str = ""
    normal_range: str = ""


class RiskTrendPoint(BaseModel):
    date: str  # YYYY-MM-DD
    avg_score: float = 0.0
    max_score: float = 0.0
    count: int = 0


class ResidentRiskProfileOut(BaseModel):
    resident_id: int
    resident_name: str = ""
    latest_score: float = 0.0
    latest_level: str = "green"
    factors: list[RiskFactor] = Field(default_factory=list)
    trend: list[RiskTrendPoint] = Field(default_factory=list)
    updated_at: datetime | None = None


class AIInferResponse(BaseModel):
    person_count: int = 0
    fall_detected: bool = False
    fall_prob: float = 0.0
    nearfall_prob: float = 0.0
    gait_unsteadiness: float = 0.0
    fall_type: str = ""
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    risk_score: float = 0.0
    level: str = "green"  # green/yellow/orange/red
    keypoints: list[list[float]] = Field(default_factory=list)
    bbox: list[float] = Field(default_factory=list)
    others_bbox: list[list[float]] = Field(default_factory=list)
    track_id: int | None = None
    tracks: list[dict] = Field(default_factory=list)
    frame_ms: int = 0
    mock: bool = False


class RiskScoreOut(BaseModel):
    id: int
    resident_id: int | None = None
    device_id: int | None = None
    score: float
    level: str
    factors_json: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class DashboardStats(BaseModel):
    total_devices: int = 0
    online_devices: int = 0
    total_residents: int = 0
    today_alerts: int = 0
    today_falls: int = 0
    avg_risk_score: float = 0.0
    latest_alerts: list[AlertOut] = Field(default_factory=list)


class LiveFrameOut(BaseModel):
    device_id: int
    image_base64: str
    captured_at: datetime


class MultimodalConfigUpdate(BaseModel):
    provider: str = "qwen"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    enabled: bool = False
    interval_seconds: int = Field(default=60, ge=10, le=86400)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=64, le=8192)
    prompt_override: str | None = None


class MultimodalConfigOut(BaseModel):
    provider: str
    model: str
    base_url: str
    api_key_masked: str
    enabled: bool
    interval_seconds: int
    temperature: float
    max_tokens: int
    prompt_override: str | None = None
    updated_at: datetime | None = None


class ProviderPreset(BaseModel):
    name: str
    label: str
    base_url: str
    models: list[str]
    default_model: str


class SnapshotOut(BaseModel):
    id: int
    device_id: int
    resident_id: int | None = None
    provider: str
    model: str
    event_type: str
    severity: str
    confidence: float
    has_issue: bool
    level: str
    summary: str
    detail_json: str | None = None
    latency_ms: int
    created_at: datetime

    class Config:
        from_attributes = True


class WebhookConfigCreate(BaseModel):
    name: str = ""
    platform: str = "custom"
    webhook_url: str
    secret: str | None = None
    enabled: bool = True
    trigger_levels: str = "red,orange"


class WebhookConfigOut(BaseModel):
    id: int
    name: str
    platform: str
    webhook_url: str = ""
    webhook_url_masked: str = ""
    secret_present: bool = False
    enabled: bool
    trigger_levels: str
    created_at: datetime

    class Config:
        from_attributes = True


class AnalyzeResponse(BaseModel):
    code: int = 0
    device_id: int
    snapshot: SnapshotOut | None = None
    result: dict = Field(default_factory=dict)