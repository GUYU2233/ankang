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
    confirmed: bool
    handled: bool
    created_at: datetime

    class Config:
        from_attributes = True


class RiskFactor(BaseModel):
    key: str
    label: str
    value: float = 0.0
    unit: str = ""
    normal_range: str = ""


class AIInferResponse(BaseModel):
    person_count: int = 0
    fall_detected: bool = False
    fall_prob: float = 0.0
    fall_type: str = ""
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    risk_score: float = 0.0
    level: str = "green"  # green/yellow/orange/red
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