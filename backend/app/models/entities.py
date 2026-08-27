from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class AlertLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


class Resident(Base):
    __tablename__ = "residents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    gender: Mapped[str] = mapped_column(String(8), default="未知")
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    address: Mapped[str | None] = mapped_column(String(255), nullable=True)
    medical_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    guardian_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    guardian_phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    devices: Mapped[list["Device"]] = relationship(back_populates="resident")


class Device(Base):
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_name: Mapped[str] = mapped_column(String(128))
    device_serial: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    vendor: Mapped[str] = mapped_column(String(32), default="sim")  # ezviz / hikvision / onvif / sim
    scene: Mapped[str] = mapped_column(String(32), default="客厅")  # 客厅/卧室/卫生间
    status: Mapped[str] = mapped_column(String(16), default="online")  # online/offline
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    access_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # RTSP 地址或本地视频路径
    channel_no: Mapped[int] = mapped_column(Integer, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    extra_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    resident_id: Mapped[int | None] = mapped_column(ForeignKey("residents.id"), nullable=True)
    resident: Mapped[Resident | None] = relationship(back_populates="devices")


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id"))
    resident_id: Mapped[int | None] = mapped_column(ForeignKey("residents.id"), nullable=True)
    level: Mapped[AlertLevel] = mapped_column(SAEnum(AlertLevel), default=AlertLevel.GREEN)
    event_type: Mapped[str] = mapped_column(String(32), default="fall_risk")  # fall_event / fall_risk / behavior_risk
    title: Mapped[str] = mapped_column(String(255), default="")
    detail_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    handled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class FallEvent(Base):
    __tablename__ = "fall_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    device_id: Mapped[int] = mapped_column(ForeignKey("devices.id"))
    resident_id: Mapped[int | None] = mapped_column(ForeignKey("residents.id"), nullable=True)
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alert_events.id"), nullable=True)
    start_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    end_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    fall_prob: Mapped[float] = mapped_column(Float, default=0.0)
    screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    resident_id: Mapped[int | None] = mapped_column(ForeignKey("residents.id"), nullable=True)
    device_id: Mapped[int | None] = mapped_column(ForeignKey("devices.id"), nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    level: Mapped[AlertLevel] = mapped_column(SAEnum(AlertLevel), default=AlertLevel.GREEN)
    factors_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class NotificationLog(Base):
    __tablename__ = "notification_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[int] = mapped_column(ForeignKey("alert_events.id"))
    channel: Mapped[str] = mapped_column(String(32), default="websocket")  # websocket / sms / voice / app
    target: Mapped[str] = mapped_column(String(128), default="家属端")
    content: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="sent")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)