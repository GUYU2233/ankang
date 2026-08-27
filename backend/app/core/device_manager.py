from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.device_adapter import DeviceAdapterFactory
from app.models.entities import Device, Resident


class DeviceManager:
    """负责设备登记、自动发现、状态巡检与默认模拟设备初始化。"""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._adapters = {}
        for vendor in ("ezviz", "hikvision", "onvif", "rtsp"):
            try:
                self._adapters[vendor] = DeviceAdapterFactory.create(vendor, self.settings)
            except ValueError:
                pass

    def adapter(self, vendor: str):
        return self._adapters.get(vendor)

    def sync_cloud_devices(self, db: Session) -> int:
        """从萤石开放平台/海康桥接同步设备（无配置或无设备时返回 0）。"""
        added = 0
        for vendor in ("ezviz", "hikvision"):
            adapter = self._adapters.get(vendor)
            if adapter is None:
                continue
            try:
                for raw in adapter.list_devices():
                    exists = db.scalar(select(Device).where(Device.device_serial == raw["device_serial"]))
                    if exists:
                        continue
                    db.add(Device(
                        device_name=raw.get("device_name", raw["device_serial"]),
                        device_serial=raw["device_serial"],
                        vendor=raw.get("vendor", vendor),
                        scene=raw.get("scene", "客厅"),
                        status=raw.get("status", "online"),
                        model=raw.get("model"),
                        channel_no=raw.get("channel_no", 1),
                        access_url=raw.get("access_url") or None,
                        extra_json=str(raw.get("extra", {})),
                        enabled=True,
                    ))
                    added += 1
            except Exception as exc:
                logger.warning(f"{vendor} 设备同步失败: {exc}")
        if added:
            db.commit()
        return added

    def ensure_demo_devices(self, db: Session) -> None:
        """无任何设备或模拟启动时，注册 2 路合成流设备以保证闭环可演示。"""
        exists = db.scalar(select(Device.id).limit(1))
        if exists:
            return
        defaults = [
            ("客厅-模拟相机01", "SIM-LIVING-001", "客厅"),
            ("卫生间-模拟相机02", "SIM-BATH-002", "卫生间"),
        ]
        for name, serial, scene in defaults:
            db.add(Device(device_name=name, device_serial=serial, vendor="sim", scene=scene,
                          status="online", model="SimCam-1S", access_url="", channel_no=1, enabled=True))
        db.commit()
        logger.info("已写入默认模拟设备: 2 路")

    def update_status(self, db: Session, device: Device) -> str:
        adapter = self._adapters.get(device.vendor)
        if adapter is None:
            device.status = "online" if device.vendor == "sim" else "offline"
            return device.status
        try:
            device.status = adapter.get_device_status({
                "device_serial": device.device_serial,
                "access_url": device.access_url,
                "channel_no": device.channel_no,
            })
        except Exception:
            device.status = "online" if device.vendor == "sim" else "offline"
        return device.status

    def demo_defaults(self) -> list[dict[str, Any]]:
        return [
            {"device_name": "客厅-模拟相机01", "device_serial": "SIM-LIVING-001", "vendor": "sim", "scene": "客厅", "model": "SimCam-1S"},
            {"device_name": "卧室-模拟相机02", "device_serial": "SIM-BED-003", "vendor": "sim", "scene": "卧室", "model": "SimCam-1S"},
        ]


_device_manager: DeviceManager | None = None


def get_device_manager() -> DeviceManager:
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager