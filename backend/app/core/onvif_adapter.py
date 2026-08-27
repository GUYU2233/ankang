from __future__ import annotations

from typing import Any

from app.core.device_adapter import DeviceAdapter, DeviceAdapterFactory, StreamAddress


@DeviceAdapterFactory.register("onvif")
@DeviceAdapterFactory.register("rtsp")
class OnvifRtspAdapter(DeviceAdapter):
    """通用 ONVIF/RTSP 适配器。无厂商 SDK 时作为兼容兜底通道。"""

    vendor = "onvif"

    def list_devices(self) -> list[dict[str, Any]]:
        return []

    def get_device_status(self, raw_device: dict[str, Any]) -> str:
        return "online" if raw_device.get("access_url") else "offline"

    def get_live_address(self, raw_device: dict[str, Any], protocol: str = "rtsp") -> StreamAddress:
        return StreamAddress(url=raw_device.get("access_url") or "", protocol="rtsp")