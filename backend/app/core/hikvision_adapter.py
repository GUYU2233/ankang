from __future__ import annotations

from typing import Any

from loguru import logger

from app.core.device_adapter import DeviceAdapter, DeviceAdapterFactory, StreamAddress


@DeviceAdapterFactory.register("hikvision")
class HikvisionAdapter(DeviceAdapter):
    """海康威视设备适配器（框架预留）。

    三种接入路径：
    1. HCNetSDK（设备网络 SDK）：经 hikvision_net_sdk_bridge 桥接服务调用 NET_DVR_* 能力；
    2. ISAPI：GET http://ip/ISAPI/... 设备管理、事件订阅；
    3. RTSP 标准取流（通用兼容通道，本框架已实现）。

    无真机时 list_devices 返回空，设备可通过 DeviceCreate 手工登记 RTSP 地址。
    """

    vendor = "hikvision"

    def list_devices(self) -> list[dict[str, Any]]:
        return []

    def get_device_status(self, raw_device: dict[str, Any]) -> str:
        url = raw_device.get("access_url") or ""
        if url:
            return "online" if url.startswith("rtsp://") else "offline"
        return self._isapi_status(raw_device)

    def _isapi_status(self, raw_device: dict[str, Any]) -> str:
        return "offline"

    def get_live_address(self, raw_device: dict[str, Any], protocol: str = "rtsp") -> StreamAddress:
        url = raw_device.get("access_url") or raw_device.get("extra", {}).get("rtsp_url", "")
        if not url:
            host = raw_device.get("extra", {}).get("ip", "")
            port = raw_device.get("extra", {}).get("port", 554)
            user = raw_device.get("extra", {}).get("username", "admin")
            pwd = raw_device.get("extra", {}).get("password", "")
            channel = raw_device.get("channel_no", 1)
            url = self.build_rtsp_url(host, port, user, pwd, channel)
        return StreamAddress(url=url, protocol="rtsp")

    @staticmethod
    def build_rtsp_url(host: str, port: int = 554, user: str = "admin", pwd: str = "", channel: int = 1, stream: str = "main") -> str:
        """生成海康设备标准 RTSP 取流地址。"""
        credentials = f"{user}:{pwd}@" if pwd else ""
        return f"rtsp://{credentials}{host}:{port}/Streaming/Channels/{channel}01?transportmode=unicast"

    def get_snapshot(self, raw_device: dict[str, Any]) -> bytes | None:
        logger.info("海康抓拍请经 HCNetSDK 桥接或 ISAPI 实现，当前返回 None")
        return None