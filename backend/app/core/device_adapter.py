from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamAddress:
    url: str
    protocol: str = "rtsp"  # rtsp / hls / flv / ezopen / local
    expired_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class DeviceAdapter(ABC):
    vendor: str = "base"

    def __init__(self, settings) -> None:
        self.settings = settings

    @abstractmethod
    def list_devices(self) -> list[dict[str, Any]]:
        """从平台/网络发现设备，返回统一设备字典列表"""

    @abstractmethod
    def get_device_status(self, raw_device: dict[str, Any]) -> str:
        """返回 online/offline"""

    @abstractmethod
    def get_live_address(self, raw_device: dict[str, Any], protocol: str = "rtsp") -> StreamAddress:
        """获取可供取流的地址（萤石返回 ezopen/hls，海康/ONVIF 返回 rtsp）"""

    def get_snapshot(self, raw_device: dict[str, Any]) -> bytes | None:
        return None

    def start_alarm_subscription(self) -> Any:
        return None

    def stop_alarm_subscription(self, handle: Any) -> None:
        pass


class DeviceAdapterFactory:
    _adapters: dict[str, type[DeviceAdapter]] = {}

    @classmethod
    def register(cls, vendor: str, adapter_cls: type[DeviceAdapter]) -> None:
        cls._adapters[vendor] = adapter_cls

    @classmethod
    def create(cls, vendor: str, settings) -> DeviceAdapter:
        adapter_cls = cls._adapters.get(vendor)
        if adapter_cls is None:
            raise ValueError(f"未注册的厂商适配器: {vendor}")
        return adapter_cls(settings)