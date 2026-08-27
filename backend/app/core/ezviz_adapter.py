from __future__ import annotations

import time
from typing import Any

import httpx
from loguru import logger

from app.core.device_adapter import DeviceAdapter, DeviceAdapterFactory, StreamAddress


@DeviceAdapterFactory.register("ezviz")
class EzvizAdapter(DeviceAdapter):
    """萤石开放平台适配器。

    通过萤石云 OpenAPI（open.ys7.com）完成 accessToken、设备管理、直播地址、
    云台控制与抓拍能力。真机联调时在 backend/.env 配置 appKey/appSecret 即可。
    """

    vendor = "ezviz"

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self._access_token: str | None = None
        self._token_expire_ts: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.settings.ezviz_app_key and self.settings.ezviz_app_secret)

    def _request(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        resp = httpx.post(f"{self.settings.ezviz_api_base}{path}", data=data, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        if str(payload.get("code")) not in ("0", "200"):
            raise RuntimeError(f"萤石开放平台接口异常: {path} code={payload.get('code')} msg={payload.get('msg')}")
        return payload.get("data") or {}

    def ensure_token(self) -> str:
        if self._access_token and time.time() < self._token_expire_ts - 60:
            return self._access_token
        if not self.configured:
            raise RuntimeError("萤石开放平台 appKey/appSecret 未配置")
        data = self._request("/api/lapp/token/get", {
            "appKey": self.settings.ezviz_app_key,
            "appSecret": self.settings.ezviz_app_secret,
        })
        self._access_token = data.get("accessToken", "")
        self._token_expire_ts = time.time() + float(data.get("expireTime", 600))
        logger.info("萤石开放平台 accessToken 已刷新")
        return self._access_token

    def list_devices(self) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        token = self.ensure_token()
        devices: list[dict[str, Any]] = []
        page_start, page_size = 0, 50
        while True:
            data = self._request("/api/lapp/device/list", {
                "accessToken": token,
                "pageStart": page_start,
                "pageSize": page_size,
            })
            rows = data if isinstance(data, list) else []
            devices.extend(rows)
            if len(rows) < page_size:
                break
            page_start += page_size
        return [
            {
                "device_name": d.get("deviceName") or d.get("deviceSerial"),
                "device_serial": d.get("deviceSerial"),
                "vendor": "ezviz",
                "model": d.get("deviceModel") or d.get("model"),
                "status": d.get("status") == 1 and "online" or "offline",
                "channel_no": d.get("channelNo", 1),
                "access_url": "",
                "scene": "客厅",
                "extra": d,
            }
            for d in devices
        ]

    def get_device_status(self, raw_device: dict[str, Any]) -> str:
        if not self.configured:
            return "offline"
        try:
            token = self.ensure_token()
            data = self._request("/api/lapp/device/status/get", {
                "accessToken": token,
                "deviceSerial": raw_device.get("device_serial"),
            })
            return "online" if str(data.get("status")) == "1" else "offline"
        except Exception as exc:
            logger.warning(f"萤石设备状态查询失败 {exc}")
            return "offline"

    def get_live_address(self, raw_device: dict[str, Any], protocol: str = "rtsp") -> StreamAddress:
        token = self.ensure_token()
        data = self._request("/api/lapp/live/address/get", {
            "accessToken": token,
            "deviceSerial": raw_device.get("device_serial"),
            "channelNo": raw_device.get("channel_no", 1),
            "protocol": 1 if protocol in ("rtmp", "hls") else 2,  # 1=rtmp/hls, 2=ezopen
        })
        url = data.get("url") or ""
        proto = "ezopen" if str(data.get("type")) == "2" else "hls"
        return StreamAddress(url=url, protocol=proto)

    def get_snapshot(self, raw_device: dict[str, Any]) -> bytes | None:
        token = self.ensure_token()
        data = self._request("/api/lapp/device/capture", {
            "accessToken": token,
            "deviceSerial": raw_device.get("device_serial"),
            "channelNo": raw_device.get("channel_no", 1),
            "quality": 1,
        })
        pic_url = data.get("picUrl") or ""
        if pic_url:
            r = httpx.get(pic_url, timeout=10)
            return r.content
        return None