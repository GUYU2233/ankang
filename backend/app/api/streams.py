from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
import httpx

from app.core.stream_service import stream_service
from app.db import SessionLocal
from app.models.entities import Device
from app.services.ai_client import ai_client

router = APIRouter(prefix="/streams", tags=["视频流"])

_VIDEO_EXTS = (".mp4", ".avi", ".mkv")


def _is_stream_source(device: Device) -> bool:
    url = (device.access_url or "").strip()
    if not url:
        return False
    return url.lower().startswith("rtsp://") or url.lower().endswith(_VIDEO_EXTS)


def _load_device(device_id: int) -> Device | None:
    """短会话加载设备后立即释放 DB 连接，避免 AI 引擎网络调用期间占用连接。"""
    with SessionLocal() as db:
        device = db.get(Device, device_id)
        if device is not None:
            # 预读常用列，避免会话关闭后触发惰性加载
            _ = (device.id, device.access_url, device.vendor, device.scene, device.device_name)
        return device


@router.get("/local-videos")
def list_local_videos():
    """列出 data/videos 下可循环播放的本地视频，供模拟设备选用。"""
    return stream_service.list_local_videos()


@router.get("/{device_id}/mjpeg")
async def get_mjpeg(device_id: int):
    device = _load_device(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    if not _is_stream_source(device):
        raise HTTPException(status_code=409, detail="该设备暂不支持连续流，请使用 frame.jpg")
    stream_id = f"device-{device.id}"
    source = stream_service.resolve_video_path((device.access_url or "").strip())
    ai_client.ensure_stream(stream_id, source, target_fps=15.0)

    async def proxy():
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
            async with client.stream("GET", ai_client.mjpeg_url(stream_id)) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(proxy(), media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{device_id}/frame.jpg")
def get_frame_jpeg(device_id: int):
    device = _load_device(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    # 视频/RTSP 源：优先取 AI 引擎带骨架标注的帧（与检测结果严格对齐）
    if _is_stream_source(device):
        buf = ai_client.get_stream_frame(f"device-{device.id}")
        if buf:
            return Response(content=buf, media_type="image/jpeg")
        # 引擎未就绪时回落本地原始帧
    try:
        packet = stream_service.get_frame(device)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"取流失败: {exc}")
    frame = packet.frame
    infer = stream_service.get_latest_result(device.id)
    if infer is not None:
        frame = stream_service.draw_overlay(frame, infer)
    return Response(content=stream_service.encode_jpeg(frame), media_type="image/jpeg")


@router.get("/{device_id}/meta")
def get_frame_meta(device_id: int):
    device = _load_device(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="设备不存在")
    infer = stream_service.get_latest_result(device.id)
    if _is_stream_source(device):
        # 视频/RTSP 源：不重复本地解码，检测字段来自检测循环写入的最新结果
        meta = {
            "source": "rtsp" if (device.access_url or "").startswith("rtsp://") else "local_video",
            "state": "playing",
        }
        captured_at = None
    else:
        packet = stream_service.get_frame(device)
        meta = dict(packet.meta)
        captured_at = packet.captured_at
    if infer is not None:
        meta.update({
            "fall_detected": infer.fall_detected,
            "fall_prob": infer.fall_prob,
            "nearfall_prob": infer.nearfall_prob,
            "gait_unsteadiness": infer.gait_unsteadiness,
            "risk_score": infer.risk_score,
            "level": infer.level,
            "person_count": infer.person_count,
            "fall_type": infer.fall_type,
            "mock": infer.mock,
        })
    return {"device_id": device_id, "meta": meta, "captured_at": captured_at}
