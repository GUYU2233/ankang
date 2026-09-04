from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.runtime import runtime
from app.streaming import StreamManager

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.streams = StreamManager(runtime, default_fps=15.0)
    yield
    app.state.streams.close()


app = FastAPI(title="智护安康 AI 推理引擎", version="0.1.0", lifespan=lifespan)


def _load_metadata() -> dict:
    meta_path = MODEL_DIR / "tcn_fall.metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@app.get("/health")
def health():
    return {
        "code": 0,
        "status": "up",
        "engine": "ai-engine",
        "model": runtime.model_info(),
        "model_metadata": _load_metadata(),
        "streams": app.state.streams.health(),
    }


@app.post("/v1/infer")
async def infer(file: UploadFile = File(...), stream_id: str = Query(default="default", max_length=128)):
    data = await file.read()
    buf = np.frombuffer(data, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return {"code": 400, "message": "无法解析图片"}
    result = runtime.execute(frame_bgr, stream_id=stream_id)
    result["code"] = 0
    return result


class StreamStartRequest(BaseModel):
    source: str
    target_fps: float = 15.0
    loop_file: bool = True


@app.post("/v1/streams/{stream_id}/start")
def stream_start(stream_id: str, body: StreamStartRequest):
    status = app.state.streams.start(stream_id, body.source, body.target_fps, body.loop_file)
    return {"code": 0, "stream": status}


@app.post("/v1/streams/{stream_id}/stop")
def stream_stop(stream_id: str):
    app.state.streams.stop(stream_id)
    return {"code": 0, "message": "stopped", "stream_id": stream_id}


@app.get("/v1/streams/{stream_id}")
def stream_status(stream_id: str):
    return {"code": 0, "stream": app.state.streams.status(stream_id)}


@app.get("/v1/streams/{stream_id}/latest")
def stream_latest(stream_id: str):
    result = app.state.streams.latest(stream_id)
    return {"code": 0 if result else 404, "result": result}


@app.get("/v1/streams/{stream_id}/frame.jpg")
def stream_frame(stream_id: str):
    buf = app.state.streams.frame_jpg(stream_id)
    if buf is None:
        return Response(status_code=404)
    return Response(content=buf, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/v1/streams/{stream_id}/mjpeg")
def stream_mjpeg(stream_id: str):
    if app.state.streams.status(stream_id).get("error") == "stream not found":
        return Response(status_code=404)
    return StreamingResponse(
        app.state.streams.mjpeg(stream_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache", "X-Accel-Buffering": "no"},
    )


class RecordingRequest(BaseModel):
    alert_id: str
    post_seconds: float = 5.0


@app.post("/v1/streams/{stream_id}/recordings")
def recording_start(stream_id: str, body: RecordingRequest):
    try:
        return {"code": 0, "recording": app.state.streams.trigger_recording(stream_id, body.alert_id, body.post_seconds)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/recordings/{clip_id}")
def recording_status(clip_id: str):
    try:
        return {"code": 0, "recording": app.state.streams.recording_status(clip_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/recordings/{clip_id}/video.mp4")
def recording_video(clip_id: str):
    path = app.state.streams.recording_path(clip_id)
    if path is None:
        raise HTTPException(status_code=404, detail="回放尚未生成")
    return FileResponse(str(path), media_type="video/mp4", filename=f"{clip_id}.mp4")


@app.get("/v1/streams")
def stream_list():
    return {"code": 0, **app.state.streams.health()}
