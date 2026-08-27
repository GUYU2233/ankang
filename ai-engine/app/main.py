from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile

from app.runtime import runtime

app = FastAPI(title="智护安康 AI 推理引擎", version="0.1.0")


@app.get("/health")
def health():
    return {"code": 0, "status": "up", "engine": "ai-engine"}


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