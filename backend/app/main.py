from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api import alerts, dashboard, devices, residents, streams
from app.config import get_settings
from app.core.device_manager import get_device_manager
from app.db import SessionLocal, init_db
from app.services.detection_loop import DetectionLoop
from app.ws.alert_ws import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(f"{settings.app_name} 启动中")
    init_db()
    with SessionLocal() as db:
        mgr = get_device_manager()
        mgr.ensure_demo_devices(db)
        mgr.sync_cloud_devices(db)
    loop = DetectionLoop()
    task = None
    import asyncio
    task = asyncio.create_task(loop.run_forever())
    yield
    loop.stop()
    if task:
        task.cancel()


app = FastAPI(title=get_settings().app_name, version="0.1.0", lifespan=lifespan)

origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)
app.include_router(devices.router, prefix=get_settings().api_prefix)
app.include_router(residents.router, prefix=get_settings().api_prefix)
app.include_router(alerts.router, prefix=get_settings().api_prefix)
app.include_router(streams.router, prefix=get_settings().api_prefix)
app.include_router(dashboard.router, prefix=get_settings().api_prefix)


@app.get("/api/v1/health")
def health():
    return {"code": 0, "status": "up", "app": get_settings().app_name}