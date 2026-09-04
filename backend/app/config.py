from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "智护安康智能预警系统"
    api_prefix: str = "/api/v1"

    # 开发默认关闭；生产设置 AUTH_MODE=required 并注入高熵 API_KEY
    auth_mode: str = "disabled"
    api_key: str = ""

    # 数据存储：本地开发默认 SQLite；生产使用 MySQL 见 deploy/init.sql
    database_url: str = "sqlite:///./zhihu_ankang.db"

    # AI 推理服务
    ai_engine_url: str = "http://127.0.0.1:8100"

    # 萤石开放平台账号（需真机时填写，缺省视为未配置，自动使用模拟设备）
    ezviz_app_key: str = ""
    ezviz_app_secret: str = ""
    ezviz_api_base: str = "https://open.ys7.com"

    # 海康设备接入：无真机时留空；框架预留 HCNetSDK 桥接 + ISAPI/RTSP
    hikvision_net_sdk_bridge: str = "http://127.0.0.1:8200"

    # 通用 RTSP/本地视频模拟
    local_video_dir: str = "./data/videos"

    # 巡检/推理节奏
    detect_interval_seconds: float = 2.0
    risk_window_seconds: int = 300
    alert_confirm_frames: int = 3

    # 多模态视觉巡检（定时截图 + 大模型识别）
    multimodal_enabled: bool = False
    multimodal_provider: str = "qwen"  # qwen/glm/gpt/gemini/custom
    multimodal_model: str = ""  # 为空则用提供商默认模型
    multimodal_base_url: str = ""  # 为空则用提供商默认接入点
    multimodal_api_key: str = ""
    multimodal_interval_seconds: int = 60
    multimodal_temperature: float = 0.2
    multimodal_max_tokens: int = 800
    multimodal_timeout_seconds: float = 30.0

    # 前端跨域
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()