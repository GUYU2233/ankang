# 智护安康：面向智慧养老的老年人跌倒风险识别与智能预警系统

> 依据“XH-202617 基于多模态 AI 监测的老年人跌倒风险、心理健康、诈骗识别及预警研究”赛题建设。
> 本仓库实现**跌倒风险方向**的系统框架：以“跌倒前预判—跌倒时预警—跌倒后快速响应”为主线，基于**萤石开放平台**能力开发，并预留**海康威视设备**兼容接入层（无真机时支持 RTSP/本地视频模拟）。

## 项目结构

```
├── backend/     # FastAPI 业务服务：设备接入、流媒体管理、风险与告警、WebSocket推送、看板统计
├── ai-engine/   # AI 推理服务：骨架估计、跌倒检测、跌倒风险因子评分（占位与模拟推理，后续接入数据集训练）
├── frontend/    # Vue3 + Vite + Element Plus 家属/护工端与大屏看板
├── deploy/      # Docker Compose 一键部署（backend、ai-engine、frontend、mysql、redis、emqx、zlm）
├── tools/       # RTSP/视频模拟摄像头、数据库初始化脚本
└── docs/        # 系统设计、部署运行说明、技术开发文档、测试报告模板
```

## 快速开始（本地开发，无 Docker / 无相机）

```powershell
# 1. 后端
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py            # http://127.0.0.1:8000/docs

# 2. AI 引擎（另开一个终端）
cd ai-engine
pip install -r requirements.txt
uvicorn app.main:app --port 8100   # http://127.0.0.1:8100/docs

# 3. 前端（另开一个终端）
cd frontend
npm install
npm run dev             # http://127.0.0.1:5173
```

默认无摄像头时，后端会自动注册 2 路**模拟视频流**（合成画面 + 随机跌倒事件），可完整体验
“取流 → AI 推理 → 四级分级预警 → WebSocket 推送 → 看板展示”闭环。

## Docker 部署（推荐，含 RTSP 模拟流）

详见 `docs/03-部署与运行说明.md` 与 `deploy/docker-compose.yml`。

## 设备接入能力

| 设备来源 | 接入方式 | 状态 |
|---|---|---|
| 萤石云设备 | 萤石开放平台 OpenAPI（accessToken / 设备管理 / 直播 / 告警订阅） | 框架接入完成，需真机/账号联调 |
| 海康威视 IPC/NVR | HCNetSDK 桥接 + ISAPI + RTSP/ONVIF 统一适配 | 接口预留，无真机，支持 RTSP 手工接入 |
| 第三方 ONVIF/RTSP | RTSP 接入 + OpenCV 取流 | 支持 |
| 本地视频/合成流 | 模拟摄像机 | 支持（开发演示） |