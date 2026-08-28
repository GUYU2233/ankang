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

## 多模态视觉巡检（定时截图 + 大模型识别）

在原有“骨架时序跌倒检测”之外，新增一条独立的大模型视觉巡检链路：按固定周期对每路监控截图，
调用多模态大模型（通义千问 Qwen-VL / 智谱 GLM-4V / OpenAI GPT-4o / Google Gemini，OpenAI 兼容协议）
识别风险，结构化结果落库并接入分级告警与 WebSocket 推送。

可识别事件：`fall`（跌倒/倒地）、`posture_abnormal`（姿态/行为异常）、`floor_clutter`（地面杂物/障碍）、
`other_risk`（火灾烟雾、漏水等）、`normal`（正常）。

### 配置

在 `backend/.env` 或环境变量中设置（也可由前端通过 `PUT /api/v1/multimodal/config` 在线修改）：

```
MULTIMODAL_ENABLED=false        # 是否启用定时巡检
MULTIMODAL_PROVIDER=qwen       # qwen / glm / gpt / gemini / custom
MULTIMODAL_MODEL=              # 留空用提供商默认模型
MULTIMODAL_API_KEY=            # 或使用提供商原生环境变量（DASHSCOPE_API_KEY 等）
MULTIMODAL_INTERVAL_SECONDS=60
```

### 后端接口（供前端对接）

| 方法与路径 | 说明 |
|---|---|
| `GET /api/v1/multimodal/config` | 读取巡检配置（密钥以掩码返回） |
| `PUT /api/v1/multimodal/config` | 更新巡检配置（provider/model/api_key/间隔/启停/自定义提示词） |
| `GET /api/v1/multimodal/providers` | 支持的提供商与默认模型列表 |
| `GET /api/v1/multimodal/status` | 巡检循环状态（运行/启停/最近执行/统计） |
| `POST /api/v1/multimodal/analyze/{device_id}` | 手动触发单设备识别 |
| `GET /api/v1/multimodal/results` | 巡检历史（按设备/事件/等级过滤，分页） |
| `GET /api/v1/multimodal/results/{id}/image` | 读取某次巡检截图 |

识别出风险后会生成一条 `alert_events` 告警并经 WebSocket 推送（`source=multimodal`），
跌倒事件同步写入 `fall_events`；截图存至 `runtime/multimodal/`。

> 说明：大模型输出仅用于安全提醒，不构成医疗诊断；紧急情况请直接拨打急救电话。