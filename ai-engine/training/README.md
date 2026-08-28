# 训练与评测管线

本目录实现“骨架提取 -> 时序分类 -> 评测 -> ONNX 导出”的可复现基线。

## 数据与受试者划分

- 输入：`data/manifest.csv`（总清单）与 `data/meta/source_mapping.csv`（来源映射）。
- 按受试者划分时优先使用 `subject_id`（非 `unknown` 时）；否则改用 `source_file` 派生
  稳定键，避免训练/验证泄漏：
  - URFD 的 `fall-01-cam0` 与 `fall-01-cam1` 归入同一 `urfd:fall-01`；
  - Le2i 的每个 `source_file` 作为独立 `le2i:<file>`。
- `risk_behavior` 在二分类中归入 `non_fall`；`--classes multiclass` 时独立为第 3 类，不再并入 normal。
- 划分按受试者键分层抽样，默认 80/20。

## 标签

- `--classes binary`（默认，现网 ONNX）：`fall=1`，`nearfall/normal/risk_behavior=0`。
- `--classes ternary`：`fall=0`、`nearfall=1`、`normal=2`（`risk_behavior` 仍并入 normal）。
- `--classes multiclass`：`fall=0`、`nearfall=1`、`risk_behavior=2`、`normal=3`。在线 2 类 ONNX 暂不重训；近跌概率由启发式 `HeuristicPoseRiskHead` 输出，训练入口为 `--model prefall`。

## 用法

在 `ai-engine/` 目录下执行：

```powershell
# 安装训练依赖
pip install -r requirements-training.txt

# 抽骨架 + 训练 TCN + 验证 + 导出 ONNX
python -m training.train --model tcn --epochs 60 --device auto

# ST-GCN
python -m training.train --model stgcn --classes ternary

# 冒烟测试（只处理 24 个片段，验证链路）
python -m training.train --model tcn --epochs 2 --max-clips 24

# 复用已缓存骨架，跳过姿态提取
python -m training.train --model tcn --skip-skeleton
```

## 输出

- 骨架缓存：`data/annotations/skeleton/<video_stem>.npz`（形状 `(T, 17, 3)`）。
- checkpoint：`ai-engine/runs/<model>_<classes>_w<window>.pt`。
- 指标：同名 `.metrics.json`；混淆矩阵：同名 `.confusion.png`。
- 模型：同名 `.onnx`（输入 `skeleton` 形状 `(N, 3, T, 17)`，输出 `logits`）。

## 在线接入（15 FPS 流式）

- AI 服务侧 `StreamManager` 按设备/流持续解码并以目标 15 FPS 推理（`/v1/streams/{id}/start`），后端每 2 秒仅读取最新聚合结果（`/v1/streams/{id}/latest`），避免高频 JPEG/HTTP 与数据库压力。
- 最佳模型置于 `ai-engine/models/tcn_fall.onnx`（`.onnx` 被 Git 忽略），配套 `tcn_fall.metadata.json` 记录版本、窗口、阈值、校验和与事件指标。
- 时序窗口 32 帧缓冲按 `stream_id` 隔离；窗口未满时不触发跌倒（模型缺失时才退回几何启发式）。
- 可配置：`TEMPORAL_MODEL_PATH`、`TEMPORAL_WINDOW`（默认 32）、`FALL_THRESHOLD`（默认 0.95）。
- `/health` 返回姿态/时序模型加载状态、ONNX provider、活跃流与模型元数据。

## 阈值扫描与事件级评估

```powershell
python ai-engine/training/threshold_scan.py --checkpoint ai-engine/runs/tcn_binary_w32.pt --device cuda
```

输出 `ai-engine/runs/threshold_sweep.json`（阈值扫描）与 `event_metrics.json`（事件召回、每小时误报、检测延迟、场景细分）。

## 单独评测 / 重新导出

```powershell
python -m training.evaluate --checkpoint ai-engine/runs/tcn_binary_w32.pt
python -m training.export_onnx --checkpoint ai-engine/runs/tcn_binary_w32.pt --out ai-engine/models/tcn_fall.onnx
```

## 局限

- 当前 manifest 有 550 条记录，均可解码；训练/评测预检会跳过缺失、LFS 指针或损坏片段。
- 已覆盖 livingroom（472）、bedroom（66）、bathroom（2）、outdoor（10）；数据仍以模拟/公开受试者为主，无真实老人居家跌倒。
- 分辨率、拍摄视角和来源差异较大；正式比赛指标应按场景、受试者和来源分别报告。

## 测试

```powershell
python ai-engine/tests/test_training_invariants.py
python ai-engine/tests/test_online_inference.py
python tools/replay_e2e.py
```
