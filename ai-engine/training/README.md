# 训练与评测管线

本目录实现“骨架提取 -> 时序分类 -> 评测 -> ONNX 导出”的可复现基线。

## 数据与受试者划分

- 输入：`data/manifest.csv`（总清单）与 `data/meta/source_mapping.csv`（来源映射）。
- `manifest.csv` 的 `subject_id` 全部为 `unknown`，因此按受试者划分时改用
  `source_file` 派生稳定键，避免训练/验证之间泄漏：
  - URFD 的 `fall-01-cam0` 与 `fall-01-cam1` 归入同一 `urfd:fall-01`；
  - Le2i 的每个 `source_file` 作为独立 `le2i:<file>`。
- 划分按受试者键分层抽样，默认 80/20。

## 标签

- `--classes binary`（默认）：`fall=1`，`nearfall/normal=0`。
- `--classes ternary`：`fall=0`、`nearfall=1`、`normal=2`。

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

## 在线接入

训练完成后，将最佳模型复制到 `ai-engine/models/tcn_fall.onnx`（模型文件默认被 Git 忽略）。AI 引擎按请求中的 `stream_id`（后端默认使用设备 ID）缓存 32 帧骨架；窗口未满时使用几何启发式，窗口满后使用 ONNX 时序概率。也可通过环境变量 `TEMPORAL_MODEL_PATH` 指定模型路径。

## 单独评测 / 重新导出

```powershell
python -m training.evaluate --checkpoint ai-engine/runs/tcn_binary_w32.pt
python -m training.export_onnx --checkpoint ai-engine/runs/tcn_binary_w32.pt --out ai-engine/models/tcn_fall.onnx
```

## 局限

- 当前 manifest 有 550 条记录，其中 540 条视频可解码；仍有 10 条缺失或损坏片段会被预检跳过。
- 已覆盖 livingroom、bedroom、bathroom、outdoor；数据仍以模拟/公开受试者为主，无真实老人居家跌倒。
- 分辨率、拍摄视角和来源差异较大；正式比赛指标应按场景、受试者和来源分别报告。
