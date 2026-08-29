# YOLO26-pose 微调语料人工标注规范（CVAT）

> 面向 CVAT 人工标注员的完整操作规范。标注对象：**摔倒/躺卧主体**（多人场景只标主体）。
> 语料：`data/annotations/pose_ft` 的 Pilot 帧（约 362 帧：躺卧 242 / 过渡 80 / 站立 40，
> train 314 / val 48，来自 ~125 个视频组）。
> 产出：CVAT 导出的 **COCO Keypoints 1.0** JSON → 转换脚本 → `labels/{train,val}/*.txt` → 微调。

---

## 1. 目的与范围

- **目的**：为 YOLO26-pose 微调构建高质量人体关键点语料，重点提升**躺卧姿态**的判别能力
  （基模对站立姿态较强、对躺卧偏弱，这正是本次微调要解决的核心问题）。
- **帧来源**（P0 采样脚本 `scripts/sample_pose_ft_frames.py` 已按 `data/annotations/pose_ft/manifest.csv` 抽好）：
  - `fall` 视频：**躺卧帧**（真实躺卧窗采样，每视频 3 帧）+ **过渡帧**（`fall_start_ts` 附近 1 帧）；
  - `normal` 视频：**站立帧**（每视频 1 帧，防止微调遗忘站立姿态）。
- **标注对象**：画面中的**摔倒/躺卧主体**（即该视频的主人公）。站立帧则标画面主体。
- **本规范覆盖**：工具流程、17 点顺序与可见性语义、bbox 规则、躺卧重点点、多人/剔除/导出、
  常见错误、QC 抽查，以及目录结构与端到端命令速查。

---

## 2. 工具与流程

1. **建 CVAT 任务**
   - Labels 配置为 **Rectangle**（主体 bbox）+ **Points / Keypoints**（17 个关键点），
     建议用 **Skeleton 模板**一次搭好 17 点骨架（见第 3 节点序），保证点序固定不串位。
   - 项目/任务名建议含 `pose_ft` 与 split 标记（如 `pose_ft-pilot-val`）。
2. **导入图片**：从 `data/annotations/pose_ft/images/{train,val}` 按目录导入
   （`train` 一个任务、`val` 一个任务，或按需分批；**不要混 split**，防止换算出错）。
3. **（可选）导入预标注**：站立/过渡帧可先跑 `scripts/prelabel_standing.py` 生成
   `prelabels/prelabel.json`，在 CVAT 用「导入 COCO Keypoints 1.0 预标注」载入后逐帧修正；
   躺卧帧**刻意没有**预标注，必须人工从零标。
4. **标注**：逐帧按第 3–7 节规则标 bbox + 17 点；无法标注的帧按第 8 节写入 `skipped.txt`。
5. **导出**：`Actions → Export task dataset`，格式选 **COCO Keypoints 1.0**，
   下载的 JSON 放入 `data/annotations/pose_ft/cvat/`（多个任务就放多个 JSON，转换脚本支持目录合并）。
6. **转换**：运行 `scripts/coco_to_yolo_pose.py --input data/annotations/pose_ft/cvat`
   （详见第 11 节命令速查），生成 `labels/{train,val}/<stem>.txt`。
7. **微调**：`scripts/finetune_pose.py`（见第 11 节）。

---

## 3. 17 点 COCO 顺序全表（索引 0–16）

> **最重要的一条**：左右均指**人体自身的左/右**（面向画面时，画面左侧往往是人体右侧），
> 不是画面左/右。**左右镜像是最常见事故**，标完后可用骨架连线自查（左肩→左肘→左腕应同侧）。

| 索引 | 英文名 | 中文名 | 与对称点 |
|----:|--------|--------|----------|
| 0 | nose | 鼻尖 | 自己 |
| 1 | left_eye | 左眼 | ↔ 2 |
| 2 | right_eye | 右眼 | ↔ 1 |
| 3 | left_ear | 左耳 | ↔ 4 |
| 4 | right_ear | 右耳 | ↔ 3 |
| 5 | left_shoulder | 左肩 | ↔ 6 |
| 6 | right_shoulder | 右肩 | ↔ 5 |
| 7 | left_elbow | 左肘 | ↔ 8 |
| 8 | right_elbow | 右肘 | ↔ 7 |
| 9 | left_wrist | 左腕 | ↔ 10 |
| 10 | right_wrist | 右腕 | ↔ 9 |
| 11 | left_hip | 左髋 | ↔ 12 |
| 12 | right_hip | 右髋 | ↔ 11 |
| 13 | left_knee | 左膝 | ↔ 14 |
| 14 | right_knee | 右膝 | ↔ 13 |
| 15 | left_ankle | 左踝 | ↔ 16 |
| 16 | right_ankle | 右踝 | ↔ 15 |

- 该顺序与 `scripts/coco_to_yolo_pose.py`、`scripts/prelabel_standing.py` 中的名称数组
  **逐字一致**（`COCO_KEYPOINT_NAMES`），与 ultralytics COCO-pose 约定一致；
  CVAT Skeleton 模板请按本表 0–16 建立，切勿自定义顺序。
- 骨架连线（COCO 标准，索引 1 起）：踝-膝-髋、髋-髋、髋-肩、肩-肩、肩-肘-腕、
  耳-眼-鼻-眼-耳、耳-肩。

---

## 4. bbox 规则（Rectangle）

- **紧贴整个人体**，含四肢伸展：躺卧时伸出的手臂/腿都要包进去（含手指方向最远端）。
- 四周留 **2–3 px 余量**（防裁切，不给训练引入截断）。
- **多人时 bbox 只围主体**（摔倒/躺卧者）；路人即使贴近主体也不并入主体 bbox。
- **不因手持物/家具扩大**：手里的拐杖、靠着的沙发、床沿都不算人体，不要为容纳它们放大 bbox。
- bbox 允许出画（画面边缘的人被截断）：照实标可见部分即可，不必强行扩到画外。

---

## 5. 关键点规则（Points / Keypoints）

- **点位在关节中心**，不是肢体边缘/衣物轮廓：如腕点在手骨末端关节中心，不在袖口；
  踝点在外踝骨中心，不在袜口。
- 每点一个可见性标签 **v**（CVAT 中为 Visibility）：

| v | 语义 | 何时用 |
|---|------|--------|
| 2 | 清晰可见 | 关节明确可见、可精确定位（即使身体有正常遮挡如手在身前） |
| 1 | 被遮挡但可推断 | 关节被遮挡（被床沿、家具、被子、他人挡），但按人体结构能推断位置 |
| 0 | 出画面或完全不可定位 | 关节在画面外，或完全无法推断（如整条腿被完全盖住） |

- v=0 的点**坐标不要乱点**：标在画面外/不可推断处即可，转换脚本会将其写为 `0 0 0`。
- 头部的眼/耳在侧脸、仰躺时常常只可见一侧：另一侧按对称推断标 v=1，或 v=0。

---

## 6. 躺卧帧重点（微调核心信号）

- 躺卧帧中 **肩(5,6)、髋(11,12)、膝(13,14)、踝(15,16) 这 8 点尽量标全**。
- 被床沿/家具/被子遮挡时，也**按人体结构推断标 v=1**（躺卧躯干一般仍有明确轮廓可推）；
  只有完全无法推断才标 v=0。
- 这 8 点是模型判别「躺卧 vs 站立/蹲坐」的核心信号：**漏标/乱标会直接削弱微调效果**。
- 躺卧时躯干可能被床体切分：髋与肩分属画面不同区域，务必用骨架连线确认左右侧连贯。
- 过渡帧（半蹲/倾倒中）同理：膝、踝常在画面下方被截断或遮挡，能推则标 v=1。

---

## 7. 多人

- **只标跌倒/躺卧主体**：其他行人、背景人物一概不标（bbox 与关键点都不标）。
- 路人靠近主体时：**收紧主体 bbox** 使主体与路人分离，路人不标关键点。
- 若画面里难以判定谁是主体（如两个人都倒下），按 `manifest.csv` 该帧所属视频的事件主体判定；
  仍存疑的帧截图存档（见第 10 节 QC），宁缺勿错。

---

## 8. 剔除规则（写入 skipped.txt）

以下帧**不标注**，把文件名（一行一个，如 `livingroom_fall_243_20260827_t7.1.jpg`）追加写入
`data/annotations/pose_ft/skipped.txt`（UTF-8）：

1. **空帧**：画面中无人体（或主体完全不在画面内）。
2. **极糊**：运动模糊/低清到关节无法定位（轻微糊不影响 17 点定位的仍标）。
3. **主体 >90% 遮挡**：主体被大面积遮挡（如整床被子盖住）无法推断姿态。
4. **3D 渲染 / 卡通形象帧**：非真实人体（动画、渲染人物、玩偶拟人）不标。

> 注意：`skipped.txt` 是人工标注产物，**不要删除**；后续转换脚本之外的流程会用它做过滤记录。

---

## 9. 常见错误清单（标注自查）

1. **左右镜像**（最严重）：把画面右侧当人体右侧。→ 用骨架连线自查：左右肩-肘-腕应各成一条链。
2. **bbox 截断四肢**：躺卧时伸直的手臂/腿被 bbox 切掉。→ bbox 包住最远端，留 2–3px。
3. **关键点放在衣物轮廓而非关节中心**：腕点放袖口、踝点放袜口、膝点放裤缝。→ 点进关节中心。
4. **把阴影/反光当肢体**：地板倒影、玻璃反光、墙面影子不是人体，不标点也不扩 bbox。
5. **遗漏踝/膝**：躺卧时踝、膝常被忽略。→ 第 6 节 8 点必须优先保证。
6. **点序串位**：Skeleton 模板搭错（如左肘右肘互换）。→ 按第 3 节索引表核对模板。
7. **v 语义误用**：把「被遮挡但可推断」标成 0，或把「出画面」标成 2 乱点坐标。

---

## 10. QC（质量检查）

- 每个任务完成后**抽检 ≥10% 帧**由第二人独立复核（不参考第一人的标注）。
- 复核重点：左右侧一致性、躺卧 8 点是否标全、bbox 是否截断、v 语义是否正确。
- 两人对同一帧一致性存疑时：**截图存档**（存到 `data/annotations/pose_ft/qc/`），
  在标注群里讨论定夺后再统一修正。
- 复核通过后按第 2 节步骤 5–7 导出 → 转换 → 微调。

---

## 11. 目录结构图与命令速查

### 目录结构

```
/home/ljh/ankang/
├── scripts/
│   ├── sample_pose_ft_frames.py      # P0 采样（已跑完 Pilot，勿再重跑覆盖人工产物）
│   ├── prelabel_standing.py          # 预标注（仅站立/过渡帧，可选）
│   ├── coco_to_yolo_pose.py          # CVAT COCO Keypoints 1.0 → YOLO-pose txt
│   └── finetune_pose.py              # YOLO26-pose 微调入口
├── ai-engine/
│   ├── models/yolo26s-pose.pt        # 预训练权重（24MB，task=pose，kpt_shape=[17,3]）
│   └── runs/pose_ft/                 # 微调输出（best.pt / last.pt / results.csv）
├── backend/.venv/bin/python          # 转换脚本解释器（cv2/numpy）
├── ai-engine/.venv/bin/python        # 预标注/微调解释器（ultralytics/torch）
└── data/
    ├── manifest.csv                  # 全量视频清单（fall_start_ts 等）
    └── annotations/pose_ft/
        ├── pose_ft.yaml              # 数据集 yaml（path 绝对路径 + flip_idx）
        ├── manifest.csv              # Pilot 帧清单（filename, train_val, ...）
        ├── pilot.txt                 # 帧相对路径清单
        ├── images/
        │   ├── train/  314 张 *.jpg
        │   └── val/     48 张 *.jpg
        ├── labels/                  # ← 转换脚本产出（人工标注后）
        │   ├── train/<stem>.txt
        │   └── val/<stem>.txt
        ├── cvat/                    # ← CVAT 导出的 COCO Keypoints 1.0 JSON 放这里
        ├── prelabels/prelabel.json  # ← 预标注产出（可选）
        ├── skipped.txt              # ← 剔除清单（一行一文件名，人工维护）
        └── qc/                      # ← QC 截图存档
```

### 端到端命令序列

```bash
# 0) 采样（P0，已产出 images/ 与 manifest.csv；重跑会清空 images/ 与 manifest.csv/pilot.txt，
#    但绝不触碰 labels/、cvat/、prelabels/ 等人工产物）
/home/ljh/ankang/backend/.venv/bin/python scripts/sample_pose_ft_frames.py

# 1) 预标注（可选：仅站立/过渡帧，躺卧帧刻意不预标注；--limit N 冒烟）
/home/ljh/ankang/ai-engine/.venv/bin/python scripts/prelabel_standing.py
#    输出 data/annotations/pose_ft/prelabels/prelabel.json → CVAT「导入 COCO Keypoints 1.0 预标注」

# 2) CVAT 人工标注（见第 2 节）→ Export task dataset → COCO Keypoints 1.0
#    JSON 放入 data/annotations/pose_ft/cvat/

# 3) 转换（--input 可给单个 json 或目录；--dry-run 只检查不写文件）
/home/ljh/ankang/backend/.venv/bin/python scripts/coco_to_yolo_pose.py \
    --input data/annotations/pose_ft/cvat

# 4) 微调（Pilot 验收 smoke；全量 30 epoch 为小时级，另行决定）
/home/ljh/ankang/ai-engine/.venv/bin/python scripts/finetune_pose.py \
    --epochs 1 --batch 2 --name smoke
#    全量：--epochs 30 --batch 4（freeze=10, lr0=0.001, cos_lr, rect, workers=2 已内置）
```

### 关键约定（为什么）

- `pose_ft.yaml` 的 `path` 必须写绝对路径（ultralytics 对 cwd 下不存在的相对 path 会改按
  DATASETS_DIR 解析，相对写法必错）；`flip_idx` 必须写（fliplr 增强要交换左右对称点，缺失会静默污染标签）。
- `labels/{train,val}/<stem>.txt` 与 `images/{train,val}/<stem>.jpg` 同前缀同名（ultralytics 惯例）。
- 转换脚本输出行格式：`0 cx cy w h x1 y1 v1 ... x17 y17 v17`（56 字段，坐标均已按图宽高归一化）。
