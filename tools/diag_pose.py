import cv2
import numpy as np
from ultralytics import YOLO


def diag(path: str) -> None:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("===", path, "frames", total, "h", h)
    model = YOLO("ai-engine/models/yolov8n-pose.pt")
    for idx in [0, total // 2, total - 1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.predict(frame, verbose=False, conf=0.35, device=0, half=False)
        if not res or res[0].keypoints is None or res[0].keypoints.data.shape[0] == 0:
            print(" idx", idx, "no person")
            continue
        k = res[0].keypoints.data[0].cpu().numpy()
        conf = k[:, 2]
        sh_y = float((k[5, 1] + k[6, 1]) / 2)
        hip_y = float((k[11, 1] + k[12, 1]) / 2)
        ankle_y = float((k[15, 1] + k[16, 1]) / 2)
        body_h = abs(hip_y - sh_y)
        hip_ratio = (h - hip_y) / max(h, 1e-6)
        ankle_hip = hip_y - ankle_y
        print(
            " idx", idx,
            "sh_y", round(sh_y, 1),
            "hip_y", round(hip_y, 1),
            "ankle_y", round(ankle_y, 1),
            "body_h", round(body_h, 1),
            "hip_ratio", round(hip_ratio, 3),
            "ankle_hip", round(ankle_hip, 1),
            "conf_hip", round(float(conf[11]), 2), round(float(conf[12]), 2),
            "conf_ankle", round(float(conf[15]), 2), round(float(conf[16]), 2),
        )
    cap.release()


diag("data/videos/livingroom/normal/livingroom_adl_001_20260827.mp4")
diag("data/videos/livingroom/fall/livingroom_fall_001_20260827.mp4")
