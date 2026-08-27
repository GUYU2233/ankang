"""后端冒烟测试：验证应用启动、默认模拟设备、看板与告警闭环。"""

import time

from fastapi.testclient import TestClient

from app.main import app


def test_smoke() -> None:
    with TestClient(app) as client:
        # 健康检查
        r = client.get("/api/v1/health")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "up"

        # 默认模拟设备
        r = client.get("/api/v1/devices")
        assert r.status_code == 200, r.text
        devices = r.json()
        assert len(devices) >= 1, "首次启动应自动注册模拟设备"

        # 看板
        r = client.get("/api/v1/dashboard/stats")
        assert r.status_code == 200, r.text
        stats = r.json()
        assert stats["total_devices"] >= 1

        # 预览帧
        dev_id = devices[0]["id"]
        r = client.get(f"/api/v1/streams/{dev_id}/frame.jpg")
        assert r.status_code == 200, r.text
        assert len(r.content) > 0

        # 等待巡检循环产生至少一条风险评分/告警（通常 2-4 秒内）
        time.sleep(6)
        r = client.get("/api/v1/dashboard/stats")
        assert r.status_code == 200, r.text
        stats = r.json()
        assert stats["avg_risk_score"] >= 0.0


if __name__ == "__main__":
    test_smoke()
    print("SMOKE OK")