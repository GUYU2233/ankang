#!/usr/bin/env python3
"""初始化数据库并写入默认模拟设备。在 backend 目录下运行或指定 PYTHONPATH。"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.db import init_db, SessionLocal  # noqa: E402
from app.core.device_manager import get_device_manager  # noqa: E402


def main() -> None:
    init_db()
    with SessionLocal() as db:
        mgr = get_device_manager()
        mgr.ensure_demo_devices(db)
        print("数据库已初始化，默认模拟设备已就绪")


if __name__ == "__main__":
    main()