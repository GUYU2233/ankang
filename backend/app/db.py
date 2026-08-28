from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

settings = get_settings()

_is_sqlite = settings.database_url.startswith("sqlite")
connect_args = {"check_same_thread": False, "timeout": 30} if _is_sqlite else {}

engine_kwargs = {"connect_args": connect_args, "future": True}
if _is_sqlite:
    # 高并发轮询 + 后台巡检下，给足连接池并启用 WAL，避免 QueuePool 耗尽/读写互锁。
    engine_kwargs.update(pool_size=20, max_overflow=40, pool_timeout=10, pool_pre_ping=True)

engine = create_engine(settings.database_url, **engine_kwargs)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app.models import entities  # noqa: F401 确保模型已注册
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite(engine)


# 旧库缺失列的 ADD COLUMN DDL（SQLite 常量默认，满足 ADD COLUMN 约束）
_ALERT_MIGRATE_DDLS = {
    "status": "VARCHAR(16) NOT NULL DEFAULT 'pending'",
    "confirmed_by": "VARCHAR(64)",
    "handled_by": "VARCHAR(64)",
    "confirmed_at": "DATETIME",
    "handled_at": "DATETIME",
    "closed_at": "DATETIME",
    "confirm_note": "TEXT",
    "handle_note": "TEXT",
}


def _migrate_sqlite(eng=None) -> None:
    """对 SQLite 旧库的 alert_events 表做增量迁移：补齐状态机列并回填老布尔语义。幂等可重复。"""
    eng = eng or engine
    if not str(eng.url).startswith("sqlite"):
        return
    with eng.begin() as conn:
        has_table = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='alert_events'")
        ).scalar()
        if not has_table:
            return
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(alert_events)"))}
        missing = [name for name in _ALERT_MIGRATE_DDLS if name not in cols]
        for name in missing:
            conn.execute(text(f"ALTER TABLE alert_events ADD COLUMN {name} {_ALERT_MIGRATE_DDLS[name]}"))
        if "status" in missing:
            # 将老布尔语义回填进状态机
            conn.execute(text(
                "UPDATE alert_events SET status = CASE "
                "WHEN handled = 1 THEN 'handled' WHEN confirmed = 1 THEN 'confirmed' ELSE 'pending' END"
            ))
        # create_all 对已存在表不会补建索引，必须显式创建
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_alert_events_status ON alert_events (status)"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
