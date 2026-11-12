import functools
import inspect
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from functools import wraps

from sqlmodel import SQLModel

from backend.common.config_loader import read_env


@functools.lru_cache
def init_db():
    db_url = read_env("DB_URL", "sqlite:///backend/records.db")
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite:") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    SQLModel.metadata.create_all(engine)
    ensure_uuid_index(engine)
    ensure_artifacts_column(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_uuid_index(engine):
    """create_all() skips indexes on existing tables, so apply it idempotently.

    shortcut: single-process deployment creates the index with raw SQL,
    when multi-dialect migrations are needed switch to Alembic.
    """
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_tasks_uuid ON tasks (uuid)")


def ensure_artifacts_column(engine):
    """Add the private artifact-reference column to pre-existing `tasks` tables.

    `create_all()` only creates missing tables, so an upgrade would leave old
    databases without the column and every download would fall back to hashing
    the whole cache. Idempotent, and existing rows keep their data.
    """
    if engine.dialect.name != "sqlite":
        # shortcut: SQLite-only ALTER; use Alembic once other dialects are supported.
        return
    with engine.begin() as conn:
        columns = [row[1] for row in conn.exec_driver_sql("PRAGMA table_info(tasks)").fetchall()]
        if columns and "artifacts" not in columns:
            conn.exec_driver_sql("ALTER TABLE tasks ADD COLUMN artifacts JSON")


def new_db_session():
    """Create a session; the caller owns it and must close it."""
    return init_db()()


def get_db_session():
    """FastAPI dependency that yields a session and closes it when the request ends."""
    session = new_db_session()
    try:
        yield session
    finally:
        session.close()


def handle_database_errors(func):
    params = list(inspect.signature(func).parameters)
    session_pos = params.index("session") if "session" in params else None

    @wraps(func)
    def wrapper(*args, **kwargs):
        if session_pos is None or "session" in kwargs or len(args) > session_pos:
            return func(*args, **kwargs)
        session = new_db_session()
        try:
            return func(*args, **kwargs, session=session)
        except Exception as e:
            print(f"Database error has occurred: {e}")
            try:
                session.rollback()
            except Exception:
                pass
            raise
        finally:
            session.close()
    return wrapper
