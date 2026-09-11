from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

_is_sqlite = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    # Job threads hold their own sessions, so the connection must be allowed to
    # cross threads. SQLAlchemy's pool still hands out one connection per thread.
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):
        # WAL lets the API keep reading while a training job writes progress.
        # Without it the default rollback journal locks readers out and job
        # polling from the UI intermittently 500s mid-training.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401  (registers mappers)

    Base.metadata.create_all(bind=engine)
