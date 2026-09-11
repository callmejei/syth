"""Settings for the single-process CML Application build.

Everything is project-local: SQLite on disk, artifacts under ``data/``, no
Postgres, Redis or S3 endpoint to reach. A CML Application gets one container
and one port, so there is nothing external to point at.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings

# The project root as CML lays it out (/home/cdsw when running there).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # SQLite replaces Postgres. Written from both the request thread and the
    # job threads, so db.py opens it in WAL mode with a busy timeout.
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'synthforge.db').as_posix()}"

    # Jobs run in an in-process thread pool instead of Celery workers. Two at a
    # time matches the old `--concurrency=2`; a CML Application is one container,
    # so this is the whole compute budget.
    job_workers: int = 2

    # Apache Atlas. Off by default: a CML Application usually cannot reach the
    # catalog, and the client falls back to fixtures/atlas_mock.json anyway.
    # Set ATLAS_ENABLED=true (plus ATLAS_BASE_URL) if yours is reachable.
    atlas_enabled: bool = False
    atlas_base_url: str = "http://localhost:21000"
    atlas_user: str = "admin"
    atlas_password: str = "admin"
    atlas_timeout_seconds: float = 5.0

    # Classifications exported from your real Atlas and imported once. Takes
    # precedence over the bundled demo fixture.
    # See scripts/import_classifications.py.
    local_catalog_path: Path = PROJECT_ROOT / "data" / "catalog" / "classifications.json"

    schema_yaml_path: Path = PROJECT_ROOT / "schema" / "schema.yaml"
    storage_dir: Path = PROJECT_ROOT / "data"

    # Built React bundle served by the same FastAPI process.
    static_dir: Path = PROJECT_ROOT / "static"

    # CML injects the port an Application must bind. 8100 is its default.
    app_port: int = int(os.environ.get("CDSW_APP_PORT", "8100"))

    class Config:
        env_file = ".env"


settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)
