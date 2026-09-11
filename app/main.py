from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.db import SessionLocal, init_db
from app.seed import seed
from app.worker import requeue_orphans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Synthforge",
    description=(
        "Synthetic data platform POC. SPN engine, Apache Atlas driven data "
        "protection policy, relational integrity across tables."
    ),
    version="0.1.0",
)

# No CORS middleware: the UI is served by this same process on this same
# origin, so there is no cross-origin request to permit.

app.include_router(router, prefix="/api")


@app.on_event("startup")
def startup() -> None:
    init_db()
    with SessionLocal() as db:
        seed(db)
    orphans = requeue_orphans()
    if orphans:
        logger.warning("failed %d job(s) left running by a previous process", orphans)
    logger.info("Synthforge ready on port %s", settings.app_port)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# --------------------------------------------------------------------------
# UI
#
# The React bundle is built ahead of time into static/ and served from here.
# CML gives an Application one port, so the API and the UI share it.
# --------------------------------------------------------------------------

_assets = settings.static_dir / "assets"
if _assets.is_dir():
    app.mount("/assets", StaticFiles(directory=_assets), name="assets")


@app.get("/{path:path}")
def spa(path: str):
    """Serve the built UI, falling back to index.html for client-side routes."""
    index = settings.static_dir / "index.html"
    if not index.is_file():
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "UI bundle not found. Build it once with: python build_ui.py "
                    "(or npm --prefix frontend ci && npm --prefix frontend run build). "
                    "The API itself is up at /api and /docs."
                )
            },
        )

    candidate = (settings.static_dir / path).resolve()
    # Only serve real files from inside static/; anything else is a client-side
    # route and must render the SPA shell.
    if path and candidate.is_file() and candidate.is_relative_to(settings.static_dir.resolve()):
        return FileResponse(candidate)

    if path.startswith("api/"):
        raise HTTPException(404, "not found")

    return FileResponse(index)
