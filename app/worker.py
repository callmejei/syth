"""TRAIN / GENERATE jobs, run in an in-process thread pool.

A CML Application is a single container running a single process, so there is
no broker to talk to and no second container to run workers in. This module
keeps the exact call shape the API used with Celery -- ``train_task.delay(job_id)``
returning something with an ``.id`` -- so the routes are unchanged.

Threads rather than processes: the heavy lifting is numpy/scikit-learn, which
releases the GIL, and a forked process would double the resident memory of a
container whose limit is whatever the CML Application was given.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app.config import settings
from app.db import SessionLocal
from app.models import Configuration, DataSource, Job, ModelArtifact
from app.synth.pipeline import run_generation, run_training, storage_for
from app.synth.report import sanitize

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(
    max_workers=settings.job_workers, thread_name_prefix="synthforge-job"
)

# Futures are retained so a job that finishes is not garbage collected before
# its exception (if any) has been logged.
_futures: dict[str, object] = {}


class _Task:
    """Gives a plain function the ``.delay()`` shape the API already calls."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def delay(self, job_id: str) -> "_Handle":
        handle = _Handle(str(uuid.uuid4()))
        future = _executor.submit(self._run, job_id)
        _futures[handle.id] = future
        future.add_done_callback(lambda f: _futures.pop(handle.id, None))
        return handle

    def _run(self, job_id: str):
        try:
            return self._fn(job_id)
        except Exception:  # noqa: BLE001
            # The job function already recorded FAILED on the row; this is so
            # the traceback reaches the Application log rather than dying with
            # the future.
            logger.exception("job %s failed", job_id)


class _Handle:
    def __init__(self, task_id: str):
        self.id = task_id


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _update(job_id: str, **fields) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        db.commit()


def _log(job_id: str, message: str, progress: float | None = None) -> None:
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        entries = list(job.logs or [])
        entries.append({"at": _now().isoformat(), "message": message})
        job.logs = entries[-200:]
        job.message = message
        if progress is not None:
            job.progress = round(float(progress), 4)
        db.commit()


def _train(job_id: str) -> dict:
    _update(job_id, status="RUNNING", started_at=_now())
    _log(job_id, "Job started", 0.0)
    try:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None:
                raise RuntimeError(f"job {job_id} not found")
            configuration = db.get(Configuration, job.configuration_id)
            if configuration is None:
                raise RuntimeError("configuration not found")
            data_source = db.get(DataSource, configuration.data_source_id)
            if data_source is None:
                raise RuntimeError("configuration has no data source")

            org_id = job.org_id
            project_id = job.project_id
            params = dict(job.params or {})
            table_names = list(configuration.selected_tables or [])
            data_args = dict(configuration.data_args or {})
            training_params = dict(configuration.training_params or {})
            storage_path = data_source.storage_path
            configuration_id = configuration.id
            data_source_id = data_source.id
            model_name = params.get("model_name") or f"model-{job_id[:8]}"
            engine_name = configuration.model_type or "SPN"

        output_dir = storage_for("models", job_id)

        result = run_training(
            storage_path=storage_path,
            table_names=table_names,
            data_args=data_args,
            training_params=training_params,
            output_dir=output_dir,
            engine=engine_name,
            policy_mode=params.get("policy_mode", "balanced"),
            enforce_policy=params.get("enforce_policy", True),
            progress=lambda p, m: _log(job_id, m, p),
        )

        with SessionLocal() as db:
            artifact = ModelArtifact(
                org_id=org_id,
                project_id=project_id,
                name=model_name,
                configuration_id=configuration_id,
                data_source_id=data_source_id,
                model_type=engine_name,
                artifact_path=output_dir,
                status="Completed",
                stats=sanitize(
                    {
                        "report": result["report"],
                        "policy": result["policy"],
                        "train_stats": result["train_stats"],
                        "params": result["params"],
                        "dp_forced": result["dp_forced"],
                    }
                ),
            )
            db.add(artifact)
            db.flush()
            artifact_id = artifact.id

            job = db.get(Job, job_id)
            if job:
                job.status = "COMPLETED"
                job.progress = 1.0
                job.model_id = artifact_id
                job.finished_at = _now()
                job.result = sanitize(
                    {
                        "model_id": artifact_id,
                        "artifact_path": output_dir,
                        "overall_fidelity": result["report"].get("overall_fidelity"),
                        "utility_ratio": (result["report"].get("utility") or {}).get(
                            "mean_utility_ratio"
                        ),
                        "dp_forced": result["dp_forced"],
                    }
                )
            configuration = db.get(Configuration, configuration_id)
            if configuration:
                configuration.status = "Ready"
            db.commit()

        return {"model_id": artifact_id}

    except Exception as exc:  # noqa: BLE001
        logger.exception("training job failed")
        _log(job_id, f"FAILED: {exc}")
        _update(
            job_id,
            status="FAILED",
            finished_at=_now(),
            result={"error": str(exc), "traceback": traceback.format_exc()[-4000:]},
        )
        raise


def _generate(job_id: str) -> dict:
    _update(job_id, status="RUNNING", started_at=_now())
    _log(job_id, "Job started", 0.0)
    try:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if job is None:
                raise RuntimeError(f"job {job_id} not found")
            artifact = db.get(ModelArtifact, job.model_id)
            if artifact is None:
                raise RuntimeError("model not found")
            params = dict(job.params or {})
            artifact_path = artifact.artifact_path

        output_dir = storage_for("synthetic", job_id)
        result = run_generation(
            artifact_path=artifact_path,
            n_rows=params.get("n_rows") or 1000,
            output_dir=output_dir,
            seed=int(params.get("seed", 0)),
            fmt=params.get("format", "parquet"),
            progress=lambda p, m: _log(job_id, m, p),
        )

        _update(
            job_id,
            status="COMPLETED",
            progress=1.0,
            finished_at=_now(),
            result=result,
        )
        return result

    except Exception as exc:  # noqa: BLE001
        logger.exception("generation job failed")
        _log(job_id, f"FAILED: {exc}")
        _update(
            job_id,
            status="FAILED",
            finished_at=_now(),
            result={"error": str(exc), "traceback": traceback.format_exc()[-4000:]},
        )
        raise


train_task = _Task(_train)
generate_task = _Task(_generate)


def requeue_orphans() -> int:
    """Fail jobs left RUNNING by a previous process.

    The thread pool dies with the Application. A CML Application restart would
    otherwise leave rows stuck at RUNNING forever, and the UI polls them as
    live work that is never coming back.
    """
    with SessionLocal() as db:
        stale = db.query(Job).filter(Job.status.in_(["RUNNING", "QUEUED"])).all()
        for job in stale:
            job.status = "FAILED"
            job.finished_at = _now()
            job.result = {"error": "interrupted by an application restart"}
        count = len(stale)
        if count:
            db.commit()
    return count


__all__ = ["generate_task", "requeue_orphans", "train_task"]
