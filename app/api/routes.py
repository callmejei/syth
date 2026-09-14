"""REST API. Endpoint shape follows the screens in the scan."""

from __future__ import annotations

import io
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.atlas.client import AtlasClient, reset_health_cache
from app.atlas.inference import infer_column
from app.atlas.policy import build_policy
from app.synth.schema_infer import infer_schema, recommend_engine, validate_schema
from app.config import settings
from app.core.schema_loader import DatasetContext, load_registry, reload_registry
from app.synth.preprocess import infer_column_kind
from app.db import get_db
from app.models import (
    AuditLog,
    ComputeNode,
    ComputeProfile,
    Configuration,
    DataSource,
    Job,
    ModelArtifact,
    ModelType,
    Organization,
    OrgSetting,
    Project,
)
from app.seed import DEFAULT_ORG
from app.synth.pipeline import specs_from_config, storage_for
from app.worker import generate_task, train_task

router = APIRouter()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def current_org(db: Session) -> Organization:
    org = db.scalar(select(Organization).where(Organization.name == DEFAULT_ORG))
    if org is None:
        raise HTTPException(500, "organization not seeded")
    return org


def audit(db: Session, action: str, entity_type: str, entity_id: str, detail: dict) -> None:
    db.add(
        AuditLog(action=action, entity_type=entity_type, entity_id=entity_id, detail=detail)
    )


def _profile_specs(db: Session, name: str) -> tuple[int, int, int]:
    profile = db.scalar(select(ComputeProfile).where(ComputeProfile.name == name))
    if profile is None:
        return 14, 28, 0
    return profile.vcpu, profile.ram_gb, profile.gpu


def _dataset_context(db: Session, data_source_id: str | None, tables: list[str] | None = None):
    if not data_source_id:
        return DatasetContext(tables=[])
    source = db.get(DataSource, data_source_id)
    if source is None:
        return DatasetContext(tables=[])
    ctx = DatasetContext.from_datasource(source.tables)
    if tables:
        ctx.tables = [t for t in ctx.tables if t.name in tables]
    return ctx


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def _read_table(storage_path: str, table_name: str):
    """Load an uploaded table, or None if it cannot be read."""
    path = Path(storage_path) / f"{table_name}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None


def _derive_column_domains(
    storage_path: str,
    table_name: str,
    categorical: list[str],
    numerical: list[str],
    datetime_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Profile a table into per-column domains the model can be given up front.

    Emitted in the shape schema.yaml already defines (`min_val`, `max_val`,
    `force_categories` under column_kwargs), so it lands in the Data Arguments
    tree where a user can see and correct it.
    """
    path = Path(storage_path) / f"{table_name}.parquet"
    if not path.exists():
        return {}
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return {}

    out: dict[str, Any] = {}

    for column in categorical:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().unique().tolist()
        if 0 < len(values) <= 60:
            out[column] = {"force_categories": [_jsonable(v) for v in values]}

    for column in numerical:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        low, high = float(series.min()), float(series.max())
        # Widen slightly so the declared range is a bound rather than a
        # description of the exact observed extremes.
        #
        # Do NOT slam the lower bound to zero just because the column is
        # non-negative. The declared range is what the histogram spans, and a
        # range far wider than the data wastes almost every bin on values that
        # never occur -- noise then fills those bins and the model emits them.
        #
        # The lower bound is therefore padded proportionally to the VALUE, not
        # to the span. Padding by a fraction of the span is what broke this:
        # on a loan column of 3.5k-1M, 5% of the span is ~50k against a minimum
        # of 3,578, so the bound collapsed to zero and the generator produced
        # loans of S$0.01 where the real first percentile is S$8,466.
        span = max(high - low, 1e-9)
        if low > 0:
            low_bound = low * 0.95
        elif low == 0:
            low_bound = 0.0
        else:
            low_bound = low - 0.1 * span
        high_bound = high + 0.05 * span
        entry: dict[str, Any] = {"min_val": low_bound, "max_val": high_bound}

        # Heavy right tail (money, durations): model it in log space, or DP's
        # equal-width bins put nearly all mass in the first bin.
        if low_bound >= 0 and series.median() > 0:
            tail_ratio = series.quantile(0.99) / max(series.median(), 1e-9)
            if tail_ratio > 8:
                entry["transform"] = "log"
        out[column] = entry

    # Datetime columns need a declared range too, or the DP guarantee stays
    # conditional for the whole table however well the other columns are
    # declared.
    for column in datetime_columns or []:
        if column not in frame.columns:
            continue
        series = pd.to_datetime(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        span = series.max() - series.min()
        pad = span * 0.05 if span.total_seconds() > 0 else pd.Timedelta(days=1)
        out[column] = {
            "min_val": (series.min() - pad).isoformat(),
            "max_val": (series.max() + pad).isoformat(),
        }

    return out


def _derive_max_degree(
    storage_path: str, table_name: str, foreign_keys: list[dict[str, Any]]
) -> int:
    """Upper bound on rows of this table per parent row.

    Taken just above the observed maximum rather than an arbitrary constant,
    because this is the span of the DP degree histogram.
    """
    if not foreign_keys:
        return 1
    path = Path(storage_path) / f"{table_name}.parquet"
    if not path.exists():
        return 200
    try:
        frame = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return 200

    child_column = foreign_keys[0]["child_column_names"][0]
    if child_column not in frame.columns:
        return 200

    counts = frame.groupby(child_column).size()
    if counts.empty:
        return 1
    # A little headroom above the busiest parent, so the bound is a bound and
    # not a description of this particular extract.
    return int(max(4, min(2000, round(counts.max() * 1.25))))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


@router.get("/schema/groups")
def schema_groups() -> dict[str, Any]:
    registry = load_registry()
    return {"groups": registry.groups, "parameter_count": len(registry.params)}


@router.get("/schema/tree")
def schema_tree(
    data_source_id: str | None = None,
    tables: str | None = None,
    include_skipped: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """The Data Arguments tree, resolved against a dataset."""
    table_list = [t for t in (tables or "").split(",") if t]
    ctx = _dataset_context(db, data_source_id, table_list or None)
    registry = load_registry()
    return {
        "sections": registry.build_tree(ctx, include_skipped=include_skipped),
        "tables": ctx.table_names,
    }


@router.post("/schema/reload")
def schema_reload() -> dict[str, Any]:
    registry = reload_registry()
    return {"reloaded": True, "parameter_count": len(registry.params)}


# --------------------------------------------------------------------------
# download helpers
# --------------------------------------------------------------------------


def _read_file(path: Path) -> pd.DataFrame:
    """Load one stored table file. Distinct from _read_table, which resolves a
    table by name within a data source's directory."""
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _encode(frame: pd.DataFrame, fmt: str) -> bytes:
    """Serialise a frame to the requested wire format."""
    if fmt == "parquet":
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        return buffer.getvalue()
    return frame.to_csv(index=False).encode("utf-8")


def _single_file_response(frame: pd.DataFrame, stem: str, fmt: str) -> Response:
    payload = _encode(frame, fmt)
    media = (
        "application/vnd.apache.parquet"
        if fmt == "parquet"
        else "text/csv; charset=utf-8"
    )
    return Response(
        content=payload,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{stem}.{fmt}"',
            "Content-Length": str(len(payload)),
        },
    )


def _zip_response(items: list[tuple[str, pd.DataFrame]], stem: str, fmt: str) -> Response:
    """Bundle several tables into one archive.

    Held in memory deliberately: a POC dataset is measured in megabytes, and a
    temp file would need cleaning up after the response is flushed.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for table, frame in items:
            archive.writestr(f"{table}.{fmt}", _encode(frame, fmt))
    payload = buffer.getvalue()
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{stem}.zip"',
            "Content-Length": str(len(payload)),
        },
    )


def _safe_stem(name: str) -> str:
    """Filename-safe slug. Dataset names are free text and reach a header."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
    return cleaned or "download"


# --------------------------------------------------------------------------
# data sources
# --------------------------------------------------------------------------


@router.get("/data-sources")
def list_data_sources(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    sources = db.scalars(select(DataSource).order_by(DataSource.created_at.desc())).all()
    used_by: dict[str, list[str]] = {}
    for configuration in db.scalars(select(Configuration)).all():
        if configuration.data_source_id:
            used_by.setdefault(configuration.data_source_id, []).append(configuration.name)
    return [
        {
            "id": s.id,
            "name": s.name,
            "version": s.version,
            "tables": [t["name"] for t in (s.tables or {}).get("tables", [])],
            "table_count": len((s.tables or {}).get("tables", [])),
            "row_count": sum(t.get("rows", 0) for t in (s.tables or {}).get("tables", [])),
            "atlas_source": (s.atlas_metadata or {}).get("source"),
            # How many of this dataset's tables the resolved source actually had
            # something to say about. The source name alone is misleading: a
            # fallback to the bundled fixture reports "fixture" whether it
            # covered every table or, as is usual for a dataset that is not the
            # bundled demo, none of them.
            "atlas_tables_classified": len(
                ((s.atlas_metadata or {}).get("tables") or {})
            ),
            "project_id": s.project_id,
            "used_by": used_by.get(s.id, []),
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sources
    ]


@router.get("/data-sources/{source_id}")
def get_data_source(source_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    source = db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(404, "data source not found")
    return {
        "id": source.id,
        "name": source.name,
        "version": source.version,
        "storage_path": source.storage_path,
        "tables": (source.tables or {}).get("tables", []),
        "atlas": source.atlas_metadata or {},
        "created_at": source.created_at.isoformat() if source.created_at else None,
    }


@router.post("/data-sources/upload")
async def upload_data_source(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    project_id: str | None = Form(None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Accept one or more csv/parquet files; each file becomes a table."""
    org = current_org(db)
    source = DataSource(
        org_id=org.id, name=name, version="v1", project_id=project_id or None
    )
    db.add(source)
    db.flush()

    directory = Path(storage_for("raw", source.id))
    tables: list[dict[str, Any]] = []

    for upload in files:
        filename = Path(upload.filename or "table.csv").name
        table_name = Path(filename).stem
        target = directory / filename
        with open(target, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)

        try:
            if target.suffix.lower() == ".parquet":
                frame = pd.read_parquet(target)
            else:
                frame = pd.read_csv(target)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"could not read {filename}: {exc}") from exc

        # Normalise to parquet so training has one code path.
        frame.to_parquet(directory / f"{table_name}.parquet", index=False)

        tables.append(
            {
                "name": table_name,
                "file": filename,
                "rows": int(len(frame)),
                "columns": [
                    {
                        "name": column,
                        "dtype": str(frame[column].dtype),
                        "nulls": float(frame[column].isna().mean()),
                        "unique": int(frame[column].nunique()),
                    }
                    for column in frame.columns
                ],
            }
        )

    source.storage_path = str(directory)
    source.tables = {"tables": tables}

    # Resolve Atlas classifications immediately, so the UI can show them.
    atlas = AtlasClient().classify_tables([t["name"] for t in tables])
    source.atlas_metadata = atlas.to_json()

    audit(db, "upload", "data_source", source.id, {"name": name, "tables": len(tables)})
    db.commit()

    return {"id": source.id, "name": source.name, "tables": tables,
            "atlas": source.atlas_metadata}


def _dependents(source_id: str, db: Session) -> tuple[list[Any], list[Any], list[Any]]:
    """Everything that would dangle if this dataset went away.

    Models are reached two ways: directly, and through their configuration. A
    model trained before ``data_source_id`` was recorded has only the second
    link, and missing it means the delete fails on a foreign key instead of
    telling the user what is in the way.
    """
    configurations = db.scalars(
        select(Configuration).where(Configuration.data_source_id == source_id)
    ).all()
    config_ids = {c.id for c in configurations}

    models = [
        m
        for m in db.scalars(select(ModelArtifact)).all()
        if m.data_source_id == source_id or m.configuration_id in config_ids
    ]
    model_ids = {m.id for m in models}

    jobs = [
        j
        for j in db.scalars(select(Job)).all()
        if j.configuration_id in config_ids or j.model_id in model_ids
    ]
    return configurations, models, jobs


@router.get("/data-sources/{source_id}/download")
def download_data_source(
    source_id: str,
    table: str | None = None,
    fmt: str = Query("csv", pattern="^(csv|parquet)$"),
    db: Session = Depends(get_db),
) -> Response:
    """Download the uploaded data, one table or all of them as a zip.

    Serves the normalised parquet written at upload time rather than the file
    the user handed us, so what comes back is exactly what training reads.
    """
    source = db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(404, "data source not found")

    directory = Path(source.storage_path or storage_for("raw", source.id))
    names = [t["name"] for t in (source.tables or {}).get("tables", [])]
    if table is not None:
        if table not in names:
            raise HTTPException(404, f"table '{table}' not in this dataset")
        names = [table]
    if not names:
        raise HTTPException(404, "this dataset has no tables")

    frames: list[tuple[str, pd.DataFrame]] = []
    for name in names:
        path = directory / f"{name}.parquet"
        if not path.exists():
            raise HTTPException(410, f"the file for '{name}' is no longer on disk")
        frames.append((name, _read_file(path)))

    stem = _safe_stem(source.name)
    if len(frames) == 1:
        return _single_file_response(frames[0][1], f"{stem}_{frames[0][0]}", fmt)
    return _zip_response(frames, stem, fmt)


@router.get("/data-sources/{source_id}/usage")
def data_source_usage(source_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """What would break if this dataset were deleted. Drives the confirm dialog."""
    source = db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(404, "data source not found")
    configurations, models, jobs = _dependents(source_id, db)
    return {
        "configurations": [{"id": c.id, "name": c.name} for c in configurations],
        "models": [{"id": m.id, "name": m.name} for m in models],
        "job_count": len(jobs),
    }


@router.delete("/data-sources/{source_id}")
def delete_data_source(
    source_id: str,
    cascade: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Delete a dataset and its files.

    Refuses by default when configurations or models depend on it: silently
    orphaning a trained model would leave a report that cannot be traced back
    to its input. ``cascade=true`` removes the dependants too.
    """
    source = db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(404, "data source not found")

    configurations, models, jobs = _dependents(source_id, db)

    if (configurations or models) and not cascade:
        raise HTTPException(
            409,
            {
                "message": "this dataset is in use",
                "configurations": [c.name for c in configurations],
                "models": [m.name for m in models],
            },
        )

    removed = {"configurations": 0, "models": 0, "jobs": 0}

    # Order matters: jobs reference models, models reference configurations.
    for job in jobs:
        # Generated output lives outside the row; drop it with the record.
        shutil.rmtree(Path(settings.storage_dir) / "synthetic" / job.id, ignore_errors=True)
        db.delete(job)
        removed["jobs"] += 1
    db.flush()

    for model in models:
        shutil.rmtree(Path(settings.storage_dir) / "models" / model.id, ignore_errors=True)
        db.delete(model)
        removed["models"] += 1
    db.flush()

    for configuration in configurations:
        db.delete(configuration)
        removed["configurations"] += 1

    if source.storage_path:
        shutil.rmtree(Path(source.storage_path), ignore_errors=True)

    name = source.name
    db.delete(source)
    audit(db, "delete", "data_source", source_id, {"name": name, **removed})
    db.commit()
    return {"deleted": source_id, "name": name, **removed}


@router.post("/data-sources/{source_id}/refresh-atlas")
def refresh_atlas(source_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    source = db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(404, "data source not found")
    table_names = [t["name"] for t in (source.tables or {}).get("tables", [])]
    # An explicit refresh should re-probe rather than trust the cached verdict.
    reset_health_cache()
    atlas = AtlasClient().classify_tables(table_names)
    source.atlas_metadata = atlas.to_json()
    audit(db, "refresh_atlas", "data_source", source_id, {"source": atlas.source})
    db.commit()
    return source.atlas_metadata


@router.get("/data-sources/{source_id}/policy")
def data_source_policy(
    source_id: str,
    mode: str = "balanced",
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Preview the protection policy Atlas implies for this dataset."""
    source = db.get(DataSource, source_id)
    if source is None:
        raise HTTPException(404, "data source not found")

    tables = (source.tables or {}).get("tables", [])
    table_columns = {
        t["name"]: [c["name"] for c in t.get("columns", [])] for t in tables
    }
    atlas = AtlasClient().classify_tables(list(table_columns))
    # Read the data so the preview matches what training will actually do: keys
    # are excluded from inference, and untagged columns are recognised by name
    # or value rather than falling through to MODEL.
    frames = {
        name: _read_table(source.storage_path, name) for name in table_columns
    }
    frames = {k: v for k, v in frames.items() if v is not None}
    detected = infer_schema(frames)
    policy = build_policy(
        atlas,
        table_columns,
        keys_by_table={t: detected.key_columns_of(t) for t in table_columns},
        mode=mode,
        frames=frames,
    )
    return policy.to_json()


# --------------------------------------------------------------------------
# configurations
# --------------------------------------------------------------------------


class ConfigurationCreate(BaseModel):
    name: str
    model_type: str = "SPN"
    data_source_id: str | None = None
    selected_tables: list[str] = []
    project_id: str | None = None
    compute_profile: str = "LOW"
    device_type: str = "CPU"


class ConfigurationUpdate(BaseModel):
    name: str | None = None
    model_type: str | None = None
    selected_tables: list[str] | None = None
    data_args: dict[str, Any] | None = None
    training_params: dict[str, Any] | None = None
    target_connection: dict[str, Any] | None = None
    compute_profile: str | None = None
    device_type: str | None = None
    status: str | None = None


@router.get("/configurations")
def list_configurations(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(Configuration).order_by(Configuration.updated_at.desc())).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "model_type": c.model_type,
            "status": c.status,
            "data_source_id": c.data_source_id,
            "project_id": c.project_id,
            "selected_tables": c.selected_tables,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in rows
    ]


@router.post("/configurations")
def create_configuration(
    payload: ConfigurationCreate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    org = current_org(db)
    vcpu, ram, gpu = _profile_specs(db, payload.compute_profile)

    model_type = db.scalar(select(ModelType).where(ModelType.name == payload.model_type))
    if model_type is None or not model_type.enabled:
        raise HTTPException(
            400,
            f"model type '{payload.model_type}' is not enabled. "
            f"{model_type.licence_note if model_type else ''}",
        )

    configuration = Configuration(
        org_id=org.id,
        project_id=payload.project_id,
        name=payload.name,
        model_type=payload.model_type,
        data_source_id=payload.data_source_id,
        selected_tables=payload.selected_tables,
        compute_profile=payload.compute_profile,
        device_type=payload.device_type,
        vcpu=vcpu,
        ram_gb=ram,
        gpu=gpu,
        training_params={
            "spn_config": {"beta": 100000, "private": False, "epsilon": 2.0},
            # An engine named in the request was chosen; one that came from the
            # field default was not. Auto-configure overrides only the latter.
            "engine_pinned": "model_type" in payload.model_fields_set,
        },
        target_connection={"need_parquet": True, "connection": "MinIO: Internal-Source-Connection"},
    )
    db.add(configuration)
    db.flush()
    audit(db, "create", "configuration", configuration.id, {"name": payload.name})
    db.commit()
    return {"id": configuration.id, "name": configuration.name, "status": configuration.status}


@router.get("/configurations/{configuration_id}")
def get_configuration(configuration_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    configuration = db.get(Configuration, configuration_id)
    if configuration is None:
        raise HTTPException(404, "configuration not found")
    return {
        "id": configuration.id,
        "name": configuration.name,
        "model_type": configuration.model_type,
        "status": configuration.status,
        "data_source_id": configuration.data_source_id,
        "selected_tables": configuration.selected_tables,
        "device_type": configuration.device_type,
        "compute_profile": configuration.compute_profile,
        "vcpu": configuration.vcpu,
        "ram_gb": configuration.ram_gb,
        "gpu": configuration.gpu,
        "data_args": configuration.data_args or {},
        "training_params": configuration.training_params or {},
        "target_connection": configuration.target_connection or {},
    }


@router.patch("/configurations/{configuration_id}")
def update_configuration(
    configuration_id: str, payload: ConfigurationUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    configuration = db.get(Configuration, configuration_id)
    if configuration is None:
        raise HTTPException(404, "configuration not found")

    if payload.model_type:
        # Same gate as creation: a disabled engine cannot be selected here
        # either, or the licence position could be bypassed with a PATCH.
        model_type = db.scalar(
            select(ModelType).where(ModelType.name == payload.model_type)
        )
        if model_type is None or not model_type.enabled:
            raise HTTPException(
                400,
                f"model type '{payload.model_type}' is not enabled. "
                f"{model_type.licence_note if model_type else ''}",
            )
        # Record that the engine was chosen, so Auto-configure leaves it alone.
        params = dict(configuration.training_params or {})
        params["engine_pinned"] = True
        configuration.training_params = params

    changed: dict[str, Any] = {}
    for field_name, value in payload.model_dump(exclude_unset=True).items():
        if value is None:
            continue
        setattr(configuration, field_name, value)
        changed[field_name] = value

    if payload.compute_profile:
        vcpu, ram, gpu = _profile_specs(db, payload.compute_profile)
        configuration.vcpu, configuration.ram_gb, configuration.gpu = vcpu, ram, gpu

    audit(db, "update", "configuration", configuration_id, {"fields": list(changed)})
    db.commit()
    return {"id": configuration.id, "updated": list(changed)}


def _catalog_gap_message(atlas: AtlasClient, coverage: dict[str, Any]) -> str:
    """Explain which tables the catalog does not cover, and what to do next.

    Returned as a readable string rather than a structured payload because it is
    rendered straight into a banner in the UI.
    """
    missing = coverage["tables_missing"]
    ungoverned = coverage["tables_ungoverned"]
    known = coverage["tables_classified"]

    lines: list[str] = []
    if not settings.atlas_enabled:
        lines.append(
            "Apache Atlas is not enabled for this deployment (ATLAS_ENABLED=false), "
            "so there is no catalog to configure from."
        )
    elif coverage["source"] not in ("atlas",):
        lines.append(
            f"Apache Atlas returned nothing usable ({coverage['detail'] or 'no detail'})."
        )

    if missing:
        lines.append(
            f"Not catalogued at all: {', '.join(sorted(missing))}. "
            "No entity for these tables was found."
        )
    if ungoverned:
        lines.append(
            f"Catalogued but carrying no classifications: {', '.join(sorted(ungoverned))}. "
            "The entity exists, but no column has been tagged."
        )
    if known:
        lines.append(f"Already classified, and usable: {', '.join(sorted(known))}.")

    if coverage["source"] == "fixture":
        lines.append(
            "The only classifications available are the bundled demo fixture, which "
            "is ILLUSTRATIVE sample data and must not be used to govern your tables."
        )

    lines.append(
        "To fix this, either tag these tables in Atlas, or export your "
        "classifications once and import them with "
        "scripts/import_classifications.py (an Atlas export or a "
        "table,column,classifications spreadsheet both work) -- no connectivity "
        "needed. Otherwise configure the Relational Schema by hand, or re-run "
        "Auto-configure with require_catalog=false to derive it from the data "
        "alone, in which case the policy rests on profiling and inference rather "
        "than on your catalog."
    )
    return " ".join(lines)


@router.post("/configurations/{configuration_id}/autoconfigure")
def autoconfigure(
    configuration_id: str,
    require_catalog: bool = True,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Populate the Relational Schema and column types from Atlas + profiling.

    This is the step that makes the catalog integration pay off: instead of
    hand-filling the Data Arguments tree for every table, we derive it.

    It is called "Auto-configure from Atlas", so by default it refuses to run
    when Atlas has nothing to say about the selected tables. Classification
    lookup falls back silently -- to an imported catalog, then to the bundled
    demo fixture -- which is right for generation (a run should not die because
    the catalog is down) and wrong here: a schema derived entirely from profiling
    would be presented as one derived from the catalog, and nobody downstream
    could tell the difference.

    Pass require_catalog=false to derive it from the data anyway. The response
    then says so explicitly.
    """
    configuration = db.get(Configuration, configuration_id)
    if configuration is None:
        raise HTTPException(404, "configuration not found")
    source = db.get(DataSource, configuration.data_source_id)
    if source is None:
        raise HTTPException(400, "configuration has no data source")

    tables = (source.tables or {}).get("tables", [])
    available = [t["name"] for t in tables]
    selected = set(configuration.selected_tables or available)
    tables = [t for t in tables if t["name"] in selected]

    if not tables:
        raise HTTPException(
            400,
            f"none of the selected tables exist in this data source. "
            f"Selected: {', '.join(sorted(selected)) or '(none)'}. "
            f"Available: {', '.join(available) or '(none)'}.",
        )

    atlas = AtlasClient().classify_tables([t["name"] for t in tables])

    table_args: dict[str, Any] = {}
    by_name = {t["name"]: t for t in tables}

    coverage = atlas.coverage(
        list(by_name),
        {t["name"]: [c["name"] for c in t.get("columns", [])] for t in tables},
    )
    if require_catalog and not coverage["complete"]:
        raise HTTPException(409, _catalog_gap_message(atlas, coverage))

    # Keys come from the data, not from column names alone. A key must actually
    # be unique and non-null, and a foreign key is only accepted once its values
    # are found in the parent -- which is what stops a shared column like
    # customer_id being taken for the primary key of every table that carries it.
    all_frames = {
        name: _read_table(source.storage_path, name) for name in by_name
    }
    all_frames = {k: v for k, v in all_frames.items() if v is not None}
    detected = infer_schema(all_frames)
    primary_keys: dict[str, str] = dict(detected.primary_keys)

    for table in tables:
        name = table["name"]
        foreign_keys = [
            {
                "parent_table_name": fk.parent_table,
                "parent_column_names": [fk.parent_column],
                "child_column_names": [fk.child_column],
            }
            for fk in detected.foreign_keys_of(name)
        ]

        # Column typing.
        #
        # This reads the actual values via infer_column_kind rather than judging
        # by dtype and cardinality alone. Guessing from the pandas dtype gets
        # dates wrong in the common case: a CSV date column arrives as `object`,
        # and because dates of birth are close to unique it was being written off
        # as a high-cardinality identifier and replaced with a placeholder
        # ("Date_of_Birth_0"). One inference implementation, used everywhere.
        categorical, numerical, datetime_columns, timedelta_columns, non_std = [], [], [], [], []
        classification = atlas.tables.get(name)
        frame = all_frames.get(name)
        key_columns = detected.key_columns_of(name)

        for column in table.get("columns", []):
            column_name = column["name"]
            column_classification = (
                classification.columns.get(column_name) if classification else None
            )
            # A direct identifier is never modelled, whatever its type.
            if column_classification and (
                column_classification.is_direct_identifier or column_classification.is_pci
            ):
                non_std.append(column_name)
                continue

            # The catalog may not have been told about this table. Fall back to
            # recognising the column, so an untagged full_name is typed as a
            # placeholder here rather than being learned as a categorical.
            if column_classification is None or not column_classification.is_classified:
                if column_name not in key_columns:
                    series = frame[column_name] if frame is not None and column_name in frame.columns else None
                    if infer_column(column_name, series) is not None:
                        non_std.append(column_name)
                        continue

            kind = None
            if frame is not None and column_name in frame.columns:
                kind = infer_column_kind(frame[column_name])
            else:
                dtype = column["dtype"]
                ratio = column.get("unique", 0) / max(table.get("rows", 1), 1)
                if "datetime" in dtype:
                    kind = "datetime"
                elif dtype.startswith(("int", "float", "uint")):
                    kind = "categorical" if ratio < 0.02 else "numerical"
                else:
                    kind = "non_std" if ratio > 0.5 else "categorical"

            # Atlas knows the declared type; trust it over inference when the
            # catalog says this is a date and the values are parseable.
            declared = (column_classification.data_type or "").lower() if column_classification else ""
            if declared in {"date", "datetime", "timestamp"} and kind != "non_std":
                kind = "datetime"

            {
                "datetime": datetime_columns,
                "timedelta": timedelta_columns,
                "numerical": numerical,
                "categorical": categorical,
                "non_std": non_std,
            }[kind].append(column_name)

        datetime_candidates = [c for c in datetime_columns]

        # Derive the column domain. Under differential privacy the domain has
        # to come from somewhere other than the training rows, or the released
        # bin edges and category lists are an unbudgeted disclosure -- and
        # equal-width bins over a measured range make heavy-tailed columns
        # unusable. Profiling them here at least makes the bins sensible and
        # puts the values somewhere a human can review and override; replace
        # them with catalog-declared ranges for an unconditional guarantee.
        column_kwargs = _derive_column_domains(
            source.storage_path, name, categorical, numerical, datetime_columns
        )

        # Declared cardinality bound for this table as a child. The DP degree
        # histogram spans 0..max_degree, so leaving it at the default (200)
        # when customers really have 2-20 orders spreads noise across a range
        # an order of magnitude too wide and flattens the distribution.
        max_degree = _derive_max_degree(source.storage_path, name, foreign_keys)

        table_args[name] = {
            "column_kwargs": column_kwargs,
            "max_degree": max_degree,
            "primary_key": primary_keys.get(name),
            "foreign_keys": foreign_keys,
            "timeseries": bool(foreign_keys and datetime_candidates),
            "static_ids": [fk["child_column_names"][0] for fk in foreign_keys][:1],
            "sortby": datetime_candidates[:1],
            "categorical_columns": categorical,
            "numerical_columns": numerical,
            "datetime_columns": datetime_columns,
            "timedelta_columns": timedelta_columns,
            "non_std_columns": non_std,
        }

    # Topological order.
    specs = specs_from_config({"table_args": table_args}, list(by_name))
    from app.synth.relational import topological_order

    order = topological_order(specs)

    data_args = dict(configuration.data_args or {})
    data_args["table_args"] = table_args
    data_args["order"] = order
    configuration.data_args = data_args
    configuration.status = "Ready"

    audit(
        db,
        "autoconfigure",
        "configuration",
        configuration_id,
        {"atlas_source": atlas.source, "tables": list(table_args)},
    )
    db.commit()

    # beta is the minimum rows needed to build a SUM node. The schema default
    # (100,000) assumes a table with millions of rows; on a smaller extract the
    # root factorises immediately and the model learns no correlations at all.
    # Derive it from the smallest selected table so the structure can actually
    # be learned, and leave headroom for several levels of clustering.
    row_counts = [t.get("rows", 0) for t in tables if t.get("rows")]
    beta = None
    if row_counts:
        beta = max(50, min(min(row_counts) // 20, 100_000))
        params = dict(configuration.training_params or {})
        spn_config = dict(params.get("spn_config") or {})
        spn_config["beta"] = beta
        params["spn_config"] = spn_config
        configuration.training_params = params
        db.commit()

    # Pick the engine from the shape of the data -- unless the user has chosen
    # one. The old condition overwrote any value that differed from the
    # recommendation, which is exactly the case where the choice was deliberate:
    # selecting ARF on a relational dataset and then running Auto-configure put
    # it silently back to SPN. The recommendation is still returned, so the UI
    # can show that the two disagree.
    recommended, rationale = recommend_engine(detected)
    pinned = bool((configuration.training_params or {}).get("engine_pinned"))
    if not configuration.model_type or not pinned:
        configuration.model_type = recommended
        db.commit()

    # Confirm the schema we just wrote actually validates, so Auto-configure can
    # never hand back a configuration that training will reject.
    issues = validate_schema(table_args, all_frames, detected)

    return {
        "atlas_source": atlas.source,
        "atlas_detail": atlas.detail,
        "order": order,
        "table_args": table_args,
        "beta": beta,
        "engine": recommended,
        "engine_rationale": rationale,
        "detected_schema": detected.to_json(),
        "schema_issues": [i.to_json() for i in issues],
        # Where this configuration actually came from. Without it a schema
        # derived purely by profiling is indistinguishable from one the
        # governance team curated.
        "catalog_coverage": coverage,
        "derived_from_catalog": coverage["complete"] and atlas.is_trusted,
        "requires_review": not (coverage["complete"] and atlas.is_trusted),
    }


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


class TrainRequest(BaseModel):
    model_name: str | None = None
    policy_mode: str = "balanced"
    enforce_policy: bool = True
    compute_profile: str = "LOW"
    # Escape hatch for a schema the validator rejects. The run proceeds, and the
    # report records that the schema was known-invalid.
    allow_schema_errors: bool = False


class GenerateRequest(BaseModel):
    model_id: str
    # None means "same shape as the source dataset", which is the default a
    # synthetic copy should have. Give a dict to size specific tables, or an int
    # to size the root table and let children follow.
    n_rows: dict[str, int] | int | None = None
    seed: int = 0
    format: str = "parquet"
    compute_profile: str = "LOW"


@router.post("/configurations/{configuration_id}/train")
def start_training(
    configuration_id: str, payload: TrainRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    configuration = db.get(Configuration, configuration_id)
    if configuration is None:
        raise HTTPException(404, "configuration not found")
    if not configuration.data_source_id:
        raise HTTPException(400, "configuration has no data source")
    if not configuration.selected_tables:
        raise HTTPException(400, "configuration has no selected tables")

    # Validate the relational schema up front. Training a model on a schema the
    # data contradicts wastes the run and, worse, produces a model whose
    # per-column scores look healthy while every join is broken.
    if not payload.allow_schema_errors:
        source = db.get(DataSource, configuration.data_source_id)
        frames = {
            name: _read_table(source.storage_path, name)
            for name in configuration.selected_tables
        }
        frames = {k: v for k, v in frames.items() if v is not None}
        if frames:
            issues = validate_schema(
                (configuration.data_args or {}).get("table_args") or {}, frames
            )
            blocking = [i for i in issues if i.level == "error"]
            if blocking:
                raise HTTPException(
                    400,
                    {
                        "message": (
                            f"the relational schema is not valid for this data "
                            f"({len(blocking)} blocking issue(s)). Run Auto-configure to "
                            "derive it, or set allow_schema_errors to train anyway."
                        ),
                        "issues": [i.to_json() for i in blocking],
                    },
                )

    job = Job(
        org_id=configuration.org_id,
        project_id=configuration.project_id,
        job_type="TRAIN",
        configuration_id=configuration_id,
        compute_profile=payload.compute_profile,
        params=payload.model_dump(),
        status="QUEUED",
    )
    db.add(job)
    db.flush()
    job_id = job.id
    audit(db, "train", "configuration", configuration_id, {"job_id": job_id})
    db.commit()

    task = train_task.delay(job_id)
    with db.begin_nested():
        pass
    stored = db.get(Job, job_id)
    if stored:
        stored.celery_task_id = task.id
        db.commit()

    return {"job_id": job_id, "status": "QUEUED"}


@router.post("/generate")
def start_generation(payload: GenerateRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    artifact = db.get(ModelArtifact, payload.model_id)
    if artifact is None:
        raise HTTPException(404, "model not found")

    job = Job(
        org_id=artifact.org_id,
        project_id=artifact.project_id,
        job_type="GENERATE",
        configuration_id=artifact.configuration_id,
        model_id=payload.model_id,
        compute_profile=payload.compute_profile,
        params=payload.model_dump(),
        status="QUEUED",
    )
    db.add(job)
    db.flush()
    job_id = job.id
    audit(db, "generate", "model", payload.model_id, {"job_id": job_id})
    db.commit()

    task = generate_task.delay(job_id)
    stored = db.get(Job, job_id)
    if stored:
        stored.celery_task_id = task.id
        db.commit()

    return {"job_id": job_id, "status": "QUEUED"}


@router.get("/jobs")
def list_jobs(limit: int = 50, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit)).all()
    return [
        {
            "id": j.id,
            "job_type": j.job_type,
            "status": j.status,
            "progress": j.progress,
            "message": j.message,
            "configuration_id": j.configuration_id,
            "model_id": j.model_id,
            "compute_profile": j.compute_profile,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        }
        for j in rows
    ]


@router.get("/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id,
        "job_type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "logs": job.logs or [],
        "result": job.result or {},
        "model_id": job.model_id,
        "configuration_id": job.configuration_id,
    }


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


@router.get("/models")
def list_models(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(ModelArtifact).order_by(ModelArtifact.created_at.desc())
    ).all()
    out = []
    for m in rows:
        configuration = db.get(Configuration, m.configuration_id)
        source = db.get(DataSource, m.data_source_id) if m.data_source_id else None
        out.append(
            {
                "id": m.id,
                "name": m.name,
                "configuration": configuration.name if configuration else None,
                "configuration_id": m.configuration_id,
                "dataset": source.name if source else None,
                "model_type": m.model_type,
                "status": m.status,
                "created_by": m.created_by,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "fidelity": (m.stats or {}).get("report", {}).get("overall_fidelity"),
                "dp_forced": (m.stats or {}).get("dp_forced", False),
            }
        )
    return out


@router.get("/models/{model_id}")
def get_model(model_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    artifact = db.get(ModelArtifact, model_id)
    if artifact is None:
        raise HTTPException(404, "model not found")
    return {
        "id": artifact.id,
        "name": artifact.name,
        "model_type": artifact.model_type,
        "status": artifact.status,
        "artifact_path": artifact.artifact_path,
        "stats": artifact.stats or {},
    }


@router.get("/models/{model_id}/report")
def get_report(model_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    artifact = db.get(ModelArtifact, model_id)
    if artifact is None:
        raise HTTPException(404, "model not found")
    return (artifact.stats or {}).get("report", {})


@router.get("/jobs/{job_id}/preview")
def preview_output(job_id: str, table: str, limit: int = 20, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None or job.job_type != "GENERATE":
        raise HTTPException(404, "generation job not found")
    files = (job.result or {}).get("files") or []
    match = next((f for f in files if f["table"] == table), None)
    if match is None:
        raise HTTPException(404, f"table '{table}' not in output")
    path = Path(match["path"])
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    head = frame.head(limit)
    return {
        "table": table,
        "columns": list(head.columns),
        "rows": head.astype(object).where(pd.notna(head), None).to_dict("records"),
        "total_rows": int(len(frame)),
    }


@router.get("/jobs/{job_id}/download")
def download_output(
    job_id: str,
    table: str | None = None,
    fmt: str = Query("csv", pattern="^(csv|parquet)$"),
    db: Session = Depends(get_db),
) -> Response:
    """Download generated synthetic data, one table or all as a zip."""
    job = db.get(Job, job_id)
    if job is None or job.job_type != "GENERATE":
        raise HTTPException(404, "generation job not found")
    files = (job.result or {}).get("files") or []
    if not files:
        raise HTTPException(404, "this job produced no output")
    if table is not None:
        files = [f for f in files if f["table"] == table]
        if not files:
            raise HTTPException(404, f"table '{table}' not in output")

    frames: list[tuple[str, pd.DataFrame]] = []
    for entry in files:
        path = Path(entry["path"])
        if not path.exists():
            raise HTTPException(410, f"the file for '{entry['table']}' is no longer on disk")
        frames.append((entry["table"], _read_file(path)))

    stem = f"synthetic_{job_id[:8]}"
    if len(frames) == 1:
        return _single_file_response(frames[0][1], f"{stem}_{frames[0][0]}", fmt)
    return _zip_response(frames, stem, fmt)


# --------------------------------------------------------------------------
# admin
# --------------------------------------------------------------------------


@router.get("/admin/model-types")
def list_model_types(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(ModelType)).all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "code": m.code,
            "description": m.description,
            "enabled": m.enabled,
            "show_progress": m.show_progress,
            "quick_train": m.quick_train,
            "default_gpu": m.default_gpu,
            "parquet_dataset": m.parquet_dataset,
            "is_beta": m.is_beta,
            "multi_table": m.multi_table,
            "licence_note": m.licence_note,
        }
        for m in rows
    ]


class ModelTypeUpdate(BaseModel):
    enabled: bool | None = None
    show_progress: bool | None = None
    quick_train: bool | None = None
    default_gpu: bool | None = None
    parquet_dataset: bool | None = None
    is_beta: bool | None = None
    multi_table: bool | None = None


@router.patch("/admin/model-types/{model_type_id}")
def update_model_type(
    model_type_id: str, payload: ModelTypeUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    model_type = db.get(ModelType, model_type_id)
    if model_type is None:
        raise HTTPException(404, "model type not found")
    fields = payload.model_dump(exclude_unset=True)
    if fields.get("enabled") and model_type.licence_note.startswith("NOT ENABLED"):
        raise HTTPException(
            400,
            "This engine cannot be enabled: " + model_type.licence_note,
        )
    for key, value in fields.items():
        if value is not None:
            setattr(model_type, key, value)
    audit(db, "update", "model_type", model_type_id, fields)
    db.commit()
    return {"id": model_type.id, "updated": list(fields)}


@router.get("/admin/compute")
def get_compute(db: Session = Depends(get_db)) -> dict[str, Any]:
    profiles = db.scalars(select(ComputeProfile)).all()
    nodes = db.scalars(select(ComputeNode)).all()
    setting = db.get(OrgSetting, "compute")
    return {
        "profiles": [
            {
                "id": p.id,
                "name": p.name,
                "node_name": p.node_name,
                "vcpu": p.vcpu,
                "ram_gb": p.ram_gb,
                "gpu": p.gpu,
            }
            for p in profiles
        ],
        "nodes": [
            {
                "id": n.id,
                "name": n.name,
                "max_vcpu": n.max_vcpu,
                "max_ram_gb": n.max_ram_gb,
                "max_gpu": n.max_gpu,
            }
            for n in nodes
        ],
        "settings": setting.value if setting else {},
    }


class ComputeProfileUpdate(BaseModel):
    vcpu: int | None = None
    ram_gb: int | None = None
    gpu: int | None = None


@router.patch("/admin/compute/profiles/{profile_id}")
def update_profile(
    profile_id: str, payload: ComputeProfileUpdate, db: Session = Depends(get_db)
) -> dict[str, Any]:
    profile = db.get(ComputeProfile, profile_id)
    if profile is None:
        raise HTTPException(404, "profile not found")
    node = db.scalar(select(ComputeNode).where(ComputeNode.name == profile.node_name))
    fields = payload.model_dump(exclude_unset=True)

    # Enforce the node ceiling, as the Compute Management screen does.
    if node:
        if fields.get("vcpu") and fields["vcpu"] > node.max_vcpu:
            raise HTTPException(400, f"vCPU exceeds node limit ({node.max_vcpu})")
        if fields.get("ram_gb") and fields["ram_gb"] > node.max_ram_gb:
            raise HTTPException(400, f"RAM exceeds node limit ({node.max_ram_gb} GB)")
        if fields.get("gpu") and fields["gpu"] > node.max_gpu:
            raise HTTPException(400, f"GPU exceeds node limit ({node.max_gpu})")

    for key, value in fields.items():
        if value is not None:
            setattr(profile, key, value)
    audit(db, "update", "compute_profile", profile_id, fields)
    db.commit()
    return {"id": profile.id, "updated": list(fields)}


@router.get("/admin/settings/{key}")
def get_setting(key: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    setting = db.get(OrgSetting, key)
    return setting.value if setting else {}


@router.put("/admin/settings/{key}")
def put_setting(key: str, value: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
    setting = db.get(OrgSetting, key)
    if setting is None:
        setting = OrgSetting(key=key, value=value)
        db.add(setting)
    else:
        setting.value = value
    audit(db, "update", "setting", key, value)
    db.commit()
    return value


@router.get("/admin/audit-logs")
def audit_logs(limit: int = 100, db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)).all()
    return [
        {
            "id": a.id,
            "actor": a.actor,
            "action": a.action,
            "entity_type": a.entity_type,
            "entity_id": a.entity_id,
            "detail": a.detail,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in rows
    ]


@router.get("/admin/atlas/status")
def atlas_status() -> dict[str, Any]:
    client = AtlasClient()
    health = client.ping()
    return {
        "enabled": settings.atlas_enabled,
        "base_url": settings.atlas_base_url,
        **health,
    }


# --------------------------------------------------------------------------
# projects
# --------------------------------------------------------------------------


@router.get("/projects")
def list_projects(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    rows = db.scalars(select(Project)).all()
    return [
        {"id": p.id, "name": p.name, "description": p.description} for p in rows
    ]


@router.get("/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """A project and everything assigned to it."""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")

    sources = db.scalars(
        select(DataSource).where(DataSource.project_id == project_id)
    ).all()
    configurations = db.scalars(
        select(Configuration)
        .where(Configuration.project_id == project_id)
        .order_by(Configuration.updated_at.desc())
    ).all()
    models = db.scalars(
        select(ModelArtifact)
        .where(ModelArtifact.project_id == project_id)
        .order_by(ModelArtifact.created_at.desc())
    ).all()
    configuration_names = {c.id: c.name for c in db.scalars(select(Configuration)).all()}

    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "data_sources": [
            {
                "id": s.id,
                "name": s.name,
                "tables": [t["name"] for t in (s.tables or {}).get("tables", [])],
                "rows": sum(t.get("rows", 0) for t in (s.tables or {}).get("tables", [])),
                "atlas_source": (s.atlas_metadata or {}).get("source"),
            }
            for s in sources
        ],
        "configurations": [
            {
                "id": c.id,
                "name": c.name,
                "model_type": c.model_type,
                "status": c.status,
                "tables": len(c.selected_tables or []),
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            }
            for c in configurations
        ],
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "model_type": m.model_type,
                "status": m.status,
                "configuration": configuration_names.get(m.configuration_id),
                "fidelity": (m.stats or {}).get("report", {}).get("overall_fidelity"),
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in models
        ],
    }


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


@router.post("/projects")
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> dict[str, Any]:
    org = current_org(db)
    project = Project(org_id=org.id, name=payload.name, description=payload.description)
    db.add(project)
    db.flush()
    audit(db, "create", "project", project.id, {"name": payload.name})
    db.commit()
    return {"id": project.id, "name": project.name}
