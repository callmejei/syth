"""Persistence model.

Mirrors the object hierarchy visible in the vendor screenshots:
Organization > Project > (DataSource, Configuration) > Model > Job.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    projects: Mapped[list[Project]] = relationship(back_populates="organization")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    organization: Mapped[Organization] = relationship(back_populates="projects")


class DataSource(Base):
    """A dataset: one or more related tables uploaded together.

    `tables` holds [{name, file, rows, columns:[{name, dtype, ...}]}].
    `atlas_metadata` caches the classifications resolved from Apache Atlas.
    """

    __tablename__ = "data_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    version: Mapped[str] = mapped_column(String(64), default="v1")
    storage_path: Mapped[str] = mapped_column(Text, default="")
    tables: Mapped[dict] = mapped_column(JSON, default=dict)
    atlas_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    atlas_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Configuration(Base):
    """The 'Configurations' object: model type + dataset + the full argument tree.

    `data_args` / `training_params` are the values for the parameters defined in
    schema.yaml. We store them sparsely -- only what differs from initialValue --
    exactly like the UI's JSON view.
    """

    __tablename__ = "configurations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    model_type: Mapped[str] = mapped_column(String(64), default="SPN")
    data_source_id: Mapped[str | None] = mapped_column(ForeignKey("data_sources.id"), nullable=True)
    selected_tables: Mapped[list] = mapped_column(JSON, default=list)
    device_type: Mapped[str] = mapped_column(String(16), default="CPU")
    compute_profile: Mapped[str] = mapped_column(String(16), default="LOW")
    vcpu: Mapped[int] = mapped_column(Integer, default=14)
    ram_gb: Mapped[int] = mapped_column(Integer, default=28)
    gpu: Mapped[int] = mapped_column(Integer, default=0)
    data_args: Mapped[dict] = mapped_column(JSON, default=dict)
    training_params: Mapped[dict] = mapped_column(JSON, default=dict)
    target_connection: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="Draft")  # Draft | Ready
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ModelArtifact(Base):
    """A trained model, i.e. a row on the project's Models tab."""

    __tablename__ = "model_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    configuration_id: Mapped[str] = mapped_column(ForeignKey("configurations.id"))
    data_source_id: Mapped[str | None] = mapped_column(ForeignKey("data_sources.id"), nullable=True)
    model_type: Mapped[str] = mapped_column(String(64), default="SPN")
    artifact_path: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="Pending")
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    created_by: Mapped[str] = mapped_column(String(255), default="DataCraft Admin")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Job(Base):
    """Train / Generate / Delta job, as listed under 'My jobs'."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    project_id: Mapped[str | None] = mapped_column(ForeignKey("projects.id"), nullable=True)
    job_type: Mapped[str] = mapped_column(String(16))  # TRAIN | GENERATE | DELTA
    configuration_id: Mapped[str | None] = mapped_column(
        ForeignKey("configurations.id"), nullable=True
    )
    model_id: Mapped[str | None] = mapped_column(ForeignKey("model_artifacts.id"), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    logs: Mapped[list] = mapped_column(JSON, default=list)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    compute_profile: Mapped[str] = mapped_column(String(16), default="LOW")
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ModelType(Base):
    """Admin > Model types. Feature toggles per engine."""

    __tablename__ = "model_types"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    code: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    show_progress: Mapped[bool] = mapped_column(Boolean, default=True)
    quick_train: Mapped[bool] = mapped_column(Boolean, default=False)
    default_gpu: Mapped[bool] = mapped_column(Boolean, default=False)
    parquet_dataset: Mapped[bool] = mapped_column(Boolean, default=True)
    is_beta: Mapped[bool] = mapped_column(Boolean, default=False)
    multi_table: Mapped[bool] = mapped_column(Boolean, default=True)
    licence_note: Mapped[str] = mapped_column(Text, default="")


class ComputeProfile(Base):
    """Admin > Compute settings > Compute Profiles (LOW / MEDIUM / HIGH)."""

    __tablename__ = "compute_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(16), unique=True)
    node_name: Mapped[str] = mapped_column(String(64), default="Node-1")
    vcpu: Mapped[int] = mapped_column(Integer, default=14)
    ram_gb: Mapped[int] = mapped_column(Integer, default=28)
    gpu: Mapped[int] = mapped_column(Integer, default=0)


class ComputeNode(Base):
    """Admin > Compute settings > Compute Management > Compute Profile Nodes."""

    __tablename__ = "compute_nodes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    max_vcpu: Mapped[int] = mapped_column(Integer, default=16)
    max_ram_gb: Mapped[int] = mapped_column(Integer, default=32)
    max_gpu: Mapped[int] = mapped_column(Integer, default=0)


class OrgSetting(Base):
    __tablename__ = "org_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)


class AuditLog(Base):
    """Admin > Audit Logs. Every config/policy change lands here -- this is the
    control your compliance team will actually ask about."""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    actor: Mapped[str] = mapped_column(String(255), default="DataCraft Admin")
    action: Mapped[str] = mapped_column(String(64))
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
