"""Seed reference data matching the Admin screens in the scan."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ComputeNode, ComputeProfile, ModelType, Organization, OrgSetting, Project

DEFAULT_ORG = "Default Synthforge Organization"

# Model types page. Two engines are implemented, one per kind of dataset:
# the SPN for related tables, ARF for standalone ones. Auto-configure detects
# which a dataset is and selects accordingly. The rest are listed as disabled so
# the roadmap is visible in the product, and so the reason SDV-based engines were
# rejected is recorded where someone will actually read it.
MODEL_TYPES = [
    {
        "name": "SPN",
        "code": "SPN",
        "description": (
            "Sum-Product Network. The engine for RELATED tables: children are "
            "sampled conditioned on their parent, so foreign keys and "
            "cardinality survive. The only engine offering differential privacy."
        ),
        "enabled": True,
        "show_progress": True,
        "quick_train": True,
        "default_gpu": False,
        "parquet_dataset": True,
        "is_beta": False,
        "multi_table": True,
        "licence_note": "Apache-2.0 clean: implemented in-house on numpy/scipy/scikit-learn.",
    },
    {
        "name": "ARF",
        "code": "ARF",
        "description": (
            "Adversarial Random Forest. The engine for INDEPENDENT tables with "
            "no foreign keys: built for interactions inside a single wide table, "
            "and it reports its own convergence."
        ),
        "enabled": True,
        "show_progress": True,
        "quick_train": False,
        "default_gpu": False,
        "parquet_dataset": True,
        "is_beta": True,
        "multi_table": False,
        "licence_note": (
            "Apache-2.0 clean: implemented in-house on scikit-learn. NOTE: no "
            "differential privacy -- the engine learns split thresholds, not "
            "counts, so the SPN epsilon accounting does not transfer. Training on "
            "columns Atlas marks SENSITIVE will warn loudly."
        ),
    },
    {
        "name": "CTGAN",
        "code": "CTGAN",
        "description": "Conditional tabular GAN.",
        "enabled": False,
        "show_progress": True,
        "quick_train": False,
        "default_gpu": True,
        "parquet_dataset": True,
        "is_beta": True,
        "multi_table": False,
        "licence_note": (
            "NOT ENABLED. The reference implementations (sdv, ctgan) are under "
            "the Business Source Licence, which requires a commercial licence "
            "for production use. Enabling this would reintroduce the vendor "
            "cost this POC exists to avoid."
        ),
    },
    {
        "name": "TVAE",
        "code": "TVAE",
        "description": "Tabular variational autoencoder.",
        "enabled": False,
        "show_progress": True,
        "quick_train": False,
        "default_gpu": True,
        "parquet_dataset": True,
        "is_beta": True,
        "multi_table": False,
        "licence_note": "NOT ENABLED. Same Business Source Licence constraint as CTGAN.",
    },
]

COMPUTE_PROFILES = [
    {"name": "LOW", "vcpu": 14, "ram_gb": 28, "gpu": 0},
    {"name": "MEDIUM", "vcpu": 14, "ram_gb": 28, "gpu": 0},
    {"name": "HIGH", "vcpu": 16, "ram_gb": 32, "gpu": 0},
]

ORG_SETTINGS = [
    {
        "key": "compute",
        "value": {
            "selection_required": True,
            "default_profile": "LOW",
            "user_experience": {"TRAIN": "LOW", "GENERATE": "LOW", "DELTA": "LOW"},
        },
    },
    {
        "key": "governance",
        "value": {
            "atlas_enabled": True,
            "policy_mode": "balanced",
            "enforce_policy": True,
            "block_on_unclassified": False,
        },
    },
]


def seed(db: Session) -> Organization:
    org = db.scalar(select(Organization).where(Organization.name == DEFAULT_ORG))
    if org is None:
        org = Organization(name=DEFAULT_ORG)
        db.add(org)
        db.flush()

    if not db.scalar(select(Project).where(Project.org_id == org.id)):
        db.add(
            Project(
                org_id=org.id,
                name="POC test",
                description="Synthetic data POC evaluation",
            )
        )

    for blob in MODEL_TYPES:
        existing = db.scalar(select(ModelType).where(ModelType.name == blob["name"]))
        if existing is None:
            db.add(ModelType(**blob))
        else:
            existing.licence_note = blob["licence_note"]
            existing.description = blob["description"]

    for blob in COMPUTE_PROFILES:
        if not db.scalar(select(ComputeProfile).where(ComputeProfile.name == blob["name"])):
            db.add(ComputeProfile(**blob))

    if not db.scalar(select(ComputeNode).where(ComputeNode.name == "Node-1")):
        db.add(ComputeNode(name="Node-1", max_vcpu=16, max_ram_gb=32, max_gpu=0))

    for blob in ORG_SETTINGS:
        if not db.get(OrgSetting, blob["key"]):
            db.add(OrgSetting(**blob))

    db.commit()
    return org


__all__ = ["DEFAULT_ORG", "seed"]
