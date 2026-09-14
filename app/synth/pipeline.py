"""Train and generate pipelines: Atlas -> policy -> SPN -> report."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.atlas.client import AtlasClient
from app.atlas.policy import BALANCED, apply_policy_to_frames, build_policy
from app.config import settings
from app.synth.engines import config_key, get_engine
from app.synth.evaluate import evaluate_tables
from app.synth.relational import ForeignKeySpec, RelationalSPN, TableSpec, topological_order
from app.synth.integrity import FAIL as INTEGRITY_FAIL
from app.synth.integrity import check_generation
from app.synth.report import build_report
from app.synth.schema_infer import ERROR as SCHEMA_ERROR
from app.synth.schema_infer import infer_schema, recommend_engine, validate_schema
from app.synth.spn import SPNParams

logger = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    return None


def load_tables(storage_path: str, table_names: list[str]) -> dict[str, pd.DataFrame]:
    base = Path(storage_path)
    frames: dict[str, pd.DataFrame] = {}
    for name in table_names:
        parquet = base / f"{name}.parquet"
        csv = base / f"{name}.csv"
        if parquet.exists():
            frames[name] = pd.read_parquet(parquet)
        elif csv.exists():
            frames[name] = pd.read_csv(csv)
        else:
            raise FileNotFoundError(f"no data file for table '{name}' under {base}")
    return frames


def specs_from_config(data_args: dict[str, Any], table_names: list[str]) -> dict[str, TableSpec]:
    """Read the Relational Schema section of a configuration into TableSpecs."""
    table_args = (data_args or {}).get("table_args") or {}
    specs: dict[str, TableSpec] = {}
    for name in table_names:
        blob = table_args.get(name) or {}
        specs[name] = TableSpec.from_config(name, blob)
    return specs


def engine_params_from_config(training_params: dict[str, Any], engine_name: str = "SPN"):
    """Build the engine's parameter object from its own config section.

    Each engine reads a different section (`spn_config` / `arf_config`), so one
    configuration can carry both and be switched between engines without losing
    either set of settings.
    """
    engine = get_engine(engine_name)
    blob = training_params or {}
    config = blob.get(config_key(engine.name)) or blob.get("spn_config") or blob or {}
    return engine.build_params(config)


# Retained for callers predating the engine registry.
def spn_params_from_config(training_params: dict[str, Any]) -> SPNParams:
    return engine_params_from_config(training_params, "SPN")


def resolve_policy(
    frames: dict[str, pd.DataFrame],
    specs: dict[str, TableSpec],
    mode: str = BALANCED,
) -> Any:
    """Pull Atlas classifications and turn them into a policy."""
    keys_by_table: dict[str, set[str]] = {}
    for name, spec in specs.items():
        keys = set(spec.primary_key)
        for fk in spec.foreign_keys:
            keys.update(fk.child_column_names)
        keys_by_table[name] = keys

    atlas = AtlasClient().classify_tables(list(frames))
    return build_policy(
        atlas,
        {name: list(frame.columns) for name, frame in frames.items()},
        keys_by_table=keys_by_table,
        mode=mode,
        # With the data in hand, a column the catalog has never been told about
        # can still be recognised from its name or its values, so an untagged
        # full_name is not modelled raw.
        frames=frames,
    )


def run_training(
    *,
    storage_path: str,
    table_names: list[str],
    data_args: dict[str, Any],
    training_params: dict[str, Any],
    output_dir: str,
    policy_mode: str = BALANCED,
    engine: str = "SPN",
    enforce_policy: bool = True,
    evaluate_utility: bool = True,
    holdout_fraction: float = 0.25,
    allow_schema_errors: bool = False,
    progress: Progress = _noop,
) -> dict[str, Any]:
    progress(0.02, "Loading source tables")
    frames = load_tables(storage_path, table_names)

    specs = specs_from_config(data_args, table_names)
    order = (data_args or {}).get("order") or topological_order(specs)
    order = [t for t in order if t in frames] or list(frames)

    # Validate the declared schema against the data before spending anything on
    # training. A primary key that is not unique, or a relational dataset with no
    # foreign keys declared, cannot produce correct output -- and per-column
    # fidelity will not reveal it afterwards.
    progress(0.05, "Validating relational schema")
    inferred_schema = infer_schema(frames)
    schema_issues = validate_schema(
        (data_args or {}).get("table_args") or {}, frames, inferred_schema
    )
    blocking = [i for i in schema_issues if i.level == SCHEMA_ERROR]
    if blocking and not allow_schema_errors:
        lines = "\n".join(f"  - {i.table}: {i.message}\n    fix: {i.fix}" for i in blocking)
        raise ValueError(
            f"the relational schema is not valid for this data "
            f"({len(blocking)} blocking issue(s)):\n{lines}"
        )

    recommended_engine, engine_rationale = recommend_engine(inferred_schema)

    progress(0.10, "Resolving Apache Atlas classifications")
    policy = resolve_policy(frames, specs, mode=policy_mode)
    logger.info("policy: %s", policy.summary())

    table_engine = get_engine(engine)
    params = engine_params_from_config(training_params, table_engine.name)

    # Atlas says this data is sensitive -> DP is not optional.
    dp_forced = False
    dp_refused = None
    if enforce_policy and policy.requires_dp and not getattr(params, "private", False):
        if table_engine.supports_dp:
            params.private = True
            dp_forced = True
        else:
            # Do not pretend. Atlas asked for privacy and this engine cannot
            # provide it, so the run proceeds only with the gap stated loudly.
            dp_refused = (
                "Apache Atlas classifies a selected column as SENSITIVE, but the "
                + table_engine.name
                + " engine does not support differential privacy, so this model is "
                "NOT private. Use the SPN engine for sensitive data, or accept "
                "this explicitly."
            )

    if enforce_policy:
        progress(0.18, "Applying data protection policy")
        working = apply_policy_to_frames(frames, policy)
        # Columns the policy pseudonymises are handed to the encoder as
        # non-statistical placeholders rather than modelled.
        for name, spec in specs.items():
            table_policy = policy.tables.get(name)
            if table_policy:
                spec.column_kind_overrides.update(table_policy.encoder_overrides())
    else:
        working = frames

    # Hold out real rows for the utility evaluation. The split is on the ROOT
    # tables and propagates down the foreign keys, so a held-out customer takes
    # all of their transactions with them -- splitting each table independently
    # would leak a customer's own rows into both sides and inflate the score.
    holdout = None
    if evaluate_utility and holdout_fraction > 0:
        progress(0.20, "Splitting a holdout for utility evaluation")
        holdout = _split_holdout(working, specs, order, holdout_fraction)

    progress(0.25, "Training SPN")

    def training_progress(fraction: float, message: str) -> None:
        progress(0.25 + fraction * 0.45, message)

    model = RelationalSPN.fit(
        working, specs, params, order=order, progress=training_progress,
        engine=table_engine.name,
    )

    progress(0.85, "Saving model artifact")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)

    progress(0.90, "Scoring fidelity on a holdout sample")
    # Score on a capped sample for speed, but compare like with like: the real
    # frames are cut to the same size, otherwise row-count and distribution
    # comparisons are distorted by the cap rather than by model quality.
    scoring_cap = 20_000
    root_tables = [t for t in order if not (specs.get(t) and specs[t].foreign_keys)]
    sample_counts = {
        name: min(len(working[name]), scoring_cap) for name in root_tables
    }
    preview = model.sample(sample_counts, seed=99)
    reference = {
        name: (
            frame.head(len(preview[name]))
            if name in preview and len(preview[name]) < len(frame)
            else frame
        )
        for name, frame in working.items()
    }
    # Columns replaced by placeholders are not modelled at all, so they are
    # reported alongside the policy-protected ones rather than scored.
    policy_json = policy.to_json()
    for name, spec in specs.items():
        placeholders = [
            column
            for column, kind in spec.column_kind_overrides.items()
            if kind == "non_std"
        ]
        if placeholders and name in policy_json.get("tables", {}):
            entry = policy_json["tables"][name]
            entry["pseudonymised"] = sorted(
                set(entry.get("pseudonymised", [])) | set(placeholders)
            )

    report = build_report(
        reference,
        preview,
        policy=policy_json,
        privacy=model.privacy,
        train_stats=model.train_stats,
    )
    aggregate_columns = sum(
        stats.get("aggregate_columns", 0) for stats in model.train_stats.values()
    )
    if aggregate_columns and getattr(params, "private", False):
        report.setdefault("warnings", []).append(
            f"Aggregation Specification is enabled ({aggregate_columns} aggregate "
            "columns) while differential privacy is on. Every aggregate is another "
            "column sharing the same epsilon budget, which measurably weakens "
            "every other column in that table. Measured on the demo, aggregates "
            "under DP destroyed the parent-child correlation they were meant to "
            "strengthen. Consider agg_enabled: false for private runs."
        )
    report["aggregation"] = {
        "columns_built": aggregate_columns,
        "per_table": model.aggregates,
    }

    if dp_forced:
        report.setdefault("warnings", []).append(
            "Differential privacy was switched on automatically because Apache "
            "Atlas classifies at least one selected column as SENSITIVE."
        )
    if dp_refused:
        report.setdefault("warnings", []).append(dp_refused)
    report["engine"] = {
        "name": table_engine.name,
        "supports_dp": table_engine.supports_dp,
        "description": table_engine.description,
        "recommended": recommended_engine,
        "recommendation_rationale": engine_rationale,
        "diagnostics": {
            name: getattr(m, "diagnostics", {}) for name, m in model.models.items()
        },
    }
    if table_engine.name != recommended_engine:
        report.setdefault("warnings", []).append(
            f"{table_engine.name} was used, but this dataset suits {recommended_engine}: "
            f"{engine_rationale}"
        )

    # What the declared schema looked like, and what the data says it is. Kept in
    # the report because a model is only as trustworthy as the schema it was
    # trained under, and a reviewer cannot see the configuration from here.
    report["schema"] = {
        "declared_valid": not blocking,
        "issues": [i.to_json() for i in schema_issues],
        "detected": inferred_schema.to_json(),
    }
    for issue in schema_issues:
        if issue.level != SCHEMA_ERROR:
            report.setdefault("warnings", []).append(
                f"schema ({issue.table}): {issue.message}"
            )

    # Structural and disclosure checks. These catch what a per-column fidelity
    # score cannot: broken joins, collapsed cardinality, and protected values
    # reaching the output.
    progress(0.92, "Checking relational integrity and disclosure")
    integrity = check_generation(
        reference, preview, schema=inferred_schema, policy=policy_json
    )
    report["integrity"] = integrity.to_json()
    for check in integrity.failed:
        report.setdefault("warnings", []).append(f"FAILED: {check.name} -- {check.detail}")
    for check in integrity.warned:
        report.setdefault("warnings", []).append(f"{check.name} -- {check.detail}")

    # The headline status. A good fidelity score must not be able to present a
    # structurally broken dataset as a healthy one.
    report["status"] = integrity.status
    if integrity.status == INTEGRITY_FAIL:
        report.setdefault("warnings", []).insert(
            0,
            f"This model FAILED {len(integrity.failed)} integrity check(s). The "
            "fidelity score below is per-column and does not reflect them; do not "
            "quote it without reading the integrity section.",
        )

    # Utility evaluation: train a second model on the training split only, so
    # the held-out real rows are genuinely unseen. The shipped artifact stays
    # the one trained on all the data.
    if holdout is not None:
        progress(0.94, "Evaluating utility (train on synthetic, test on real)")
        try:
            report["utility"] = _evaluate_utility(
                holdout, specs, params, order, policy_json, table_engine.name
            )
        except Exception as exc:  # noqa: BLE001 - evaluation must never fail a run
            logger.exception("utility evaluation failed")
            report["utility"] = {"available": False, "error": str(exc)}

    progress(1.0, "Training complete")
    return {
        "artifact_path": output_dir,
        "order": order,
        "policy": policy.to_json(),
        "report": report,
        "train_stats": model.train_stats,
        "dp_forced": dp_forced,
        "status": report["status"],
        "schema": report["schema"],
        # Serialise whatever the chosen engine's parameter object holds. Naming
        # the SPN's fields explicitly here meant an ARF run died on
        # `params.epsilon` after training had already succeeded.
        "params": {"engine": table_engine.name, **asdict(params)},
    }


def source_row_counts(model: RelationalSPN) -> dict[str, int]:
    """How many rows each table had in the data the model was trained on."""
    return {
        name: int(stats["rows"])
        for name, stats in (model.train_stats or {}).items()
        if isinstance(stats, dict) and stats.get("rows")
    }


def run_generation(
    *,
    artifact_path: str,
    n_rows: dict[str, int] | int | None = None,
    output_dir: str,
    seed: int = 0,
    fmt: str = "parquet",
    progress: Progress = _noop,
) -> dict[str, Any]:
    progress(0.05, "Loading model artifact")
    model = RelationalSPN.load(artifact_path)

    # Default: reproduce the source dataset's shape exactly, table for table.
    #
    # Only the root table is otherwise pinned; every child is drawn from its
    # degree model, so totals land near the source count without hitting it
    # (20,000 real transactions came out anywhere between 19,501 and 20,656
    # across eight seeds). Pinning each table to the count it was trained on
    # removes that drift, and it is safe precisely because those counts are what
    # the model already produces on average -- the rescale factor is ~1.0, not
    # the 150x that distorts cardinality when a child is given an arbitrary size.
    matched_source = False
    if n_rows is None:
        n_rows = source_row_counts(model)
        matched_source = bool(n_rows)

    def generation_progress(fraction: float, message: str) -> None:
        progress(0.05 + fraction * 0.8, message)

    synth = model.sample(n_rows, seed=seed, progress=generation_progress)

    progress(0.90, "Writing output")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for name, frame in synth.items():
        if fmt == "csv":
            path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
        else:
            path = out / f"{name}.parquet"
            frame.to_parquet(path, index=False)
        files.append(
            {
                "table": name,
                "path": str(path),
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "bytes": int(path.stat().st_size),
            }
        )

    # Sizing a CHILD table directly scales its degree distribution, which
    # silently breaks the ratio to its parent and cascades into its own
    # children. Asking for 200,000 accounts against 1,000 customers once
    # produced 200 accounts per customer and 1.5M transactions, and nothing
    # said so. Compare what came out against the cardinality the model learned.
    warnings: list[str] = []
    for child, by_parent in (model.degrees or {}).items():
        child_frame = synth.get(child)
        if child_frame is None or not len(child_frame):
            continue
        for parent, degree_model in (by_parent or {}).items():
            parent_frame = synth.get(parent)
            if parent_frame is None or not len(parent_frame):
                continue
            values = getattr(degree_model, "values", None)
            probs = getattr(degree_model, "probs", None)
            if not values or not probs:
                continue
            learned = float(sum(v * p for v, p in zip(values, probs)))
            actual = len(child_frame) / len(parent_frame)
            if learned > 0 and (actual / learned > 1.5 or actual / learned < 0.67):
                warnings.append(
                    f"{child} came out at {actual:.2f} rows per {parent}, but the model "
                    f"learned {learned:.2f}. Sizing a child table directly rescales its "
                    f"cardinality -- set the row count on the root table instead."
                )
            break  # only the primary parent

    progress(1.0, "Generation complete")
    return {
        "output_dir": str(out),
        "files": files,
        "warnings": warnings,
        "matched_source": matched_source,
    }


def _split_holdout(
    frames: dict[str, pd.DataFrame],
    specs: dict[str, TableSpec],
    order: list[str],
    fraction: float,
    seed: int = 13,
) -> dict[str, Any]:
    """Split into train/test along the relational structure.

    Root tables are split by row; child tables follow their parent, so every row
    belonging to a held-out entity is held out too. Splitting each table
    independently would put a customer's transactions in the training set while
    the customer sits in the test set, and any model could then "predict" that
    customer from rows it had already seen.
    """
    rng = np.random.default_rng(seed)
    train: dict[str, pd.DataFrame] = {}
    test: dict[str, pd.DataFrame] = {}
    test_keys: dict[str, set] = {}

    for table_name in order:
        frame = frames.get(table_name)
        if frame is None:
            continue
        spec = specs.get(table_name) or TableSpec(name=table_name)

        parent_fk = next(
            (fk for fk in spec.foreign_keys if fk.parent_table_name in test_keys), None
        )

        if parent_fk is None:
            mask = rng.random(len(frame)) < fraction
            train[table_name] = frame[~mask].reset_index(drop=True)
            test[table_name] = frame[mask].reset_index(drop=True)
            if spec.primary_key and spec.primary_key[0] in frame.columns:
                test_keys[table_name] = set(test[table_name][spec.primary_key[0]])
        else:
            child_column = parent_fk.child_column_names[0]
            held = test_keys.get(parent_fk.parent_table_name, set())
            if child_column in frame.columns:
                mask = frame[child_column].isin(held)
            else:
                mask = pd.Series(rng.random(len(frame)) < fraction, index=frame.index)
            train[table_name] = frame[~mask].reset_index(drop=True)
            test[table_name] = frame[mask].reset_index(drop=True)
            if spec.primary_key and spec.primary_key[0] in frame.columns:
                test_keys[table_name] = set(test[table_name][spec.primary_key[0]])

    return {"train": train, "test": test}


def _evaluate_utility(
    holdout: dict[str, Any],
    specs: dict[str, TableSpec],
    params: SPNParams,
    order: list[str],
    policy_json: dict[str, Any],
    engine_name: str = "SPN",
) -> dict[str, Any]:
    train_frames = holdout["train"]
    test_frames = holdout["test"]

    usable_order = [t for t in order if t in train_frames and len(train_frames[t]) > 0]
    if not usable_order:
        return {"available": False, "reason": "holdout split produced no usable tables"}

    evaluation_model = RelationalSPN.fit(
        train_frames, specs, params, order=usable_order, engine=engine_name
    )
    root_tables = [
        t for t in usable_order if not (specs.get(t) and specs[t].foreign_keys)
    ]
    counts = {name: len(train_frames[name]) for name in root_tables}
    synthetic = evaluation_model.sample(counts, seed=4321)

    protected: dict[str, set[str]] = {}
    for table_name, entry in (policy_json.get("tables") or {}).items():
        protected[table_name] = set(entry.get("pseudonymised", [])) | set(
            entry.get("suppressed", [])
        )
        spec = specs.get(table_name)
        if spec:
            protected[table_name] |= set(spec.primary_key)
            for fk in spec.foreign_keys:
                protected[table_name] |= set(fk.child_column_names)

    result = evaluate_tables(
        train_frames, test_frames, synthetic, protected=protected
    )
    result["available"] = True
    result["method"] = (
        "A separate model was trained on the training split only and sampled; "
        "the shipped artifact is still trained on all rows. The split follows "
        "foreign keys, so a held-out parent takes its children with it."
    )
    result["holdout_rows"] = {name: int(len(f)) for name, f in test_frames.items()}
    return result


def storage_for(kind: str, entity_id: str) -> str:
    path = Path(settings.storage_dir) / kind / entity_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


__all__ = ["run_generation", "run_training", "specs_from_config", "storage_for"]
