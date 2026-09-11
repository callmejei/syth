"""Fidelity and privacy report comparing real vs synthetic data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _tvd(real: pd.Series, synth: pd.Series) -> float:
    real_dist = real.value_counts(normalize=True)
    synth_dist = synth.value_counts(normalize=True)
    left, right = real_dist.align(synth_dist, fill_value=0.0)
    return float(np.abs(left - right).sum() / 2)


def _numeric_stats(real: pd.Series, synth: pd.Series) -> dict[str, Any]:
    real_values = pd.to_numeric(real, errors="coerce").dropna()
    synth_values = pd.to_numeric(synth, errors="coerce").dropna()
    if real_values.empty or synth_values.empty:
        return {"comparable": False}

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    real_q = real_values.quantile(quantiles).to_numpy()
    synth_q = synth_values.quantile(quantiles).to_numpy()
    spread = max(real_values.std(), 1e-9)
    # Mean absolute quantile gap, scaled by the real spread: 0 is perfect.
    divergence = float(np.mean(np.abs(real_q - synth_q)) / spread)

    return {
        "comparable": True,
        "real_mean": float(real_values.mean()),
        "synth_mean": float(synth_values.mean()),
        "real_median": float(real_values.median()),
        "synth_median": float(synth_values.median()),
        "real_std": float(real_values.std()),
        "synth_std": float(synth_values.std()),
        "quantile_divergence": divergence,
        "score": float(max(0.0, 1.0 - divergence)),
    }


def column_report(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    protected: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Per-column comparison.

    Protected columns (suppressed, pseudonymised, or surrogate keys) are
    reported but deliberately excluded from the fidelity score: for those, a
    total mismatch with the source is the intended outcome, and scoring them as
    failures would penalise the tool for doing its job.
    """
    protected = protected or set()
    rows: list[dict[str, Any]] = []

    for column in real.columns:
        if column not in synth.columns:
            rows.append({"column": column, "status": "missing_in_synthetic"})
            continue

        real_series, synth_series = real[column], synth[column]
        entry: dict[str, Any] = {
            "column": column,
            "real_nulls": float(real_series.isna().mean()),
            "synth_nulls": float(synth_series.isna().mean()),
        }

        if column in protected:
            # Report the overlap as a privacy signal instead of a fidelity one.
            real_values = set(real_series.dropna().astype(str))
            synth_values = set(synth_series.dropna().astype(str))
            overlap = len(real_values & synth_values)
            entry.update(
                {
                    "kind": "protected",
                    "scored": False,
                    "value_overlap": overlap,
                    "value_overlap_rate": overlap / max(len(synth_values), 1),
                    "note": (
                        "protected column: pseudonymised or suppressed by policy, "
                        "excluded from fidelity scoring. Overlap should be 0."
                    ),
                }
            )
            rows.append(entry)
            continue

        entry["scored"] = True

        is_datetime = pd.api.types.is_datetime64_any_dtype(
            real_series
        ) or pd.api.types.is_timedelta64_dtype(real_series)

        if is_datetime:
            # Compare as distributions over time, not as exact-string categories:
            # every timestamp is unique, so a categorical comparison always
            # reports total divergence regardless of model quality.
            entry["kind"] = "datetime"
            real_numeric = _to_epoch(real_series)
            synth_numeric = _to_epoch(synth_series)
            entry.update(_numeric_stats(real_numeric, synth_numeric))
            for key in ("real_mean", "synth_mean", "real_median", "synth_median"):
                if key in entry and pd.notna(entry[key]):
                    entry[key] = _from_epoch(entry[key], real_series)
        elif pd.api.types.is_numeric_dtype(real_series) and not pd.api.types.is_bool_dtype(
            real_series
        ):
            entry["kind"] = "numerical"
            entry.update(_numeric_stats(real_series, synth_series))
        else:
            entry["kind"] = "categorical"
            real_cardinality = int(real_series.nunique())
            # A column with as many distinct values as rows is an identifier in
            # all but name; comparing category sets is meaningless.
            if real_cardinality > 0.5 * max(len(real_series), 1):
                entry.update(
                    {
                        "kind": "high_cardinality",
                        "scored": False,
                        "real_cardinality": real_cardinality,
                        "synth_cardinality": int(synth_series.nunique()),
                        "note": (
                            "near-unique column treated as an identifier; "
                            "excluded from fidelity scoring"
                        ),
                    }
                )
                rows.append(entry)
                continue

            tvd = _tvd(real_series.dropna().astype(str), synth_series.dropna().astype(str))
            entry["tvd"] = tvd
            entry["score"] = float(max(0.0, 1.0 - tvd))
            entry["real_cardinality"] = real_cardinality
            entry["synth_cardinality"] = int(synth_series.nunique())

        rows.append(entry)
    return rows


def _to_epoch(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()
    return pd.to_datetime(series, errors="coerce").astype("int64").where(
        pd.to_datetime(series, errors="coerce").notna()
    ) / 1e9


def _from_epoch(value: float, like: pd.Series) -> Any:
    try:
        if pd.api.types.is_timedelta64_dtype(like):
            return str(pd.to_timedelta(value, unit="s"))
        return str(pd.to_datetime(value, unit="s"))
    except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
        return value


def correlation_delta(real: pd.DataFrame, synth: pd.DataFrame) -> float | None:
    """Mean absolute difference between the two correlation matrices."""
    numeric = [
        c for c in real.columns
        if c in synth.columns
        and pd.api.types.is_numeric_dtype(real[c])
        and pd.api.types.is_numeric_dtype(synth[c])
    ]
    if len(numeric) < 2:
        return None
    real_corr = real[numeric].corr(numeric_only=True).to_numpy()
    synth_corr = synth[numeric].corr(numeric_only=True).to_numpy()
    mask = ~(np.isnan(real_corr) | np.isnan(synth_corr))
    if not mask.any():
        return None
    return float(np.abs(real_corr[mask] - synth_corr[mask]).mean())


def privacy_checks(
    real: pd.DataFrame, synth: pd.DataFrame, suppressed: list[str] | None = None
) -> dict[str, Any]:
    """Cheap, explainable privacy floors. Not a formal guarantee."""
    suppressed = suppressed or []
    comparable = [
        c for c in real.columns
        if c in synth.columns and not pd.api.types.is_datetime64_any_dtype(real[c])
    ][:8]

    exact_match_rate = 0.0
    if comparable:
        real_rows = set(map(tuple, real[comparable].astype(str).to_numpy()))
        synth_rows = list(map(tuple, synth[comparable].astype(str).to_numpy()))
        matches = sum(1 for row in synth_rows if row in real_rows)
        exact_match_rate = matches / max(len(synth_rows), 1)

    return {
        "exact_row_match_rate": exact_match_rate,
        "suppressed_columns": suppressed,
        "note": (
            "Exact-match rate is a sanity floor, not a privacy guarantee. It "
            "does not measure membership inference or attribute disclosure "
            "risk. Treat a non-zero rate as a red flag, not a zero rate as "
            "proof of safety."
        ),
    }


def sanitize(value: Any) -> Any:
    """Make a structure safe to store as JSON.

    NaN and Infinity are valid Python floats and valid in Python's json module
    by default, but they are NOT valid JSON, and Postgres rejects them outright.
    A single NaN buried in a report is enough to fail the whole insert, so
    everything is normalised on the way out.
    """
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def build_report(
    real_tables: dict[str, pd.DataFrame],
    synth_tables: dict[str, pd.DataFrame],
    *,
    policy: dict[str, Any] | None = None,
    privacy: dict[str, Any] | None = None,
    train_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tables: dict[str, Any] = {}
    scores: list[float] = []

    for name, real in real_tables.items():
        synth = synth_tables.get(name)
        if synth is None:
            continue

        suppressed: list[str] = []
        protected: set[str] = set()
        if policy:
            table_policy = (policy.get("tables") or {}).get(name) or {}
            suppressed = table_policy.get("suppressed", [])
            protected = set(table_policy.get("pseudonymised", [])) | set(suppressed)

        columns = column_report(real, synth, protected=protected)
        column_scores = [
            c["score"]
            for c in columns
            if c.get("scored") and isinstance(c.get("score"), float)
        ]
        table_score = float(np.mean(column_scores)) if column_scores else 0.0
        scores.append(table_score)

        leaked = [
            c["column"]
            for c in columns
            if c.get("kind") == "protected" and c.get("value_overlap", 0) > 0
        ]

        tables[name] = {
            "real_rows": int(len(real)),
            "synth_rows": int(len(synth)),
            "fidelity_score": table_score,
            "scored_columns": len(column_scores),
            "protected_columns": sorted(protected),
            "protected_columns_leaked": leaked,
            "correlation_delta": correlation_delta(real, synth),
            "columns": columns,
            "privacy": privacy_checks(real, synth, suppressed),
            "train_stats": (train_stats or {}).get(name, {}),
        }

    warnings: list[str] = []
    for name, stats in (train_stats or {}).items():
        if stats.get("warning"):
            warnings.append(f"{name}: {stats['warning']}")
    for name, table in tables.items():
        if table["protected_columns_leaked"]:
            warnings.append(
                f"{name}: protected columns reproduced real values "
                f"({', '.join(table['protected_columns_leaked'])}). "
                "Investigate before releasing this dataset."
            )
        if table["scored_columns"] == 0:
            warnings.append(
                f"{name}: no columns were eligible for fidelity scoring "
                "(all protected or near-unique)."
            )

    dp_enabled = False
    conditional_tables: list[str] = []
    if privacy:
        for table_name, entry in privacy.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("enabled"):
                dp_enabled = True
                if not entry.get("domain_is_public", False):
                    conditional_tables.append(table_name)

    if dp_enabled and conditional_tables:
        warnings.append(
            "Differential privacy guarantee is CONDITIONAL for "
            f"{', '.join(conditional_tables)}: column ranges or category lists "
            "were measured from the data rather than declared, so the released "
            "domain is not covered by the epsilon budget. Declare them (from "
            "Atlas or the configuration) for an unconditional guarantee."
        )

    return {
        "overall_fidelity": float(np.mean(scores)) if scores else 0.0,
        "tables": tables,
        "policy": policy or {},
        "differential_privacy": {
            "enabled": dp_enabled,
            "unconditional": dp_enabled and not conditional_tables,
            "conditional_tables": conditional_tables,
            "detail": privacy or {},
            "caveat": (
                ""
                if not dp_enabled
                else (
                    "Pure epsilon-DP (Laplace, delta = 0). The budget covers leaf "
                    "parameters, sum weights, node-size decisions, independence "
                    "tests and row clustering."
                    + (
                        " Column domains were declared, so the guarantee is "
                        "unconditional."
                        if not conditional_tables
                        else " Column domains for "
                        f"{', '.join(conditional_tables)} were measured from the "
                        "data, so for those tables the guarantee is conditional "
                        "on the domain being public knowledge."
                    )
                )
            ),
        },
        "warnings": warnings,
    }


__all__ = ["build_report", "column_report", "correlation_delta", "privacy_checks"]
