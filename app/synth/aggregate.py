"""Aggregation Specification: summarising a child table onto its parent row.

This implements the `agg_*` parameters from schema.yaml, which were previously
rendered in the UI and ignored by the engine.

Why it matters. A parent row on its own says little about its children: a
customer record does not know that this customer spends heavily, buys in bursts,
or has been winding down. Folding summaries of the child table onto the parent
row before the parent is modelled means the parent's own distribution carries
that shape, so a synthetic "high-value, high-frequency" customer is generated as
a coherent whole rather than as a random parent with random children stapled to
it.

Parameter sources, following the schema's own containment:

    global_default_args          applies to every column
    default_discrete_kwargs      categorical child columns
    default_continuous_kwargs    numeric child columns
    default_ts_numerical_kwargs  numeric columns of a time-series child
    roller_args.temporal         the reduced set used inside a rolling window

Ordering matters for several of these -- first/last values, argmax positions,
end-minus-start, monotonicity -- so when the child table is a time series the
rows are sorted by its `sortby` column within each parent before anything is
computed. Without that, "first value" is whatever order the file happened to be
in.

COST. Every aggregate is another column in the parent's model. Under
differential privacy the per-column budget is the total divided by the column
count, so turning on twenty aggregates measurably weakens every other column in
that table. `max_columns` caps the total and the report says what was dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Prefix marking a generated aggregate. Stripped before output; never a real
# column of the user's data.
AGG_PREFIX = "__agg__"

# Default ceiling on generated aggregate columns per parent table.
DEFAULT_MAX_COLUMNS = 24


@dataclass
class AggregationConfig:
    """The `agg_*` switches, resolved for one child table."""

    # any column type
    include_na_ratio: bool = True
    include_unique_ratio: bool = False
    k_top: int = 0
    k_first: int = 1
    k_last: int = 1

    # categorical
    include_n_unique: bool = False
    proportions_of: list[Any] = field(default_factory=list)

    # numeric
    include_mean: bool = True
    include_std: bool = True
    include_sum: bool = True
    include_log_mean: bool = False
    include_log_std: bool = False
    include_skew: bool = False
    include_kurt: bool = False
    quantiles: list[float] = field(default_factory=list)
    include_end_start_diff: bool = True
    include_increase_ratio: bool = True
    include_decrease_ratio: bool = False
    check_increasing: bool = False
    check_decreasing: bool = False
    first_max_position: bool = False
    first_min_position: bool = False
    last_max_position: bool = False
    last_min_position: bool = False
    autocorr_lags: list[int] = field(default_factory=list)

    max_columns: int = DEFAULT_MAX_COLUMNS

    # Opt-in, and deliberately so.
    #
    # Measured on the customer sample: enabling the schema's own default switches
    # adds 23 columns to the parent model and changes nothing detectable without
    # differential privacy, because the parent->child signal already travels via
    # the per-child context columns and the degree model. With DP at epsilon=2 it
    # is worse than useless -- 36 columns instead of 13 cuts the per-column
    # budget by ~2.8x and the segment->income correlation collapses from
    # 32k-84k to 37k-38k.
    #
    # The parameters are implemented faithfully; they are just not a good default.
    # Turn on with `agg_enabled: true` when you want the parent table to carry
    # child summaries in its own right.
    enabled: bool = False

    @classmethod
    def from_config(cls, blob: dict[str, Any], timeseries: bool = False) -> AggregationConfig:
        """Read the aggregation switches out of a table's configuration.

        Later sources win, matching the schema's global -> per-type layering.
        """
        blob = blob or {}
        merged: dict[str, Any] = {}
        sources = ["global_default_args", "default_discrete_kwargs", "default_continuous_kwargs"]
        if timeseries:
            sources += ["default_ts_numerical_kwargs", "default_ts_mixed_kwargs"]

        for key in sources:
            section = blob.get(key)
            if isinstance(section, dict):
                merged.update(section)
        # A table may also set the switches directly.
        merged.update({k: v for k, v in blob.items() if k.startswith("agg_")})

        def pick(name: str, default):
            # Accept both `agg_include_mean` and the bare `include_mean` used
            # inside roller_args.temporal.
            for candidate in (f"agg_{name}", name):
                if candidate in merged and merged[candidate] is not None:
                    return merged[candidate]
            return default

        config = cls(
            max_columns=int(pick("max_columns", DEFAULT_MAX_COLUMNS)),
            enabled=_as_bool(pick("enabled", False)),
        )
        for name in (
            "include_na_ratio", "include_unique_ratio", "include_n_unique",
            "include_mean", "include_std", "include_sum", "include_log_mean",
            "include_log_std", "include_skew", "include_kurt",
            "include_end_start_diff", "include_increase_ratio",
            "include_decrease_ratio", "check_increasing", "check_decreasing",
            "first_max_position", "first_min_position", "last_max_position",
            "last_min_position",
        ):
            setattr(config, name, _as_bool(pick(name, getattr(config, name))))
        for name in ("k_top", "k_first", "k_last"):
            setattr(config, name, max(0, int(pick(name, getattr(config, name)) or 0)))
        config.quantiles = _as_floats(pick("quantiles", []))
        config.autocorr_lags = [int(x) for x in (_as_floats(pick("autocorr_lags", [])) or [])]
        config.proportions_of = pick("proportions_of", []) or []
        return config

    def to_json(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, list) else v)
            for k, v in self.__dict__.items()
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        # The scanned schema contains "tue", "feise", "felse" for booleans.
        return value.strip().lower() in {"true", "tue", "ture", "yes", "1"}
    return bool(value)


def _as_floats(value: Any) -> list[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


class Aggregator:
    """Computes the configured summaries of a child table per parent row."""

    def __init__(self, config: AggregationConfig, child_name: str):
        self.config = config
        self.child_name = child_name
        self.built: list[str] = []
        self.dropped: list[str] = []

    def column_name(self, column: str, stat: str) -> str:
        return f"{AGG_PREFIX}{self.child_name}__{column}__{stat}"

    # -- the aggregate set for one column --

    def _numeric_aggregates(self, series: pd.Series) -> dict[str, Any]:
        config = self.config
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        out: dict[str, Any] = {}
        if values.size == 0:
            return out

        if config.include_mean:
            out["mean"] = float(np.mean(values))
        if config.include_std:
            out["std"] = float(np.std(values)) if values.size > 1 else 0.0
        if config.include_sum:
            out["sum"] = float(np.sum(values))
        if config.include_log_mean:
            # log1p keeps zero at zero; negatives are clipped rather than
            # producing NaN and poisoning the column.
            out["log_mean"] = float(np.mean(np.log1p(np.clip(values, 0, None))))
        if config.include_log_std:
            logged = np.log1p(np.clip(values, 0, None))
            out["log_std"] = float(np.std(logged)) if values.size > 1 else 0.0
        if config.include_skew and values.size > 2:
            out["skew"] = float(pd.Series(values).skew())
        if config.include_kurt and values.size > 3:
            out["kurt"] = float(pd.Series(values).kurt())
        for q in config.quantiles:
            if 0.0 <= q <= 1.0:
                out[f"q{int(round(q * 100)):02d}"] = float(np.quantile(values, q))

        # Order-dependent features. Meaningful only because the caller sorted
        # the group first.
        if config.include_end_start_diff and values.size >= 2:
            out["end_start_diff"] = float(values[-1] - values[0])
        if values.size >= 2:
            steps = np.diff(values)
            if config.include_increase_ratio:
                out["increase_ratio"] = float(np.mean(steps > 0))
            if config.include_decrease_ratio:
                out["decrease_ratio"] = float(np.mean(steps < 0))
            if config.check_increasing:
                out["is_increasing"] = bool(np.all(steps >= 0))
            if config.check_decreasing:
                out["is_decreasing"] = bool(np.all(steps <= 0))
        if config.first_max_position:
            out["first_max_pos"] = float(np.argmax(values) / max(values.size - 1, 1))
        if config.first_min_position:
            out["first_min_pos"] = float(np.argmin(values) / max(values.size - 1, 1))
        if config.last_max_position:
            out["last_max_pos"] = float(
                (values.size - 1 - np.argmax(values[::-1])) / max(values.size - 1, 1)
            )
        if config.last_min_position:
            out["last_min_pos"] = float(
                (values.size - 1 - np.argmin(values[::-1])) / max(values.size - 1, 1)
            )
        for lag in config.autocorr_lags:
            if 0 < lag < values.size:
                series_values = pd.Series(values)
                value = series_values.autocorr(lag=lag)
                out[f"autocorr{lag}"] = float(value) if pd.notna(value) else 0.0

        if config.k_first:
            for index in range(min(config.k_first, values.size)):
                out[f"first{index + 1}"] = float(values[index])
        if config.k_last:
            for index in range(min(config.k_last, values.size)):
                out[f"last{index + 1}"] = float(values[-(index + 1)])

        return out

    def _categorical_aggregates(self, series: pd.Series) -> dict[str, Any]:
        config = self.config
        values = series.dropna()
        out: dict[str, Any] = {}
        if values.empty:
            return out

        if config.include_n_unique:
            out["n_unique"] = int(values.nunique())
        if config.k_top:
            counts = values.value_counts()
            for index in range(min(config.k_top, len(counts))):
                out[f"top{index + 1}"] = counts.index[index]
        if config.k_first:
            for index in range(min(config.k_first, len(values))):
                out[f"first{index + 1}"] = values.iloc[index]
        if config.k_last:
            for index in range(min(config.k_last, len(values))):
                out[f"last{index + 1}"] = values.iloc[-(index + 1)]
        for category in config.proportions_of:
            out[f"prop_{category}"] = float((values == category).mean())
        return out

    def _shared_aggregates(self, series: pd.Series) -> dict[str, Any]:
        config = self.config
        out: dict[str, Any] = {}
        if config.include_na_ratio:
            out["na_ratio"] = float(series.isna().mean())
        if config.include_unique_ratio and len(series):
            out["unique_ratio"] = float(series.nunique(dropna=True) / len(series))
        return out

    # -- main entry point --

    def build(
        self,
        child_frame: pd.DataFrame,
        parent_frame: pd.DataFrame,
        child_key: list[str],
        parent_key: list[str],
        sortby: list[str] | None = None,
        exclude_columns: set[str] | None = None,
    ) -> pd.DataFrame:
        """One row per parent row, columns = configured aggregates.

        Parents with no children get 0 for counts and NaN elsewhere, which the
        encoder then models as a genuine null rather than a fabricated zero.
        """
        exclude = set(exclude_columns or set()) | set(child_key)
        empty = pd.DataFrame(index=parent_frame.index)

        if not self.config.enabled:
            # Row count still reaches the parent via the degree column; nothing
            # else is generated.
            return empty

        if not child_key or not parent_key:
            return empty
        if not all(c in child_frame.columns for c in child_key):
            return empty
        if not all(c in parent_frame.columns for c in parent_key):
            return empty

        working = child_frame
        if sortby:
            order = [c for c in sortby if c in working.columns]
            if order:
                working = working.sort_values(child_key + order)

        candidate_columns = [
            c for c in working.columns
            if c not in exclude and not c.startswith(("__agg__", "__ctx__", "__deg__"))
        ]

        # Pre-convert datetimes to epoch seconds: a mean of timestamps is not
        # something pandas gives us directly, and doing it per group would
        # repeat the parse for every parent.
        numeric_view: dict[str, pd.Series] = {}
        kinds: dict[str, str] = {}
        for column in candidate_columns:
            series = working[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                numeric_view[column] = (
                    pd.to_datetime(series, errors="coerce").astype("int64") / 1e9
                )
                kinds[column] = "numeric"
            elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                numeric_view[column] = series
                kinds[column] = "numeric"
            else:
                kinds[column] = "categorical"

        # One pass over the groups, building every statistic for every column.
        # Grouping once per column and letting pandas expand the returned dicts
        # does not work: it collapses them into a MultiIndex Series, so the stat
        # names are lost and every column arrives named "0".
        records: dict[Any, dict[str, Any]] = {}
        for key, group in working.groupby(child_key[0], sort=False):
            row: dict[str, Any] = {"count": float(len(group))}
            for column in candidate_columns:
                if kinds[column] == "numeric":
                    values = numeric_view[column].loc[group.index]
                    stats = self._numeric_aggregates(values)
                else:
                    stats = self._categorical_aggregates(group[column])
                stats.update(self._shared_aggregates(group[column]))
                for stat, value in stats.items():
                    row[self.column_name(column, stat)] = value
            records[key] = row

        if not records:
            return empty

        table = pd.DataFrame.from_dict(records, orient="index")
        table.index.name = None

        # Apply the cap round-robin across source columns rather than in
        # declaration order. Taking them in order lets the first child column
        # consume the entire budget -- a cap of 10 spent eight slots on
        # `order_date` statistics and left `amount`, the column anyone would
        # actually want summarised, with one. Deterministic either way, so a
        # config change does not silently reshuffle what is kept.
        by_column: dict[str, list[str]] = {}
        for name in table.columns:
            if name == "count":
                continue
            source_column = name[len(f"{AGG_PREFIX}{self.child_name}__"):].rsplit("__", 1)[0]
            by_column.setdefault(source_column, []).append(name)

        interleaved: list[str] = []
        depth = 0
        while any(len(names) > depth for names in by_column.values()):
            for names in by_column.values():
                if len(names) > depth:
                    interleaved.append(names[depth])
            depth += 1

        keep = ["count"] + interleaved
        if len(keep) > self.config.max_columns:
            self.dropped = keep[self.config.max_columns :]
            keep = keep[: self.config.max_columns]
        table = table[keep]

        # Align to the parent's row order.
        lookup = parent_frame[parent_key[0]]
        aligned = table.reindex(lookup.to_numpy())
        aligned.index = parent_frame.index
        if "count" in aligned.columns:
            aligned["count"] = aligned["count"].fillna(0.0)

        aligned = aligned.rename(
            columns={"count": f"{AGG_PREFIX}{self.child_name}__count"}
        )
        self.built = list(aligned.columns)
        return aligned


def strip_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop generated aggregate columns before the data is returned."""
    drop = [c for c in frame.columns if c.startswith(AGG_PREFIX)]
    return frame.drop(columns=drop) if drop else frame


__all__ = [
    "AGG_PREFIX",
    "AggregationConfig",
    "Aggregator",
    "DEFAULT_MAX_COLUMNS",
    "strip_aggregates",
]
