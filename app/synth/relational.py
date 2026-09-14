"""Multi-table synthesis with referential integrity.

Generating each table independently and hoping the keys line up does not work:
foreign keys land on parents that do not exist, and the number of children per
parent collapses to noise. This module does what the vendor's IRG/relational
models do in principle -- generate parents first, then generate children
*conditioned on* their parent -- using the SPN as the per-table density model.

Per table we learn three things:

  1. A row model      -- SPN over the table's own (non-key) columns.
  2. A degree model   -- distribution of "how many children does a parent have",
                         learned per foreign key. This is what preserves
                         cardinality; without it, a customer with 200 card
                         transactions becomes a customer with 3.
  3. Parent context   -- aggregates of the child table folded back onto the
                         parent row before the parent is modelled, so the parent
                         "knows" how busy it is. This is the schema.yaml
                         "Aggregation Specification" group in miniature.

Processing order comes from the `order` parameter (topological: parents first),
matching the constraint documented in schema.yaml.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.synth.aggregate import AGG_PREFIX, AggregationConfig, Aggregator
from app.synth.dp import PrivacyLedger
from app.synth.engines import get_engine
from app.synth.preprocess import TableEncoder, stable_seed
from app.synth.spn import SPN, SPNParams

# Declared upper bound on children per parent, used as the DP support for the
# degree histogram. Overridden per table via TableSpec.max_degree.
MAX_DEGREE = 200

# Share of the epsilon budget reserved for the degree models. They are a
# separate data-dependent release from the per-table SPNs, so they need their
# own allocation rather than riding along on one.
DEGREE_BUDGET_FRACTION = 0.1


def _dp_settings(params: Any) -> tuple[bool, float]:
    """Read (private, epsilon) from any engine's parameter object.

    Only the SPN carries these; ARF has no epsilon at all. Reading them
    defensively keeps the relational layer engine-agnostic instead of assuming
    an SPN-shaped params object.
    """
    return bool(getattr(params, "private", False)), float(getattr(params, "epsilon", 0.0))


def _with_fields(params: Any, **updates: Any) -> Any:
    """dataclasses.replace, but only for fields the object actually has."""
    from dataclasses import fields as dataclass_fields

    try:
        allowed = {f.name for f in dataclass_fields(params)}
    except TypeError:
        return params
    filtered = {k: v for k, v in updates.items() if k in allowed}
    return replace(params, **filtered) if filtered else params


@dataclass
class ForeignKeySpec:
    parent_table_name: str
    parent_column_names: list[str]
    child_column_names: list[str]
    nullable: bool = False
    unique: bool = False

    @classmethod
    def from_config(cls, blob: dict[str, Any]) -> ForeignKeySpec:
        return cls(
            parent_table_name=blob.get("parent_table_name", ""),
            parent_column_names=_as_list(blob.get("parent_column_names")),
            child_column_names=_as_list(blob.get("child_column_names")),
            nullable=bool(blob.get("nullable", False)),
            unique=bool(blob.get("unique", False)),
        )


@dataclass
class TableSpec:
    name: str
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeySpec] = field(default_factory=list)
    timeseries: bool = False
    static_ids: list[str] = field(default_factory=list)
    sortby: list[str] = field(default_factory=list)
    column_kind_overrides: dict[str, str] = field(default_factory=dict)
    # Public column domains: {"balance": {"min": 0, "max": 1e6},
    #                         "segment": {"categories": [...]}}
    declared_domain: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Declared upper bound on how many rows of THIS table one parent row may
    # have. Public modelling choice, used as the DP support for the degree
    # histogram; keep it close to the real range or DP noise dominates.
    max_degree: int = MAX_DEGREE
    # Raw configuration blob for this table, read by AggregationConfig so the
    # Aggregation Specification parameters reach the aggregator.
    aggregation_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, name: str, blob: dict[str, Any]) -> TableSpec:
        return cls(
            name=name,
            declared_domain=_declared_domain(blob),
            primary_key=_as_list(blob.get("primary_key")),
            foreign_keys=[
                ForeignKeySpec.from_config(fk)
                for fk in (blob.get("foreign_keys") or [])
                if isinstance(fk, dict) and fk.get("parent_table_name")
            ],
            timeseries=bool(blob.get("timeseries", False)),
            static_ids=_as_list(blob.get("static_ids")),
            sortby=_as_list(blob.get("sortby")),
            column_kind_overrides=_kind_overrides(blob),
            max_degree=int(blob.get("max_degree") or MAX_DEGREE),
            aggregation_args=dict(blob or {}),
        )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _declared_domain(blob: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Read declared column domains out of the configuration.

    Maps onto the per-column parameters schema.yaml already defines --
    `min_val`, `max_val` and `force_categories` under column_kwargs -- so the
    domain is expressed where a user would expect to set it, and can be
    populated from Atlas by the autoconfigure step.
    """
    out: dict[str, dict[str, Any]] = {}
    column_kwargs = blob.get("column_kwargs") or {}
    if not isinstance(column_kwargs, dict):
        return out

    for column, settings in column_kwargs.items():
        if not isinstance(settings, dict):
            continue
        entry: dict[str, Any] = {}
        if settings.get("min_val") is not None:
            entry["min"] = settings["min_val"]
        if settings.get("max_val") is not None:
            entry["max"] = settings["max_val"]
        categories = settings.get("force_categories")
        if isinstance(categories, list) and categories:
            entry["categories"] = categories
        if settings.get("transform"):
            entry["transform"] = settings["transform"]
        if settings.get("nullable") is not None:
            entry["nullable"] = bool(settings["nullable"])
        if entry:
            out[column] = entry
    return out


def _kind_overrides(blob: dict[str, Any]) -> dict[str, str]:
    """Map the explicit *_columns parameters onto encoder type overrides."""
    mapping = {
        "categorical_columns": "categorical",
        "numerical_columns": "numerical",
        "datetime_columns": "datetime",
        "timedelta_columns": "timedelta",
        "non_std_columns": "non_std",
    }
    out: dict[str, str] = {}
    for key, kind in mapping.items():
        for column in _as_list(blob.get(key)):
            out[column] = kind
    return out


@dataclass
class DegreeModel:
    """Distribution over the number of child rows per parent row."""

    values: list[int]
    probs: list[float]

    @classmethod
    def fit(
        cls,
        counts: np.ndarray,
        ledger: PrivacyLedger | None = None,
        rng: np.random.Generator | None = None,
        cap: int = 200,
    ) -> DegreeModel:
        """Fit the degree distribution.

        Under DP the support must be the declared range 0..cap rather than the
        observed values -- releasing "the distinct child counts that occur"
        would disclose, for instance, that exactly one parent has 87 children.
        Counts are clipped to the cap, which is a declared modelling bound.
        """
        if counts.size == 0:
            return cls(values=[0], probs=[1.0])

        if ledger is not None and ledger.enabled and rng is not None:
            # The support width drives the damage here. Every empty bin still
            # receives noise of order 1/epsilon, so a support far wider than the
            # data's actual range fills with spurious mass and inflates the mean
            # -- a 0..200 support over degrees that are really 1..4 puts most of
            # the probability on noise. Two mitigations: the caller declares a
            # cap close to the real range, and we bin rather than using one
            # value per integer, so real counts stay large relative to noise.
            cap = max(int(cap), 1)
            clipped = np.clip(counts.astype(int), 0, cap)
            n_bins = min(16, cap + 1)
            edges = np.unique(np.linspace(0, cap + 1, n_bins + 1).astype(int))
            histogram, _ = np.histogram(clipped, bins=edges)
            noisy = ledger.laplace_counts(
                histogram.astype(float), ledger.leaf_epsilon(), rng, "degree_model"
            )
            # Stability thresholding: a bin whose noisy count is within the
            # noise scale carries no evidence, so drop it instead of letting it
            # contribute degrees. This is post-processing and costs no budget.
            scale = 1.0 / max(ledger.leaf_epsilon(), 1e-9)
            noisy = np.where(noisy > scale, noisy, 0.0)
            if noisy.sum() <= 0:
                noisy = histogram.astype(float) + 1e-9

            values: list[int] = []
            probs_out: list[float] = []
            total = noisy.sum()
            for index, weight in enumerate(noisy):
                if weight <= 0:
                    continue
                low, high = int(edges[index]), int(edges[index + 1])
                span = list(range(low, max(high, low + 1)))
                for value in span:
                    values.append(value)
                    probs_out.append(float(weight / total / len(span)))
            if not values:
                return cls(values=[0], probs=[1.0])
            normaliser = sum(probs_out)
            return cls(
                values=values, probs=[p / normaliser for p in probs_out]
            )

        uniques, freq = np.unique(counts.astype(int), return_counts=True)
        return cls(
            values=[int(v) for v in uniques],
            probs=[float(p) for p in freq / freq.sum()],
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.values, size=n, p=self.probs)

    def to_json(self) -> dict[str, Any]:
        return {"values": self.values, "probs": self.probs}

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> DegreeModel:
        return cls(values=blob["values"], probs=blob["probs"])


class RelationalSPN:
    """Container for the per-table SPNs plus the wiring between them."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.specs: dict[str, TableSpec] = {}
        self.encoders: dict[str, TableEncoder] = {}
        self.models: dict[str, SPN] = {}
        self.degrees: dict[str, dict[str, DegreeModel]] = {}
        self.modelled_columns: dict[str, list[str]] = {}
        self.pk_dtypes: dict[str, str] = {}
        self.privacy: dict[str, Any] = {}
        self.degree_privacy: dict[str, Any] = {}
        self.dp_enabled: bool = False
        self.train_stats: dict[str, Any] = {}
        # child table -> [(context column name, parent table, parent column)]
        self.context_columns: dict[str, list[tuple[str, str, str]]] = {}
        # Namespace for minted primary keys. Synthetic keys deliberately have no
        # relationship to real ones, but they should be REPRODUCIBLE: the same
        # model, seed and namespace must give the same identities every time, so
        # a refreshed test dataset keeps the customers a test suite refers to.
        # Change the namespace to deliberately mint a fresh population.
        self.key_namespace: str = "v1"
        # Which density model backs every table. One engine per model; mixing
        # them across tables would make the privacy story incoherent.
        self.engine_name: str = "SPN"
        # parent table -> child table -> which aggregates were built
        self.aggregates: dict[str, Any] = {}

    # -- fit --

    @classmethod
    def fit(
        cls,
        tables: dict[str, pd.DataFrame],
        specs: dict[str, TableSpec],
        params: SPNParams,
        order: list[str] | None = None,
        progress: Any = None,
        engine: str = "SPN",
    ) -> RelationalSPN:
        table_engine = get_engine(engine)
        if getattr(params, "private", False) and not table_engine.supports_dp:
            raise ValueError(
                f"engine '{table_engine.name}' does not support differential "
                "privacy. Either use the SPN engine, or turn privacy off "
                "deliberately -- this is not downgraded silently because a run "
                "that asked for privacy must not quietly return a model without it."
            )
        model = cls()
        model.engine_name = table_engine.name
        model.specs = specs
        model.order = order or topological_order(specs)

        children_of = _children_index(specs)

        # Carve the degree models' budget out of the total before the SPNs see
        # it, so the sum of everything released stays within params.epsilon.
        n_foreign_keys = max(sum(len(s.foreign_keys) for s in specs.values()), 1)
        private, epsilon = _dp_settings(params)
        degree_ledger = PrivacyLedger(
            epsilon_total=epsilon * DEGREE_BUDGET_FRACTION,
            enabled=private,
            n_columns=n_foreign_keys,
            max_depth=1,
            domain_is_public=True,
            domain_note=f"degree support declared as 0..{MAX_DEGREE}",
        )
        if private:
            params = replace(
                params, epsilon=epsilon * (1.0 - DEGREE_BUDGET_FRACTION)
            )
        model.degree_privacy = degree_ledger.report()
        model.dp_enabled = private

        for position, table_name in enumerate(model.order):
            if progress:
                progress(
                    position / max(len(model.order), 1),
                    f"Training SPN for {table_name}",
                )
            frame = tables[table_name]
            spec = specs.get(table_name) or TableSpec(name=table_name)

            # Columns the SPN actually models: drop keys, they are assigned
            # structurally rather than sampled.
            key_columns = set(spec.primary_key)
            for fk in spec.foreign_keys:
                key_columns.update(fk.child_column_names)

            modelled = [c for c in frame.columns if c not in key_columns]

            # Fold summaries of each child table onto the parent row, driven by
            # the Aggregation Specification parameters. The parent's own
            # distribution then carries the shape of its children, so a
            # high-value high-frequency customer is generated as a coherent
            # whole rather than a random parent with random children attached.
            context = pd.DataFrame(index=frame.index)
            context_domain: dict[str, dict[str, Any]] = {}
            aggregate_report: dict[str, Any] = {}

            for child_name, fk in children_of.get(table_name, []):
                child_frame = tables.get(child_name)
                child_spec = specs.get(child_name) or TableSpec(name=child_name)
                if child_frame is None:
                    continue

                # The degree column keeps its own name and linear domain: the
                # generator depends on it, and a log-spaced DP bin at the top of
                # a count range inverts to thousands of children.
                counts = _child_counts(child_frame, fk, frame, spec)
                if counts is not None:
                    degree_column = f"__deg__{child_name}"
                    context[degree_column] = counts.values
                    context_domain[degree_column] = {
                        "min": 0,
                        "max": child_spec.max_degree,
                    }

                aggregation = AggregationConfig.from_config(
                    child_spec.aggregation_args, timeseries=child_spec.timeseries
                )
                aggregator = Aggregator(aggregation, child_name)
                summaries = aggregator.build(
                    child_frame,
                    frame,
                    child_key=fk.child_column_names,
                    parent_key=fk.parent_column_names,
                    sortby=child_spec.sortby if child_spec.timeseries else None,
                    exclude_columns=set(child_spec.primary_key),
                )
                for column in summaries.columns:
                    # `count` is already carried by the degree column.
                    if column.endswith("__count"):
                        continue
                    context[column] = summaries[column].values

                aggregate_report[child_name] = {
                    "columns": [c for c in summaries.columns if not c.endswith("__count")],
                    "dropped_at_cap": aggregator.dropped,
                    "config": aggregation.to_json(),
                }

            # Pull the parent's attributes onto each child row, so the child
            # SPN learns P(child | parent) rather than P(child). This is what
            # keeps a PRIVATE customer's transactions looking like a PRIVATE
            # customer's transactions.
            context_spec: list[tuple[str, str, str]] = []
            for fk in spec.foreign_keys:
                parent_frame = tables.get(fk.parent_table_name)
                parent_spec = specs.get(fk.parent_table_name)
                if parent_frame is None or parent_spec is None:
                    continue
                parent_columns = _context_candidates(parent_frame, parent_spec)
                if not parent_columns:
                    continue
                merged = _join_parent_context(
                    frame, parent_frame, fk, parent_columns
                )
                for parent_column in parent_columns:
                    ctx_name = f"__ctx__{fk.parent_table_name}__{parent_column}"
                    context[ctx_name] = merged[parent_column].values
                    context_spec.append(
                        (ctx_name, fk.parent_table_name, parent_column)
                    )
            model.context_columns[table_name] = context_spec
            if aggregate_report:
                model.aggregates[table_name] = aggregate_report

            training_frame = pd.concat([frame[modelled], context], axis=1)

            encoder = TableEncoder.fit(
                training_frame,
                overrides=spec.column_kind_overrides,
                declared_domain={**spec.declared_domain, **context_domain},
                apply_declared_transform=private,
            )
            matrix = encoder.transform(training_frame)

            # Tell the SPN whether its domain was declared or measured, so the
            # privacy ledger can state an unconditional or conditional epsilon.
            domain = encoder.domain_summary()
            table_params = _with_fields(
                params,
                domain_is_public=domain["all_declared"],
                domain_note=(
                    "all column domains declared"
                    if domain["all_declared"]
                    else "measured from data: "
                    + ", ".join(domain["measured_from_data"][:8])
                ),
            )
            spn = table_engine.fit(matrix, encoder.specs, table_params)

            model.encoders[table_name] = encoder
            model.models[table_name] = spn
            model.modelled_columns[table_name] = list(training_frame.columns)
            model.privacy[table_name] = table_engine.privacy_of(spn)
            if spec.primary_key:
                model.pk_dtypes[table_name] = str(frame[spec.primary_key[0]].dtype)

            # Degree models per foreign key, for use when generating children.
            degree_rng = np.random.default_rng(int(getattr(params, 'random_state', 0)) + 17)
            for fk in spec.foreign_keys:
                counts = _degree_counts(frame, fk, tables.get(fk.parent_table_name))
                model.degrees.setdefault(table_name, {})[fk.parent_table_name] = (
                    DegreeModel.fit(
                        counts,
                        ledger=degree_ledger,
                        rng=degree_rng,
                        cap=spec.max_degree,
                    )
                )

            model.train_stats[table_name] = {
                "rows": int(len(frame)),
                "modelled_columns": len(modelled),
                "context_columns": int(context.shape[1]),
                "parent_context_columns": len(context_spec),
                "aggregate_columns": int(
                    sum(len(v["columns"]) for v in aggregate_report.values())
                ),
                "nodes": table_engine.stats_of(spn),
            }
            if table_engine.name == "SPN" and getattr(params, "beta", 0) >= len(frame):
                # Faithful to the vendor semantics, but worth flagging: no SUM
                # node can form, so every column becomes independent.
                model.train_stats[table_name]["warning"] = (
                    f"beta ({params.beta:,}) >= row count ({len(frame):,}): "
                    "the model degenerates to independent columns and will not "
                    "reproduce correlations. Lower beta."
                )

        if progress:
            progress(1.0, "Training complete")
        return model

    # -- sample --

    def sample(
        self,
        n_rows: dict[str, int] | int,
        seed: int = 0,
        progress: Any = None,
        key_namespace: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        if key_namespace is not None:
            self.key_namespace = key_namespace
        rng = np.random.default_rng(seed)
        out: dict[str, pd.DataFrame] = {}
        # Per parent table, the child-row count the parent's own model predicted
        # for each generated row: {parent_table: {child_table: array}}.
        predicted_degrees: dict[str, dict[str, np.ndarray]] = {}

        for position, table_name in enumerate(self.order):
            if progress:
                progress(position / max(len(self.order), 1), f"Generating {table_name}")
            spec = self.specs.get(table_name) or TableSpec(name=table_name)
            parent_fks = [fk for fk in spec.foreign_keys if fk.parent_table_name in out]

            # Condition on the most specific parent available. A loan payment
            # belongs to a loan, which belongs to a customer; generating it
            # against the customer instead leaves its loan_id to be drawn at
            # random, so the payment's loan and its customer disagree. Later in
            # the topological order means deeper in the hierarchy, so the last
            # parent is the immediate one.
            if len(parent_fks) > 1:
                depth = {name: index for index, name in enumerate(self.order)}
                parent_fks.sort(
                    key=lambda fk: depth.get(fk.parent_table_name, -1), reverse=True
                )

            if not parent_fks:
                count = _resolve_count(n_rows, table_name, default=1000)
                frame, degrees_out = self._sample_rows(table_name, count, rng)
                predicted_degrees[table_name] = degrees_out
                frame = self._assign_primary_key(table_name, spec, frame)
                out[table_name] = frame
                continue

            # Child table: cardinality comes from the degree model, so each
            # synthetic parent gets a plausible number of children.
            primary_fk = parent_fks[0]
            parent_frame = out[primary_fk.parent_table_name]
            degree_model = self.degrees.get(table_name, {}).get(
                primary_fk.parent_table_name
            )

            degrees = self._degrees_for_parent(
                parent_frame,
                primary_fk,
                degree_model,
                rng,
                predicted=predicted_degrees.get(primary_fk.parent_table_name, {}).get(
                    table_name
                ),
            )

            total = int(degrees.sum())
            # Only an explicit per-table count pins a child table. A scalar
            # n_rows means "this many root rows"; the children that implies come
            # from the degree model, so asking for 1,200 customers yields the
            # ~12.85 transactions each that the real data carries rather than
            # 1,200 transactions in total.
            requested = n_rows.get(table_name) if isinstance(n_rows, dict) else None
            if requested is not None and total > 0:
                # Scale to the requested row count while keeping the shape of
                # the degree distribution.
                scale = requested / total
                degrees = np.maximum(
                    (degrees * scale).round().astype(int),
                    0,
                )
                total = int(degrees.sum())

                # Scaling alone cannot land on an exact count: degrees are small
                # integers, so a parent with 1 child scaled by 1.26 rounds back
                # to 1 and the total barely moves (1,500 requested came out as
                # 1,243). Close the remainder one child at a time, spread over
                # random parents so the distribution's shape survives.
                shortfall = int(requested) - total
                if shortfall > 0:
                    picks = rng.integers(0, len(degrees), size=shortfall)
                    np.add.at(degrees, picks, 1)
                elif shortfall < 0:
                    remaining = -shortfall
                    while remaining > 0:
                        givers = np.flatnonzero(degrees > 0)
                        if not len(givers):
                            break
                        take = min(remaining, len(givers))
                        degrees[rng.choice(givers, size=take, replace=False)] -= 1
                        remaining -= take
                total = int(degrees.sum())

            # Repeat each synthetic parent row by its degree, so every child row
            # knows which parent it belongs to before it is sampled.
            parent_row_index = np.repeat(np.arange(len(parent_frame)), degrees)

            frame, degrees_out = self._sample_rows(
                table_name,
                max(total, 0),
                rng,
                parent_frames=out,
                parent_row_index={primary_fk.parent_table_name: parent_row_index},
            )
            predicted_degrees[table_name] = degrees_out

            # Attach the parent keys.
            parent_keys = parent_frame[primary_fk.parent_column_names].to_numpy()
            repeated = np.repeat(parent_keys, degrees, axis=0)
            for position_in_key, child_column in enumerate(primary_fk.child_column_names):
                if position_in_key < repeated.shape[1]:
                    frame[child_column] = repeated[:, position_in_key]

            # Secondary foreign keys.
            #
            # Banking extracts are routinely denormalised: a transaction carries
            # both account_id and customer_id, and a loan payment carries both
            # loan_id and customer_id. Those paths are not independent -- the
            # customer on the transaction IS the owner of its account -- so
            # sampling the second key independently produces a row whose two
            # joins name different people. It resolves (no orphan) and is still
            # wrong, which is the worst kind of wrong.
            #
            # So whenever the primary parent already carries the column, inherit
            # it from the parent row this child belongs to. Only a genuinely
            # independent relationship falls back to sampling.
            for fk in parent_fks[1:]:
                other_parent = out[fk.parent_table_name]
                if other_parent.empty:
                    continue

                inherited = [
                    column
                    for column in fk.child_column_names
                    if column in parent_frame.columns
                ]
                if len(inherited) == len(fk.child_column_names):
                    bridged = parent_frame[inherited].to_numpy()[parent_row_index]
                    for position_in_key, child_column in enumerate(fk.child_column_names):
                        frame[child_column] = bridged[:, position_in_key]
                    continue

                picks = rng.integers(0, len(other_parent), size=len(frame))
                values = other_parent[fk.parent_column_names].to_numpy()[picks]
                for position_in_key, child_column in enumerate(fk.child_column_names):
                    if position_in_key < values.shape[1]:
                        frame[child_column] = values[:, position_in_key]

            frame = self._assign_primary_key(table_name, spec, frame)

            if spec.timeseries and spec.sortby:
                sort_columns = [c for c in spec.sortby if c in frame.columns]
                group_columns = [
                    c for c in (spec.static_ids or primary_fk.child_column_names)
                    if c in frame.columns
                ]
                if sort_columns:
                    frame = frame.sort_values(
                        by=group_columns + sort_columns
                    ).reset_index(drop=True)

            out[table_name] = frame

        if progress:
            progress(1.0, "Generation complete")
        return out

    # -- internals --

    def _sample_rows(
        self,
        table_name: str,
        count: int,
        rng: np.random.Generator,
        parent_frames: dict[str, pd.DataFrame] | None = None,
        parent_row_index: dict[str, np.ndarray] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Returns the sampled rows and the child-row counts the model predicted.

        The predicted degrees are what tie a child table's size to its parent's
        attributes, so they are returned rather than discarded.
        """
        encoder = self.encoders[table_name]
        spn = self.models[table_name]
        if count <= 0:
            return (
                pd.DataFrame(
                    columns=[
                        c
                        for c in self.modelled_columns[table_name]
                        if not c.startswith(("__deg__", "__ctx__", AGG_PREFIX))
                    ]
                ),
                {},
            )

        seed = int(rng.integers(0, 2**31 - 1))
        evidence_columns, evidence_frame = self._build_evidence(
            table_name, count, parent_frames or {}, parent_row_index or {}
        )

        if evidence_columns:
            evidence_matrix = encoder.transform(evidence_frame)
            matrix = spn.sample_conditional(evidence_matrix, evidence_columns, seed=seed)
        else:
            matrix = spn.sample(count, seed=seed)

        frame = encoder.inverse_transform(matrix)

        # Harvest the predicted per-row child counts before dropping the
        # context columns.
        degrees_out: dict[str, np.ndarray] = {}
        for column in frame.columns:
            if column.startswith("__deg__"):
                child_table = column[len("__deg__") :]
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
                values = np.nan_to_num(values, nan=0.0)
                degrees_out[child_table] = np.clip(np.round(values), 0, None).astype(int)

        drop = [
            c for c in frame.columns
            if c.startswith(("__deg__", "__ctx__", AGG_PREFIX))
        ]
        return frame.drop(columns=drop), degrees_out

    def _build_evidence(
        self,
        table_name: str,
        count: int,
        parent_frames: dict[str, pd.DataFrame],
        parent_row_index: dict[str, np.ndarray],
    ) -> tuple[list[int], pd.DataFrame]:
        """Assemble the parent-attribute evidence for `count` child rows."""
        context_spec = self.context_columns.get(table_name) or []
        if not context_spec or not parent_row_index:
            return [], pd.DataFrame()

        all_columns = self.encoders[table_name].columns
        frame = pd.DataFrame(
            {column: pd.Series([np.nan] * count) for column in all_columns}
        )

        evidence_columns: list[int] = []
        for ctx_name, parent_table, parent_column in context_spec:
            index = parent_row_index.get(parent_table)
            parent_frame = parent_frames.get(parent_table)
            if index is None or parent_frame is None:
                continue
            if parent_column not in parent_frame.columns:
                continue
            if ctx_name not in all_columns:
                continue
            if index.size != count:
                continue
            frame[ctx_name] = parent_frame[parent_column].to_numpy()[index]
            evidence_columns.append(all_columns.index(ctx_name))

        return evidence_columns, frame

    def _degrees_for_parent(
        self,
        parent_frame: pd.DataFrame,
        fk: ForeignKeySpec,
        degree_model: DegreeModel | None,
        rng: np.random.Generator,
        predicted: np.ndarray | None = None,
    ) -> np.ndarray:
        """How many children each synthetic parent row gets.

        Prefer the count the parent's own SPN predicted, because that count was
        modelled jointly with the parent's attributes and therefore preserves
        the link between them -- a high-value customer keeps their high
        transaction volume. Falling back to the marginal degree distribution
        severs that link: cardinality is then assigned at random, so the child
        table stops being dominated by the busy parents that dominate it in
        reality, which visibly biases the child table's own distributions.
        """
        # Under DP the predicted per-row degree is not usable: it comes from a
        # noised histogram over a wide count range, and the spurious mass in
        # its upper bins inflates the child table by orders of magnitude. The
        # privatised marginal is far more stable, at the cost of severing the
        # link between a parent's attributes and how many children it has.
        # That trade-off is recorded in the report rather than hidden.
        if (
            not self.dp_enabled
            and predicted is not None
            and len(predicted) == len(parent_frame)
        ):
            return np.asarray(predicted, dtype=int)
        if degree_model is None:
            return np.ones(len(parent_frame), dtype=int)
        return degree_model.sample(len(parent_frame), rng).astype(int)

    def _assign_primary_key(
        self, table_name: str, spec: TableSpec, frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Synthesise a fresh surrogate key.

        Primary keys are never sampled from the model -- reproducing a real key
        would be a direct record linkage. We mint new ones.
        """
        if not spec.primary_key:
            return frame
        key = spec.primary_key[0]
        if key in frame.columns and frame[key].notna().all():
            return frame
        dtype = self.pk_dtypes.get(table_name, "int64")
        n = len(frame)
        # Offset the key space so synthetic keys cannot be confused with real
        # ones, and shuffle so a key does not encode its own row position.
        # Sequential 1..N keys would collide with the source keys exactly and
        # leak ordering, which is why they are not used.
        key_rng = np.random.default_rng(stable_seed(table_name, self.key_namespace))
        offset = int(key_rng.integers(10**8, 9 * 10**8))
        values = offset + key_rng.permutation(n)
        if dtype.startswith("int") or dtype.startswith("uint"):
            frame[key] = values
        else:
            prefix = "".join(part[0] for part in table_name.split("_") if part).upper()
            frame[key] = [f"{prefix}{v:09d}" for v in values]
        columns = [key] + [c for c in frame.columns if c != key]
        return frame[columns]

    # -- persistence --

    def save(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "order": self.order,
            "modelled_columns": self.modelled_columns,
            "pk_dtypes": self.pk_dtypes,
            "privacy": self.privacy,
            "degree_privacy": self.degree_privacy,
            "dp_enabled": self.dp_enabled,
            "key_namespace": self.key_namespace,
            "engine": self.engine_name,
            "aggregates": self.aggregates,
            "train_stats": self.train_stats,
            "context_columns": {
                table: [list(item) for item in items]
                for table, items in self.context_columns.items()
            },
            "specs": {
                name: {
                    "primary_key": s.primary_key,
                    "foreign_keys": [
                        {
                            "parent_table_name": fk.parent_table_name,
                            "parent_column_names": fk.parent_column_names,
                            "child_column_names": fk.child_column_names,
                            "nullable": fk.nullable,
                            "unique": fk.unique,
                        }
                        for fk in s.foreign_keys
                    ],
                    "timeseries": s.timeseries,
                    "static_ids": s.static_ids,
                    "sortby": s.sortby,
                    "column_kind_overrides": s.column_kind_overrides,
                    "declared_domain": s.declared_domain,
                    "max_degree": s.max_degree,
                    "aggregation_args": s.aggregation_args,
                }
                for name, s in self.specs.items()
            },
            "degrees": {
                table: {parent: d.to_json() for parent, d in mapping.items()}
                for table, mapping in self.degrees.items()
            },
            "encoders": {
                name: encoder.to_json() for name, encoder in self.encoders.items()
            },
        }
        with open(path / "manifest.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        for name, spn in self.models.items():
            spn.save(str(path / f"spn__{name}.json"))

    @classmethod
    def load(cls, directory: str | Path) -> RelationalSPN:
        path = Path(directory)
        with open(path / "manifest.json", encoding="utf-8") as handle:
            manifest = json.load(handle)
        model = cls()
        model.order = manifest["order"]
        model.modelled_columns = manifest["modelled_columns"]
        model.pk_dtypes = manifest.get("pk_dtypes", {})
        model.privacy = manifest.get("privacy", {})
        model.degree_privacy = manifest.get("degree_privacy", {})
        model.dp_enabled = manifest.get("dp_enabled", False)
        model.key_namespace = manifest.get("key_namespace", "v1")
        model.engine_name = manifest.get("engine", "SPN")
        model.aggregates = manifest.get("aggregates", {})
        model.train_stats = manifest.get("train_stats", {})
        model.context_columns = {
            table: [tuple(item) for item in items]
            for table, items in manifest.get("context_columns", {}).items()
        }
        model.specs = {
            name: TableSpec(
                name=name,
                primary_key=blob["primary_key"],
                foreign_keys=[ForeignKeySpec(**fk) for fk in blob["foreign_keys"]],
                timeseries=blob.get("timeseries", False),
                static_ids=blob.get("static_ids", []),
                sortby=blob.get("sortby", []),
                column_kind_overrides=blob.get("column_kind_overrides", {}),
                declared_domain=blob.get("declared_domain", {}),
                max_degree=blob.get("max_degree", MAX_DEGREE),
                aggregation_args=blob.get("aggregation_args", {}),
            )
            for name, blob in manifest["specs"].items()
        }
        model.degrees = {
            table: {parent: DegreeModel.from_json(d) for parent, d in mapping.items()}
            for table, mapping in manifest.get("degrees", {}).items()
        }
        model.encoders = {
            name: TableEncoder.from_json(blob)
            for name, blob in manifest["encoders"].items()
        }
        model.models = {
            name: get_engine(model.engine_name).load(str(path / f"spn__{name}.json"))
            for name in model.order
        }
        return model


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


MAX_CONTEXT_COLUMNS = 6

# Public modelling cap on how many children one parent row may have. It is a
# declared bound, not a measurement, so it is safe to use as a DP domain. The
# log transform matters as much as the cap: degree distributions are heavily
# right-skewed (most customers have a few transactions, a few have hundreds),
# and equal-width DP bins over a raw count range put nearly all mass in the
# first bin, where Laplace noise then scatters weight into tail bins worth
# hundreds of children each -- which inflates every child table.


def _context_candidates(parent_frame: pd.DataFrame, parent_spec: TableSpec) -> list[str]:
    """Parent columns worth conditioning a child on.

    Keys carry no signal, and high-cardinality identifiers (non_std) would just
    blow up the model, so both are excluded. We cap the count because every
    context column multiplies the evidence term in conditional sampling.
    """
    excluded = set(parent_spec.primary_key)
    for fk in parent_spec.foreign_keys:
        excluded.update(fk.child_column_names)
    for column, kind in parent_spec.column_kind_overrides.items():
        if kind == "non_std":
            excluded.add(column)

    candidates: list[tuple[float, str]] = []
    for column in parent_frame.columns:
        if column in excluded:
            continue
        series = parent_frame[column]
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            continue
        # Prefer low-cardinality drivers (segment, risk_rating) and numerics;
        # skip anything that looks like a free-text identifier.
        ratio = n_unique / max(len(series), 1)
        if not pd.api.types.is_numeric_dtype(series) and ratio > 0.5:
            continue
        score = ratio if pd.api.types.is_numeric_dtype(series) else -ratio
        candidates.append((score, column))

    candidates.sort(key=lambda item: item[0])
    return [column for _, column in candidates[:MAX_CONTEXT_COLUMNS]]


def _join_parent_context(
    child_frame: pd.DataFrame,
    parent_frame: pd.DataFrame,
    fk: ForeignKeySpec,
    parent_columns: list[str],
) -> pd.DataFrame:
    """Left-join the parent's chosen columns onto each child row."""
    if not fk.child_column_names or not fk.parent_column_names:
        return pd.DataFrame(index=child_frame.index, columns=parent_columns)
    if not all(c in child_frame.columns for c in fk.child_column_names):
        return pd.DataFrame(index=child_frame.index, columns=parent_columns)
    if not all(c in parent_frame.columns for c in fk.parent_column_names):
        return pd.DataFrame(index=child_frame.index, columns=parent_columns)

    lookup = parent_frame[fk.parent_column_names + parent_columns].drop_duplicates(
        subset=fk.parent_column_names
    )
    merged = child_frame[fk.child_column_names].merge(
        lookup,
        left_on=fk.child_column_names,
        right_on=fk.parent_column_names,
        how="left",
    )
    merged.index = child_frame.index
    return merged[parent_columns]


def _children_index(
    specs: dict[str, TableSpec]
) -> dict[str, list[tuple[str, ForeignKeySpec]]]:
    index: dict[str, list[tuple[str, ForeignKeySpec]]] = {}
    for child_name, spec in specs.items():
        for fk in spec.foreign_keys:
            index.setdefault(fk.parent_table_name, []).append((child_name, fk))
    return index


def _child_counts(
    child_frame: pd.DataFrame,
    fk: ForeignKeySpec,
    parent_frame: pd.DataFrame,
    parent_spec: TableSpec,
) -> pd.Series | None:
    """Rows of `child_frame` per row of `parent_frame`."""
    if not fk.child_column_names or not fk.parent_column_names:
        return None
    if not all(c in child_frame.columns for c in fk.child_column_names):
        return None
    if not all(c in parent_frame.columns for c in fk.parent_column_names):
        return None
    grouped = child_frame.groupby(fk.child_column_names).size()
    keys = parent_frame[fk.parent_column_names]
    if len(fk.parent_column_names) == 1:
        lookup = keys.iloc[:, 0].map(grouped).fillna(0)
    else:
        index = pd.MultiIndex.from_frame(keys)
        lookup = pd.Series(grouped.reindex(index).fillna(0).to_numpy(),
                           index=parent_frame.index)
    return lookup.astype(float)


def _degree_counts(
    child_frame: pd.DataFrame,
    fk: ForeignKeySpec,
    parent_frame: pd.DataFrame | None,
) -> np.ndarray:
    """Observed number of children per parent, including parents with zero."""
    if parent_frame is None or not fk.child_column_names:
        return np.array([1])
    if not all(c in child_frame.columns for c in fk.child_column_names):
        return np.array([1])
    counts = child_frame.groupby(fk.child_column_names).size()
    if not all(c in parent_frame.columns for c in fk.parent_column_names):
        return counts.to_numpy()
    if len(fk.parent_column_names) == 1:
        aligned = parent_frame[fk.parent_column_names[0]].map(counts).fillna(0)
    else:
        index = pd.MultiIndex.from_frame(parent_frame[fk.parent_column_names])
        aligned = pd.Series(counts.reindex(index).fillna(0).to_numpy())
    return aligned.to_numpy()


def topological_order(specs: dict[str, TableSpec]) -> list[str]:
    """Parents before children, as `order` in schema.yaml requires."""
    pending = dict(specs)
    resolved: list[str] = []
    guard = 0
    while pending and guard < 100:
        guard += 1
        for name, spec in list(pending.items()):
            parents = {fk.parent_table_name for fk in spec.foreign_keys}
            parents.discard(name)
            if parents.issubset(set(resolved)):
                resolved.append(name)
                pending.pop(name)
    resolved.extend(pending.keys())  # cycles: fall back to declaration order
    return resolved


def _resolve_count(
    n_rows: dict[str, int] | int, table: str, default: int | None
) -> int | None:
    if isinstance(n_rows, dict):
        value = n_rows.get(table)
        return int(value) if value is not None else default
    if isinstance(n_rows, int):
        return int(n_rows)
    return default


__all__ = [
    "DegreeModel",
    "ForeignKeySpec",
    "RelationalSPN",
    "TableSpec",
    "topological_order",
]
