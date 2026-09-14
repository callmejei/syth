"""Maps Atlas classifications onto synthesis strategy.

This is the piece the vendor product does not have. Betterdata decides how to
treat a column by looking at the data; here the decision is driven by the
classification your governance team already curated in Atlas, and every
decision is recorded with the tag that caused it.

Two things fall out of that which matter to a compliance reviewer:

  * The policy is declarative and auditable. "national_id was suppressed
    because Atlas tags it PII" is a sentence you can put in a control document.
  * It fails closed. An unclassified column in a table that contains PII is
    treated as suspect rather than waved through, because the common failure
    mode is a new column landing in a table before anyone tags it.

Strategies
----------
SUPPRESS    drop the column; never reaches the model
PSEUDONYM   replace with a generated surrogate that preserves format/cardinality
            but has no link to the original (our non_std placeholder path)
GENERALISE  coarsen before modelling (dates to month, ages to bands)
MODEL       learn and synthesise normally
MODEL_DP    learn and synthesise, and require differential privacy to be on
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.atlas.client import AtlasResult, ColumnClassification
from app.atlas.inference import Inference, infer_column

SUPPRESS = "SUPPRESS"
PSEUDONYM = "PSEUDONYM"
GENERALISE = "GENERALISE"
MODEL = "MODEL"
MODEL_DP = "MODEL_DP"

# Strictness of the policy, exposed as an org setting.
STRICT = "strict"      # unclassified columns in a sensitive table are pseudonymised
BALANCED = "balanced"  # unclassified columns are modelled, but flagged
PERMISSIVE = "permissive"


@dataclass
class ColumnDecision:
    column: str
    strategy: str
    reason: str
    classifications: list[str] = field(default_factory=list)
    encoder_kind: str | None = None  # override passed to the TableEncoder
    # True when the strategy came from name/value inference rather than from a
    # curated catalog tag. Surfaced everywhere, so a policy resting on guesses
    # is never mistaken for one resting on your catalog.
    inferred: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "strategy": self.strategy,
            "reason": self.reason,
            "classifications": self.classifications,
            "encoder_kind": self.encoder_kind,
            "inferred": self.inferred,
        }


@dataclass
class TablePolicy:
    table: str
    decisions: list[ColumnDecision] = field(default_factory=list)

    @property
    def suppressed(self) -> list[str]:
        return [d.column for d in self.decisions if d.strategy == SUPPRESS]

    @property
    def pseudonymised(self) -> list[str]:
        return [d.column for d in self.decisions if d.strategy == PSEUDONYM]

    @property
    def generalised(self) -> list[str]:
        return [d.column for d in self.decisions if d.strategy == GENERALISE]

    @property
    def requires_dp(self) -> bool:
        return any(d.strategy == MODEL_DP for d in self.decisions)

    @property
    def inferred(self) -> list[str]:
        """Columns protected by inference rather than by a curated tag."""
        return [d.column for d in self.decisions if d.inferred]

    # A classification says what a column *is*; the encoder needs that to pick a
    # surrogate generator. Without it the style was guessed from the spelling of
    # the column name, so `full_name` got names and `cfna1` -- the same data
    # under a bank's internal naming -- got a row counter.
    _STYLE_FOR_TAG = {
        "NAME": "name",
        "EMAIL": "email",
        "PHONE": "phone",
        "NATIONAL_ID": "national_id",
        "PASSPORT": "national_id",
        "ADDRESS": "address",
    }

    def placeholder_styles(self) -> dict[str, str]:
        """Surrogate generator per column, from its classification."""
        styles: dict[str, str] = {}
        for decision in self.decisions:
            for tag in decision.classifications:
                # Inference records its guesses as "INFERRED:NAME" so they can
                # never be mistaken for a curated tag; the shape it implies is
                # the same either way.
                bare = tag.split(":", 1)[-1].strip().upper()
                style = self._STYLE_FOR_TAG.get(bare)
                if style:
                    styles[decision.column] = style
                    break
        return styles

    def encoder_overrides(self) -> dict[str, str]:
        return {
            d.column: d.encoder_kind
            for d in self.decisions
            if d.encoder_kind
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "decisions": [d.to_json() for d in self.decisions],
            "suppressed": self.suppressed,
            "pseudonymised": self.pseudonymised,
            "generalised": self.generalised,
            "inferred": self.inferred,
            "requires_dp": self.requires_dp,
        }


@dataclass
class PolicyResult:
    tables: dict[str, TablePolicy]
    source: str
    detail: str
    mode: str

    @property
    def requires_dp(self) -> bool:
        return any(t.requires_dp for t in self.tables.values())

    def summary(self) -> dict[str, Any]:
        inferred = sum(len(t.inferred) for t in self.tables.values())
        return {
            "source": self.source,
            "detail": self.detail,
            "mode": self.mode,
            "requires_dp": self.requires_dp,
            "suppressed_total": sum(len(t.suppressed) for t in self.tables.values()),
            "pseudonymised_total": sum(
                len(t.pseudonymised) for t in self.tables.values()
            ),
            "generalised_total": sum(len(t.generalised) for t in self.tables.values()),
            # How much of this policy rests on guesses rather than on the
            # catalog. A reviewer needs this number before quoting any other.
            "inferred_total": inferred,
            "catalog_backed": inferred == 0,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "tables": {k: v.to_json() for k, v in self.tables.items()},
        }


def decide_column(
    classification: ColumnClassification | None,
    column_name: str,
    *,
    table_has_sensitive: bool,
    mode: str,
    is_key: bool = False,
    inference: Inference | None = None,
) -> ColumnDecision:
    """Choose a strategy for one column.

    A curated classification always wins. `inference` is consulted only when the
    catalog has nothing to say, and never forces differential privacy -- see
    app/atlas/inference.py for why.
    """
    tags = classification.classifications if classification else []

    if is_key:
        # Keys are minted fresh by the relational layer regardless; recording
        # the decision keeps the audit trail complete.
        return ColumnDecision(
            column=column_name,
            strategy=PSEUDONYM,
            reason="primary or foreign key: surrogate key generated, original never reused",
            classifications=tags,
        )

    if classification is None or not classification.is_classified:
        # The catalog is silent. Before falling through to MODEL, check whether
        # the column looks like something that must not be modelled raw. This is
        # the floor that stops an untagged `full_name` being learned as a
        # categorical and re-emitted verbatim.
        if inference is not None and mode != PERMISSIVE:
            return ColumnDecision(
                column=column_name,
                strategy=PSEUDONYM,
                reason=(
                    f"NOT IN CATALOG -- protected by inference: {inference.reason}. "
                    "Tag this column PII in Atlas to make the decision authoritative."
                ),
                classifications=inference.tags,
                encoder_kind="non_std",
                inferred=True,
            )

        if mode == STRICT and table_has_sensitive:
            return ColumnDecision(
                column=column_name,
                strategy=PSEUDONYM,
                reason=(
                    "no Atlas classification, and this table carries sensitive "
                    "columns; strict mode fails closed"
                ),
                classifications=[],
                encoder_kind="non_std",
                inferred=True,
            )

        return ColumnDecision(
            column=column_name,
            strategy=MODEL,
            reason=(
                "no Atlas classification found -- review and tag this column"
                if table_has_sensitive
                else "no Atlas classification; treated as non-sensitive"
            ),
            classifications=[],
        )

    if classification.is_pci:
        return ColumnDecision(
            column=column_name,
            strategy=SUPPRESS,
            reason=f"Atlas classification {tags} is PCI scope: column dropped entirely",
            classifications=tags,
        )

    if classification.is_direct_identifier:
        return ColumnDecision(
            column=column_name,
            strategy=PSEUDONYM,
            reason=(
                f"Atlas classification {tags} marks a direct identifier: replaced "
                "with an unlinkable surrogate, never modelled"
            ),
            classifications=tags,
            encoder_kind="non_std",
        )

    if classification.is_quasi_identifier:
        return ColumnDecision(
            column=column_name,
            strategy=GENERALISE,
            reason=(
                f"Atlas classification {tags} marks a quasi-identifier: coarsened "
                "before modelling to limit re-identification by combination"
            ),
            classifications=tags,
            encoder_kind="categorical",
        )

    if classification.is_sensitive:
        return ColumnDecision(
            column=column_name,
            strategy=MODEL_DP,
            reason=(
                f"Atlas classification {tags} marks sensitive data: modelled, and "
                "differential privacy is required for this run"
            ),
            classifications=tags,
        )

    return ColumnDecision(
        column=column_name,
        strategy=MODEL,
        reason=f"Atlas classification {tags} is not restricted: modelled normally",
        classifications=tags,
    )


def build_policy(
    atlas: AtlasResult,
    table_columns: dict[str, list[str]],
    keys_by_table: dict[str, set[str]] | None = None,
    mode: str = BALANCED,
    frames: dict[str, Any] | None = None,
) -> PolicyResult:
    """Produce a decision for every column of every selected table.

    `frames` is optional and only improves inference: with the data in hand, a
    column whose name reveals nothing can still be caught by its values.
    """
    keys_by_table = keys_by_table or {}
    frames = frames or {}
    tables: dict[str, TablePolicy] = {}

    for table_name, columns in table_columns.items():
        table_classification = atlas.tables.get(table_name)
        column_map = table_classification.columns if table_classification else {}
        frame = frames.get(table_name)

        table_has_sensitive = any(
            c.is_direct_identifier or c.is_pci or c.is_sensitive
            for c in column_map.values()
        )

        policy = TablePolicy(table=table_name)
        keys = keys_by_table.get(table_name, set())
        for column_name in columns:
            existing = column_map.get(column_name)
            inference = None
            # Only bother inferring where the catalog is silent, and never for a
            # key -- keys are minted fresh and never modelled anyway.
            if column_name not in keys and (existing is None or not existing.is_classified):
                series = None
                if frame is not None and column_name in getattr(frame, "columns", []):
                    series = frame[column_name]
                inference = infer_column(column_name, series)

            policy.decisions.append(
                decide_column(
                    existing,
                    column_name,
                    table_has_sensitive=table_has_sensitive,
                    mode=mode,
                    is_key=column_name in keys,
                    inference=inference,
                )
            )
        tables[table_name] = policy

    return PolicyResult(
        tables=tables, source=atlas.source, detail=atlas.detail, mode=mode
    )


def apply_policy_to_frames(
    frames: dict[str, Any], policy: PolicyResult
) -> dict[str, Any]:
    """Drop suppressed columns and coarsen generalised ones, before training."""
    import pandas as pd

    out: dict[str, Any] = {}
    for table_name, frame in frames.items():
        table_policy = policy.tables.get(table_name)
        if table_policy is None:
            out[table_name] = frame
            continue

        working = frame.copy()

        for column in table_policy.suppressed:
            if column in working.columns:
                working = working.drop(columns=[column])

        for column in table_policy.generalised:
            if column not in working.columns:
                continue
            series = working[column]
            # A date column read from CSV is `object`, so parse before deciding
            # how to coarsen it -- otherwise a quasi-identifier date falls
            # through to the categorical branch and is never generalised at all.
            if not pd.api.types.is_datetime64_any_dtype(
                series
            ) and not pd.api.types.is_numeric_dtype(series):
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.95:
                    series = parsed

            if pd.api.types.is_datetime64_any_dtype(series):
                # Dates of birth become the year only. Year-month leaves ~12x
                # more categories than the model can learn from a typical
                # table, and month-of-birth adds little analytical value while
                # materially narrowing a re-identification search.
                working[column] = series.dt.year.astype("Int64").astype(str)
            elif pd.api.types.is_numeric_dtype(series):
                # Numeric quasi-identifiers (age) become bands.
                try:
                    working[column] = pd.cut(series, bins=10).astype(str)
                except (ValueError, TypeError):
                    pass

        out[table_name] = working
    return out


__all__ = [
    "BALANCED",
    "GENERALISE",
    "MODEL",
    "MODEL_DP",
    "PERMISSIVE",
    "PSEUDONYM",
    "STRICT",
    "SUPPRESS",
    "ColumnDecision",
    "PolicyResult",
    "TablePolicy",
    "apply_policy_to_frames",
    "build_policy",
    "decide_column",
]
