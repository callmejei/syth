"""Structural and disclosure checks on generated data.

The fidelity score compares marginals, quantiles and correlation matrices. It is
per-column by construction, so it cannot see that a join is broken. That gap is
not hypothetical: an ARF run on a six-table dataset reported 0.946 fidelity with
no warnings while every one of its seven foreign keys was 100% orphaned, and
every synthetic row carried a real customer's name.

So a run is not judged by fidelity alone. These checks answer the questions a
reviewer actually asks -- can the tables still be joined, and did anything real
get out -- and they return a status that a good fidelity score cannot override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# A synthetic key that resolves to no parent row makes the child unusable in any
# analysis that joins. There is no tolerable rate for this.
ORPHAN_TOLERANCE = 0.0
# Children per parent, as a ratio of the real mean.
CARDINALITY_BAND = (0.75, 1.25)
# Fraction of synthetic rows allowed to carry a value that exists in the real
# column, for a protected column whose real values are near-unique per row
# (a national ID, an email). Any reuse there is a real value escaping.
PROTECTED_LEAK_TOLERANCE = 0.01
# The same, for a protected column drawn from a vocabulary (a person's name).
# A surrogate generator with its own name list collides with real names by
# coincidence; well above this rate suggests it is reproducing them instead.
VOCABULARY_COLLISION_TOLERANCE = 0.40
# Verbatim whole-row copies.
ROW_COPY_TOLERANCE = 0.01


@dataclass
class Check:
    name: str
    status: str
    detail: str
    table: str = ""
    value: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "table": self.table,
            "value": self.value,
        }


@dataclass
class IntegrityResult:
    checks: list[Check] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(c.status == FAIL for c in self.checks):
            return FAIL
        if any(c.status == WARN for c in self.checks):
            return WARN
        return PASS

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warned(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": sum(1 for c in self.checks if c.status == PASS),
            "failed": len(self.failed),
            "warned": len(self.warned),
            "checks": [c.to_json() for c in self.checks],
        }


def _protected_columns(policy: dict[str, Any], table: str) -> list[str]:
    """Columns the policy promised not to reproduce."""
    entry = ((policy or {}).get("tables") or {}).get(table) or {}
    return sorted(set(entry.get("pseudonymised") or []) | set(entry.get("suppressed") or []))


def check_generation(
    real: dict[str, Any],
    synthetic: dict[str, Any],
    *,
    schema: Any = None,
    policy: dict[str, Any] | None = None,
) -> IntegrityResult:
    """Run every structural and disclosure check we can on a generated set.

    `schema` is an InferredSchema; when omitted the relational checks are
    skipped rather than guessed at.
    """
    result = IntegrityResult()
    policy = policy or {}

    # ---------------------------------------------------------------- keys
    if schema is not None:
        for table, pk in getattr(schema, "primary_keys", {}).items():
            frame = synthetic.get(table)
            if frame is None or pk not in getattr(frame, "columns", []):
                continue
            duplicates = len(frame) - int(frame[pk].nunique())
            result.checks.append(
                Check(
                    name=f"primary key {table}.{pk} unique",
                    status=PASS if duplicates == 0 else FAIL,
                    detail=f"{duplicates:,} duplicate value(s)",
                    table=table,
                    value=float(duplicates),
                )
            )

        for fk in getattr(schema, "foreign_keys", []):
            child = synthetic.get(fk.child_table)
            parent = synthetic.get(fk.parent_table)
            if child is None or parent is None:
                continue
            if fk.child_column not in child.columns or fk.parent_column not in parent.columns:
                continue

            kids = child[fk.child_column].dropna()
            if kids.empty:
                continue
            parents = set(parent[fk.parent_column].dropna().unique())
            orphan_rate = float((~kids.isin(parents)).mean())
            label = f"{fk.child_table}.{fk.child_column} -> {fk.parent_table}.{fk.parent_column}"
            result.checks.append(
                Check(
                    name=f"foreign key {label} resolves",
                    status=PASS if orphan_rate <= ORPHAN_TOLERANCE else FAIL,
                    detail=(
                        f"{orphan_rate:.1%} of child rows reference a parent that does "
                        "not exist"
                        + (
                            ". The tables were generated independently -- check that "
                            "this foreign key is declared in the configuration."
                            if orphan_rate > 0.5
                            else ""
                        )
                    ),
                    table=fk.child_table,
                    value=orphan_rate,
                )
            )

            # Cardinality: a child table that collapses to one row per parent has
            # lost the distribution even when every value is individually right.
            real_child = real.get(fk.child_table)
            if real_child is not None and fk.child_column in real_child.columns:
                real_mean = float(real_child.groupby(fk.child_column).size().mean())
                synth_mean = float(child.groupby(fk.child_column).size().mean())
                ratio = synth_mean / real_mean if real_mean else 0.0
                low, high = CARDINALITY_BAND
                result.checks.append(
                    Check(
                        name=f"children per parent for {label}",
                        status=PASS if low <= ratio <= high else WARN,
                        detail=(
                            f"real={real_mean:.2f} synthetic={synth_mean:.2f} "
                            f"ratio={ratio:.2f}"
                        ),
                        table=fk.child_table,
                        value=ratio,
                    )
                )

        # A redundant key path (a child carrying both its parent's key and its
        # grandparent's) must agree, or the two joins tell different stories.
        by_child: dict[str, list[Any]] = {}
        for fk in getattr(schema, "foreign_keys", []):
            by_child.setdefault(fk.child_table, []).append(fk)
        for table, fks in by_child.items():
            if len(fks) < 2:
                continue
            frame = synthetic.get(table)
            if frame is None:
                continue
            for direct in fks:
                for other in fks:
                    if direct is other:
                        continue
                    bridge = synthetic.get(direct.parent_table)
                    if bridge is None or other.child_column not in getattr(
                        bridge, "columns", []
                    ):
                        continue
                    if (
                        direct.child_column not in frame.columns
                        or other.child_column not in frame.columns
                    ):
                        continue
                    merged = frame[[direct.child_column, other.child_column]].merge(
                        bridge[[direct.parent_column, other.child_column]],
                        left_on=direct.child_column,
                        right_on=direct.parent_column,
                        how="inner",
                        suffixes=("_child", "_bridge"),
                    )
                    left = f"{other.child_column}_child"
                    right = f"{other.child_column}_bridge"
                    if not len(merged) or left not in merged or right not in merged:
                        continue
                    mismatch = float((merged[left] != merged[right]).mean())
                    result.checks.append(
                        Check(
                            name=(
                                f"{table}.{other.child_column} agrees with "
                                f"{direct.parent_table}.{other.child_column}"
                            ),
                            status=PASS if mismatch <= 0.01 else WARN,
                            detail=(
                                f"{mismatch:.1%} of rows disagree on the same "
                                "relationship reached two ways"
                            ),
                            table=table,
                            value=mismatch,
                        )
                    )

    # ------------------------------------------------------------ disclosure
    for table, frame in synthetic.items():
        real_frame = real.get(table)
        if real_frame is None or frame is None:
            continue

        for column in _protected_columns(policy, table):
            if column not in frame.columns or column not in real_frame.columns:
                continue
            real_series = real_frame[column].dropna().astype(str)
            real_values = set(real_series)
            synth_values = frame[column].dropna().astype(str)
            if synth_values.empty or not real_values:
                continue
            leak = float(synth_values.isin(real_values).mean())

            # Whether a shared value is a disclosure depends on what the column
            # is. A national ID or an email is near-unique per person, so any
            # reuse means a real individual's value was emitted. A name is drawn
            # from a vocabulary -- 378 distinct names across 2,000 real customers
            # here -- so a surrogate generator working from its own name list
            # will collide with the real data by coincidence, and that is not
            # linkage: the surrogate is attached to a freshly minted key with no
            # correspondence to any real row. Judging both by the same threshold
            # either misses real leaks or cries wolf over common names.
            uniqueness = len(real_values) / max(len(real_series), 1)
            near_unique = uniqueness > 0.9

            if near_unique:
                status = PASS if leak <= PROTECTED_LEAK_TOLERANCE else FAIL
                detail = (
                    f"{leak:.2%} of synthetic rows carry a value that exists in the real "
                    f"column, which is near-unique per row ({uniqueness:.0%} distinct) -- "
                    "any reuse is a real value escaping"
                )
            else:
                status = PASS if leak <= VOCABULARY_COLLISION_TOLERANCE else WARN
                detail = (
                    f"{leak:.2%} of synthetic rows share a value with the real column. "
                    f"This column is vocabulary-drawn ({len(real_values):,} distinct "
                    f"values across {len(real_series):,} rows), so coincidental "
                    "collisions are expected; surrogates carry minted keys, so a shared "
                    "value is not row-level linkage"
                )
            result.checks.append(
                Check(
                    name=f"protected column {table}.{column} not reproduced",
                    status=status,
                    detail=detail,
                    table=table,
                    value=leak,
                )
            )

        # Verbatim row copies, ignoring key columns (which are minted, so they
        # never match and would mask a copied row).
        key_columns = set()
        if schema is not None:
            key_columns = set(getattr(schema, "key_columns_of", lambda _t: set())(table))
        compared = [c for c in frame.columns if c not in key_columns and c in real_frame.columns]
        if compared:
            real_rows = set(map(tuple, real_frame[compared].astype(str).to_numpy()))
            synth_rows = list(map(tuple, frame[compared].astype(str).to_numpy()))
            if synth_rows:
                copied = sum(1 for row in synth_rows if row in real_rows) / len(synth_rows)
                result.checks.append(
                    Check(
                        name=f"{table} rows not copied verbatim",
                        status=PASS if copied <= ROW_COPY_TOLERANCE else WARN,
                        detail=f"{copied:.2%} of synthetic rows match a real row exactly",
                        table=table,
                        value=copied,
                    )
                )

    return result


__all__ = [
    "FAIL",
    "PASS",
    "WARN",
    "Check",
    "IntegrityResult",
    "check_generation",
]
