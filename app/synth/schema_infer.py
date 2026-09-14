"""Relational structure detection and schema validation.

Two jobs, both of which the platform previously left to the user:

**Detection.** Work out each table's primary key, which of its columns are
foreign keys into other tables, and therefore whether the dataset is relational
at all. That answer drives the engine default, and it is what the Auto-configure
button fills the Relational Schema in with.

**Validation.** Refuse a schema that cannot be true. This exists because of a
real incident: a configuration declared `primary_key = "customer_id"` on all six
tables of a relational dataset and no foreign keys at all. Every table then
minted its own private `customer_id` space, so all seven joins came out 100%
orphaned -- and because per-column marginals were still excellent, the run
reported 0.946 fidelity with no warnings. Structural nonsense has to be caught
before training, not inferred from a disappointing report afterwards.

A declared primary key that is not unique in the data is not a preference to be
honoured; it is a contradiction, and generation cannot be correct under it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A child column must have at least this fraction of its values present in the
# parent's key for the pair to be treated as a genuine foreign key. Not 1.0:
# real extracts carry a tail of rows whose parent was filtered out.
CONTAINMENT_THRESHOLD = 0.90

ERROR = "error"
WARNING = "warning"


@dataclass
class SchemaIssue:
    level: str
    table: str
    message: str
    fix: str

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "table": self.table,
            "message": self.message,
            "fix": self.fix,
        }


@dataclass
class ForeignKey:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    containment: float

    def to_json(self) -> dict[str, Any]:
        return {
            "child_table": self.child_table,
            "child_column": self.child_column,
            "parent_table": self.parent_table,
            "parent_column": self.parent_column,
            "containment": round(self.containment, 4),
        }


@dataclass
class InferredSchema:
    primary_keys: dict[str, str] = field(default_factory=dict)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    @property
    def is_relational(self) -> bool:
        return bool(self.foreign_keys)

    def foreign_keys_of(self, table: str) -> list[ForeignKey]:
        return [fk for fk in self.foreign_keys if fk.child_table == table]

    def key_columns_of(self, table: str) -> set[str]:
        keys = {self.primary_keys[table]} if table in self.primary_keys else set()
        keys |= {fk.child_column for fk in self.foreign_keys_of(table)}
        return keys

    def to_json(self) -> dict[str, Any]:
        return {
            "primary_keys": self.primary_keys,
            "foreign_keys": [fk.to_json() for fk in self.foreign_keys],
            "is_relational": self.is_relational,
        }


def _singular(table: str) -> str:
    lowered = table.lower()
    if lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if lowered.endswith("ses"):
        return lowered[:-2]
    if lowered.endswith("s"):
        return lowered[:-1]
    return lowered


def _looks_like_key(column: str) -> bool:
    lowered = column.lower()
    return lowered.endswith(("_id", "_key", "_no", "_code")) or lowered in {
        "id",
        "key",
        "cif",
        "uuid",
        "guid",
    }


def _is_unique(frame, column: str) -> bool:
    series = frame[column]
    if series.isna().any():
        return False
    return int(series.nunique()) == len(frame) and len(frame) > 0


def infer_schema(frames: dict[str, Any]) -> InferredSchema:
    """Derive primary and foreign keys from the data itself."""
    schema = InferredSchema()

    # Pass 1: primary keys. A key must actually be unique and non-null, so a
    # shared column like customer_id is eliminated in every child table
    # automatically -- it only survives in the table it identifies.
    candidates: dict[str, list[str]] = {}
    for table, frame in frames.items():
        if frame is None or not len(frame):
            continue
        unique_keyish = [
            c for c in frame.columns if _looks_like_key(c) and _is_unique(frame, c)
        ]
        candidates[table] = unique_keyish
        preferred = f"{_singular(table)}_id"
        for column in unique_keyish:
            if column.lower() == preferred:
                schema.primary_keys[table] = column
                break

    # Anything still undecided: take a unique key-like column that is not
    # already another table's primary key, so `loan_payments` picks payment_id
    # rather than loan_id.
    claimed = set(schema.primary_keys.values())
    for table, unique_keyish in candidates.items():
        if table in schema.primary_keys:
            continue
        remaining = [c for c in unique_keyish if c not in claimed]
        if remaining:
            schema.primary_keys[table] = remaining[0]
        elif unique_keyish:
            schema.primary_keys[table] = unique_keyish[0]

    # Pass 2: foreign keys. A column matching another table's primary key by
    # name, confirmed by checking that its values actually live there.
    for child, frame in frames.items():
        if frame is None or not len(frame):
            continue
        child_pk = schema.primary_keys.get(child)
        for parent, parent_pk in schema.primary_keys.items():
            if parent == child or parent_pk not in frame.columns:
                continue
            if parent_pk == child_pk:
                continue
            parent_frame = frames.get(parent)
            if parent_frame is None or parent_pk not in parent_frame.columns:
                continue

            child_values = frame[parent_pk].dropna()
            if child_values.empty:
                continue
            parent_values = set(parent_frame[parent_pk].dropna().unique())
            containment = float(child_values.isin(parent_values).mean())
            if containment >= CONTAINMENT_THRESHOLD:
                schema.foreign_keys.append(
                    ForeignKey(
                        child_table=child,
                        child_column=parent_pk,
                        parent_table=parent,
                        parent_column=parent_pk,
                        containment=containment,
                    )
                )

    _order_by_depth(schema)
    return schema


def _table_depth(schema: InferredSchema) -> dict[str, int]:
    """How far each table sits from a root: 0 for a table with no parents."""
    parents: dict[str, set[str]] = {}
    for fk in schema.foreign_keys:
        parents.setdefault(fk.child_table, set()).add(fk.parent_table)

    depth = {table: 0 for table in parents}
    for fk in schema.foreign_keys:
        depth.setdefault(fk.parent_table, 0)

    # Relax until stable. Bounded by the number of tables, so a cyclic
    # declaration cannot spin here.
    for _ in range(len(depth) + 1):
        changed = False
        for table, table_parents in parents.items():
            candidate = 1 + max((depth.get(p, 0) for p in table_parents), default=-1)
            if candidate > depth.get(table, 0):
                depth[table] = candidate
                changed = True
        if not changed:
            break
    return depth


def _order_by_depth(schema: InferredSchema) -> None:
    """Put each child's immediate parent first among its foreign keys.

    Everything downstream treats foreign key [0] as *the* parent: the degree
    model is learned against it, max_degree is sized from it, and generation
    conditions the child on it. So the order is not cosmetic.

    A loan payment belongs to a loan, which belongs to a customer. Ordered by
    discovery instead, `customer_id` came first, so max_degree was sized from
    payments-per-customer (max 34 -> 42) while generation drew degrees per
    *loan* (max 15 -> 19). The histogram spanned more than twice the range it
    should and the child table came out 2.3x too large.
    """
    depth = _table_depth(schema)
    schema.foreign_keys.sort(
        key=lambda fk: (fk.child_table, -depth.get(fk.parent_table, 0), fk.parent_table)
    )


def validate_schema(
    table_args: dict[str, Any], frames: dict[str, Any], inferred: InferredSchema | None = None
) -> list[SchemaIssue]:
    """Check a declared schema against the data. Errors must block training."""
    inferred = inferred or infer_schema(frames)
    issues: list[SchemaIssue] = []

    declared_pks = {
        table: (args or {}).get("primary_key")
        for table, args in table_args.items()
    }

    # Driven by the tables that exist, not by the ones that were declared. A
    # configuration with no Relational Schema at all is the most common way to
    # get an orphaned dataset, and iterating the declaration would skip it
    # entirely.
    for table in frames:
        args = table_args.get(table) or {}
        frame = frames.get(table)
        if frame is None:
            continue

        if not args and inferred.foreign_keys_of(table):
            listed = ", ".join(
                f"{fk.child_column} -> {fk.parent_table}.{fk.parent_column}"
                for fk in inferred.foreign_keys_of(table)
            )
            issues.append(
                SchemaIssue(
                    ERROR,
                    table,
                    (
                        f"nothing is declared for {table}, but the data shows "
                        f"{len(inferred.foreign_keys_of(table))} foreign key(s) "
                        f"({listed}). Generated as-is it would be an independent table "
                        "with every join orphaned."
                    ),
                    "run Auto-configure to derive the relational schema",
                )
            )
            continue

        declared_pk = args.get("primary_key")
        declared_fks = args.get("foreign_keys") or []

        if declared_pk:
            if declared_pk not in frame.columns:
                issues.append(
                    SchemaIssue(
                        ERROR,
                        table,
                        f"declared primary key '{declared_pk}' is not a column of {table}",
                        f"set primary_key to one of: {', '.join(map(str, frame.columns))}",
                    )
                )
            elif not _is_unique(frame, declared_pk):
                distinct = int(frame[declared_pk].nunique())
                suggestion = inferred.primary_keys.get(table)
                issues.append(
                    SchemaIssue(
                        ERROR,
                        table,
                        (
                            f"declared primary key '{declared_pk}' is not unique: "
                            f"{distinct:,} distinct values across {len(frame):,} rows. "
                            "Keys are minted fresh per table, so declaring a shared "
                            "column as the primary key gives this table its own "
                            "private key space and orphans every join into it."
                        ),
                        (
                            f"set primary_key to '{suggestion}'"
                            if suggestion
                            else "declare the column that uniquely identifies a row"
                        )
                        + (
                            f" and declare '{declared_pk}' as a foreign key"
                            if any(
                                fk.child_column == declared_pk
                                for fk in inferred.foreign_keys_of(table)
                            )
                            else ""
                        ),
                    )
                )

        # The specific shape of the incident: a foreign key declared as this
        # table's primary key, while the table it points at declares it too.
        if declared_pk:
            other_owners = [
                other
                for other, pk in declared_pks.items()
                if other != table and pk == declared_pk
            ]
            if other_owners and not _is_unique(frame, declared_pk):
                issues.append(
                    SchemaIssue(
                        ERROR,
                        table,
                        (
                            f"'{declared_pk}' is declared as the primary key of "
                            f"{table} and also of {', '.join(other_owners)}. A column "
                            "cannot be the primary key of more than one table; here it "
                            "is a foreign key."
                        ),
                        f"declare '{declared_pk}' as a foreign key in {table}",
                    )
                )

        # Relational structure exists in the data but was not declared.
        expected = inferred.foreign_keys_of(table)
        if expected and not declared_fks:
            listed = ", ".join(
                f"{fk.child_column} -> {fk.parent_table}.{fk.parent_column}"
                for fk in expected
            )
            issues.append(
                SchemaIssue(
                    ERROR,
                    table,
                    (
                        f"{table} has {len(expected)} foreign key(s) in the data but none "
                        f"declared ({listed}). Without them the table is generated "
                        "independently and every join is orphaned."
                    ),
                    "run Auto-configure, or declare the foreign keys above",
                )
            )

        # Declared foreign keys that the data does not support.
        for fk in declared_fks:
            child_columns = fk.get("child_column_names") or []
            parent_table = fk.get("parent_table_name")
            parent_columns = fk.get("parent_column_names") or []
            if not child_columns or not parent_table or not parent_columns:
                issues.append(
                    SchemaIssue(
                        ERROR, table, f"incomplete foreign key declaration: {fk}",
                        "each foreign key needs parent_table_name, parent_column_names "
                        "and child_column_names",
                    )
                )
                continue
            child_column, parent_column = child_columns[0], parent_columns[0]
            parent_frame = frames.get(parent_table)
            if child_column not in frame.columns:
                issues.append(
                    SchemaIssue(
                        ERROR, table,
                        f"foreign key column '{child_column}' is not a column of {table}",
                        "correct the child_column_names",
                    )
                )
                continue
            if parent_frame is None or parent_column not in parent_frame.columns:
                issues.append(
                    SchemaIssue(
                        ERROR, table,
                        f"foreign key points at {parent_table}.{parent_column}, "
                        "which does not exist",
                        "correct the parent_table_name / parent_column_names",
                    )
                )
                continue
            child_values = frame[child_column].dropna()
            if child_values.empty:
                continue
            parent_values = set(parent_frame[parent_column].dropna().unique())
            containment = float(child_values.isin(parent_values).mean())
            if containment < CONTAINMENT_THRESHOLD:
                issues.append(
                    SchemaIssue(
                        WARNING, table,
                        (
                            f"only {containment:.1%} of {table}.{child_column} values exist "
                            f"in {parent_table}.{parent_column}; the source data itself has "
                            "orphans here"
                        ),
                        "confirm the relationship, or accept that the same orphan rate "
                        "will appear in the output",
                    )
                )

    # A table nothing can join to, in an otherwise relational dataset.
    if inferred.is_relational:
        for table in frames:
            if table in inferred.primary_keys:
                continue
            if frames.get(table) is None:
                continue
            issues.append(
                SchemaIssue(
                    WARNING, table,
                    f"no unique key could be found for {table}, so it cannot be joined to",
                    "add a primary key column, or generate this table on its own",
                )
            )

    return issues


def recommend_engine(inferred: InferredSchema) -> tuple[str, str]:
    """Pick the engine that suits the dataset's shape.

    The split is deliberate and each engine owns one regime:

        relational   -> SPN, whose conditional sampling exists for exactly this
                        (children drawn from P(child | parent)), and which is the
                        only engine that can deliver differential privacy, so
                        relational data is not cut off from it.
        independent  -> ARF, which reports its own convergence and is built for
                        interactions inside a single wide table.

    Measured on this dataset: SPN on the correctly-configured relational schema
    gave 0.967 fidelity with 0 orphans across all 7 foreign keys and
    children-per-parent ratios of 0.80-1.10.

    One caveat worth keeping visible: on a single table the SPN also measured
    better than ARF (utility 1.008 vs 0.747, detection 0.510 vs 0.645 on the
    1,500-row customer extract), and it is far cheaper to fit. The recommendation
    is therefore advisory -- the engine can be changed, and the training report
    always records which engine was recommended and which was used.
    """
    if inferred.is_relational:
        return (
            "SPN",
            (
                f"dataset is relational ({len(inferred.foreign_keys)} foreign key(s) "
                "detected): the SPN samples children conditioned on their parent, keeps "
                "foreign keys intact, and is the only engine that supports differential "
                "privacy."
            ),
        )
    return (
        "ARF",
        (
            "dataset has no foreign keys, so each table is modelled on its own: ARF is "
            "built for interactions within a single wide table and reports its own "
            "convergence. Switch to the SPN if this run needs differential privacy."
        ),
    )


__all__ = [
    "CONTAINMENT_THRESHOLD",
    "ERROR",
    "WARNING",
    "ForeignKey",
    "InferredSchema",
    "SchemaIssue",
    "infer_schema",
    "recommend_engine",
    "validate_schema",
]
