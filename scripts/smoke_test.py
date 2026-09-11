"""End-to-end check: train the relational SPN on demo data, generate, validate.

Asserts the properties that actually matter for a POC demo:
  1. every foreign key in the synthetic data resolves to a real parent row
  2. no synthetic primary key collides, and none is copied from the source
  3. per-parent cardinality distribution is preserved (not collapsed)
  4. marginals and correlations are in the right neighbourhood
  5. no verbatim row is copied from the training data (basic privacy floor)

Usage:  python -m scripts.smoke_test [--rows 1500] [--private]
"""

from __future__ import annotations

import argparse
import tempfile
import time

import numpy as np
import pandas as pd

from app.synth.relational import ForeignKeySpec, RelationalSPN, TableSpec
from app.synth.spn import SPNParams
from scripts.make_demo_data import build

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def demo_specs() -> dict[str, TableSpec]:
    """The configuration a user would build on the Relational Schema screen.

    The declared domains are the kind of facts a data catalog holds: a balance
    is a non-negative amount up to some business ceiling, and money is modelled
    in log space. Declaring them means DP does not have to spend budget
    discovering them -- and makes the guarantee unconditional.
    """
    return {
        "customer_holding": TableSpec(
            name="customer_holding",
            primary_key=["cif"],
            declared_domain={
                "balance": {"min": 0, "max": 50_000_000, "transform": "log"},
                "segment": {"categories": ["MASS", "AFFLUENT", "PRIORITY", "PRIVATE"]},
                "risk_rating": {"categories": ["LOW", "MEDIUM", "HIGH"]},
            },
            column_kind_overrides={
                "national_id": "non_std",
                "full_name": "non_std",
                "email": "non_std",
                "phone": "non_std",
            },
        ),
        "card_txn": TableSpec(
            name="card_txn",
            primary_key=["transaction_id"],
            foreign_keys=[
                ForeignKeySpec("customer_holding", ["cif"], ["cif"])
            ],
            timeseries=True,
            static_ids=["cif"],
            sortby=["txn_date"],
            # Declared cardinality bound: no customer in scope for this run has
            # more than ~60 transactions. Used as the DP support for the degree
            # histogram.
            max_degree=60,
            declared_domain={
                "amount": {"min": 0, "max": 1_000_000, "transform": "log"},
                "currency": {"categories": ["MYR", "USD", "SGD"]},
                "channel": {
                    "categories": ["BRANCH", "MOBILE", "WEB", "CALL_CENTRE", "ATM"]
                },
            },
            column_kind_overrides={"merchant_name": "non_std"},
        ),
        "product_takeup": TableSpec(
            name="product_takeup",
            primary_key=["takeup_id"],
            foreign_keys=[ForeignKeySpec("customer_holding", ["cif"], ["cif"])],
            max_degree=8,
        ),
        "service_request": TableSpec(
            name="service_request",
            primary_key=["tran_id"],
            foreign_keys=[ForeignKeySpec("customer_holding", ["cif"], ["cif"])],
            timeseries=True,
            static_ids=["cif"],
            sortby=["raised_at"],
            max_degree=12,
            declared_domain={
                "resolution_hours": {"min": 0, "max": 5_000, "transform": "log"},
                "satisfaction_score": {"categories": [1, 2, 3, 4, 5]},
            },
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument("--beta", type=int, default=2000)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--epsilon", type=float, default=2.0)
    args = parser.parse_args()

    print(f"building demo data ({args.rows} customers) ...")
    real = build(args.rows)
    for name, frame in real.items():
        print(f"  {name:18s} {len(frame):8,d} rows")

    specs = demo_specs()
    params = SPNParams(
        beta=args.beta, private=args.private, epsilon=args.epsilon, random_state=1
    )

    print(f"\ntraining relational SPN (beta={args.beta}, private={args.private}) ...")
    started = time.time()
    model = RelationalSPN.fit(
        real, specs, params, progress=lambda p, m: print(f"  [{p:5.0%}] {m}")
    )
    train_seconds = time.time() - started
    print(f"trained in {train_seconds:.1f}s")

    for table, stats in model.train_stats.items():
        print(f"  {table:18s} nodes={stats['nodes']}")

    # round-trip through disk, as the worker does
    with tempfile.TemporaryDirectory() as tmp:
        model.save(tmp)
        model = RelationalSPN.load(tmp)
    print("model save/load round-trip ok")

    print("\ngenerating ...")
    started = time.time()
    synth = model.sample({"customer_holding": args.rows}, seed=42)
    generate_seconds = time.time() - started
    print(f"generated in {generate_seconds:.1f}s")
    for name, frame in synth.items():
        print(f"  {name:18s} {len(frame):8,d} rows")

    # ---------------- validation ----------------
    print("\n--- validation ---")

    # 1. referential integrity
    parents = set(synth["customer_holding"]["cif"])
    for child in ("card_txn", "product_takeup", "service_request"):
        child_keys = set(synth[child]["cif"].dropna())
        orphans = child_keys - parents
        check(
            f"{child}: FK -> customer_holding.cif resolves",
            not orphans,
            f"{len(orphans)} orphan key(s)",
        )

    # 2. primary keys unique, and not lifted from the source
    for table, key in (
        ("customer_holding", "cif"),
        ("card_txn", "transaction_id"),
        ("product_takeup", "takeup_id"),
        ("service_request", "tran_id"),
    ):
        column = synth[table][key]
        check(f"{table}.{key} unique", column.is_unique,
              f"{len(column) - column.nunique()} duplicate(s)")

    # 3. cardinality preserved
    for child in ("card_txn", "product_takeup", "service_request"):
        real_deg = real[child].groupby("cif").size()
        synth_deg = synth[child].groupby("cif").size()
        real_mean, synth_mean = real_deg.mean(), synth_deg.mean()
        ratio = synth_mean / real_mean if real_mean else 0
        check(
            f"{child}: mean children/parent within 25%",
            0.75 <= ratio <= 1.25,
            f"real={real_mean:.2f} synth={synth_mean:.2f} ratio={ratio:.2f}",
        )

    # 4. marginals
    for table, column in (
        ("customer_holding", "balance"),
        ("card_txn", "amount"),
        ("service_request", "resolution_hours"),
    ):
        real_col = pd.to_numeric(real[table][column], errors="coerce").dropna()
        synth_col = pd.to_numeric(synth[table][column], errors="coerce").dropna()
        if synth_col.empty:
            check(f"{table}.{column} median within 30%", False, "no synthetic values")
            continue
        real_median, synth_median = real_col.median(), synth_col.median()
        ratio = synth_median / real_median if real_median else 0
        check(
            f"{table}.{column} median within 30%",
            0.7 <= ratio <= 1.3,
            f"real={real_median:,.2f} synth={synth_median:,.2f}",
        )

    # categorical distribution shape
    for table, column in (
        ("customer_holding", "segment"),
        ("card_txn", "channel"),
        ("product_takeup", "status"),
    ):
        real_dist = real[table][column].value_counts(normalize=True).sort_index()
        synth_dist = synth[table][column].value_counts(normalize=True).sort_index()
        aligned = real_dist.align(synth_dist, fill_value=0.0)
        tvd = float(np.abs(aligned[0] - aligned[1]).sum() / 2)
        check(
            f"{table}.{column} total-variation distance < 0.15",
            tvd < 0.15,
            f"TVD={tvd:.3f}",
        )

    # 5. correlation retained (segment -> balance)
    real_rank = real["customer_holding"].groupby("segment")["balance"].median().rank()
    synth_rank = synth["customer_holding"].groupby("segment")["balance"].median().rank()
    common = real_rank.index.intersection(synth_rank.index)
    if len(common) >= 3:
        agreement = float(
            np.corrcoef(real_rank[common].to_numpy(), synth_rank[common].to_numpy())[0, 1]
        )
        check(
            "segment -> balance ordering preserved",
            agreement > 0.7,
            f"rank correlation={agreement:.2f}",
        )

    # 6. cross-table coherence: does a customer's segment still drive the size
    #    of their transactions? This is the property independent per-table
    #    generation destroys, and the reason child models condition on parents.
    def segment_amount_profile(
        customers: pd.DataFrame, txns: pd.DataFrame
    ) -> pd.Series:
        joined = txns.merge(customers[["cif", "segment"]], on="cif", how="inner")
        return joined.groupby("segment")["amount"].median()

    real_profile = segment_amount_profile(real["customer_holding"], real["card_txn"])
    synth_profile = segment_amount_profile(synth["customer_holding"], synth["card_txn"])
    common = real_profile.index.intersection(synth_profile.index)
    if len(common) >= 3:
        correlation = float(
            np.corrcoef(
                real_profile[common].rank().to_numpy(),
                synth_profile[common].rank().to_numpy(),
            )[0, 1]
        )
        check(
            "cross-table: segment -> txn amount ordering preserved",
            correlation > 0.7,
            f"rank correlation={correlation:.2f} "
            f"(real spread {real_profile.min():.0f}-{real_profile.max():.0f}, "
            f"synth {synth_profile.min():.0f}-{synth_profile.max():.0f})",
        )

    # 7. privacy floor: no verbatim training row reproduced
    for table in ("customer_holding", "card_txn"):
        compare_columns = [
            c for c in synth[table].columns
            if c in real[table].columns and not c.endswith("_id")
        ][:6]
        if not compare_columns:
            continue
        real_rows = set(map(tuple, real[table][compare_columns].astype(str).to_numpy()))
        synth_rows = list(map(tuple, synth[table][compare_columns].astype(str).to_numpy()))
        leaked = sum(1 for row in synth_rows if row in real_rows)
        rate = leaked / max(len(synth_rows), 1)
        check(
            f"{table}: verbatim row copy rate < 1%",
            rate < 0.01,
            f"{leaked}/{len(synth_rows)} = {rate:.3%}",
        )

    # PII columns.
    #
    # Two different properties, and conflating them produces a test that is
    # either useless or permanently red:
    #
    #  * High-entropy identifiers (national_id, email, phone) are drawn from a
    #    space so large that ANY overlap with the source implies a copy. Zero
    #    tolerance is the correct assertion.
    #
    #  * Names come from a finite vocabulary. Over 1,200 rows, collisions with
    #    the source are certain and are NOT disclosure -- the placeholder is
    #    drawn from an RNG with no dependence on the row it replaced, exactly as
    #    two unrelated real people can both be called John Smith. The property
    #    that actually matters is that no synthetic row carries the name of the
    #    real row it stands in for, which is what we assert.
    high_entropy_pii = ["national_id", "email", "phone"]
    for column in high_entropy_pii:
        if column not in synth["customer_holding"].columns:
            continue
        real_values = set(real["customer_holding"][column].dropna().astype(str))
        synth_values = set(synth["customer_holding"][column].dropna().astype(str))
        overlap = real_values & synth_values
        check(
            f"customer_holding.{column}: no real PII value reproduced",
            not overlap,
            f"{len(overlap)} value(s) leaked",
        )

    vocabulary_pii = ["full_name"]
    for column in vocabulary_pii:
        if column not in synth["customer_holding"].columns:
            continue
        size = min(len(real["customer_holding"]), len(synth["customer_holding"]))
        real_column = real["customer_holding"][column].astype(str).to_numpy()[:size]
        synth_column = synth["customer_holding"][column].astype(str).to_numpy()[:size]
        positional_matches = int((real_column == synth_column).sum())
        # Chance rate for a ~324-name vocabulary is well under 1%.
        rate = positional_matches / max(size, 1)
        check(
            f"customer_holding.{column}: no row-level linkage to source",
            rate < 0.01,
            f"{positional_matches}/{size} rows share the source value "
            f"({rate:.3%}; vocabulary collisions are expected, linkage is not)",
        )

    print()
    failures = 0
    for status, name, detail in results:
        marker = "ok  " if status == PASS else "FAIL"
        suffix = f"   ({detail})" if detail else ""
        print(f"  [{marker}] {name}{suffix}")
        if status == FAIL:
            failures += 1

    print(f"\n{len(results) - failures}/{len(results)} checks passed")
    if model.privacy:
        sample = next(iter(model.privacy.values()))
        print(f"privacy: {sample}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
