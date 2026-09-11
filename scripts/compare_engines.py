"""Head-to-head engine comparison on the same data, split and metrics.

Settles "which engine is better for this table" by measurement rather than by
argument. Both engines see the same training split and are scored on the same
held-out real rows with the same harness:

    utility ratio    train on synthetic, test on real, vs train on real
    detection AUC    can a classifier separate real from synthetic (0.5 = no)
    marginals        median per group, to show where each one drifts

Usage:
    python -m scripts.compare_engines --table customers
    python -m scripts.compare_engines --table customers --engines SPN ARF
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from app.synth.engines import get_engine
from app.synth.evaluate import evaluate_tables
from app.synth.preprocess import TableEncoder
from app.synth.relational import TableSpec

SAMPLE_DIR = Path("/data/sample")

# Columns a policy would protect anyway; excluded so the comparison is about
# density modelling rather than about placeholder generators.
PROTECTED = {"full_name", "email", "phone", "national_id", "customer_id", "order_id"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="customers")
    parser.add_argument("--engines", nargs="+", default=["SPN", "ARF"])
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--group-by", default="segment")
    parser.add_argument("--measure", default="annual_income")
    args = parser.parse_args()

    path = SAMPLE_DIR / f"{args.table}.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found; run make_customer_sample first")

    frame = pd.read_parquet(path)
    columns = [c for c in frame.columns if c not in PROTECTED]
    frame = frame[columns]

    # One split, reused by every engine.
    shuffled = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - args.holdout))
    train, test = shuffled.iloc[:cut], shuffled.iloc[cut:]
    print(f"{args.table}: {len(train):,} train / {len(test):,} holdout, {len(columns)} columns\n")

    results = []
    for name in args.engines:
        engine = get_engine(name)
        # Each engine gets settings suited to it; using one number for both
        # would handicap whichever it does not suit.
        config = (
            {"beta": max(len(train) // 10, 50), "private": False}
            if engine.name == "SPN"
            else {"n_trees": 60, "max_rounds": 5}
        )
        params = engine.build_params(config, random_state=1)

        encoder = TableEncoder.fit(train)
        matrix = encoder.transform(train)

        started = time.time()
        model = engine.fit(matrix, encoder.specs, params)
        fit_seconds = time.time() - started

        synthetic = encoder.inverse_transform(model.sample(len(train), seed=99))

        scores = evaluate_tables(
            {args.table: train}, {args.table: test}, {args.table: synthetic}
        )
        entry = scores["tables"].get(args.table, {})
        detection = (entry.get("detection") or {}).get("detection_auc")

        results.append(
            {
                "engine": engine.name,
                "fit_seconds": fit_seconds,
                "utility": scores.get("mean_utility_ratio"),
                "detection_auc": detection,
                "targets": scores.get("targets_evaluated"),
                "stats": engine.stats_of(model),
                "diagnostics": getattr(model, "diagnostics", {}),
                "synthetic": synthetic,
                "supports_dp": engine.supports_dp,
            }
        )

    print("=" * 74)
    print(f"{'engine':8s}{'fit s':>8s}{'utility':>10s}{'detect AUC':>12s}{'targets':>9s}  DP")
    print("=" * 74)
    for row in results:
        utility = f"{row['utility']:.3f}" if row["utility"] is not None else "n/a"
        auc = f"{row['detection_auc']:.3f}" if row["detection_auc"] is not None else "n/a"
        print(
            f"{row['engine']:8s}{row['fit_seconds']:8.1f}{utility:>10s}{auc:>12s}"
            f"{row['targets'] or 0:>9d}  {'yes' if row['supports_dp'] else 'no'}"
        )

    if args.group_by in frame.columns and args.measure in frame.columns:
        print(f"\nmedian {args.measure} by {args.group_by}")
        header = f"  {args.group_by:12s}{'real':>12s}"
        for row in results:
            header += f"{row['engine']:>12s}"
        print(header)
        keys = sorted(train[args.group_by].dropna().unique())
        real_median = train.groupby(args.group_by)[args.measure].median()
        for key in keys:
            line = f"  {str(key):12s}{real_median.get(key, float('nan')):12,.0f}"
            for row in results:
                value = row["synthetic"].groupby(args.group_by)[args.measure].median()
                line += f"{value.get(key, float('nan')):12,.0f}"
            print(line)

    for row in results:
        if row["diagnostics"]:
            accuracy = row["diagnostics"].get("discriminator_oob_accuracy")
            print(
                f"\n{row['engine']} diagnostics: rounds={row['diagnostics'].get('rounds')}"
                f" converged={row['diagnostics'].get('converged')}"
                f" oob_accuracy={[round(a, 3) for a in (accuracy or [])]}"
            )
        else:
            print(f"\n{row['engine']} structure: {row['stats']}")

    print(
        "\nutility 1.0 = a model trained on synthetic does as well on real data as one "
        "trained on real.\ndetection 0.5 = a classifier cannot tell real from synthetic."
    )


if __name__ == "__main__":
    main()
