"""How much of schema.yaml does this POC actually implement?

There are three different answers to that question and conflating them
overstates what has been built:

  PARSED    the loader reads the parameter and knows its type, options,
            documentation and group. All 460 keys are parsed.
  SURFACED  the parameter is rendered in the Data Arguments UI, so a user can
            set it. Anything typed `skipped` in the source is deliberately not.
  CONSUMED  the training pipeline actually reads the value and changes its
            behaviour because of it. This is the only tier that means the
            parameter *works*.

A parameter can be surfaced but not consumed -- the UI will happily accept it
and the engine will ignore it. That gap is what this report measures.

Usage:
    python -m scripts.schema_coverage
    python -m scripts.schema_coverage --group "Relational Schema" --verbose
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from app.config import settings
from app.core.schema_loader import DatasetContext, TableContext, load_registry

# The parameters the pipeline genuinely reads, verified by hand against the
# source rather than by name matching.
#
# A grep for the parameter name is far too generous: `base`, `metric`, `p`, `tol`
# and `unit` all appear in unrelated code, which inflated the first version of
# this report from ~25 consumed parameters to 91. If you add a parameter to the
# engine, add it here too -- the number is only worth anything if it is honest.
CONSUMED = {
    # Relational Schema -- read by app/synth/relational.py TableSpec.from_config
    "table_args": "TableSpec per table",
    "order": "processing order (parents first)",
    "primary_key": "surrogate key minted per table",
    "foreign_keys": "FK-aware generation",
    "parent_table_name": "ForeignKeySpec",
    "parent_column_names": "ForeignKeySpec",
    "child_column_names": "ForeignKeySpec",
    "timeseries": "sorts rows within a static id",
    "static_ids": "grouping for time-series sort",
    "sortby": "sort column for time-series",

    # Column Type Inference -- read by TableEncoder.fit / _kind_overrides
    "categorical_columns": "encoder type override",
    "numerical_columns": "encoder type override",
    "datetime_columns": "encoder type override",
    "timedelta_columns": "encoder type override",
    "non_std_columns": "placeholder column",
    "max_n_category": "category cap before 'other' bucket",
    "unique_threshold": "categorical vs numerical decision",
    "force_min_category": "categorical vs numerical decision",

    # Column Data Specification -- read as the declared DP domain
    "column_kwargs": "per-column settings container",
    "min_val": "declared domain lower bound",
    "max_val": "declared domain upper bound",
    "force_categories": "declared category list",

    # Training Arguments -- SPN_Config
    "spn_config": "SPNParams",
    "beta": "min leaf size for a SUM node",
    "private": "differential privacy on/off",
    "epsilon": "DP budget",
    "p_value_threshold": "independence test threshold",
    "max_clusters": "branching factor at SUM nodes",

    # Aggregation Specification -- read by app/synth/aggregate.py
    "global_default_args": "aggregation defaults for every column",
    "default_discrete_kwargs": "aggregation defaults for categorical columns",
    "default_continuous_kwargs": "aggregation defaults for numeric columns",
    "default_ts_numerical_kwargs": "aggregation defaults, time-series numeric",
    "default_ts_mixed_kwargs": "aggregation defaults, time-series mixed",
    "agg_include_na_ratio": "null fraction per parent",
    "agg_include_unique_ratio": "distinct/count per parent",
    "agg_include_n_unique": "distinct count per parent",
    "agg_k_top": "k most frequent child values",
    "agg_k_first": "first k child values (ordered)",
    "agg_k_last": "last k child values (ordered)",
    "agg_include_mean": "mean per parent",
    "agg_include_std": "standard deviation per parent",
    "agg_include_sum": "sum per parent",
    "agg_include_log_mean": "mean of log1p",
    "agg_include_log_std": "std of log1p",
    "agg_include_skew": "skewness",
    "agg_include_kurt": "kurtosis",
    "agg_quantiles": "quantiles per parent",
    "agg_include_end_start_diff": "last minus first",
    "agg_include_increase_ratio": "fraction of increasing steps",
    "agg_include_decrease_ratio": "fraction of decreasing steps",
    "agg_check_increasing": "monotonic increasing flag",
    "agg_check_decreasing": "monotonic decreasing flag",
    "agg_first_max_position": "normalised argmax position",
    "agg_first_min_position": "normalised argmin position",
    "agg_last_max_position": "normalised last-argmax position",
    "agg_last_min_position": "normalised last-argmin position",
    "agg_autocorr_lags": "autocorrelation at the given lags",
    "agg_proportions_of": "proportion of the listed categories",

    # Read, stored, but currently without effect on output
    "drop_columns": "columns removed before training",
    "target_columns": "accepted by the utility evaluator, never populated",
}

# Extensions of our own, not present in the vendor schema.
OUR_EXTENSIONS = {
    "agg_enabled": "master switch for aggregation (off: it costs DP budget)",
    "agg_max_columns": "cap on generated aggregate columns per parent",
    "transform": "log transform for heavy-tailed columns under DP",
    "nullable": "declared nullability, so DP does not invent nulls",
    "max_degree": "declared children-per-parent bound for the DP degree histogram",
}

def is_consumed(name: str) -> bool:
    """Is this parameter on the hand-verified consumed list?"""
    return name.split("#")[0] in CONSUMED


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=None, help="only report this group")
    parser.add_argument("--verbose", action="store_true", help="list every parameter")
    args = parser.parse_args()

    raw = yaml.safe_load(
        open(settings.schema_yaml_path, encoding="utf-8", errors="replace")
    )
    registry = load_registry()

    ctx = DatasetContext(
        tables=[
            TableContext("customers", ["customer_id", "segment", "annual_income"]),
            TableContext("orders", ["order_id", "customer_id", "amount"]),
        ]
    )

    # Everything the UI actually renders.
    surfaced: set[str] = set()

    def walk(nodes):
        for node in nodes:
            surfaced.add(node["key"])
            for child_key in ("children", "itemSchema"):
                if node.get(child_key):
                    walk(node[child_key])

    for section in registry.build_tree(ctx):
        walk(section["fields"])

    groups: dict[str, dict[str, list[str]]] = {}
    totals = {"parsed": 0, "skipped": 0, "surfaced": 0, "consumed": 0}

    for name, definition in raw.items():
        if not isinstance(definition, dict):
            continue
        type_token = str(definition.get("type", "")).strip().lower()
        if type_token == "group-tag":
            continue

        totals["parsed"] += 1
        group = definition.get("groupTag") or "(no group)"
        bucket = groups.setdefault(
            group, {"skipped": [], "surfaced_only": [], "consumed": []}
        )

        skipped = type_token.startswith("skipped")
        bare = name.split("#")[0]
        in_ui = bare in surfaced
        consumed = is_consumed(name)

        if consumed:
            totals["consumed"] += 1
            bucket["consumed"].append(name)
        elif skipped:
            totals["skipped"] += 1
            bucket["skipped"].append(name)
        elif in_ui:
            totals["surfaced"] += 1
            bucket["surfaced_only"].append(name)
        else:
            totals["skipped"] += 1
            bucket["skipped"].append(name)

    print("=" * 74)
    print("schema.yaml coverage")
    print("=" * 74)
    print(f"  parameters in source        {totals['parsed']}")
    print(f"  CONSUMED by the engine      {totals['consumed']}"
          f"   ({totals['consumed'] / totals['parsed']:.0%})")
    print(f"  surfaced in UI, ignored     {totals['surfaced']}"
          f"   ({totals['surfaced'] / totals['parsed']:.0%})")
    print(f"  not surfaced (skipped)      {totals['skipped']}"
          f"   ({totals['skipped'] / totals['parsed']:.0%})")

    print("\nby group")
    print(f"  {'group':<34}{'consumed':>9}{'ignored':>9}{'skipped':>9}")
    order = [g["name"] for g in registry.groups] + ["(no group)"]
    for group in order:
        if group not in groups:
            continue
        if args.group and group != args.group:
            continue
        bucket = groups[group]
        print(
            f"  {group[:33]:<34}{len(bucket['consumed']):>9}"
            f"{len(bucket['surfaced_only']):>9}{len(bucket['skipped']):>9}"
        )

    if args.verbose or args.group:
        for group in order:
            if group not in groups:
                continue
            if args.group and group != args.group:
                continue
            bucket = groups[group]
            print(f"\n--- {group} ---")
            for label, names in (
                ("CONSUMED", bucket["consumed"]),
                ("surfaced but ignored", bucket["surfaced_only"]),
                ("not surfaced", bucket["skipped"]),
            ):
                if not names:
                    continue
                print(f"  {label}:")
                for name in sorted(names)[:40]:
                    note = f"  <- {CONSUMED[name.split(chr(35))[0]]}" if name.split("#")[0] in CONSUMED else ""
                    print(f"    {name}{note}")
                if len(names) > 40:
                    print(f"    ... and {len(names) - 40} more")


    print("\nOur own additions, not in the vendor schema:")
    for name, why in OUR_EXTENSIONS.items():
        print(f"    {name:<14} {why}")


if __name__ == "__main__":
    main()
