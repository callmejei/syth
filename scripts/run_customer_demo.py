"""End-to-end demo on the customers/orders sample, driven through the API.

Does exactly what a user would do in the UI, in order:
  upload -> auto-configure from Atlas -> train -> generate -> compare.

Usage:
    python -m scripts.run_customer_demo --rows 4000
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import httpx
import pandas as pd

BASE = "http://localhost:8000/api"
SAMPLE_DIR = Path("/data/sample")


def wait_for_job(job_id: str, label: str, timeout_s: int = 600) -> dict:
    deadline = time.time() + timeout_s
    last_message = ""
    while time.time() < deadline:
        job = httpx.get(f"{BASE}/jobs/{job_id}", timeout=30).json()
        if job.get("message") and job["message"] != last_message:
            last_message = job["message"]
            print(f"    [{job.get('progress', 0):5.0%}] {last_message}")
        if job["status"] in ("COMPLETED", "FAILED"):
            return job
        time.sleep(2)
    raise TimeoutError(f"{label} did not finish in {timeout_s}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4000, help="synthetic customers")
    parser.add_argument("--beta", type=int, default=400)
    parser.add_argument("--name", default="customer-demo")
    parser.add_argument(
        "--no-dp",
        action="store_true",
        help="skip policy enforcement (and therefore DP), to show the trade-off",
    )
    args = parser.parse_args()

    print("1. Uploading the sample dataset")
    files = [
        ("files", (Path(p).name, open(p, "rb"), "application/octet-stream"))
        for p in sorted(glob.glob(str(SAMPLE_DIR / "*.parquet")))
    ]
    if not files:
        raise SystemExit(f"no parquet files in {SAMPLE_DIR}; run make_customer_sample first")
    upload = httpx.post(
        f"{BASE}/data-sources/upload",
        data={"name": f"{args.name}-source"},
        files=files,
        timeout=300,
    ).json()
    data_source_id = upload["id"]
    print(f"   tables: {[t['name'] for t in upload['tables']]}")
    print(f"   classifications from: {upload['atlas']['source']}")

    print("\n2. Creating a configuration")
    configuration = httpx.post(
        f"{BASE}/configurations",
        json={
            "name": args.name,
            "model_type": "SPN",
            "data_source_id": data_source_id,
            "selected_tables": [t["name"] for t in upload["tables"]],
        },
        timeout=60,
    ).json()
    configuration_id = configuration["id"]

    print("\n3. Auto-configuring from Atlas")
    derived = httpx.post(
        f"{BASE}/configurations/{configuration_id}/autoconfigure", timeout=300
    ).json()
    print(f"   source: {derived['atlas_source']}   order: {' -> '.join(derived['order'])}")
    for table, blob in derived["table_args"].items():
        foreign = [
            f"{fk['parent_table_name']}.{fk['parent_column_names'][0]}"
            for fk in blob["foreign_keys"]
        ]
        print(
            f"   {table:12s} pk={blob['primary_key']}"
            f" fk={foreign or '-'}"
            f" protected={blob['non_std_columns']}"
        )

    httpx.patch(
        f"{BASE}/configurations/{configuration_id}",
        json={"training_params": {"spn_config": {"beta": args.beta, "private": False}}},
        timeout=60,
    )

    print("\n4. Training")
    started = httpx.post(
        f"{BASE}/configurations/{configuration_id}/train",
        json={
            "model_name": f"{args.name}-model",
            "enforce_policy": not args.no_dp,
        },
        timeout=60,
    ).json()
    job = wait_for_job(started["job_id"], "training")
    if job["status"] != "COMPLETED":
        print("   FAILED:", job["result"].get("error"))
        raise SystemExit(1)
    model_id = job["result"]["model_id"]

    report = httpx.get(f"{BASE}/models/{model_id}/report", timeout=60).json()
    print(f"\n   fidelity        {report['overall_fidelity']:.1%}")
    dp = report["differential_privacy"]
    print(f"   differential privacy: {'on' if dp['enabled'] else 'off'}")
    utility = report.get("utility") or {}
    if utility.get("available"):
        print(f"   utility ratio   {utility.get('mean_utility_ratio')}")
        print(f"   detection AUC   {utility.get('mean_detection_auc')}")

    print("\n5. Generating synthetic data")
    generated = httpx.post(
        f"{BASE}/generate",
        json={"model_id": model_id, "n_rows": {"customers": args.rows}, "seed": 7},
        timeout=60,
    ).json()
    job = wait_for_job(generated["job_id"], "generation")
    if job["status"] != "COMPLETED":
        print("   FAILED:", job["result"].get("error"))
        raise SystemExit(1)

    out_dir = Path(job["result"]["output_dir"])
    for entry in job["result"]["files"]:
        print(f"   {entry['table']:12s} {entry['rows']:8,d} rows -> {entry['path']}")

    # ---------------- comparison ----------------
    real_customers = pd.read_parquet(SAMPLE_DIR / "customers.parquet")
    real_orders = pd.read_parquet(SAMPLE_DIR / "orders.parquet")
    synth_customers = pd.read_parquet(out_dir / "customers.parquet")
    synth_orders = pd.read_parquet(out_dir / "orders.parquet")

    print("\n" + "=" * 68)
    print("REAL vs SYNTHETIC")
    print("=" * 68)

    print("\nmedian annual_income by segment")
    print(f"  {'segment':10s} {'real':>12s} {'synthetic':>12s}")
    real_income = real_customers.groupby("segment")["annual_income"].median()
    synth_income = synth_customers.groupby("segment")["annual_income"].median()
    for segment in sorted(set(real_income.index) | set(synth_income.index)):
        print(
            f"  {segment:10s} {real_income.get(segment, float('nan')):12,.0f}"
            f" {synth_income.get(segment, float('nan')):12,.0f}"
        )

    print("\nmean orders per customer by segment")
    print(f"  {'segment':10s} {'real':>12s} {'synthetic':>12s}")
    real_join = real_orders.merge(real_customers[["customer_id", "segment"]], on="customer_id")
    synth_join = synth_orders.merge(
        synth_customers[["customer_id", "segment"]], on="customer_id"
    )
    real_deg = real_join.groupby(["segment", "customer_id"]).size().groupby("segment").mean()
    synth_deg = synth_join.groupby(["segment", "customer_id"]).size().groupby("segment").mean()
    for segment in sorted(set(real_deg.index) | set(synth_deg.index)):
        print(
            f"  {segment:10s} {real_deg.get(segment, float('nan')):12.1f}"
            f" {synth_deg.get(segment, float('nan')):12.1f}"
        )

    print("\nmedian order amount by segment")
    print(f"  {'segment':10s} {'real':>12s} {'synthetic':>12s}")
    real_amount = real_join.groupby("segment")["amount"].median()
    synth_amount = synth_join.groupby("segment")["amount"].median()
    for segment in sorted(set(real_amount.index) | set(synth_amount.index)):
        print(
            f"  {segment:10s} {real_amount.get(segment, float('nan')):12,.2f}"
            f" {synth_amount.get(segment, float('nan')):12,.2f}"
        )

    print("\nintegrity checks")
    orphans = set(synth_orders["customer_id"]) - set(synth_customers["customer_id"])
    print(f"  orders pointing at a customer that exists : {'YES' if not orphans else 'NO'}")
    print(f"  customer_id unique                        : "
          f"{'YES' if synth_customers['customer_id'].is_unique else 'NO'}")
    for column in ("full_name", "email", "phone", "national_id"):
        leaked = set(real_customers[column].dropna().astype(str)) & set(
            synth_customers[column].dropna().astype(str)
        )
        print(f"  no real {column:12s} reproduced        : "
              f"{'YES' if not leaked else f'NO ({len(leaked)})'}")

    print("\nsample synthetic customers")
    columns = ["customer_id", "full_name", "email", "city", "segment",
               "annual_income", "credit_score"]
    print(synth_customers[columns].head(5).to_string(index=False))

    print("\nsample synthetic orders")
    print(
        synth_orders[["order_id", "customer_id", "order_date", "amount", "category", "status"]]
        .head(5)
        .to_string(index=False)
    )

    print(f"\nsynthetic files: {out_dir}")


if __name__ == "__main__":
    main()
