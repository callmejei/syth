"""Generate the demo dataset: the four related tables from the screenshots.

    customer_holding (parent, PK cif)
      |- card_txn        (PK transaction_id, FK cif)  time-series
      |- product_takeup  (PK takeup_id,      FK cif)
      `- service_request (PK tran_id,        FK cif)  time-series

Deliberately built with structure a synthesiser has to earn:
  - skewed money distributions (log-normal, not gaussian)
  - correlations (segment drives balance drives txn amount)
  - realistic cardinality spread (a few very active customers, a long tail)
  - genuine PII columns so the Atlas classification path has something to bite
  - nulls in the places real banking data has them

Usage:  python -m scripts.make_demo_data --rows 2000 --out /data/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ["MASS", "AFFLUENT", "PRIORITY", "PRIVATE"]
SEGMENT_WEIGHTS = [0.62, 0.24, 0.11, 0.03]
CHANNELS = ["BRANCH", "MOBILE", "WEB", "CALL_CENTRE", "ATM"]
PRODUCTS = ["SAVINGS", "CURRENT", "FD", "CREDIT_CARD", "MORTGAGE", "UNIT_TRUST"]
MERCHANT_CATEGORIES = [
    "GROCERY", "FUEL", "DINING", "TRAVEL", "UTILITIES",
    "ONLINE_RETAIL", "HEALTHCARE", "ENTERTAINMENT",
]
REQUEST_TYPES = ["CARD_LOST", "DISPUTE", "LIMIT_INCREASE", "STATEMENT", "COMPLAINT"]
FIRST_NAMES = [
    "Aarav", "Priya", "Wei", "Siti", "John", "Mei", "Rahul", "Nurul",
    "David", "Ananya", "Hafiz", "Grace", "Kumar", "Lina", "Tan", "Farah",
]
LAST_NAMES = [
    "Rahman", "Chen", "Sharma", "Lim", "Tan", "Abdullah", "Nair", "Wong",
    "Ibrahim", "Krishnan", "Lee", "Menon", "Yusof", "Goh", "Das", "Ong",
]


def build(rows: int, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_customers = rows

    # ---- customer_holding (parent) ----
    cif = np.array([f"CIF{i:08d}" for i in range(1, n_customers + 1)])
    segment = rng.choice(SEGMENTS, size=n_customers, p=SEGMENT_WEIGHTS)

    # Balance depends on segment -- a real correlation the model must capture.
    segment_mu = {"MASS": 8.2, "AFFLUENT": 10.1, "PRIORITY": 11.4, "PRIVATE": 13.0}
    mu = np.array([segment_mu[s] for s in segment])
    balance = np.round(rng.lognormal(mean=mu, sigma=0.7), 2)

    age = np.clip(rng.normal(41, 13, n_customers), 18, 92).astype(int)
    birth_year = 2026 - age
    dob = pd.to_datetime(
        {
            "year": birth_year,
            "month": rng.integers(1, 13, n_customers),
            "day": rng.integers(1, 29, n_customers),
        }
    )

    full_name = [
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n_customers)
    ]
    email = [
        f"{name.split()[0].lower()}.{name.split()[1].lower()}{i}@example.com"
        for i, name in enumerate(full_name)
    ]
    # National ID: high-cardinality direct identifier.
    national_id = [f"{rng.integers(10, 99)}{rng.integers(10**10, 10**11 - 1)}" for _ in range(n_customers)]
    phone = [f"+60{rng.integers(10**8, 10**9 - 1)}" for _ in range(n_customers)]

    risk_rating = rng.choice(["LOW", "MEDIUM", "HIGH"], n_customers, p=[0.7, 0.25, 0.05])
    onboarded_at = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        rng.integers(0, 2400, n_customers), unit="D"
    )
    # Real data has holes.
    email_arr = np.array(email, dtype=object)
    email_arr[rng.random(n_customers) < 0.04] = None

    customer_holding = pd.DataFrame(
        {
            "cif": cif,
            "full_name": full_name,
            "national_id": national_id,
            "email": email_arr,
            "phone": phone,
            "dob": dob,
            "segment": segment,
            "risk_rating": risk_rating,
            "balance": balance,
            "onboarded_at": onboarded_at,
        }
    )

    # ---- card_txn (child, time-series) ----
    # Activity scales with segment: PRIVATE customers transact far more.
    activity = {"MASS": 6, "AFFLUENT": 14, "PRIORITY": 26, "PRIVATE": 48}
    lam = np.array([activity[s] for s in segment])
    n_txn_per_customer = rng.poisson(lam)
    n_txn_per_customer = np.clip(n_txn_per_customer, 0, None)
    total_txn = int(n_txn_per_customer.sum())

    txn_cif = np.repeat(cif, n_txn_per_customer)
    txn_segment = np.repeat(segment, n_txn_per_customer)
    txn_balance = np.repeat(balance, n_txn_per_customer)

    # Amount correlates with the customer's balance.
    amount_mu = np.log(np.maximum(txn_balance, 100)) * 0.45
    amount = np.round(rng.lognormal(mean=amount_mu, sigma=0.9), 2)
    amount = np.clip(amount, 1.0, None)

    txn_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 590, total_txn), unit="D"
    ) + pd.to_timedelta(rng.integers(0, 86400, total_txn), unit="s")

    card_txn = pd.DataFrame(
        {
            "transaction_id": np.arange(1, total_txn + 1),
            "cif": txn_cif,
            "txn_date": txn_date,
            "amount": amount,
            "currency": rng.choice(["MYR", "USD", "SGD"], total_txn, p=[0.88, 0.08, 0.04]),
            "merchant_category": rng.choice(MERCHANT_CATEGORIES, total_txn),
            "merchant_name": [f"MERCH-{rng.integers(1000, 9999)}" for _ in range(total_txn)],
            "channel": rng.choice(CHANNELS, total_txn, p=[0.08, 0.46, 0.28, 0.04, 0.14]),
            "is_fraud": rng.random(total_txn) < 0.012,
        }
    ).sort_values(["cif", "txn_date"]).reset_index(drop=True)
    _ = txn_segment

    # ---- product_takeup (child) ----
    n_products = rng.integers(1, 5, n_customers)
    total_products = int(n_products.sum())
    pt_cif = np.repeat(cif, n_products)
    product_takeup = pd.DataFrame(
        {
            "takeup_id": np.arange(1, total_products + 1),
            "cif": pt_cif,
            "product": rng.choice(PRODUCTS, total_products),
            "open_date": pd.to_datetime("2020-01-01")
            + pd.to_timedelta(rng.integers(0, 2100, total_products), unit="D"),
            "status": rng.choice(["ACTIVE", "DORMANT", "CLOSED"], total_products,
                                 p=[0.78, 0.14, 0.08]),
            "monthly_fee": np.round(rng.choice([0.0, 5.0, 12.0, 25.0], total_products), 2),
        }
    )

    # ---- service_request (child, time-series) ----
    n_requests = rng.poisson(1.6, n_customers)
    total_requests = int(n_requests.sum())
    sr_cif = np.repeat(cif, n_requests)
    raised_at = pd.to_datetime("2025-03-01") + pd.to_timedelta(
        rng.integers(0, 500, total_requests), unit="D"
    )
    resolution_hours = np.round(np.abs(rng.lognormal(2.1, 1.0, total_requests)), 1)
    service_request = pd.DataFrame(
        {
            "tran_id": np.arange(1, total_requests + 1),
            "cif": sr_cif,
            "request_type": rng.choice(REQUEST_TYPES, total_requests),
            "channel": rng.choice(CHANNELS, total_requests),
            "raised_at": raised_at,
            "resolution_hours": resolution_hours,
            "satisfaction_score": rng.choice(
                [1, 2, 3, 4, 5, None], total_requests, p=[0.05, 0.08, 0.17, 0.3, 0.3, 0.1]
            ),
        }
    ).sort_values(["cif", "raised_at"]).reset_index(drop=True)

    return {
        "customer_holding": customer_holding,
        "card_txn": card_txn,
        "product_takeup": product_takeup,
        "service_request": service_request,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2000, help="number of customers")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="/data/demo")
    args = parser.parse_args()

    tables = build(args.rows, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
        frame.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:18s} {len(frame):8,d} rows x {frame.shape[1]} cols")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
