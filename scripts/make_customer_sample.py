"""A small, readable customer dataset to demo synthesis on.

Two tables, deliberately simple enough to eyeball:

    customers (parent, PK customer_id)
      `- orders  (child,  PK order_id, FK customer_id)

Built with structure a synthesiser has to actually earn, so that "did it work?"
is answerable by looking rather than by trusting a score:

  * segment drives income drives credit_score          (a chain of correlations)
  * segment drives how many orders a customer places   (cardinality signal)
  * income drives order amount                         (cross-table correlation)
  * real PII: name, email, phone, national id          (for the policy to act on)
  * money is log-normal, not gaussian                  (a real distribution shape)
  * some columns are genuinely null sometimes          (as real extracts are)

Usage:
    python -m scripts.make_customer_sample --customers 4000 --out /data/sample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ["BASIC", "PLUS", "PREMIUM", "VIP"]
SEGMENT_WEIGHTS = [0.55, 0.28, 0.13, 0.04]

CITIES = [
    "Kuala Lumpur", "Penang", "Johor Bahru", "Ipoh", "Kuching",
    "Kota Kinabalu", "Malacca", "Shah Alam",
]
CITY_WEIGHTS = [0.30, 0.16, 0.14, 0.10, 0.09, 0.08, 0.07, 0.06]

CATEGORIES = [
    "Electronics", "Groceries", "Fashion", "Home", "Beauty", "Sports", "Books",
]
PAYMENT_METHODS = ["CARD", "WALLET", "BANK_TRANSFER", "CASH_ON_DELIVERY"]
ORDER_STATUS = ["DELIVERED", "SHIPPED", "CANCELLED", "RETURNED"]

FIRST_NAMES = [
    "Aisha", "Wei Jie", "Ravi", "Siti", "Daniel", "Mei Ling", "Arjun", "Nurul",
    "Kumar", "Grace", "Hafiz", "Priya", "Jason", "Farah", "Lim", "Anand",
    "Chloe", "Zainab", "Marcus", "Divya",
]
LAST_NAMES = [
    "Tan", "Rahman", "Nair", "Lim", "Abdullah", "Wong", "Sharma", "Ibrahim",
    "Chen", "Menon", "Yusof", "Goh", "Krishnan", "Lee", "Ong", "Das",
]


def build(n_customers: int, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # ---------------- customers ----------------
    customer_id = np.array([f"C{i:07d}" for i in range(1, n_customers + 1)])
    segment = rng.choice(SEGMENTS, size=n_customers, p=SEGMENT_WEIGHTS)

    # Income depends on segment -- the first link in the chain.
    income_mu = {"BASIC": 10.2, "PLUS": 10.9, "PREMIUM": 11.6, "VIP": 12.4}
    mu = np.array([income_mu[s] for s in segment])
    annual_income = np.round(rng.lognormal(mean=mu, sigma=0.45), 2)

    # Credit score follows income, with noise and a hard range.
    score_base = 480 + 260 * (np.log(annual_income) - 10.0) / 2.5
    credit_score = np.clip(
        np.round(score_base + rng.normal(0, 45, n_customers)), 300, 850
    ).astype(int)

    age = np.clip(rng.normal(38, 12, n_customers), 18, 85).astype(int)
    dob = pd.to_datetime(
        {
            "year": 2026 - age,
            "month": rng.integers(1, 13, n_customers),
            "day": rng.integers(1, 29, n_customers),
        }
    )

    full_name = [
        f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" for _ in range(n_customers)
    ]
    email = np.array(
        [
            f"{name.split()[0].lower().replace(' ', '')}.{name.split()[-1].lower()}"
            f"{rng.integers(1, 999)}@example.com"
            for name in full_name
        ],
        dtype=object,
    )
    phone = [f"+601{rng.integers(10**7, 10**8 - 1)}" for _ in range(n_customers)]
    national_id = [
        f"{rng.integers(60, 99)}{rng.integers(10, 13):02d}{rng.integers(10, 29):02d}"
        f"-{rng.integers(10, 15)}-{rng.integers(1000, 9999)}"
        for _ in range(n_customers)
    ]

    signup_date = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1800, n_customers), unit="D"
    )
    gender = rng.choice(["F", "M", "X"], n_customers, p=[0.48, 0.49, 0.03])
    city = rng.choice(CITIES, n_customers, p=CITY_WEIGHTS)
    is_active = rng.random(n_customers) < 0.82

    # Real extracts have holes.
    email[rng.random(n_customers) < 0.05] = None

    customers = pd.DataFrame(
        {
            "customer_id": customer_id,
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "national_id": national_id,
            "dob": dob,
            "gender": gender,
            "city": city,
            "segment": segment,
            "annual_income": annual_income,
            "credit_score": credit_score,
            "signup_date": signup_date,
            "is_active": is_active,
        }
    )

    # ---------------- orders ----------------
    # Order volume rises sharply with segment: VIPs are the heavy shoppers, so
    # the order table is dominated by them, exactly as in real retail data.
    orders_lambda = {"BASIC": 2.0, "PLUS": 5.0, "PREMIUM": 11.0, "VIP": 22.0}
    lam = np.array([orders_lambda[s] for s in segment])
    n_orders_each = rng.poisson(lam)
    total_orders = int(n_orders_each.sum())

    order_customer = np.repeat(customer_id, n_orders_each)
    order_income = np.repeat(annual_income, n_orders_each)

    # Basket size scales with income.
    amount_mu = np.log(order_income) * 0.42
    amount = np.round(np.clip(rng.lognormal(amount_mu, 0.75, total_orders), 5, None), 2)

    order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 560, total_orders), unit="D"
    )

    orders = pd.DataFrame(
        {
            "order_id": np.arange(1, total_orders + 1),
            "customer_id": order_customer,
            "order_date": order_date,
            "amount": amount,
            "category": rng.choice(CATEGORIES, total_orders),
            "payment_method": rng.choice(
                PAYMENT_METHODS, total_orders, p=[0.52, 0.26, 0.14, 0.08]
            ),
            "status": rng.choice(ORDER_STATUS, total_orders, p=[0.80, 0.10, 0.06, 0.04]),
            "items": rng.integers(1, 9, total_orders),
        }
    ).sort_values(["customer_id", "order_date"]).reset_index(drop=True)

    return {"customers": customers, "orders": orders}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="/data/sample")
    args = parser.parse_args()

    tables = build(args.customers, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, frame in tables.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
        frame.to_csv(out / f"{name}.csv", index=False)
        print(f"  {name:12s} {len(frame):8,d} rows x {frame.shape[1]} cols")

    customers, orders = tables["customers"], tables["orders"]
    print("\nStructure the synthesiser has to reproduce:")
    print("  median income by segment:")
    for segment, value in customers.groupby("segment")["annual_income"].median().items():
        print(f"     {segment:8s} {value:12,.0f}")
    print("  mean orders per customer by segment:")
    joined = orders.merge(customers[["customer_id", "segment"]], on="customer_id")
    per_customer = joined.groupby(["segment", "customer_id"]).size().groupby("segment").mean()
    for segment, value in per_customer.items():
        print(f"     {segment:8s} {value:6.1f}")
    print("  median order amount by segment:")
    for segment, value in joined.groupby("segment")["amount"].median().items():
        print(f"     {segment:8s} {value:8,.2f}")

    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
