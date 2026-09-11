"""Verify the platform is reading classifications from a live Atlas."""

from __future__ import annotations

import sys

import httpx

BASE = "http://localhost:8000/api"


def main() -> None:
    data_source_id = sys.argv[1] if len(sys.argv) > 1 else open("/data/ds_id").read().strip()

    status = httpx.get(f"{BASE}/admin/atlas/status", timeout=30).json()
    print(f"atlas reachable: {status.get('reachable')}  ({status.get('base_url')})")

    refreshed = httpx.post(
        f"{BASE}/data-sources/{data_source_id}/refresh-atlas", timeout=180
    ).json()
    print(f"data source source: {refreshed.get('source')}  {refreshed.get('detail', '')}")

    policy = httpx.get(
        f"{BASE}/data-sources/{data_source_id}/policy", timeout=180
    ).json()
    print(f"policy source: {policy.get('source')}   requires_dp={policy.get('requires_dp')}")
    print(
        f"suppressed={policy.get('suppressed_total')} "
        f"pseudonymised={policy.get('pseudonymised_total')} "
        f"generalised={policy.get('generalised_total')}"
    )

    for table_name, entry in policy.get("tables", {}).items():
        print(f"\n{table_name}")
        for decision in entry["decisions"]:
            tags = ",".join(decision["classifications"]) or "-"
            print(f"   {decision['column']:20s} {decision['strategy']:11s} {tags}")


if __name__ == "__main__":
    main()
