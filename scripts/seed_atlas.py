"""Seed a live Apache Atlas with the demo tables and their classifications.

Creates the classification typedefs, then registers the four demo tables and
their columns with exactly the tags in fixtures/atlas_mock.json -- so the
policy engine behaves identically whether it reads Atlas or the fixture.

Run once Atlas reports healthy (it takes 5-10 minutes on first boot):

    docker compose --profile atlas up -d atlas
    docker compose exec backend python -m scripts.seed_atlas

Idempotent: re-running updates rather than duplicating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from app.config import settings

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
DEFAULT_FIXTURE = FIXTURES_DIR / "atlas_mock.json"
CLUSTER = "poc"

CLASSIFICATIONS = [
    ("PII", "Personally identifiable information"),
    ("NAME", "Personal name"),
    ("NATIONAL_ID", "Government issued identity number"),
    ("EMAIL", "Email address"),
    ("PHONE", "Telephone number"),
    ("QUASI_IDENTIFIER", "Re-identifying in combination with other columns"),
    ("DATE_OF_BIRTH", "Date of birth"),
    ("SENSITIVE", "Commercially or personally sensitive"),
    ("FINANCIAL", "Financial value"),
    ("IDENTIFIER", "Key or surrogate identifier"),
    ("INTERNAL", "Internal, non-restricted"),
    ("PCI", "Payment card industry scope"),
    ("GENDER", "Gender; re-identifying in combination"),
    ("CREDIT_SCORE", "Internal credit score"),
]


def wait_for_atlas(client: httpx.Client, attempts: int = 60) -> bool:
    for attempt in range(attempts):
        try:
            response = client.get(f"{settings.atlas_base_url}/api/atlas/admin/version")
            if response.status_code == 200:
                print(f"Atlas is up: {response.json()}")
                return True
        except Exception:  # noqa: BLE001
            pass
        print(f"  waiting for Atlas ... ({attempt + 1}/{attempts})")
        time.sleep(10)
    return False


def ensure_classifications(client: httpx.Client) -> None:
    """Create any classification typedefs that are not already defined.

    Atlas rejects the whole batch with 409 if even one name is taken, so
    posting the full list and treating 409 as "already done" silently skips
    genuinely new tags -- and then every entity carrying one fails to load.
    Ask what exists first and send only the difference.
    """
    existing: set[str] = set()
    response = client.get(
        f"{settings.atlas_base_url}/api/atlas/v2/types/typedefs",
        params={"type": "classification"},
    )
    if response.status_code == 200:
        existing = {
            d.get("name")
            for d in (response.json() or {}).get("classificationDefs", [])
        }

    missing = [(n, d) for n, d in CLASSIFICATIONS if n not in existing]
    if not missing:
        print(f"all {len(CLASSIFICATIONS)} classification typedefs already exist")
        return

    defs = [
        {"name": name, "description": description, "superTypes": [], "attributeDefs": []}
        for name, description in missing
    ]
    response = client.post(
        f"{settings.atlas_base_url}/api/atlas/v2/types/typedefs",
        json={"classificationDefs": defs},
    )
    if response.status_code in (200, 201):
        print(f"created {len(defs)} typedefs: {', '.join(n for n, _ in missing)}")
    else:
        print(f"typedef create returned {response.status_code}: {response.text[:300]}")
        raise SystemExit(1)


def build_entities(blob: dict, database: str) -> tuple[list[dict], list[dict]]:
    """Return (tables, columns).

    They must be loaded in that order. A table entity carrying a `columns` list
    is rejected outright if those columns do not exist yet, and creating both in
    one bulk call does not resolve the cycle. Creating the tables first and then
    the columns -- each pointing at its table -- lets Atlas build the
    relationship from the column side.
    """
    tables: list[dict] = []
    columns: list[dict] = []

    for table_name, table_blob in (blob.get("tables") or {}).items():
        table_qn = f"{database}.{table_name}@{CLUSTER}"

        tables.append(
            {
                "typeName": "hive_table",
                "attributes": {
                    "qualifiedName": table_qn,
                    "name": table_name,
                    "owner": "synthforge-poc",
                    "temporary": False,
                },
            }
        )

        for column_name, column_blob in (table_blob.get("columns") or {}).items():
            columns.append(
                {
                    "typeName": "hive_column",
                    "attributes": {
                        "qualifiedName": f"{table_qn}.{column_name}",
                        "name": column_name,
                        "type": column_blob.get("data_type", "string"),
                        "comment": column_blob.get("description", ""),
                        "table": {
                            "typeName": "hive_table",
                            "uniqueAttributes": {"qualifiedName": table_qn},
                        },
                    },
                    "classifications": [
                        {"typeName": tag}
                        for tag in column_blob.get("classifications", [])
                    ],
                }
            )

    return tables, columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DEFAULT_FIXTURE),
        help="classification file to load into Atlas",
    )
    parser.add_argument(
        "--database", default="retailbank", help="Atlas database qualifier"
    )
    args = parser.parse_args()

    auth = (settings.atlas_user, settings.atlas_password)
    with httpx.Client(timeout=30.0, auth=auth) as client:
        if not wait_for_atlas(client):
            print("Atlas did not become healthy; aborting.", file=sys.stderr)
            raise SystemExit(1)

        ensure_classifications(client)

        with open(args.input, encoding="utf-8") as handle:
            blob = json.load(handle)
        print(f"loading {args.input} as database '{args.database}'")

        tables, columns = build_entities(blob, args.database)

        for label, entities in (("tables", tables), ("columns", columns)):
            response = client.post(
                f"{settings.atlas_base_url}/api/atlas/v2/entity/bulk",
                json={"entities": entities},
            )
            if response.status_code not in (200, 201):
                print(
                    f"{label} bulk create failed {response.status_code}: "
                    f"{response.text[:500]}"
                )
                raise SystemExit(1)
            result = response.json() or {}
            mutated = result.get("mutatedEntities") or {}
            created = len(mutated.get("CREATE") or [])
            updated = len(mutated.get("UPDATE") or [])
            print(f"  {label}: {created} created, {updated} updated")

        print("Atlas seeded.")


if __name__ == "__main__":
    main()
