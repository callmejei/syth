"""Import column classifications from a file, for a machine with no catalog access.

The point of the Atlas integration is that generation is driven by the tags your
governance team already curated. That does NOT require a live connection to the
office Atlas -- it requires the tags. This imports them from whatever form you
can actually get them out in, and writes the local catalog the platform reads.

Accepted inputs (auto-detected):

  1. Atlas entity export / search response (JSON)
     Anything containing Atlas entities with `classifications`, including:
       - the response body of POST /api/atlas/v2/search/basic
       - GET /api/atlas/v2/entity/bulk?guid=...
       - the entities.json inside an Atlas export ZIP
     Ask your Atlas admin for an export of the hive_table / hive_column entities
     for the tables in scope; it is a read-only operation.

  2. CSV, for when all you can get is a spreadsheet from the governance team:
       table,column,classifications,terms,description
       customer_holding,national_id,"PII;NATIONAL_ID",National Registration ID,
       customer_holding,balance,"SENSITIVE;FINANCIAL",,Total relationship balance
     Semicolons or pipes separate multiple classifications.

  3. The platform's own catalog format, for round-tripping.

Usage:
    python -m scripts.import_classifications --input /data/catalog/export.json
    python -m scripts.import_classifications --input /data/catalog/tags.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.config import settings

SPLITTERS = (";", "|", ",")


def _split_tags(raw: str) -> list[str]:
    if not raw:
        return []
    for splitter in SPLITTERS:
        if splitter in raw:
            return [part.strip().upper() for part in raw.split(splitter) if part.strip()]
    return [raw.strip().upper()]


def from_csv(path: Path) -> dict[str, Any]:
    tables: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"qualified_name": "", "guid": "", "columns": {}}
    )
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {(f or "").strip().lower() for f in (reader.fieldnames or [])}
        if not {"table", "column"} <= fields:
            raise ValueError(
                "CSV must have at least 'table' and 'column' columns; "
                f"found: {sorted(fields)}"
            )
        for row in reader:
            row = { (k or "").strip().lower(): (v or "").strip() for k, v in row.items() }
            table, column = row.get("table"), row.get("column")
            if not table or not column:
                continue
            tables[table]["qualified_name"] = row.get("qualified_name") or table
            tables[table]["columns"][column] = {
                "classifications": _split_tags(row.get("classifications", "")),
                "terms": _split_tags(row.get("terms", "")) if row.get("terms") else [],
                "data_type": row.get("data_type") or None,
                "description": row.get("description", ""),
            }
    return {"tables": dict(tables)}


def _iter_entities(blob: Any):
    """Yield entity dicts from the various shapes Atlas hands back."""
    if isinstance(blob, list):
        for item in blob:
            yield from _iter_entities(item)
        return
    if not isinstance(blob, dict):
        return

    if "typeName" in blob and ("attributes" in blob or "guid" in blob):
        yield blob

    for key in ("entities", "entity", "referredEntities", "searchResults"):
        value = blob.get(key)
        if isinstance(value, dict):
            # referredEntities is a guid -> entity map
            for item in value.values():
                yield from _iter_entities(item)
        elif isinstance(value, list):
            for item in value:
                yield from _iter_entities(item)


def from_atlas_export(blob: Any) -> dict[str, Any]:
    """Rebuild table -> column -> classifications from Atlas entities."""
    columns_by_table: dict[str, dict[str, Any]] = defaultdict(dict)
    table_meta: dict[str, dict[str, str]] = {}

    entities = list(_iter_entities(blob))
    if not entities:
        raise ValueError("no Atlas entities found in this file")

    for entity in entities:
        type_name = str(entity.get("typeName", "")).lower()
        attributes = entity.get("attributes") or {}
        name = attributes.get("name")
        qualified = attributes.get("qualifiedName") or ""

        if "column" not in type_name:
            if name and ("table" in type_name or "dataset" in type_name):
                table_meta[name] = {
                    "qualified_name": qualified,
                    "guid": entity.get("guid", ""),
                }
            continue

        if not name:
            continue

        # Derive the owning table: prefer the relationship, else the
        # qualifiedName, which is conventionally db.table.column@cluster.
        table_name = None
        table_ref = attributes.get("table") or (
            entity.get("relationshipAttributes") or {}
        ).get("table")
        if isinstance(table_ref, dict):
            unique = table_ref.get("uniqueAttributes") or {}
            table_qualified = unique.get("qualifiedName") or ""
            if table_qualified:
                table_name = table_qualified.split("@")[0].split(".")[-1]
            table_name = table_name or table_ref.get("displayText")
        if not table_name and qualified:
            stem = qualified.split("@")[0]
            parts = stem.split(".")
            if len(parts) >= 2:
                table_name = parts[-2]
        if not table_name:
            continue

        columns_by_table[table_name][name] = {
            "classifications": [
                c.get("typeName")
                for c in (entity.get("classifications") or [])
                if c.get("typeName")
            ],
            "terms": [
                t.get("displayText")
                for t in (entity.get("meanings") or [])
                if t.get("displayText")
            ],
            "data_type": attributes.get("dataType") or attributes.get("type"),
            "description": attributes.get("comment") or attributes.get("description") or "",
        }

    tables = {}
    for table_name, columns in columns_by_table.items():
        meta = table_meta.get(table_name, {})
        tables[table_name] = {
            "qualified_name": meta.get("qualified_name", table_name),
            "guid": meta.get("guid", ""),
            "columns": columns,
        }
    return {"tables": tables}


def load(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        return from_csv(path)

    with open(path, encoding="utf-8") as handle:
        blob = json.load(handle)

    # Already in our format?
    if isinstance(blob, dict) and isinstance(blob.get("tables"), dict):
        sample = next(iter(blob["tables"].values()), None)
        if isinstance(sample, dict) and "columns" in sample:
            return {"tables": blob["tables"]}

    return from_atlas_export(blob)


def summarise(catalog: dict[str, Any]) -> None:
    tables = catalog.get("tables") or {}
    total_columns = sum(len(t.get("columns") or {}) for t in tables.values())
    tagged = 0
    tag_counts: dict[str, int] = defaultdict(int)
    for table in tables.values():
        for column in (table.get("columns") or {}).values():
            tags = column.get("classifications") or []
            if tags:
                tagged += 1
            for tag in tags:
                tag_counts[tag] += 1

    print(f"  tables:          {len(tables)}")
    print(f"  columns:         {total_columns}")
    print(f"  classified:      {tagged}")
    if total_columns and tagged < total_columns:
        print(
            f"  UNCLASSIFIED:    {total_columns - tagged} "
            "(strict policy mode will fail these closed)"
        )
    if tag_counts:
        print("  classifications:")
        for tag, count in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
            print(f"      {tag:24s} {count}")

    for name, table in tables.items():
        columns = table.get("columns") or {}
        print(f"    {name}: {len(columns)} column(s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Atlas export JSON, or a CSV")
    parser.add_argument(
        "--output",
        default=None,
        help=f"where to write the local catalog (default {settings.local_catalog_path})",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        print(f"input not found: {source}", file=sys.stderr)
        raise SystemExit(1)

    try:
        catalog = load(source)
    except Exception as exc:  # noqa: BLE001
        print(f"could not read {source}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not catalog.get("tables"):
        print("no tables found in the input", file=sys.stderr)
        raise SystemExit(1)

    print(f"parsed {source.name}:")
    summarise(catalog)

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    target = Path(args.output) if args.output else Path(settings.local_catalog_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    catalog["_source"] = str(source)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)

    print(f"\nwritten to {target}")
    print("The platform will now use these classifications. Refresh a data source")
    print("(Data sources -> Refresh Atlas) to pick them up.")


if __name__ == "__main__":
    main()
