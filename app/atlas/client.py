"""Apache Atlas client.

Betterdata infers what a column *is* from the data. You already know, because
Atlas holds the classifications your governance team curated. This client pulls
those classifications so the synthesiser can be driven by declared sensitivity
rather than by guesswork.

Falls back to a local fixture when Atlas is unreachable, so the POC demos while
the (slow) Atlas container warms up. The fallback is always reported in
`source`, never silently substituted -- a policy decision made from mock data
must be visibly mock.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Circuit breaker.
#
# When Atlas is down or its hostname does not resolve, every request still has
# to time out. A classification lookup fans out to (tables x entity types)
# requests, so an unreachable Atlas turned a single API call into a 45-second
# hang. We probe once, cache the verdict briefly, and skip the fan-out entirely
# while the circuit is open.
_HEALTH_TTL_SECONDS = 60.0
_HEALTH_LOCK = threading.Lock()
_health_cache: dict[str, tuple[float, bool, str]] = {}


def _cached_health(base_url: str) -> tuple[bool, str] | None:
    with _HEALTH_LOCK:
        entry = _health_cache.get(base_url)
    if not entry:
        return None
    checked_at, reachable, reason = entry
    if time.monotonic() - checked_at > _HEALTH_TTL_SECONDS:
        return None
    return reachable, reason


def _store_health(base_url: str, reachable: bool, reason: str) -> None:
    with _HEALTH_LOCK:
        _health_cache[base_url] = (time.monotonic(), reachable, reason)


def reset_health_cache() -> None:
    """Force the next call to re-probe (used by the refresh endpoint)."""
    with _HEALTH_LOCK:
        _health_cache.clear()

FIXTURE_PATH = Path(__file__).resolve().parent.parent.parent / "fixtures" / "atlas_mock.json"

# Atlas ships some of these; the rest are conventional names we also honour.
DIRECT_IDENTIFIER_TAGS = {
    "PII", "DIRECT_IDENTIFIER", "IDENTIFIER", "NRIC", "NATIONAL_ID",
    "PASSPORT", "EMAIL", "PHONE", "NAME", "ADDRESS",
}
PCI_TAGS = {"PCI", "PCI_DSS", "CARD_NUMBER", "PAN", "CVV"}
QUASI_IDENTIFIER_TAGS = {
    "QUASI_IDENTIFIER", "QID", "DOB", "DATE_OF_BIRTH", "POSTCODE",
    "ZIPCODE", "GENDER", "AGE",
}
SENSITIVE_TAGS = {
    "SENSITIVE", "CONFIDENTIAL", "RESTRICTED", "FINANCIAL",
    "HEALTH", "PHI", "SALARY", "CREDIT_SCORE",
}


@dataclass
class ColumnClassification:
    column: str
    classifications: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    data_type: str | None = None
    description: str = ""

    @property
    def is_direct_identifier(self) -> bool:
        return bool(self._upper() & DIRECT_IDENTIFIER_TAGS)

    @property
    def is_pci(self) -> bool:
        return bool(self._upper() & PCI_TAGS)

    @property
    def is_quasi_identifier(self) -> bool:
        return bool(self._upper() & QUASI_IDENTIFIER_TAGS)

    @property
    def is_sensitive(self) -> bool:
        return bool(self._upper() & SENSITIVE_TAGS)

    @property
    def is_classified(self) -> bool:
        return bool(self.classifications)

    def _upper(self) -> set[str]:
        return {c.upper() for c in self.classifications}

    def to_json(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "classifications": self.classifications,
            "terms": self.terms,
            "data_type": self.data_type,
            "description": self.description,
            "is_direct_identifier": self.is_direct_identifier,
            "is_pci": self.is_pci,
            "is_quasi_identifier": self.is_quasi_identifier,
            "is_sensitive": self.is_sensitive,
        }


@dataclass
class TableClassification:
    table: str
    qualified_name: str = ""
    guid: str = ""
    columns: dict[str, ColumnClassification] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "qualified_name": self.qualified_name,
            "guid": self.guid,
            "columns": {k: v.to_json() for k, v in self.columns.items()},
        }


@dataclass
class AtlasResult:
    tables: dict[str, TableClassification]
    source: str  # "atlas" | "fixture" | "empty"
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "detail": self.detail,
            "tables": {k: v.to_json() for k, v in self.tables.items()},
        }


class AtlasClient:
    def __init__(
        self,
        base_url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        enabled: bool | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (base_url or settings.atlas_base_url).rstrip("/")
        self.auth = (user or settings.atlas_user, password or settings.atlas_password)
        self.enabled = settings.atlas_enabled if enabled is None else enabled
        self.timeout = timeout or settings.atlas_timeout_seconds

    # -- health --

    def ping(self, use_cache: bool = True) -> dict[str, Any]:
        if not self.enabled:
            return {"reachable": False, "reason": "ATLAS_ENABLED is false"}

        if use_cache:
            cached = _cached_health(self.base_url)
            if cached is not None:
                reachable, reason = cached
                return {"reachable": reachable, "reason": reason, "cached": True}

        try:
            # Deliberately short: this is a liveness probe, not a data call.
            with httpx.Client(timeout=min(self.timeout, 3.0), auth=self.auth) as client:
                response = client.get(f"{self.base_url}/api/atlas/admin/version")
                response.raise_for_status()
                _store_health(self.base_url, True, "ok")
                return {"reachable": True, "version": response.json()}
        except Exception as exc:  # noqa: BLE001 - report, never raise, on health
            reason = f"{type(exc).__name__}: {exc}"
            _store_health(self.base_url, False, reason)
            return {"reachable": False, "reason": reason}

    # -- classification lookup --

    def classify_tables(self, table_names: list[str]) -> AtlasResult:
        """Resolve classifications for each table, falling back to the fixture."""
        if self.enabled:
            # Probe once before fanning out. Without this, an unreachable Atlas
            # costs (tables x entity types) timeouts per request.
            health = self.ping()
            if not health.get("reachable"):
                return self._from_fixture(
                    table_names,
                    detail=f"Atlas unreachable ({health.get('reason', 'unknown')})",
                )
            try:
                tables = self._fetch_from_atlas(table_names)
                if tables:
                    return AtlasResult(tables=tables, source="atlas")
                logger.info("Atlas returned no entities for %s", table_names)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Atlas lookup failed, using fixture: %s", exc)
                return self._from_fixture(
                    table_names, detail=f"Atlas unreachable ({type(exc).__name__})"
                )
        return self._from_fixture(
            table_names,
            detail="Atlas disabled" if not self.enabled else "no matching entities in Atlas",
        )

    def _fetch_from_atlas(self, table_names: list[str]) -> dict[str, TableClassification]:
        out: dict[str, TableClassification] = {}
        with httpx.Client(timeout=self.timeout, auth=self.auth) as client:
            for table_name in table_names:
                entity = self._search_table(client, table_name)
                if not entity:
                    continue
                out[table_name] = self._build_table_classification(
                    client, table_name, entity
                )
        return out

    def _search_table(
        self, client: httpx.Client, table_name: str
    ) -> dict[str, Any] | None:
        """Basic search across the table-ish entity types."""
        for type_name in ("hive_table", "rdbms_table", "DataSet"):
            payload = {
                "typeName": type_name,
                "excludeDeletedEntities": True,
                "limit": 5,
                "query": table_name,
            }
            try:
                response = client.post(
                    f"{self.base_url}/api/atlas/v2/search/basic", json=payload
                )
                if response.status_code != 200:
                    continue
                entities = (response.json() or {}).get("entities") or []
            except Exception:  # noqa: BLE001
                continue
            for entity in entities:
                name = (entity.get("attributes") or {}).get("name") or entity.get(
                    "displayText"
                )
                if name and str(name).lower() == table_name.lower():
                    return entity
        return None

    def _build_table_classification(
        self, client: httpx.Client, table_name: str, entity: dict[str, Any]
    ) -> TableClassification:
        guid = entity.get("guid", "")
        table = TableClassification(
            table=table_name,
            qualified_name=(entity.get("attributes") or {}).get("qualifiedName", ""),
            guid=guid,
        )
        if not guid:
            return table

        response = client.get(f"{self.base_url}/api/atlas/v2/entity/guid/{guid}")
        if response.status_code != 200:
            return table
        blob = response.json() or {}
        referred = blob.get("referredEntities") or {}
        entity_body = blob.get("entity") or {}
        column_refs = (entity_body.get("relationshipAttributes") or {}).get(
            "columns"
        ) or (entity_body.get("attributes") or {}).get("columns") or []

        for ref in column_refs:
            column_guid = ref.get("guid")
            column_entity = referred.get(column_guid)
            if not column_entity:
                column_entity = self._fetch_entity(client, column_guid)
            if not column_entity:
                continue
            attributes = column_entity.get("attributes") or {}
            column_name = attributes.get("name")
            if not column_name:
                continue
            table.columns[column_name] = ColumnClassification(
                column=column_name,
                classifications=[
                    c.get("typeName")
                    for c in (column_entity.get("classifications") or [])
                    if c.get("typeName")
                ],
                terms=[
                    t.get("displayText")
                    for t in (column_entity.get("meanings") or [])
                    if t.get("displayText")
                ],
                data_type=attributes.get("dataType") or attributes.get("type"),
                description=attributes.get("comment") or attributes.get("description") or "",
            )
        return table

    def _fetch_entity(
        self, client: httpx.Client, guid: str | None
    ) -> dict[str, Any] | None:
        if not guid:
            return None
        try:
            response = client.get(f"{self.base_url}/api/atlas/v2/entity/guid/{guid}")
            if response.status_code != 200:
                return None
            return (response.json() or {}).get("entity")
        except Exception:  # noqa: BLE001
            return None

    # -- file fallbacks --

    def _from_fixture(self, table_names: list[str], detail: str = "") -> AtlasResult:
        """Resolve from a file when there is no live Atlas.

        Two sources, in order of trust:
          1. A catalog imported from your own Atlas export or a governance
             spreadsheet (scripts/import_classifications.py). These are YOUR
             classifications and are treated as real.
          2. The bundled demo fixture, which is illustrative only.

        Which one was used is always reported, never assumed -- a policy built
        from demo tags must never look like a policy built from your catalog.
        """
        local_path = Path(settings.local_catalog_path)
        if local_path.exists():
            source_label = "local-catalog"
            path = local_path
            detail = (
                f"imported catalog: {local_path}"
                + (f" ({detail})" if detail else "")
            )
        elif FIXTURE_PATH.exists():
            source_label = "fixture"
            path = FIXTURE_PATH
        else:
            return AtlasResult(tables={}, source="empty", detail=detail)

        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)

        tables: dict[str, TableClassification] = {}
        for table_name in table_names:
            entry = (blob.get("tables") or {}).get(table_name)
            if not entry:
                continue
            table = TableClassification(
                table=table_name,
                qualified_name=entry.get("qualified_name", ""),
                guid=entry.get("guid", ""),
            )
            for column_name, column_blob in (entry.get("columns") or {}).items():
                table.columns[column_name] = ColumnClassification(
                    column=column_name,
                    classifications=column_blob.get("classifications", []),
                    terms=column_blob.get("terms", []),
                    data_type=column_blob.get("data_type"),
                    description=column_blob.get("description", ""),
                )
            tables[table_name] = table

        missing = [name for name in table_names if name not in tables]
        if missing and source_label == "local-catalog":
            detail += f" | no entry for: {', '.join(missing)}"

        return AtlasResult(
            tables=tables,
            source=source_label,
            detail=detail
            or (
                "using bundled demo fixture -- these are ILLUSTRATIVE tags, not "
                "your catalog"
            ),
        )


__all__ = [
    "AtlasClient",
    "AtlasResult",
    "ColumnClassification",
    "TableClassification",
]
