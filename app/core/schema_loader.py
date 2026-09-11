"""Turns schema.yaml into the nested 'Data Arguments' tree the UI renders.

The YAML is a flat map of parameter-name -> definition. The UI tree is built by
resolving four kinds of indirection:

  1. groupTag        -- assigns a parameter to one of the 12 section headers
  2. object/fields   -- static child lists, or the dynamic keyword `tables`
  3. extension       -- `extendFrom` another definition, overriding options/fields
                        (the `#table` suffix scopes a parameter to one table)
  4. wildcards       -- `dataArgs.table_args.*` is the template applied to every
                        table under table_args

Dynamic option sources (`tables`, `table-columns`) are resolved against a
DatasetContext so the tree the frontend receives is fully concrete.

The source YAML was reconstructed by OCR, so token-level typos are expected and
normalised here rather than by editing the source file -- keeping schema.yaml
byte-identical to what was scanned means we can re-diff it against the vendor's
original later.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.config import settings

# --- OCR normalisation -------------------------------------------------------
# Values seen mangled in the scan. Keys are lowercased raw tokens.
_OPTION_SOURCE_FIXES = {
    "tables": "tables",
    "table-columns": "table-columns",
    "teble-columns": "table-columns",
    "table_columns": "table-columns",
    "tabte-columns": "table-columns",
}

_TAG_FIXES = {
    "high impact": "High Impact",
    "high impect": "High Impact",
    "medium impact": "Medium Impact",
    "medlum impact": "Medium Impact",
    "low impact": "Low Impact",
    "low impect": "Low Impact",
    "low lmpact": "Low Impact",
}

_BOOL_FIXES = {
    "true": True,
    "false": False,
    "felse": False,
    "flase": False,
    "ture": True,
}

_TYPE_FIXES = {
    "text": "text",
    "toxt": "text",
    "number": "number",
    "boolean": "boolean",
    "select": "select",
    "object": "object",
    "objects-array": "objects-array",
    "group-tag": "group-tag",
    "extension": "extension",
    "swappable-fields": "swappable-fields",
    "select-table-columns": "select-table-columns",
}

# Parameter types that exist in the config model but are deliberately not
# rendered (the vendor UI marks these `skipped`).
SKIPPED = "skipped"


def _norm_tag(tag: Any) -> str | None:
    if not isinstance(tag, str):
        return None
    return _TAG_FIXES.get(tag.strip().lower(), tag.strip())


def _norm_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _BOOL_FIXES.get(value.strip().lower(), value)
    return value


def _norm_type(raw: Any) -> tuple[str, bool]:
    """Return (type, is_skipped). `skipped # number` means skipped-but-typed."""
    if not isinstance(raw, str):
        return "text", False
    token = raw.split("#")[0].strip().lower()
    if token.startswith(SKIPPED):
        # Recover the real type from the trailing comment when present.
        comment = raw.split("#", 1)[1].strip().lower() if "#" in raw else ""
        inner = comment.split()[0] if comment else "text"
        return _TYPE_FIXES.get(inner, "text"), True
    return _TYPE_FIXES.get(token, token or "text"), False


def _norm_option_source(raw: Any) -> str | None:
    if isinstance(raw, str):
        return _OPTION_SOURCE_FIXES.get(raw.strip().lower(), raw.strip())
    return None


# --- context -----------------------------------------------------------------


@dataclass
class TableContext:
    name: str
    columns: list[str] = field(default_factory=list)


@dataclass
class DatasetContext:
    """What the dynamic option sources resolve against."""

    tables: list[TableContext] = field(default_factory=list)

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def columns_of(self, table_name: str | None) -> list[str]:
        if table_name is None:
            # Union of all columns, used when scope is ambiguous.
            seen: list[str] = []
            for t in self.tables:
                for c in t.columns:
                    if c not in seen:
                        seen.append(c)
            return seen
        for t in self.tables:
            if t.name == table_name:
                return list(t.columns)
        return []

    @classmethod
    def from_datasource(cls, tables: Any) -> DatasetContext:
        """Build from the DataSource.tables JSON blob."""
        out: list[TableContext] = []
        if isinstance(tables, dict):
            items = tables.get("tables", [])
        else:
            items = tables or []
        for item in items:
            if not isinstance(item, dict):
                continue
            cols = item.get("columns") or []
            names = [
                c.get("name") if isinstance(c, dict) else str(c)
                for c in cols
            ]
            out.append(TableContext(name=item.get("name", ""), columns=[n for n in names if n]))
        return cls(tables=out)


# --- the registry ------------------------------------------------------------


class SchemaRegistry:
    """Parsed schema.yaml, with tree resolution."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.groups: list[dict[str, Any]] = []
        self.params: dict[str, dict[str, Any]] = {}
        self._wildcards: dict[str, dict[str, Any]] = {}

        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            type_token = str(value.get("type", "")).split("#")[0].strip().lower()
            if type_token == "group-tag":
                self.groups.append(
                    {
                        "name": key,
                        "documentation": (value.get("documentation") or "").strip(),
                    }
                )
            if "*" in key:
                self._wildcards[key] = value
            self.params[key] = value

    # -- lookup helpers --

    def get(self, name: str) -> dict[str, Any] | None:
        if name in self.params:
            return self.params[name]
        # `foo#table` falls back to the base definition `foo`.
        if "#" in name:
            base = name.split("#", 1)[0]
            return self.params.get(base)
        return None

    def wildcard_for(self, prefix: str) -> dict[str, Any] | None:
        """Find `dataArgs.table_args.*` given `table_args`."""
        for key, value in self._wildcards.items():
            if key.startswith("dataArgs.") and key.endswith(".*"):
                if key[len("dataArgs.") : -2] == prefix:
                    return value
        for key, value in self._wildcards.items():
            if key.endswith(".*") and key.rsplit(".", 2)[-2] == prefix:
                return value
        return None

    def _resolve_extension(self, definition: dict[str, Any]) -> dict[str, Any]:
        """Collapse `type: extension` into its base definition."""
        seen: set[str] = set()
        merged = dict(definition)
        while str(merged.get("type", "")).strip().lower() == "extension":
            parent_name = merged.get("extendFrom")
            if not parent_name or parent_name in seen:
                break
            seen.add(parent_name)
            parent = self.get(parent_name)
            if not parent:
                break
            base = dict(parent)
            base.update({k: v for k, v in merged.items() if k not in ("type", "extendFrom")})
            merged = base
        return merged

    # -- tree building --

    def build_tree(
        self,
        ctx: DatasetContext,
        roots: list[str] | None = None,
        include_skipped: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the grouped Data Arguments tree.

        Output shape mirrors the UI: a list of group sections, each with the
        parameter nodes that carry that groupTag.
        """
        roots = roots or self._default_roots()
        nodes: list[dict[str, Any]] = []
        for name in roots:
            node = self.build_node(
                name, ctx, path=name, table_scope=None, include_skipped=include_skipped
            )
            if node:
                nodes.append(node)

        # Bucket by groupTag, preserving the group order declared in the YAML.
        by_group: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            by_group.setdefault(node.get("groupTag") or "Ungrouped", []).append(node)

        sections: list[dict[str, Any]] = []
        for group in self.groups:
            items = by_group.pop(group["name"], [])
            if items:
                sections.append(
                    {
                        "name": group["name"],
                        "documentation": group["documentation"],
                        "fields": items,
                    }
                )
        for name, items in by_group.items():
            sections.append({"name": name, "documentation": "", "fields": items})
        return sections

    def _default_roots(self) -> list[str]:
        """Top-level params: everything with a groupTag that is not a fragment
        referenced only as a child of something else."""
        child_names: set[str] = set()
        for definition in self.params.values():
            if not isinstance(definition, dict):
                continue
            fields = definition.get("fields")
            if isinstance(fields, list):
                child_names.update(str(f) for f in fields)
            schemas = definition.get("schemas")
            if isinstance(schemas, list):
                child_names.update(str(s) for s in schemas)

        roots = []
        for name, definition in self.params.items():
            if "*" in name or "#" in name or "." in name:
                continue
            if not isinstance(definition, dict):
                continue
            if str(definition.get("type", "")).strip().lower() == "group-tag":
                continue
            if not definition.get("groupTag"):
                continue
            if name in child_names:
                continue
            roots.append(name)
        return roots

    def build_node(
        self,
        name: str,
        ctx: DatasetContext,
        path: str,
        table_scope: str | None,
        include_skipped: bool = False,
        _depth: int = 0,
    ) -> dict[str, Any] | None:
        if _depth > 12:
            return None
        raw_def = self.get(name)
        if raw_def is None:
            return None

        definition = self._resolve_extension(raw_def)
        node_type, is_skipped = _norm_type(definition.get("type"))
        if is_skipped and not include_skipped:
            return None
        if definition.get("hidden") and not include_skipped:
            # `hidden` fields still exist in the model; the vendor UI reveals
            # them only via conditions. We emit them flagged so the frontend can
            # decide, rather than dropping them.
            pass

        label = definition.get("label")
        if label == "same-as-key" or not label:
            label = name
        # `#table` scoping: display the bare name.
        display_name = name.split("#")[0]

        tags = [t for t in (_norm_tag(t) for t in definition.get("tags", []) or []) if t]

        node: dict[str, Any] = {
            "key": display_name,
            "path": path,
            "label": str(label),
            "type": node_type,
            "skipped": is_skipped,
            "hidden": bool(definition.get("hidden", False)),
            "tags": tags,
            "highImpact": "High Impact" in tags,
            "groupTag": definition.get("groupTag"),
            "documentation": (definition.get("documentation") or "").strip(),
            "required": bool(definition.get("required", False)),
            "multi": bool(definition.get("multi", False)),
            "conditional": bool(definition.get("conditional", False)),
            "tableScope": table_scope,
        }

        if "initialValue" in definition:
            node["initialValue"] = _norm_bool(definition["initialValue"])
        for numeric_key in ("min", "max", "step"):
            if numeric_key in definition:
                node[numeric_key] = definition[numeric_key]

        conditions = definition.get("conditions")
        if isinstance(conditions, list):
            node["conditions"] = [
                {
                    "field": str(c.get("field", "")),
                    "is": _norm_bool(c.get("is")),
                }
                for c in conditions
                if isinstance(c, dict)
            ]

        # --- options ---
        options = definition.get("options")
        source = _norm_option_source(options)
        if source == "tables":
            node["options"] = [{"label": t, "value": t} for t in ctx.table_names]
            node["optionSource"] = "tables"
        elif source == "table-columns":
            node["options"] = [
                {"label": c, "value": c} for c in ctx.columns_of(table_scope)
            ]
            node["optionSource"] = "table-columns"
        elif isinstance(options, list):
            node["options"] = [
                {
                    "label": str(o.get("label")) if isinstance(o, dict) else str(o),
                    "value": (o.get("value") if isinstance(o, dict) else o),
                }
                for o in options
            ]

        if node_type == "select-table-columns":
            # Options depend on the sibling field named by `tableField`; the
            # frontend resolves this reactively.
            node["type"] = "select"
            node["tableField"] = definition.get("tableField")
            node["optionSource"] = "dynamic-table-columns"
            node["options"] = []

        # --- children ---
        fields = definition.get("fields")
        if isinstance(fields, str) and fields.strip().lower() == "tables":
            # One object node per table, from the wildcard template.
            template = self.wildcard_for(display_name)
            children: list[dict[str, Any]] = []
            for table in ctx.tables:
                child_path = f"{path}.{table.name}"
                sub_fields = (template or {}).get("fields") or []
                sub_children = []
                for sub_name in sub_fields:
                    sub_node = self.build_node(
                        str(sub_name),
                        ctx,
                        path=f"{child_path}.{str(sub_name).split('#')[0]}",
                        table_scope=table.name,
                        include_skipped=include_skipped,
                        _depth=_depth + 1,
                    )
                    if sub_node:
                        sub_children.append(sub_node)
                children.append(
                    {
                        "key": table.name,
                        "path": child_path,
                        "label": table.name,
                        "type": "object",
                        "tags": [],
                        "highImpact": False,
                        "tableScope": table.name,
                        "documentation": "",
                        "children": sub_children,
                    }
                )
            node["children"] = children
            node["perTable"] = True
        elif isinstance(fields, list):
            children = []
            # `foo` and `foo#table` both resolve to the value path `foo`, so a
            # fields list naming both would produce two nodes writing to the
            # same location. Keep the first and drop the duplicate.
            seen_paths: set[str] = set()
            for sub_name in fields:
                child_path = f"{path}.{str(sub_name).split('#')[0]}"
                if child_path in seen_paths:
                    continue
                sub_node = self.build_node(
                    str(sub_name),
                    ctx,
                    path=child_path,
                    table_scope=table_scope,
                    include_skipped=include_skipped,
                    _depth=_depth + 1,
                )
                if sub_node:
                    seen_paths.add(child_path)
                    children.append(sub_node)
            node["children"] = children

        # objects-array: a repeatable item (foreign_keys is the key example).
        schemas = definition.get("schemas")
        if isinstance(schemas, list):
            item_fields = []
            for sub_name in schemas:
                if not isinstance(sub_name, str):
                    continue
                sub_node = self.build_node(
                    sub_name,
                    ctx,
                    path=f"{path}[].{sub_name}",
                    table_scope=table_scope,
                    include_skipped=include_skipped,
                    _depth=_depth + 1,
                )
                if sub_node:
                    item_fields.append(sub_node)
            node["type"] = "objects-array"
            node["itemSchema"] = item_fields
            node.setdefault("initialValue", [])

        return node

    # -- defaults --

    def defaults_for(self, ctx: DatasetContext) -> dict[str, Any]:
        """Collect initialValues into a nested dict, for a fresh configuration."""
        tree = self.build_tree(ctx)
        out: dict[str, Any] = {}

        def walk(nodes: list[dict[str, Any]]) -> None:
            for node in nodes:
                if "initialValue" in node and not node.get("children"):
                    _set_path(out, node["path"], node["initialValue"])
                for child_key in ("children",):
                    if node.get(child_key):
                        walk(node[child_key])

        for section in tree:
            walk(section["fields"])
        return out


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in re.split(r"\.", path) if p]
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
        if not isinstance(cursor, dict):
            return
    if parts:
        cursor[parts[-1]] = value


OVERLAY_PATH = Path(__file__).with_name("schema_overlay.yaml")


def apply_overlay(raw: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Repair known OCR damage in the scanned schema.

    Kept separate from schema.yaml so the scan stays byte-identical and can be
    diffed against the vendor's original later. See schema_overlay.yaml for the
    rationale behind each correction.
    """
    merged = copy.deepcopy(raw)

    for name, definition in (overlay.get("restore") or {}).items():
        merged.setdefault(name, copy.deepcopy(definition))

    for name, patch in (overlay.get("patch") or {}).items():
        target = merged.get(name)
        if isinstance(target, dict):
            target.update(copy.deepcopy(patch))
        else:
            merged[name] = copy.deepcopy(patch)

    for name, schemas in (overlay.get("schemas") or {}).items():
        target = merged.get(name)
        if isinstance(target, dict):
            target["schemas"] = list(schemas)

    label_fixes = overlay.get("labels") or {}
    if label_fixes:
        for definition in merged.values():
            if isinstance(definition, dict):
                label = definition.get("label")
                if isinstance(label, str) and label in label_fixes:
                    definition["label"] = label_fixes[label]

    return merged


def _load_overlay() -> dict[str, Any]:
    if not OVERLAY_PATH.exists():
        return {}
    with open(OVERLAY_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def load_registry(path: str | None = None) -> SchemaRegistry:
    yaml_path = Path(path) if path else settings.schema_yaml_path
    with open(yaml_path, encoding="utf-8", errors="replace") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"schema.yaml did not parse to a mapping: {yaml_path}")
    return SchemaRegistry(apply_overlay(raw, _load_overlay()))


def reload_registry() -> SchemaRegistry:
    load_registry.cache_clear()
    return load_registry()


__all__ = [
    "DatasetContext",
    "SchemaRegistry",
    "TableContext",
    "load_registry",
    "reload_registry",
]
