"""Compile LlamaIndex ``MetadataFilters`` to a SQL ``WHERE`` clause body.

Filters apply only to declared metadata columns — the engine cannot index
into the serialized ``_node_content`` JSON catch-all.
"""

from __future__ import annotations

from collections.abc import Iterable

from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from llama_index.vector_stores.infino._arrow import quote_str, sql_value

_BINARY_OPERATORS: dict[FilterOperator, str] = {
    FilterOperator.EQ: "=",
    FilterOperator.NE: "!=",
    FilterOperator.GT: ">",
    FilterOperator.GTE: ">=",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "<=",
}

_CONDITION_JOIN: dict[FilterCondition, str] = {
    FilterCondition.AND: " AND ",
    FilterCondition.OR: " OR ",
}


def compile_filters(filters: MetadataFilters, allowed: Iterable[str]) -> str:
    """Render a ``MetadataFilters`` tree as a SQL predicate string."""
    return _compile_group(filters, set(allowed))


def _compile_group(group: MetadataFilters, allowed: set[str]) -> str:
    parts = [_compile_node(child, allowed) for child in group.filters]
    if not parts:
        return "TRUE"
    condition = group.condition or FilterCondition.AND
    if condition == FilterCondition.NOT:
        return f"NOT ({_join(' AND ', parts)})"
    return _join(_CONDITION_JOIN[condition], parts)


def _join(joiner: str, parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return joiner.join(f"({p})" for p in parts)


def _compile_node(
    child: MetadataFilter | MetadataFilters,
    allowed: set[str],
) -> str:
    if isinstance(child, MetadataFilters):
        return _compile_group(child, allowed)
    if child.key not in allowed:
        raise ValueError(
            f"cannot filter on {child.key!r}: not a declared metadata column "
            f"(declared: {sorted(allowed)})"
        )
    return _compile_comparison(child)


def _compile_comparison(f: MetadataFilter) -> str:
    op = f.operator
    key = f.key
    value = f.value

    if op in _BINARY_OPERATORS:
        return f"{key} {_BINARY_OPERATORS[op]} {sql_value(value)}"
    if op in (FilterOperator.IN, FilterOperator.ANY):
        items = ", ".join(sql_value(v) for v in _as_list(value))
        return f"{key} IN ({items})"
    if op == FilterOperator.NIN:
        items = ", ".join(sql_value(v) for v in _as_list(value))
        return f"{key} NOT IN ({items})"
    if op == FilterOperator.ALL:
        parts = [f"{key} = {sql_value(v)}" for v in _as_list(value)]
        return " AND ".join(parts) if parts else "TRUE"
    if op == FilterOperator.CONTAINS:
        return _like(key, _as_str(value), case_sensitive=True)
    if op == FilterOperator.TEXT_MATCH:
        return _like(key, _as_str(value), case_sensitive=True)
    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        return _like(key, _as_str(value), case_sensitive=False)
    if op == FilterOperator.IS_EMPTY:
        return f"({key} IS NULL OR {key} = '')"
    raise ValueError(f"unsupported filter operator: {op}")


def _like(key: str, value: str, *, case_sensitive: bool) -> str:
    pattern = "%" + _escape_like(value) + "%"
    op = "LIKE" if case_sensitive else "ILIKE"
    return f"{key} {op} {quote_str(pattern)} ESCAPE '\\'"


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _as_list(value: object) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a list value, got {type(value).__name__}")
    return list(value)


def _as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string value, got {type(value).__name__}")
    return value
