"""Compile LlamaIndex ``MetadataFilters`` to a SQL predicate.

Filters apply only to declared metadata columns; keys that live only in the
serialized node content cannot be filtered on.
"""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from llama_index.vector_stores.infino._arrow import quote_ident, quote_str, sql_value

_COMPARISONS: dict[FilterOperator, str] = {
    FilterOperator.EQ: "=",
    FilterOperator.NE: "!=",
    FilterOperator.GT: ">",
    FilterOperator.GTE: ">=",
    FilterOperator.LT: "<",
    FilterOperator.LTE: "<=",
}

_ARRAY_OPERATORS = frozenset({FilterOperator.CONTAINS, FilterOperator.ANY, FilterOperator.ALL})


def compile_filters(filters: MetadataFilters, columns: Mapping[str, pa.DataType]) -> str:
    """Render a filter tree as a SQL predicate over ``columns`` (name to Arrow type).

    Raises ``ValueError`` for an undeclared key or an operator the column's
    type does not support.
    """
    parts = [
        compile_filters(child, columns)
        if isinstance(child, MetadataFilters)
        else _compile_filter(child, columns)
        for child in filters.filters
    ]
    if not parts:
        return "TRUE"
    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.NOT:
        return f"NOT ({_join(' AND ', parts)})"
    return _join(" OR " if condition == FilterCondition.OR else " AND ", parts)


def _join(joiner: str, parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return joiner.join(f"({p})" for p in parts)


def _compile_filter(f: MetadataFilter, columns: Mapping[str, pa.DataType]) -> str:
    if f.key not in columns:
        raise ValueError(
            f"cannot filter on {f.key!r}: not a declared metadata column "
            f"(declared: {sorted(columns)})"
        )
    op = f.operator
    key = quote_ident(f.key)
    is_list = pa.types.is_list(columns[f.key]) or pa.types.is_large_list(columns[f.key])

    if op == FilterOperator.IS_EMPTY:
        return f"({key} IS NULL OR cardinality({key}) = 0)" if is_list else f"{key} IS NULL"
    if op in _ARRAY_OPERATORS:
        if not is_list:
            raise ValueError(f"{op.name} needs a list column; {f.key!r} is {columns[f.key]}")
        if op == FilterOperator.CONTAINS:
            return f"array_has({key}, {sql_value(f.value)})"
        items = _as_list(f.value)
        if not items:
            return "FALSE" if op == FilterOperator.ANY else "TRUE"
        function = "array_has_any" if op == FilterOperator.ANY else "array_has_all"
        return f"{function}({key}, [{', '.join(sql_value(v) for v in items)}])"
    if is_list:
        raise ValueError(f"{op.name} does not apply to list column {f.key!r}")
    if op in _COMPARISONS:
        return f"{key} {_COMPARISONS[op]} {sql_value(f.value)}"
    if op in (FilterOperator.IN, FilterOperator.NIN):
        items = _as_list(f.value)
        if not items:
            return "FALSE" if op == FilterOperator.IN else "TRUE"
        negate = "NOT " if op == FilterOperator.NIN else ""
        return f"{key} {negate}IN ({', '.join(sql_value(v) for v in items)})"
    if op == FilterOperator.TEXT_MATCH:
        return _like(key, f.value, "LIKE")
    if op == FilterOperator.TEXT_MATCH_INSENSITIVE:
        return _like(key, f.value, "ILIKE")
    raise ValueError(f"unsupported filter operator: {op.name}")


def _like(key: str, value: object, operator: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string value, got {type(value).__name__}")
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{key} {operator} {quote_str('%' + escaped + '%')} ESCAPE '\\'"


def _as_list(value: object) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected a list value, got {type(value).__name__}")
    return list(value)
