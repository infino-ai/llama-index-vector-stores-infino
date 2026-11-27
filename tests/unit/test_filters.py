from __future__ import annotations

import pyarrow as pa
import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from llama_index.vector_stores.infino._filters import compile_filters

COLUMNS = {
    "cat": pa.large_utf8(),
    "year": pa.int64(),
    "tags": pa.list_(pa.large_utf8()),
}


def _compile(key: str, value, op: FilterOperator = FilterOperator.EQ) -> str:
    return compile_filters(
        MetadataFilters(filters=[MetadataFilter(key=key, value=value, operator=op)]), COLUMNS
    )


def _group(*filters, condition=FilterCondition.AND) -> MetadataFilters:
    return MetadataFilters(filters=list(filters), condition=condition)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        (FilterOperator.EQ, '"year" = 2020'),
        (FilterOperator.NE, '"year" != 2020'),
        (FilterOperator.GT, '"year" > 2020'),
        (FilterOperator.GTE, '"year" >= 2020'),
        (FilterOperator.LT, '"year" < 2020'),
        (FilterOperator.LTE, '"year" <= 2020'),
    ],
)
def test_comparisons(op, expected):
    assert _compile("year", 2020, op) == expected


def test_in_and_not_in():
    assert _compile("cat", ["ml", "nlp"], FilterOperator.IN) == "\"cat\" IN ('ml', 'nlp')"
    assert _compile("cat", ["ml"], FilterOperator.NIN) == "\"cat\" NOT IN ('ml')"


def test_empty_in_matches_nothing_and_empty_nin_matches_everything():
    assert _compile("cat", [], FilterOperator.IN) == "FALSE"
    assert _compile("cat", [], FilterOperator.NIN) == "TRUE"


def test_array_operators_on_list_column():
    assert _compile("tags", "a", FilterOperator.CONTAINS) == "array_has(\"tags\", 'a')"
    assert _compile("tags", ["a", "b"], FilterOperator.ANY) == (
        "array_has_any(\"tags\", ['a', 'b'])"
    )
    assert _compile("tags", ["a", "b"], FilterOperator.ALL) == (
        "array_has_all(\"tags\", ['a', 'b'])"
    )
    assert _compile("tags", [], FilterOperator.ANY) == "FALSE"
    assert _compile("tags", [], FilterOperator.ALL) == "TRUE"


@pytest.mark.parametrize("op", [FilterOperator.CONTAINS, FilterOperator.ANY, FilterOperator.ALL])
def test_array_operators_reject_scalar_columns(op):
    with pytest.raises(ValueError, match="needs a list column"):
        _compile("cat", ["a"], op)


def test_scalar_operators_reject_list_columns():
    with pytest.raises(ValueError, match="does not apply to list column"):
        _compile("tags", "a")


def test_is_empty_depends_on_column_type():
    assert _compile("year", None, FilterOperator.IS_EMPTY) == '"year" IS NULL'
    assert _compile("tags", None, FilterOperator.IS_EMPTY) == (
        '("tags" IS NULL OR cardinality("tags") = 0)'
    )


def test_text_match_escapes_like_wildcards():
    assert _compile("cat", "100%_x", FilterOperator.TEXT_MATCH) == (
        r""""cat" LIKE '%100\%\_x%' ESCAPE '\'"""
    )
    assert _compile("cat", "GPT", FilterOperator.TEXT_MATCH_INSENSITIVE) == (
        "\"cat\" ILIKE '%GPT%' ESCAPE '\\'"
    )


def test_conditions_and_nesting():
    ml = MetadataFilter(key="cat", value="ml")
    nlp = MetadataFilter(key="cat", value="nlp")
    recent = MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)
    either = _group(ml, nlp, condition=FilterCondition.OR)
    assert compile_filters(_group(either, recent), COLUMNS) == (
        '(("cat" = \'ml\') OR ("cat" = \'nlp\')) AND ("year" >= 2024)'
    )
    assert compile_filters(_group(ml, condition=FilterCondition.NOT), COLUMNS) == (
        "NOT (\"cat\" = 'ml')"
    )
    assert compile_filters(_group(), COLUMNS) == "TRUE"


def test_rejects_undeclared_column():
    with pytest.raises(ValueError, match="not a declared metadata column"):
        _compile("unknown", 1)


def test_escapes_single_quotes_in_values():
    assert _compile("cat", "O'Reilly") == "\"cat\" = 'O''Reilly'"
