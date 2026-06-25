from __future__ import annotations

import pytest
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from llama_index.vector_stores.infino._filters import compile_filters

ALLOWED = {"cat", "year", "tag"}


def _f(key: str, value, op: FilterOperator = FilterOperator.EQ) -> MetadataFilter:
    return MetadataFilter(key=key, value=value, operator=op)


def _wrap(*filters, condition=FilterCondition.AND) -> MetadataFilters:
    return MetadataFilters(filters=list(filters), condition=condition)


def test_equality_and_comparison_operators():
    cases = [
        (FilterOperator.EQ, "ml", "cat = 'ml'"),
        (FilterOperator.NE, "ml", "cat != 'ml'"),
        (FilterOperator.GT, 2020, "cat > 2020"),
        (FilterOperator.GTE, 2020, "cat >= 2020"),
        (FilterOperator.LT, 2020, "cat < 2020"),
        (FilterOperator.LTE, 2020, "cat <= 2020"),
    ]
    for op, value, expected in cases:
        assert compile_filters(_wrap(_f("cat", value, op)), ALLOWED) == expected


def test_set_membership_operators():
    in_ = compile_filters(_wrap(_f("cat", ["ml", "nlp"], FilterOperator.IN)), ALLOWED)
    nin = compile_filters(_wrap(_f("cat", ["ml"], FilterOperator.NIN)), ALLOWED)
    any_ = compile_filters(_wrap(_f("cat", ["ml"], FilterOperator.ANY)), ALLOWED)
    assert in_ == "cat IN ('ml', 'nlp')"
    assert nin == "cat NOT IN ('ml')"
    # ANY is treated as IN for scalar columns (no list-typed columns in our schema).
    assert any_ == "cat IN ('ml')"


def test_all_operator_becomes_and_of_equals():
    out = compile_filters(_wrap(_f("tag", ["a", "b"], FilterOperator.ALL)), ALLOWED)
    assert out == "tag = 'a' AND tag = 'b'"


def test_text_match_uses_escaped_like():
    out = compile_filters(_wrap(_f("tag", "100%", FilterOperator.TEXT_MATCH)), ALLOWED)
    # %/_ escaped, wrapped in % wildcards, ESCAPE clause emitted.
    assert out == r"tag LIKE '%100\%%' ESCAPE '\'"


def test_text_match_insensitive_uses_ilike():
    out = compile_filters(
        _wrap(_f("tag", "GPT", FilterOperator.TEXT_MATCH_INSENSITIVE)), ALLOWED
    )
    assert "ILIKE" in out
    assert "'%GPT%'" in out


def test_is_empty_handles_null_and_empty_string():
    out = compile_filters(_wrap(_f("tag", "", FilterOperator.IS_EMPTY)), ALLOWED)
    assert out == "(tag IS NULL OR tag = '')"


def test_and_or_combinations():
    and_out = compile_filters(
        _wrap(_f("cat", "ml"), _f("year", 2024, FilterOperator.GTE)),
        ALLOWED,
    )
    or_out = compile_filters(
        _wrap(_f("cat", "ml"), _f("cat", "nlp"), condition=FilterCondition.OR),
        ALLOWED,
    )
    assert and_out == "(cat = 'ml') AND (year >= 2024)"
    assert or_out == "(cat = 'ml') OR (cat = 'nlp')"


def test_nested_groups_compose():
    inner = _wrap(_f("cat", "ml"), _f("cat", "nlp"), condition=FilterCondition.OR)
    out = compile_filters(_wrap(inner, _f("year", 2024, FilterOperator.GTE)), ALLOWED)
    assert out == "((cat = 'ml') OR (cat = 'nlp')) AND (year >= 2024)"


def test_not_condition_wraps_in_not():
    out = compile_filters(_wrap(_f("cat", "ml"), condition=FilterCondition.NOT), ALLOWED)
    assert out == "NOT (cat = 'ml')"


def test_rejects_undeclared_column():
    with pytest.raises(ValueError, match="not a declared metadata column"):
        compile_filters(_wrap(_f("unknown", 1)), ALLOWED)


def test_quotes_escape_single_quotes():
    out = compile_filters(_wrap(_f("cat", "O'Reilly")), ALLOWED)
    assert out == "cat = 'O''Reilly'"


def test_empty_group_yields_true():
    out = compile_filters(_wrap(), ALLOWED)
    assert out == "TRUE"
