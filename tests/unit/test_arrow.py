from __future__ import annotations

import math

import pyarrow as pa
import pytest
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from llama_index.vector_stores.infino._arrow import (
    NODE_CONTENT_COLUMN,
    NODE_TYPE_COLUMN,
    SCORE_COLUMN,
    quote_ident,
    quote_str,
    rows_to_nodes,
    scores,
    sql_value,
)


def test_quote_str_escapes_single_quotes():
    assert quote_str("o'reilly") == "'o''reilly'"


def test_quote_ident_escapes_double_quotes():
    assert quote_ident("my-table") == '"my-table"'
    assert quote_ident('a"b') == '"a""b"'


def test_sql_value_renders_primitives():
    assert sql_value(5) == "5"
    assert sql_value(3.14) == "3.14"
    assert sql_value(True) == "true"
    assert sql_value("x") == "'x'"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_sql_value_rejects_non_finite_floats(value):
    with pytest.raises(ValueError, match="finite"):
        sql_value(value)


def test_sql_value_rejects_other_types():
    with pytest.raises(TypeError):
        sql_value(None)


def test_rows_to_nodes_rebuilds_serialized_nodes():
    node = TextNode(text="hello world", id_="n1", metadata={"src": "doc1"})
    serialized = node_to_metadata_dict(node, remove_text=True)
    table = pa.table(
        {
            "text": ["hello world"],
            NODE_CONTENT_COLUMN: [serialized[NODE_CONTENT_COLUMN]],
            NODE_TYPE_COLUMN: [serialized[NODE_TYPE_COLUMN]],
            SCORE_COLUMN: [0.42],
        }
    )
    [rebuilt] = rows_to_nodes(table)
    assert rebuilt.node_id == "n1"
    assert rebuilt.get_content() == "hello world"
    assert rebuilt.metadata == {"src": "doc1"}
    assert scores(table) == [0.42]


def test_scores_is_none_without_a_score_column():
    assert scores(pa.table({"text": ["a"]})) is None
