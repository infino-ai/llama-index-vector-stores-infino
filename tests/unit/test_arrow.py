from __future__ import annotations

import pyarrow as pa
from llama_index.core.schema import TextNode

from llama_index.vector_stores.infino._arrow import (
    NODE_CONTENT_COLUMN,
    NODE_TYPE_COLUMN,
    SCORE_COLUMN,
    encode_node,
    quote_str,
    rows_to_results,
    sql_value,
    vector_array,
)


def test_quote_str_escapes_single_quotes():
    assert quote_str("o'reilly") == "'o''reilly'"


def test_sql_value_handles_primitive_types():
    assert sql_value(5) == "5"
    assert sql_value(3.14) == "3.14"
    assert sql_value(True) == "true"
    assert sql_value(False) == "false"
    assert sql_value("x") == "'x'"


def test_vector_array_is_fixed_size_list_float32():
    arr = vector_array([[0.1, 0.2], [0.3, 0.4]], dim=2)
    assert arr.type == pa.list_(pa.float32(), 2)


def test_encode_decode_roundtrip_via_arrow_table():
    node = TextNode(text="hello world", id_="n1", embedding=[0.5] * 4, metadata={"src": "doc1"})
    node.relationships = {}
    row = encode_node(node, declared_keys=["src"])
    assert row["text"] == "hello world"
    assert row["src"] == "doc1"
    assert row[NODE_CONTENT_COLUMN]
    assert row[NODE_TYPE_COLUMN] == "TextNode"

    table = pa.table(
        {
            "node_id": [node.node_id],
            "text": [row["text"]],
            "src": [row["src"]],
            NODE_CONTENT_COLUMN: [row[NODE_CONTENT_COLUMN]],
            NODE_TYPE_COLUMN: [row[NODE_TYPE_COLUMN]],
            SCORE_COLUMN: [0.42],
        }
    )
    nodes, ids, scores = rows_to_results(table, node_id_column="node_id", text_column="text")
    assert ids == [node.node_id]
    assert scores == [0.42]
    assert nodes[0].get_content() == "hello world"
    assert nodes[0].metadata == {"src": "doc1"}


def test_rows_to_results_handles_empty_table():
    empty = pa.table({"node_id": [], "text": []})
    nodes, ids, scores = rows_to_results(empty, node_id_column="node_id", text_column="text")
    assert nodes == [] and ids == [] and scores is None
