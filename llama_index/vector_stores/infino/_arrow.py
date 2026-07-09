"""Arrow + SQL helpers shared by the store.

The schema contract — column names and SQL-quoting rules — lives here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pyarrow as pa
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)

NODE_CONTENT_COLUMN = "_node_content"
NODE_TYPE_COLUMN = "_node_type"
SCORE_COLUMN = "score"


def quote_str(value: str) -> str:
    """Quote a string as a SQL literal, escaping embedded single quotes."""
    return "'" + value.replace("'", "''") + "'"


def sql_value(value: Any) -> str:
    """Render a typed filter value as a SQL literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return quote_str(value)
    raise TypeError(f"unsupported filter value type: {type(value).__name__}")


def vector_array(vectors: Sequence[Sequence[float]], dim: int) -> pa.Array:
    """Pack embeddings into a ``fixed_size_list<float32, dim>`` array."""
    return pa.array(vectors, type=pa.list_(pa.float32(), dim))


def encode_node(node: BaseNode, declared_keys: Sequence[str]) -> dict[str, Any]:
    """Project a node onto the store's columns (caller adds ``node_id``)."""
    metadata = node_to_metadata_dict(node, remove_text=True)
    return {
        "text": node.get_content(metadata_mode=MetadataMode.NONE),
        "embedding": node.get_embedding(),
        "ref_doc_id": node.ref_doc_id or "",
        **{k: node.metadata.get(k) for k in declared_keys},
        NODE_CONTENT_COLUMN: metadata[NODE_CONTENT_COLUMN],
        NODE_TYPE_COLUMN: metadata[NODE_TYPE_COLUMN],
    }


def decode_node(row: dict[str, Any], *, text_column: str) -> BaseNode:
    """Rebuild a node from a result row, falling back to a bare TextNode."""
    payload = {
        NODE_CONTENT_COLUMN: row[NODE_CONTENT_COLUMN],
        NODE_TYPE_COLUMN: row[NODE_TYPE_COLUMN],
    }
    try:
        return metadata_dict_to_node(payload, text=row[text_column])
    except ValueError:
        return TextNode(text=row[text_column] or "", id_=row.get("node_id"))


def rows_to_results(
    table: pa.Table,
    *,
    node_id_column: str,
    text_column: str,
) -> tuple[list[BaseNode], list[str], list[float] | None]:
    """Convert an Arrow result into LlamaIndex's node/id/score triple."""
    n = table.num_rows
    if n == 0:
        return [], [], None

    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    scores = cols.get(SCORE_COLUMN)

    nodes: list[BaseNode] = []
    ids: list[str] = []
    for i in range(n):
        row = {name: vals[i] for name, vals in cols.items()}
        node = decode_node(row, text_column=text_column)
        ids.append(cast(str, row.get(node_id_column) or node.node_id))
        nodes.append(node)
    return nodes, ids, scores
