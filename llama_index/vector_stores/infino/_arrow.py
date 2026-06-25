"""Arrow + SQL helpers shared by the store.

Schema contract — column names and SQL-quoting rules — lives here.
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

# Engine error string when a query has no matching rows — see vector_stores/infino issue.
_EMPTY_RESULT_MARKER = "at least one RecordBatch"


def sql_lit(value: str) -> str:
    """Quote a string as a SQL literal."""
    return "'" + value.replace("'", "''") + "'"


def sql_literal(value: Any) -> str:
    """Render any filter value as a SQL literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return sql_lit(value)
    raise TypeError(f"unsupported filter value type: {type(value).__name__}")


def vector_array(vectors: Sequence[Sequence[float]], dim: int) -> pa.Array:
    """Pack embeddings into a ``fixed_size_list<float32, dim>`` array."""
    return pa.array(vectors, type=pa.list_(pa.float32(), dim))


def vector_literal(embedding: Sequence[float]) -> str:
    """Render an embedding as the SQL TVF accepts (comma-separated floats)."""
    return ",".join(str(float(x)) for x in embedding)


def encode_node(
    node: BaseNode,
    declared_keys: Sequence[str],
) -> dict[str, Any]:
    """Project a node onto the store's columns.

    Returns a dict keyed by user-facing field names: ``text``, ``embedding``,
    ``ref_doc_id``, the declared metadata keys, plus the two ``_node_*``
    catch-alls. Caller adds the ``node_id``.
    """
    metadata = node_to_metadata_dict(node, remove_text=True)
    return {
        "text": node.get_content(metadata_mode=MetadataMode.NONE),
        "embedding": node.get_embedding(),
        "ref_doc_id": node.ref_doc_id or "",
        **{k: node.metadata.get(k) for k in declared_keys},
        NODE_CONTENT_COLUMN: metadata[NODE_CONTENT_COLUMN],
        NODE_TYPE_COLUMN: metadata["_node_type"],
    }


def decode_node(
    row: dict[str, Any],
    *,
    text_column: str,
) -> BaseNode:
    """Rebuild a node from a result row's columns."""
    payload = {
        "_node_content": row[NODE_CONTENT_COLUMN],
        "_node_type": row[NODE_TYPE_COLUMN],
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
    """Convert an Arrow result into LlamaIndex's ``VectorStoreQueryResult`` triple."""
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
        node_id = cast(str, row.get(node_id_column) or node.node_id)
        ids.append(node_id)
        nodes.append(node)
    return nodes, ids, scores


def is_empty_result_error(exc: BaseException) -> bool:
    """Engine raises a ``ValueError`` carrying this string when a SQL query
    yields zero matching rows; treat as an empty result."""
    return isinstance(exc, ValueError) and _EMPTY_RESULT_MARKER in str(exc)
