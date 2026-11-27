"""SQL quoting and Arrow-to-node conversion shared by the store."""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node

NODE_ID_COLUMN = "node_id"
REF_DOC_ID_COLUMN = "ref_doc_id"
TEXT_COLUMN = "text"
VECTOR_COLUMN = "embedding"
NODE_CONTENT_COLUMN = "_node_content"
NODE_TYPE_COLUMN = "_node_type"
SCORE_COLUMN = "score"


def quote_ident(name: str) -> str:
    """Quote a table or column name as a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def quote_str(value: str) -> str:
    """Quote a string as a SQL literal."""
    return "'" + value.replace("'", "''") + "'"


def sql_value(value: Any) -> str:
    """Render a filter value as a SQL literal; raises ``TypeError`` for other types."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"filter value must be finite, got {value}")
        return repr(value)
    if isinstance(value, str):
        return quote_str(value)
    raise TypeError(f"unsupported filter value type: {type(value).__name__}")


def rows_to_nodes(table: pa.Table) -> list[BaseNode]:
    """Rebuild the nodes a result table holds, in row order."""
    texts = table.column(TEXT_COLUMN).to_pylist()
    contents = table.column(NODE_CONTENT_COLUMN).to_pylist()
    types = table.column(NODE_TYPE_COLUMN).to_pylist()
    return [
        metadata_dict_to_node({NODE_CONTENT_COLUMN: c, NODE_TYPE_COLUMN: t}, text=text)
        for text, c, t in zip(texts, contents, types, strict=True)
    ]


def scores(table: pa.Table) -> list[float] | None:
    """The result's ``score`` column, or ``None`` when it has none."""
    if SCORE_COLUMN not in table.column_names:
        return None
    return [float(s) for s in table.column(SCORE_COLUMN).to_pylist()]
