"""The :class:`InfinoVectorStore` LlamaIndex vector store.

One Infino table holds each node's id, text, embedding, ``ref_doc_id``,
declared metadata columns, and the serialized node; vector, BM25, hybrid
(RRF), and SQL retrieval all run over it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import pyarrow as pa
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import node_to_metadata_dict

import infino
from llama_index.vector_stores.infino._arrow import (
    NODE_CONTENT_COLUMN,
    NODE_ID_COLUMN,
    NODE_TYPE_COLUMN,
    REF_DOC_ID_COLUMN,
    SCORE_COLUMN,
    TEXT_COLUMN,
    VECTOR_COLUMN,
    quote_ident,
    quote_str,
    rows_to_nodes,
    scores,
)
from llama_index.vector_stores.infino._filters import compile_filters

Metric = Literal["cosine", "l2sq", "l2", "negdot", "dot"]

DEFAULT_MMR_FETCH_K = 20


class InfinoVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by a single Infino table.

    Args:
        connection: an :class:`infino.Connection` to durable storage (a local
            path, object storage, or Infino Cloud); ``memory://`` cannot delete,
            so ``add`` fails on it.
        table_name: table to open; created if absent.
        dim: embedding dimension, in ``[16, 4096]``.
        metric: ``"cosine"``, ``"l2sq"`` / ``"l2"``, or ``"negdot"`` / ``"dot"``.
        metadata_columns: metadata keys stored as filterable columns; the rest
            round-trips in the serialized node. Fixed at table creation, and
            named apart from the store's own columns.
    """

    flat_metadata: ClassVar[bool] = False

    _connection: infino.Connection = PrivateAttr()
    _table: infino.Table = PrivateAttr()
    _table_name: str = PrivateAttr()
    _dim: int = PrivateAttr()
    _metric: Metric = PrivateAttr()
    _metadata_columns: list[pa.Field] = PrivateAttr()

    def __init__(
        self,
        connection: infino.Connection,
        table_name: str,
        *,
        dim: int,
        metric: Metric = "cosine",
        metadata_columns: Sequence[pa.Field] = (),
    ) -> None:
        super().__init__(stores_text=True)
        self._connection = connection
        self._table_name = table_name
        self._dim = dim
        self._metric = metric
        self._metadata_columns = list(metadata_columns)
        schema = self._build_schema()
        names = [*schema.names, SCORE_COLUMN]
        if len(set(names)) != len(names):
            raise ValueError(
                f"metadata column names must be distinct and not one of the store's "
                f"own columns: {names}"
            )
        self._table = self._open_or_create(schema)

    @property
    def client(self) -> infino.Connection:
        return self._connection

    @property
    def table(self) -> infino.Table:
        return self._table

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        """Upsert nodes by ``node_id``.

        Deletes then appends, which is not atomic: a failed append leaves the
        replaced nodes deleted. Raises ``ValueError`` on a repeated ``node_id``.
        """
        ids = [n.node_id for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("add() got the same node_id more than once")
        if not ids:
            return []
        self._table.delete(_in(NODE_ID_COLUMN, ids))
        self._append(nodes)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._table.delete(f"{quote_ident(REF_DOC_ID_COLUMN)} = {quote_str(ref_doc_id)}")

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        where = self._where(node_ids=node_ids, filters=filters)
        if where is not None:
            self._table.delete(where)

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        where = self._where(node_ids=node_ids, filters=filters)
        if where is None:
            raise ValueError("get_nodes requires node_ids or filters")
        table = self._connection.query_sql(
            f"SELECT {self._select_list(score=False)} "
            f"FROM {quote_ident(self._table_name)} WHERE {where}"
        )
        return rows_to_nodes(table)

    def clear(self) -> None:
        self._connection.drop_table(self._table_name, purge=True)
        self._table = self._open_or_create(self._build_schema())

    def count(self) -> int:
        """Total row count."""
        table = self._connection.query_sql(
            f"SELECT COUNT(*) AS n FROM {quote_ident(self._table_name)}"
        )
        return int(table.column("n")[0].as_py())

    def optimize(self) -> None:
        """Compact the table's files; raises on Infino Cloud, which compacts tables itself."""
        self._table.optimize()

    def gc(self, grace_secs: float) -> None:
        """Delete files no longer referenced and older than ``grace_secs``."""
        self._table.gc(grace_secs)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Run a query in ``DEFAULT``, ``TEXT_SEARCH``, ``HYBRID`` or ``MMR`` mode.

        ``kwargs`` (``vector_store_kwargs`` on a retriever) may carry
        ``filter_query`` / ``filter_mode`` for a full-text pre-filter on the
        text in vector modes, and ``embed_model`` / ``mmr_fetch_k`` for MMR.
        Raises ``NotImplementedError`` for other modes and ``ValueError`` for
        query fields Infino does not support.
        """
        self._check_supported(query)
        # VectorStoreIndex sends node_ids=[] for stores that keep their own text,
        # so an empty list here means no constraint.
        where = self._where(
            node_ids=query.node_ids or None,
            doc_ids=query.doc_ids or None,
            filters=query.filters,
        )
        if kwargs.get("filter_query") is not None and where is not None:
            raise ValueError("`filter_query` cannot be combined with filters, node_ids or doc_ids")
        mode = query.mode
        if mode == VectorStoreQueryMode.DEFAULT:
            table = self._vector_search(query, query.similarity_top_k, where, kwargs)
            return self._result(table, distance=True)
        if mode == VectorStoreQueryMode.TEXT_SEARCH:
            return self._text_query(query, where, kwargs)
        if mode == VectorStoreQueryMode.HYBRID:
            return self._hybrid_query(query, where, kwargs)
        if mode == VectorStoreQueryMode.MMR:
            return self._mmr_query(query, where, kwargs)
        raise NotImplementedError(f"query mode {mode.value!r} is not supported")

    async def async_add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        return await asyncio.to_thread(self.add, nodes, **add_kwargs)

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        await asyncio.to_thread(self.delete, ref_doc_id, **delete_kwargs)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        await asyncio.to_thread(self.delete_nodes, node_ids, filters, **delete_kwargs)

    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        return await asyncio.to_thread(self.get_nodes, node_ids, filters)

    async def aclear(self) -> None:
        await asyncio.to_thread(self.clear)

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return await asyncio.to_thread(self.query, query, **kwargs)

    def search_by_sql(self, sql: str) -> VectorStoreQueryResult:
        """Run SQL and map its rows to nodes, with ``score`` as the similarity.

        The result must include ``text``, ``_node_content`` and
        ``_node_type``; raises ``ValueError`` when one is missing.
        """
        table = self._connection.query_sql(sql)
        required = [TEXT_COLUMN, NODE_CONTENT_COLUMN, NODE_TYPE_COLUMN]
        missing = [c for c in required if c not in table.column_names]
        if missing:
            raise ValueError(f"search_by_sql result is missing columns {missing}")
        return self._result(table, distance=False)

    def _check_supported(self, query: VectorStoreQuery) -> None:
        if query.alpha is not None:
            raise ValueError("`alpha` is not supported: HYBRID fuses with RRF, which has no weight")
        if query.sparse_top_k is not None:
            raise ValueError("`sparse_top_k` is not supported: HYBRID takes one top-k")
        if query.output_fields:
            raise ValueError("`output_fields` is not supported: results carry whole nodes")
        if query.embedding_field not in (None, VECTOR_COLUMN):
            raise ValueError(f"`embedding_field` must be {VECTOR_COLUMN!r}, the store's vector")

    def _vector_search(
        self,
        query: VectorStoreQuery,
        top_k: int,
        where: str | None,
        kwargs: dict[str, Any],
    ) -> pa.Table:
        embedding = _require_embedding(query)
        if where is not None:
            tvf = (
                f"vector_search({quote_str(self._table_name)}, "
                f"{quote_str(VECTOR_COLUMN)}, {quote_str(_vector_literal(embedding))}, "
                f"{top_k})"
            )
            return self._search_where(tvf, where, top_k, ascending=True)
        filter_query = kwargs.get("filter_query")
        if filter_query is None:
            return self._table.vector_search(
                VECTOR_COLUMN, embedding, top_k, projection=self._projection()
            )
        return self._table.vector_search(
            VECTOR_COLUMN,
            embedding,
            top_k,
            filter_column=TEXT_COLUMN,
            filter_query=filter_query,
            filter_mode=kwargs.get("filter_mode"),
            projection=self._projection(),
        )

    def _text_query(
        self, query: VectorStoreQuery, where: str | None, kwargs: dict[str, Any]
    ) -> VectorStoreQueryResult:
        query_str = _require_query_str(query, "TEXT_SEARCH")
        _reject_filter_query(kwargs, "TEXT_SEARCH")
        top_k = query.similarity_top_k
        if where is None:
            table = self._table.bm25_search(
                TEXT_COLUMN, query_str, top_k, projection=self._projection()
            )
        else:
            tvf = (
                f"bm25_search({quote_str(self._table_name)}, "
                f"{quote_str(TEXT_COLUMN)}, {quote_str(query_str)}, {top_k})"
            )
            table = self._search_where(tvf, where, top_k, ascending=False)
        return self._result(table, distance=False)

    def _hybrid_query(
        self, query: VectorStoreQuery, where: str | None, kwargs: dict[str, Any]
    ) -> VectorStoreQueryResult:
        query_str = _require_query_str(query, "HYBRID")
        embedding = _require_embedding(query)
        _reject_filter_query(kwargs, "HYBRID")
        top_k = query.hybrid_top_k or query.similarity_top_k
        if where is None:
            table = self._table.hybrid_search(
                TEXT_COLUMN,
                query_str,
                VECTOR_COLUMN,
                embedding,
                top_k,
                projection=self._projection(),
            )
        else:
            tvf = (
                f"hybrid_search({quote_str(self._table_name)}, "
                f"{quote_str(TEXT_COLUMN)}, {quote_str(query_str)}, "
                f"{quote_str(VECTOR_COLUMN)}, {quote_str(_vector_literal(embedding))}, "
                f"{top_k})"
            )
            table = self._search_where(tvf, where, top_k, ascending=False)
        return self._result(table, distance=False)

    def _mmr_query(
        self, query: VectorStoreQuery, where: str | None, kwargs: dict[str, Any]
    ) -> VectorStoreQueryResult:
        from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings

        embed_model = kwargs.get("embed_model")
        if embed_model is None:
            raise ValueError(
                "MMR requires `embed_model` in vector_store_kwargs: Infino does not "
                "return stored vectors, so candidates are re-embedded"
            )
        fetch_k = int(kwargs.get("mmr_fetch_k", DEFAULT_MMR_FETCH_K))
        candidates = self._result(self._vector_search(query, fetch_k, where, kwargs), distance=True)
        if not candidates.nodes:
            return candidates
        embeddings = embed_model.get_text_embedding_batch(
            [n.get_content(metadata_mode=MetadataMode.EMBED) for n in candidates.nodes]
        )
        _, selected = get_top_k_mmr_embeddings(
            query_embedding=_require_embedding(query),
            embeddings=embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=list(range(len(candidates.nodes))),
            mmr_threshold=query.mmr_threshold,
        )
        similarities = candidates.similarities or []
        return VectorStoreQueryResult(
            nodes=[candidates.nodes[i] for i in selected],
            ids=[candidates.nodes[i].node_id for i in selected],
            similarities=[similarities[i] for i in selected],
        )

    def _search_where(self, tvf: str, where: str, top_k: int, *, ascending: bool) -> pa.Table:
        # A WHERE over a search function ranks within exactly the rows it keeps.
        order = "ASC" if ascending else "DESC"
        return self._connection.query_sql(
            f"SELECT {self._select_list(score=True)} FROM {tvf} WHERE {where} "
            f"ORDER BY {SCORE_COLUMN} {order} LIMIT {top_k}"
        )

    def _projection(self) -> list[str]:
        return [
            TEXT_COLUMN,
            *(f.name for f in self._metadata_columns),
            NODE_CONTENT_COLUMN,
            NODE_TYPE_COLUMN,
            SCORE_COLUMN,
        ]

    def _select_list(self, *, score: bool) -> str:
        columns = self._projection() if score else self._projection()[:-1]
        return ", ".join(quote_ident(c) for c in columns)

    def _where(
        self,
        *,
        node_ids: Sequence[str] | None = None,
        doc_ids: Sequence[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> str | None:
        parts = []
        if node_ids is not None:
            parts.append(_in(NODE_ID_COLUMN, node_ids))
        if doc_ids is not None:
            parts.append(_in(REF_DOC_ID_COLUMN, doc_ids))
        if filters is not None:
            columns = {f.name: f.type for f in self._metadata_columns}
            parts.append(compile_filters(filters, columns))
        return " AND ".join(f"({p})" for p in parts) if parts else None

    def _append(self, nodes: Sequence[BaseNode]) -> None:
        serialized = [node_to_metadata_dict(n, remove_text=True) for n in nodes]
        columns: dict[str, list[Any]] = {
            NODE_ID_COLUMN: [n.node_id for n in nodes],
            REF_DOC_ID_COLUMN: [n.ref_doc_id or "" for n in nodes],
            TEXT_COLUMN: [n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes],
            VECTOR_COLUMN: [n.get_embedding() for n in nodes],
            **{f.name: [n.metadata.get(f.name) for n in nodes] for f in self._metadata_columns},
            NODE_CONTENT_COLUMN: [s[NODE_CONTENT_COLUMN] for s in serialized],
            NODE_TYPE_COLUMN: [s[NODE_TYPE_COLUMN] for s in serialized],
        }
        self._table.append(pa.Table.from_pydict(columns, schema=self._table.schema()))

    def _result(self, table: pa.Table, *, distance: bool) -> VectorStoreQueryResult:
        nodes = rows_to_nodes(table)
        raw = scores(table)
        similarities = (
            [_distance_to_similarity(self._metric, d) for d in raw]
            if distance and raw is not None
            else raw
        )
        return VectorStoreQueryResult(
            nodes=nodes, ids=[n.node_id for n in nodes], similarities=similarities
        )

    def _build_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(NODE_ID_COLUMN, pa.large_utf8(), nullable=False),
                pa.field(REF_DOC_ID_COLUMN, pa.large_utf8(), nullable=False),
                pa.field(TEXT_COLUMN, pa.large_utf8(), nullable=False),
                pa.field(VECTOR_COLUMN, pa.list_(pa.float32(), self._dim), nullable=False),
                *self._metadata_columns,
                pa.field(NODE_CONTENT_COLUMN, pa.large_utf8(), nullable=False),
                pa.field(NODE_TYPE_COLUMN, pa.large_utf8(), nullable=False),
            ]
        )

    def _open_or_create(self, schema: pa.Schema) -> infino.Table:
        if self._table_name in self._connection.list_tables():
            table = self._connection.open_table(self._table_name)
            stored = table.schema()
            if not stored.equals(schema):
                raise ValueError(
                    f"table {self._table_name!r} exists with a different schema; `dim` and "
                    f"`metadata_columns` are fixed at creation: "
                    + "; ".join(_schema_mismatch(stored, schema))
                    + ". Open it with the stored schema, or create a new table for the new shape."
                )
            return table
        indexes = (
            infino.IndexSpec()
            .fts(NODE_ID_COLUMN)
            .fts(REF_DOC_ID_COLUMN)
            .fts(TEXT_COLUMN)
            .vector(VECTOR_COLUMN, self._dim, self._metric)
        )
        return self._connection.create_table(self._table_name, schema, indexes)


def _in(column: str, values: Sequence[str]) -> str:
    if not values:
        return "FALSE"
    return f"{quote_ident(column)} IN ({', '.join(quote_str(v) for v in values)})"


def _require_embedding(query: VectorStoreQuery) -> list[float]:
    if query.query_embedding is None:
        raise ValueError(f"query.query_embedding is required for {query.mode.value} mode")
    return list(query.query_embedding)


def _require_query_str(query: VectorStoreQuery, mode: str) -> str:
    if not query.query_str:
        raise ValueError(f"query.query_str is required for {mode}")
    return query.query_str


def _reject_filter_query(kwargs: dict[str, Any], mode: str) -> None:
    if kwargs.get("filter_query") is not None:
        raise ValueError(
            f"`filter_query` applies to vector search only, not {mode}; use `query.filters`"
        )


def _schema_mismatch(stored: pa.Schema, declared: pa.Schema) -> list[str]:
    """Describe each column whose type differs or is missing on one side."""
    mismatches = []
    for name in dict.fromkeys([*stored.names, *declared.names]):
        s = stored.field(name).type if name in stored.names else None
        d = declared.field(name).type if name in declared.names else None
        if s != d:
            mismatches.append(f"{name} (stored {s}, declared {d})")
    return mismatches


def _vector_literal(embedding: Sequence[float]) -> str:
    return ",".join(repr(float(x)) for x in embedding)


def _distance_to_similarity(metric: Metric, distance: float) -> float:
    """Turn an engine distance into a similarity where higher is better."""
    if metric == "cosine":
        return 1.0 - distance
    if metric in ("l2", "l2sq"):
        return 1.0 / (1.0 + distance)
    return -distance
