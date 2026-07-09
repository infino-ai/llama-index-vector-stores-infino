"""The :class:`InfinoVectorStore` LlamaIndex vector store.

One Infino table holds each node's id, text, embedding, ``ref_doc_id``,
declared metadata columns, and a JSON catch-all; vector, BM25, hybrid (RRF),
and SQL retrieval all run over it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import pyarrow as pa
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

import infino
from llama_index.vector_stores.infino._arrow import (
    NODE_CONTENT_COLUMN,
    NODE_TYPE_COLUMN,
    SCORE_COLUMN,
    encode_node,
    quote_str,
    rows_to_results,
    vector_array,
)
from llama_index.vector_stores.infino._filters import compile_filters

Metric = Literal["cosine", "l2sq", "l2", "negdot", "dot"]

DEFAULT_METRIC: Metric = "cosine"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_VECTOR_COLUMN = "embedding"
DEFAULT_NODE_ID_COLUMN = "node_id"
DEFAULT_REF_DOC_ID_COLUMN = "ref_doc_id"
# IVF builder clamps n_cent <= 64 below 100K rows; 64 is the effective max.
DEFAULT_N_CENT = 64
# MMR re-embeds candidate texts; over-fetch a wider pool than the final top-k.
DEFAULT_MMR_FETCH_K = 20
# Structured filters run post-rank, so over-fetch from the vector TVF and trim.
DEFAULT_FILTER_OVERSAMPLE = 10


class InfinoVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by a single Infino table.

    Args:
        connection: a live :class:`infino.Connection` (durable; ``memory://``
            cannot delete or upsert).
        table_name: table to open; created if absent.
        dim: embedding dimension (must be in ``[16, 4096]``).
        metric: ``"cosine"`` (default), ``"l2sq"`` / ``"l2"``, ``"negdot"`` / ``"dot"``.
        text_column / vector_column / node_id_column / ref_doc_id_column:
            column names.
        metadata_columns: metadata keys promoted to filterable scalar columns;
            the rest round-trips via the JSON catch-all. Fixed at creation.
        n_cent: IVF centroid count (engine-clamped on small tables).
        filter_oversample: over-fetch multiplier for structured-filter queries.
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: ClassVar[bool] = False

    _connection: infino.Connection = PrivateAttr()
    _table: infino.Table = PrivateAttr()
    _table_name: str = PrivateAttr()
    _dim: int = PrivateAttr()
    _metric: Metric = PrivateAttr()
    _text_column: str = PrivateAttr()
    _vector_column: str = PrivateAttr()
    _node_id_column: str = PrivateAttr()
    _ref_doc_id_column: str = PrivateAttr()
    _metadata_columns: list[pa.Field] = PrivateAttr()
    _metadata_column_names: list[str] = PrivateAttr()
    _n_cent: int = PrivateAttr()
    _filter_oversample: int = PrivateAttr()

    SUPPORTED_MODES: ClassVar[frozenset[VectorStoreQueryMode]] = frozenset(
        {
            VectorStoreQueryMode.DEFAULT,
            VectorStoreQueryMode.TEXT_SEARCH,
            VectorStoreQueryMode.HYBRID,
            VectorStoreQueryMode.MMR,
        }
    )

    def __init__(
        self,
        connection: infino.Connection,
        table_name: str,
        *,
        dim: int,
        metric: Metric = DEFAULT_METRIC,
        text_column: str = DEFAULT_TEXT_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        node_id_column: str = DEFAULT_NODE_ID_COLUMN,
        ref_doc_id_column: str = DEFAULT_REF_DOC_ID_COLUMN,
        metadata_columns: Sequence[pa.Field] = (),
        n_cent: int = DEFAULT_N_CENT,
        filter_oversample: int = DEFAULT_FILTER_OVERSAMPLE,
    ) -> None:
        super().__init__(stores_text=True, is_embedding_query=True)
        self._connection = connection
        self._table_name = table_name
        self._dim = dim
        self._metric = metric
        self._text_column = text_column
        self._vector_column = vector_column
        self._node_id_column = node_id_column
        self._ref_doc_id_column = ref_doc_id_column
        self._metadata_columns = list(metadata_columns)
        self._metadata_column_names = [f.name for f in self._metadata_columns]
        self._n_cent = n_cent
        self._filter_oversample = filter_oversample
        self._table = _open_or_create(connection, table_name, self._build_schema(), n_cent, metric)

    @classmethod
    def from_params(
        cls,
        connection: infino.Connection,
        table_name: str,
        *,
        dim: int,
        **kwargs: Any,
    ) -> InfinoVectorStore:
        """Construct a store, opening or creating ``table_name``."""
        return cls(connection, table_name, dim=dim, **kwargs)

    @property
    def client(self) -> infino.Connection:
        return self._connection

    @property
    def table(self) -> infino.Table:
        return self._table

    def add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        if not nodes:
            return []
        ids = [n.node_id for n in nodes]
        # Upsert = delete-by-node_id then append; not atomic (no engine txn).
        self._delete_node_ids(ids)
        self._append(nodes, ids)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self._table.delete(f"{self._ref_doc_id_column} = {quote_str(ref_doc_id)}")

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **delete_kwargs: Any,
    ) -> None:
        predicate = self._where(node_ids=node_ids, filters=filters)
        if predicate is not None:
            self._table.delete(predicate)

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        if node_ids is None and filters is None:
            raise ValueError("get_nodes requires node_ids or filters")
        # node_ids alone → exact_match per id (the only pre-I/O prune for uuids).
        if node_ids is not None and filters is None:
            return self._get_by_ids(node_ids)
        predicate = self._where(node_ids=node_ids, filters=filters) or "TRUE"
        columns = ", ".join(self._node_projection())
        table = self._connection.query_sql(
            f"SELECT {columns} FROM {self._table_name} WHERE {predicate}"
        )
        nodes, _, _ = rows_to_results(
            table,
            node_id_column=self._node_id_column,
            text_column=self._text_column,
        )
        return nodes

    def clear(self) -> None:
        self._connection.drop_table(self._table_name, purge=True)
        self._table = _open_or_create(
            self._connection,
            self._table_name,
            self._build_schema(),
            self._n_cent,
            self._metric,
        )

    def count(self) -> int:
        """Total row count."""
        table = self._connection.query_sql(f"SELECT COUNT(*) AS n FROM {self._table_name}")
        return int(table.column("n")[0].as_py())

    def optimize(self) -> None:
        """Compact the table's superfiles."""
        self._table.optimize()

    def gc(self, grace_secs: float) -> None:
        """Reclaim storage from files older than ``grace_secs``."""
        self._table.gc(grace_secs)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        mode = query.mode
        if mode == VectorStoreQueryMode.DEFAULT:
            return self._vector_query(query, **kwargs)
        if mode == VectorStoreQueryMode.TEXT_SEARCH:
            return self._text_query(query)
        if mode == VectorStoreQueryMode.HYBRID:
            return self._hybrid_query(query)
        if mode == VectorStoreQueryMode.MMR:
            return self._mmr_query(query, **kwargs)
        raise NotImplementedError(f"query mode {mode!r} is not supported")

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        return await asyncio.to_thread(self.add, nodes, **kwargs)

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
        """Run arbitrary SQL and map rows to a ``VectorStoreQueryResult``.

        The SELECT must project the store's read columns (node id, ref doc id,
        text, declared metadata, the two ``_node_*`` catch-alls) and, for
        ranking, ``score``.
        """
        return self._to_result(self._connection.query_sql(sql), distance_metric=False)

    def _vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        embedding = _require_embedding(query)
        filter_query = kwargs.get("filter_query")
        if query.filters is not None and filter_query is not None:
            raise ValueError(
                "use either `query.filters` (structured, post-rank) or "
                "`filter_query` (text pushdown, pre-rank), not both"
            )
        if filter_query is not None:
            result = self._table.vector_search(
                self._vector_column,
                embedding,
                query.similarity_top_k,
                filter_column=kwargs.get("filter_column") or self._text_column,
                filter_query=filter_query,
                filter_mode=kwargs.get("filter_mode"),
                projection=self._search_projection(),
            )
        elif query.filters is not None:
            result = self._filtered_vector_search(query, query.filters, embedding)
        else:
            result = self._table.vector_search(
                self._vector_column,
                embedding,
                query.similarity_top_k,
                projection=self._search_projection(),
            )
        return self._to_result(result, distance_metric=True)

    def _filtered_vector_search(
        self,
        query: VectorStoreQuery,
        filters: MetadataFilters,
        embedding: list[float],
    ) -> pa.Table:
        # vector_search has no arbitrary-WHERE param, so structured filters go
        # through the SQL TVF: over-fetch, filter, then trim to top-k.
        where = compile_filters(filters, self._metadata_column_names)
        columns = ", ".join(self._search_projection())
        vector = ",".join(str(float(x)) for x in embedding)
        sql = (
            f"SELECT {columns} FROM vector_search("
            f"{quote_str(self._table_name)}, {quote_str(self._vector_column)}, "
            f"'{vector}', {query.similarity_top_k * self._filter_oversample}) "
            f"WHERE {where} ORDER BY {SCORE_COLUMN} ASC "
            f"LIMIT {query.similarity_top_k}"
        )
        return self._connection.query_sql(sql)

    def _text_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        query_str = _require_query_str(query, "TEXT_SEARCH")
        result = self._table.bm25_search(
            self._text_column,
            query_str,
            query.similarity_top_k,
            projection=self._search_projection(),
        )
        return self._to_result(result, distance_metric=False)

    def _hybrid_query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        query_str = _require_query_str(query, "HYBRID")
        embedding = _require_embedding(query)
        k = query.hybrid_top_k or query.similarity_top_k
        result = self._table.hybrid_search(
            self._text_column,
            query_str,
            self._vector_column,
            embedding,
            k,
            projection=self._search_projection(),
        )
        return self._to_result(result, distance_metric=False)

    def _mmr_query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        # Local import: keep MMR utils off the import path for non-MMR users.
        from llama_index.core.indices.query.embedding_utils import (
            get_top_k_mmr_embeddings,
        )

        embed_model = kwargs.get("embed_model")
        if embed_model is None:
            raise ValueError(
                "MMR requires `embed_model` in vector_store_kwargs: Infino "
                "does not return stored vectors, so candidate texts are re-embedded"
            )
        query_embedding = _require_embedding(query)
        fetch_k = int(kwargs.get("mmr_fetch_k", DEFAULT_MMR_FETCH_K))
        candidates = self._table.vector_search(
            self._vector_column,
            query_embedding,
            fetch_k,
            projection=self._search_projection(),
        )
        base = self._to_result(candidates, distance_metric=True)
        if not base.nodes:
            return base
        candidate_embeddings = embed_model.get_text_embedding_batch(
            [n.get_content() for n in base.nodes]
        )
        _scores, selected_idx = get_top_k_mmr_embeddings(
            query_embedding=query_embedding,
            embeddings=candidate_embeddings,
            similarity_top_k=query.similarity_top_k,
            embedding_ids=list(range(len(base.nodes))),
            mmr_threshold=query.mmr_threshold,
        )
        nodes = [base.nodes[i] for i in selected_idx]
        ids_list = base.ids or []
        ids = [ids_list[i] for i in selected_idx] if ids_list else []
        sims = (
            [base.similarities[i] for i in selected_idx] if base.similarities is not None else None
        )
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=sims)

    def _get_by_ids(self, node_ids: Sequence[str]) -> list[BaseNode]:
        # exact_match on the unique node_id returns <=1 row, so dedup the input
        # (preserving order) rather than the results.
        projection = self._node_projection()
        nodes: list[BaseNode] = []
        for node_id in dict.fromkeys(node_ids):
            table = self._table.exact_match(self._node_id_column, node_id, projection=projection)
            found, _, _ = rows_to_results(
                table,
                node_id_column=self._node_id_column,
                text_column=self._text_column,
            )
            nodes.extend(found)
        return nodes

    def _search_projection(self) -> list[str]:
        return [*self._node_projection(), SCORE_COLUMN]

    def _node_projection(self) -> list[str]:
        return [
            self._node_id_column,
            self._ref_doc_id_column,
            self._text_column,
            *self._metadata_column_names,
            NODE_CONTENT_COLUMN,
            NODE_TYPE_COLUMN,
        ]

    def _where(
        self,
        *,
        node_ids: Sequence[str] | None,
        filters: MetadataFilters | None,
    ) -> str | None:
        parts: list[str] = []
        if node_ids:
            id_list = ", ".join(quote_str(i) for i in node_ids)
            parts.append(f"{self._node_id_column} IN ({id_list})")
        if filters is not None:
            parts.append(compile_filters(filters, self._metadata_column_names))
        if not parts:
            return None
        return " AND ".join(parts) if len(parts) > 1 else parts[0]

    def _append(self, nodes: Sequence[BaseNode], ids: Sequence[str]) -> None:
        rows = [encode_node(n, self._metadata_column_names) for n in nodes]
        arrays: list[pa.Array] = [
            pa.array(ids, type=pa.large_utf8()),
            pa.array([r["ref_doc_id"] for r in rows], type=pa.large_utf8()),
            pa.array([r["text"] for r in rows], type=pa.large_utf8()),
            vector_array([r["embedding"] for r in rows], self._dim),
        ]
        for field in self._metadata_columns:
            arrays.append(pa.array([r.get(field.name) for r in rows], type=field.type))
        arrays.append(pa.array([r[NODE_CONTENT_COLUMN] for r in rows], type=pa.large_utf8()))
        arrays.append(pa.array([r[NODE_TYPE_COLUMN] for r in rows], type=pa.large_utf8()))
        batch = pa.record_batch(arrays, schema=self._table.schema())
        self._table.append(batch)

    def _delete_node_ids(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        id_list = ", ".join(quote_str(i) for i in ids)
        self._table.delete(f"{self._node_id_column} IN ({id_list})")

    def _to_result(self, table: pa.Table, *, distance_metric: bool) -> VectorStoreQueryResult:
        nodes, ids, scores = rows_to_results(
            table,
            node_id_column=self._node_id_column,
            text_column=self._text_column,
        )
        similarities = (
            [_distance_to_similarity(self._metric, s) for s in scores]
            if distance_metric and scores is not None
            else scores
        )
        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=similarities)

    def _build_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(self._node_id_column, pa.large_utf8(), nullable=False),
                pa.field(self._ref_doc_id_column, pa.large_utf8(), nullable=False),
                pa.field(self._text_column, pa.large_utf8(), nullable=False),
                pa.field(
                    self._vector_column,
                    pa.list_(pa.float32(), self._dim),
                    nullable=False,
                ),
                *self._metadata_columns,
                pa.field(NODE_CONTENT_COLUMN, pa.large_utf8(), nullable=False),
                pa.field(NODE_TYPE_COLUMN, pa.large_utf8(), nullable=False),
            ]
        )


def _open_or_create(
    connection: infino.Connection,
    table_name: str,
    schema: pa.Schema,
    n_cent: int,
    metric: Metric,
) -> infino.Table:
    if table_name in connection.list_tables():
        return connection.open_table(table_name)
    indexes = (
        infino.IndexSpec()
        .fts(schema.field(0).name)  # node_id
        .fts(schema.field(1).name)  # ref_doc_id
        .fts(schema.field(2).name)  # text
        .vector(schema.field(3).name, schema.field(3).type.list_size, n_cent, metric)
    )
    return connection.create_table(table_name, schema, indexes)


def _require_embedding(query: VectorStoreQuery) -> list[float]:
    if query.query_embedding is None:
        raise ValueError("query.query_embedding is required for this mode")
    return list(query.query_embedding)


def _require_query_str(query: VectorStoreQuery, mode: str) -> str:
    if not query.query_str:
        raise ValueError(f"query.query_str is required for {mode}")
    return query.query_str


def _distance_to_similarity(metric: Metric, distance: float) -> float:
    """Map a raw distance into a [0, 1] relevance where higher is better."""
    if metric == "cosine":
        return max(0.0, min(1.0, 1.0 - distance))
    if metric in ("l2", "l2sq"):
        return 1.0 / (1.0 + distance)
    # negdot/dot: smaller distance means more similar; negate so larger = better.
    return -distance
