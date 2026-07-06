from __future__ import annotations

import pytest
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from tests.integration.conftest import vec


def _query(**kwargs) -> VectorStoreQuery:
    kwargs.setdefault("similarity_top_k", 3)
    return VectorStoreQuery(**kwargs)


def test_default_mode_returns_vector_neighbours(store, seeded_nodes):
    store.add(seeded_nodes)
    result = store.query(_query(query_embedding=vec(0.10)))
    assert len(result.nodes) == 3
    assert result.ids[0] == "n1"
    assert all(s is not None for s in result.similarities)


def test_text_search_mode_runs_bm25(store, seeded_nodes):
    store.add(seeded_nodes)
    result = store.query(
        _query(query_str="billing", mode=VectorStoreQueryMode.TEXT_SEARCH)
    )
    assert {n.node_id for n in result.nodes} == {"n2", "n4"}


def test_hybrid_mode_returns_rrf_results(store, seeded_nodes):
    store.add(seeded_nodes)
    result = store.query(
        _query(
            query_str="billing",
            query_embedding=vec(0.30),
            mode=VectorStoreQueryMode.HYBRID,
        )
    )
    assert len(result.nodes) >= 1
    assert result.similarities is not None
    # RRF scores decrease monotonically by rank.
    sims = result.similarities
    assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1))


def test_mmr_requires_embed_model(store, seeded_nodes):
    store.add(seeded_nodes)
    with pytest.raises(ValueError, match="embed_model"):
        store.query(_query(query_embedding=vec(0.10), mode=VectorStoreQueryMode.MMR))


def test_mmr_diversifies_results(store, seeded_nodes, embed_model):
    store.add(seeded_nodes)
    result = store.query(
        _query(
            query_embedding=embed_model._signature("vector search"),
            mode=VectorStoreQueryMode.MMR,
            mmr_threshold=0.5,
        ),
        embed_model=embed_model,
    )
    assert len(result.nodes) == 3
    assert len({n.node_id for n in result.nodes}) == 3


def test_filter_query_pushdown_prunes_before_ranking(store, seeded_nodes):
    store.add(seeded_nodes)
    result = store.query(_query(query_embedding=vec(0.10)), filter_query="billing")
    assert {n.node_id for n in result.nodes} <= {"n2", "n4"}


def test_structured_filter_post_rank(store, seeded_nodes):
    store.add(seeded_nodes)
    result = store.query(
        _query(
            query_embedding=vec(0.10),
            filters=MetadataFilters(
                filters=[MetadataFilter(key="category", value="support")]
            ),
        )
    )
    assert {n.node_id for n in result.nodes} <= {"n2", "n4"}


def test_structured_and_pushdown_are_mutually_exclusive(store, seeded_nodes):
    store.add(seeded_nodes)
    with pytest.raises(ValueError, match="not both"):
        store.query(
            _query(
                query_embedding=vec(0.10),
                filters=MetadataFilters(
                    filters=[MetadataFilter(key="category", value="tech")]
                ),
            ),
            filter_query="billing",
        )


def test_search_by_sql_returns_nodes(store, seeded_nodes):
    store.add(seeded_nodes)
    columns = "node_id, ref_doc_id, text, category, year, _node_content, _node_type"
    result = store.search_by_sql(
        f"SELECT {columns} FROM docs WHERE category = 'tech'"
    )
    assert {n.node_id for n in result.nodes} == {"n1", "n3"}


def test_unsupported_mode_raises(store, seeded_nodes):
    store.add(seeded_nodes)
    with pytest.raises(NotImplementedError):
        store.query(
            _query(query_embedding=vec(0.10), mode=VectorStoreQueryMode.SPARSE)
        )


def test_empty_table_returns_empty_results(store):
    result = store.query(_query(query_embedding=vec(0.10)))
    assert result.nodes == []
    assert result.ids == []
