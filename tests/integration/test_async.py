from __future__ import annotations

from llama_index.core.vector_stores.types import VectorStoreQuery

from tests.integration.conftest import vec


async def test_async_add_and_query(store, seeded_nodes):
    ids = await store.async_add(seeded_nodes)
    assert sorted(ids) == sorted(n.node_id for n in seeded_nodes)
    result = await store.aquery(
        VectorStoreQuery(query_embedding=vec(0.10), similarity_top_k=2)
    )
    assert len(result.nodes) == 2


async def test_async_delete_and_clear(store, seeded_nodes):
    await store.async_add(seeded_nodes)
    await store.adelete("doc-a")
    survivors = await store.aget_nodes(node_ids=[n.node_id for n in seeded_nodes])
    assert sorted(n.node_id for n in survivors) == ["n2", "n4"]
    await store.aclear()
    assert await store.aget_nodes(node_ids=["n2"]) == []
