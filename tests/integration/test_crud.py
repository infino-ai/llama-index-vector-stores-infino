from __future__ import annotations

import pytest
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from tests.integration.conftest import make_node, vec


def test_add_returns_node_ids_and_persists(store, seeded_nodes):
    ids = store.add(seeded_nodes)
    assert ids == [n.node_id for n in seeded_nodes]
    fetched = store.get_nodes(node_ids=ids)
    assert sorted(n.node_id for n in fetched) == sorted(ids)


def test_add_upserts_on_existing_node_id(store, seeded_nodes):
    store.add([seeded_nodes[0]])
    replacement = make_node(
        "rewritten text", id_=seeded_nodes[0].node_id, embedding=vec(0.99)
    )
    store.add([replacement])
    fetched = store.get_nodes(node_ids=[replacement.node_id])
    assert len(fetched) == 1
    assert fetched[0].get_content() == "rewritten text"


def test_delete_by_ref_doc_id_drops_all_children(store, seeded_nodes):
    store.add(seeded_nodes)
    store.delete("doc-a")
    survivors = store.get_nodes(node_ids=[n.node_id for n in seeded_nodes])
    assert sorted(n.node_id for n in survivors) == ["n2", "n4"]


def test_delete_nodes_by_id_and_filter(store, seeded_nodes):
    store.add(seeded_nodes)
    store.delete_nodes(
        node_ids=["n1", "n2", "n3"],
        filters=MetadataFilters(filters=[MetadataFilter(key="category", value="tech")]),
    )
    survivors = store.get_nodes(node_ids=[n.node_id for n in seeded_nodes])
    assert sorted(n.node_id for n in survivors) == ["n2", "n4"]


def test_get_nodes_by_filter_only(store, seeded_nodes):
    store.add(seeded_nodes)
    found = store.get_nodes(
        filters=MetadataFilters(
            filters=[MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)]
        )
    )
    assert sorted(n.node_id for n in found) == ["n1", "n4"]


def test_get_nodes_requires_an_argument(store):
    with pytest.raises(ValueError):
        store.get_nodes()


def test_clear_empties_and_remains_reusable(store, seeded_nodes):
    store.add(seeded_nodes)
    store.clear()
    assert store.get_nodes(node_ids=["n1"]) == []
    store.add([seeded_nodes[0]])
    fetched = store.get_nodes(node_ids=["n1"])
    assert len(fetched) == 1
