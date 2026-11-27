from __future__ import annotations

import infino
import pyarrow as pa
import pytest
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from llama_index.vector_stores.infino import InfinoVectorStore
from tests.integration.conftest import DIM, make_node, vec


def test_add_returns_node_ids_and_persists(store, seeded_nodes):
    ids = store.add(seeded_nodes)
    assert ids == [n.node_id for n in seeded_nodes]
    fetched = store.get_nodes(node_ids=ids)
    assert sorted(n.node_id for n in fetched) == sorted(ids)


def test_add_upserts_on_existing_node_id(store, seeded_nodes):
    store.add([seeded_nodes[0]])
    replacement = make_node("rewritten text", id_=seeded_nodes[0].node_id, embedding=vec(0.99))
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


def test_get_nodes_by_ids_ignores_misses(store, seeded_nodes):
    store.add(seeded_nodes)
    found = store.get_nodes(node_ids=["n1", "missing", "n1"])
    assert [n.node_id for n in found] == ["n1"]


def test_count_reflects_row_total(store, seeded_nodes):
    assert store.count() == 0
    store.add(seeded_nodes)
    assert store.count() == len(seeded_nodes)
    store.clear()
    assert store.count() == 0


def test_optimize_and_gc_are_callable(store, seeded_nodes):
    store.add(seeded_nodes)
    store.optimize()
    store.gc(grace_secs=0.0)
    assert store.count() == len(seeded_nodes)


def test_table_name_needing_quotes_works(tmp_connection):
    store = InfinoVectorStore(tmp_connection, "my-docs", dim=DIM)
    store.add([make_node("x", id_="p1", embedding=vec(0.2))])
    assert [n.node_id for n in store.get_nodes(node_ids=["p1"])] == ["p1"]
    assert store.count() == 1


def test_add_rejects_repeated_node_id(store):
    node = make_node("x", id_="dup", embedding=vec(0.2))
    with pytest.raises(ValueError, match="more than once"):
        store.add([node, node])


def test_empty_node_ids_match_nothing(store, seeded_nodes):
    store.add(seeded_nodes)
    assert store.get_nodes(node_ids=[]) == []
    store.delete_nodes(
        node_ids=[], filters=MetadataFilters(filters=[MetadataFilter(key="category", value="tech")])
    )
    assert store.count() == len(seeded_nodes)


@pytest.mark.parametrize("name", ["text", "score"])
def test_metadata_column_cannot_shadow_a_store_column(tmp_connection, name):
    with pytest.raises(ValueError, match="distinct"):
        InfinoVectorStore(
            tmp_connection, "clash", dim=DIM, metadata_columns=[pa.field(name, pa.large_utf8())]
        )


def test_list_metadata_column_filters(tmp_connection):
    store = InfinoVectorStore(
        tmp_connection,
        "tagged",
        dim=DIM,
        metadata_columns=[pa.field("tags", pa.list_(pa.large_utf8()), nullable=True)],
    )
    store.add(
        [
            make_node("a", id_="t1", embedding=vec(0.1), metadata={"tags": ["x", "y"]}),
            make_node("b", id_="t2", embedding=vec(0.2), metadata={"tags": ["y"]}),
            make_node("c", id_="t3", embedding=vec(0.3), metadata={"tags": []}),
        ]
    )

    def ids(op, value):
        f = MetadataFilters(filters=[MetadataFilter(key="tags", value=value, operator=op)])
        return sorted(n.node_id for n in store.get_nodes(filters=f))

    assert ids(FilterOperator.CONTAINS, "x") == ["t1"]
    assert ids(FilterOperator.ANY, ["x", "y"]) == ["t1", "t2"]
    assert ids(FilterOperator.ALL, ["x", "y"]) == ["t1"]
    assert ids(FilterOperator.IS_EMPTY, None) == ["t3"]


def test_is_empty_on_numeric_column(store, seeded_nodes):
    store.add([*seeded_nodes, make_node("undated", id_="n5", embedding=vec(0.9))])
    f = MetadataFilters(
        filters=[MetadataFilter(key="year", value=None, operator=FilterOperator.IS_EMPTY)]
    )
    assert [n.node_id for n in store.get_nodes(filters=f)] == ["n5"]


def test_memory_connection_cannot_mutate():
    store = InfinoVectorStore(infino.connect("memory://"), "mem", dim=DIM)
    node = make_node("only in memory", id_="m1", embedding=vec(0.1))
    with pytest.raises(RuntimeError, match="no storage attached"):
        store.add([node])


def test_reopen_with_matching_schema_succeeds(tmp_connection):
    columns = [pa.field("category", pa.large_utf8(), nullable=True)]
    InfinoVectorStore(tmp_connection, "reopen", dim=DIM, metadata_columns=columns)
    store = InfinoVectorStore(tmp_connection, "reopen", dim=DIM, metadata_columns=columns)
    assert store.count() == 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"dim": DIM * 2}, "embedding"),
        ({"dim": DIM, "metadata_columns": [pa.field("extra", pa.int64())]}, "extra"),
    ],
)
def test_reopen_with_mismatched_schema_raises(tmp_connection, kwargs, expected):
    InfinoVectorStore(tmp_connection, "fixed", dim=DIM)
    with pytest.raises(ValueError, match=expected) as excinfo:
        InfinoVectorStore(tmp_connection, "fixed", **kwargs)
    assert "create a new table" in str(excinfo.value)
