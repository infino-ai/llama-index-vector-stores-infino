"""Built-wheel smoke: import the package, round-trip one node on a real engine."""

from __future__ import annotations

import infino
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery

from llama_index.vector_stores.infino import InfinoVectorStore

DIM = 16


def test_round_trip(tmp_path) -> None:
    connection = infino.connect(str(tmp_path / "db"))
    store = InfinoVectorStore.from_params(connection, "smoke", dim=DIM)
    node = TextNode(text="hello infino", id_="n", embedding=[0.1] * DIM)
    store.add([node])
    result = store.query(VectorStoreQuery(query_embedding=[0.1] * DIM, similarity_top_k=1))
    assert [n.node_id for n in result.nodes] == ["n"]
    assert result.nodes[0].get_content() == "hello infino"
