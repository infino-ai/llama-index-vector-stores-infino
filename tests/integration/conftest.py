from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import infino
import pyarrow as pa
import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

from llama_index.vector_stores.infino import InfinoVectorStore

DIM = 16


class FakeEmbedding(BaseEmbedding):
    """Deterministic embedding for tests — vector hashes characters of text."""

    def _signature(self, text: str) -> list[float]:
        seed = sum(ord(c) for c in text) or 1
        return [((seed * (i + 1)) % 97) / 97.0 for i in range(DIM)]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._signature(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._signature(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._signature(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._signature(query)


@pytest.fixture
def tmp_connection(tmp_path: Path) -> Iterator[infino.Connection]:
    yield infino.connect(str(tmp_path))


@pytest.fixture
def embed_model() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def store(tmp_connection: infino.Connection) -> InfinoVectorStore:
    return InfinoVectorStore(
        tmp_connection,
        "docs",
        dim=DIM,
        metadata_columns=[
            pa.field("category", pa.large_utf8(), nullable=True),
            pa.field("year", pa.int64(), nullable=True),
        ],
    )


def make_node(
    text: str,
    *,
    id_: str,
    embedding: list[float],
    ref_doc_id: str | None = None,
    metadata: dict | None = None,
) -> TextNode:
    node = TextNode(text=text, id_=id_, embedding=embedding)
    if metadata:
        node.metadata = metadata
    if ref_doc_id is not None:
        node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref_doc_id)
    return node


def vec(seed: float) -> list[float]:
    return [seed] * DIM


@pytest.fixture
def seeded_nodes() -> list[TextNode]:
    return [
        make_node(
            "vector search powers retrieval",
            id_="n1",
            embedding=vec(0.10),
            ref_doc_id="doc-a",
            metadata={"category": "tech", "year": 2024},
        ),
        make_node(
            "billing dispute resolution",
            id_="n2",
            embedding=vec(0.30),
            ref_doc_id="doc-b",
            metadata={"category": "support", "year": 2023},
        ),
        make_node(
            "BM25 ranks lexical matches",
            id_="n3",
            embedding=vec(0.50),
            ref_doc_id="doc-a",
            metadata={"category": "tech", "year": 2022},
        ),
        make_node(
            "subscription billing question",
            id_="n4",
            embedding=vec(0.70),
            ref_doc_id="doc-b",
            metadata={"category": "support", "year": 2024},
        ),
    ]
