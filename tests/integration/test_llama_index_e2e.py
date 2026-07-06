from __future__ import annotations

import infino
import pyarrow as pa
from llama_index.core import Document, StorageContext, VectorStoreIndex

from llama_index.vector_stores.infino import InfinoVectorStore
from tests.integration.conftest import DIM, FakeEmbedding


def test_vector_store_index_roundtrip(tmp_path):
    connection = infino.connect(str(tmp_path))
    store = InfinoVectorStore(
        connection,
        "lli_e2e",
        dim=DIM,
        metadata_columns=[pa.field("source", pa.large_utf8(), nullable=True)],
    )
    embed = FakeEmbedding()

    documents = [
        Document(text="infino is a search engine", metadata={"source": "intro"}),
        Document(text="vector retrieval scales", metadata={"source": "intro"}),
        Document(text="bm25 ranks lexical hits", metadata={"source": "ranking"}),
    ]
    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed
    )

    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("vector retrieval")
    assert len(nodes) == 2
    assert all(n.score is not None for n in nodes)
