"""The :class:`InfinoVectorStore` LlamaIndex vector store.

One Infino table holds the node id, the text, the embedding, the parent
``ref_doc_id``, declared metadata columns, and a JSON catch-all. Vector,
BM25, hybrid (RRF), and SQL retrieval all run over that one table.
"""

from __future__ import annotations

from llama_index.core.vector_stores.types import BasePydanticVectorStore


class InfinoVectorStore(BasePydanticVectorStore):
    """LlamaIndex ``BasePydanticVectorStore`` backed by a single Infino table."""

    stores_text: bool = True
    flat_metadata: bool = False
    is_embedding_query: bool = True
