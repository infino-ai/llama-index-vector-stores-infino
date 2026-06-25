"""LlamaIndex vector store integration for Infino."""

from importlib.metadata import PackageNotFoundError, version

from llama_index.vector_stores.infino.base import InfinoVectorStore

try:
    __version__ = version("llama-index-vector-stores-infino")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["InfinoVectorStore"]
