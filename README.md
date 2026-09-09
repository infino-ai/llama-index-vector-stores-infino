# llama-index-vector-stores-infino

[![PyPI](https://img.shields.io/pypi/v/llama-index-vector-stores-infino.svg)](https://pypi.org/project/llama-index-vector-stores-infino/)
[![Downloads](https://img.shields.io/pypi/dm/llama-index-vector-stores-infino.svg)](https://pypi.org/project/llama-index-vector-stores-infino/)
[![Python](https://img.shields.io/pypi/pyversions/llama-index-vector-stores-infino.svg)](https://pypi.org/project/llama-index-vector-stores-infino/)
[![License](https://img.shields.io/pypi/l/llama-index-vector-stores-infino.svg)](https://www.apache.org/licenses/LICENSE-2.0)

**LlamaIndex over [Infino](https://github.com/infino-ai/infino) — vector,
full-text (BM25), hybrid (RRF), and SQL-native retrieval over one copy of your
data on object storage.**

Most "vector database" LlamaIndex integrations expose only the vector slice of
their engine. Infino keeps your data in Apache Parquet on object storage and
runs SQL, BM25, vector, and hybrid (RRF) retrieval over it from a single
in-process engine — no separate search cluster or vector store to keep in
sync. This package surfaces that whole retrieval surface as a
`BasePydanticVectorStore`: all four `VectorStoreQueryMode`s (vector, pure
BM25, RRF hybrid, MMR), structured filters spanning the full LlamaIndex
operator set, a text-pushdown pre-filter, and a SQL escape hatch — each a
first-class path on the same table.

Infino never embeds: you bring a LlamaIndex `BaseEmbedding`, and the
integration uses the embeddings already attached to your nodes.

## Installation

```sh
pip install llama-index-vector-stores-infino
```

Requires Python 3.10+. `infino`, `llama-index-core`, `pyarrow`, and `numpy`
are installed as dependencies. Bring your own embeddings provider separately
(e.g. `pip install llama-index-embeddings-openai`).

## Quickstart

```python
import infino
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.infino import InfinoVectorStore

connection = infino.connect("./data")  # local path or S3 URI (durable storage)
store = InfinoVectorStore(connection, table_name="docs", dim=1536)

storage_context = StorageContext.from_defaults(vector_store=store)
index = VectorStoreIndex.from_documents(
    [Document(text="Infino runs search on object storage.")],
    storage_context=storage_context,
    embed_model=OpenAIEmbedding(),
)

retriever = index.as_retriever(similarity_top_k=4)
```

## Connecting: local, object storage, or hosted cloud

The store construction is identical in every case — only the `infino.connect(...)`
call differs:

```python
import infino

# Local (embedded): Parquet under a directory, no server.
connection = infino.connect("./data")

# Object storage (embedded): Parquet in your bucket, queried in place.
connection = infino.connect(
    "s3://my-bucket/rag", storage_options={"aws_region": "us-east-1"}
)

# Hosted Infino Cloud: sign up at https://platform.infino.ws for an API key.
connection = infino.connect(
    "https://api.platform.infino.ws/<database>", api_key="..."
)

store = InfinoVectorStore(connection, table_name="docs", dim=384)
```

## Core concepts

- **`InfinoVectorStore`** wraps a single Infino table — the text, its
  embedding, the node id, the parent `ref_doc_id`, declared metadata
  columns, and a JSON catch-all. The table is created on first construction
  and opened on subsequent runs.
- **Identity** — `node.node_id` is the durable id; `ref_doc_id` lets
  `delete(ref_doc_id)` drop all chunks of a source document in one call.
  Both are FTS-indexed so the engine prunes superfiles before I/O.
- **Metadata, two tiers** — keys you name in `metadata_columns=` become real
  scalar columns you can filter on; everything else round-trips losslessly
  through a JSON catch-all but isn't filterable. The schema is fixed at
  table creation — adding a new filterable key means recreating the table.
- **Storage** — `add`, `delete`, and `clear` need durable storage (a local
  path or S3 URI). A `memory://` connection cannot delete or upsert and will
  raise; use it only for read-only experiments. `dim` must be in `[16, 4096]`.
- **Scores** — vector distance is *smaller is nearer*; BM25 and RRF are
  *larger is better*. `VectorStoreQueryResult.similarities` always
  represents "higher = better" — distances are normalized for `cosine`,
  `l2`, and `l2sq`.

## Query modes

All four are first-class on the same store, selected by
`VectorStoreQueryMode`:

```python
from llama_index.core.vector_stores.types import VectorStoreQueryMode

index.as_retriever(similarity_top_k=10)                              # DEFAULT (vector)
index.as_retriever(similarity_top_k=10, vector_store_query_mode="text_search")  # BM25
index.as_retriever(similarity_top_k=10, vector_store_query_mode="hybrid")        # RRF
index.as_retriever(similarity_top_k=10, vector_store_query_mode="mmr",
                   vector_store_kwargs={"embed_model": embed_model})             # MMR
```

`HYBRID` fuses BM25 and vector search with reciprocal-rank fusion in one
engine call — no separate reranking round-trip. `MMR` re-embeds candidate
texts (Infino's vector column isn't projectable), so an `embed_model` must
be passed via `vector_store_kwargs`.

## Metadata filtering

Promote the keys you want to filter on to real columns, then pass a
`MetadataFilters`. The full LlamaIndex operator surface is supported:
`EQ` / `NE` / `GT` / `GTE` / `LT` / `LTE`, `IN` / `NIN`, `ANY` / `ALL`,
`CONTAINS`, `TEXT_MATCH` / `TEXT_MATCH_INSENSITIVE`, `IS_EMPTY`, plus
nested `AND` / `OR` / `NOT`.

```python
import pyarrow as pa
from llama_index.core.vector_stores.types import (
    FilterOperator, MetadataFilter, MetadataFilters,
)

store = InfinoVectorStore(
    connection, "papers", dim=1536,
    metadata_columns=[
        pa.field("category", pa.large_utf8(), nullable=False),
        pa.field("year", pa.int64(), nullable=False),
    ],
)

retriever = index.as_retriever(
    similarity_top_k=10,
    filters=MetadataFilters(filters=[
        MetadataFilter(key="category", value="ml"),
        MetadataFilter(key="year", value=2023, operator=FilterOperator.GTE),
    ]),
)
```

## Text-pushdown pre-filter

For a *text* predicate, push it into the kNN instead of post-filtering the
top-k. The engine prunes to rows matching the full-text terms **before**
ranking, so exactly `k` nearest *matching* rows come back — no over-fetch.
Pass it via `vector_store_kwargs`:

```python
retriever = index.as_retriever(
    similarity_top_k=10,
    vector_store_kwargs={"filter_query": "billing", "filter_mode": "and"},
)
```

`filters` (structured, post-rank `WHERE`) and `filter_query` (text,
pre-rank pushdown) are distinct paths and not combinable in one call.

## SQL escape hatch

The escape hatch for what the typed modes don't cover — joins, custom
`WHERE`, or the raw `vector_search` / `hybrid_search` table functions:

```python
qv = ",".join(str(x) for x in embed_model.get_query_embedding("fox"))
store.search_by_sql(f"""
    SELECT node_id, ref_doc_id, text, _node_content, _node_type, score
    FROM hybrid_search('docs', 'text', 'fox', 'embedding', '{qv}', 10)
    ORDER BY score DESC
""")
```

## Async

`async_add`, `aquery`, `adelete`, `adelete_nodes`, `aget_nodes`, and
`aclear` are all implemented as `asyncio.to_thread` wrappers over the
synchronous engine, so the event loop is never blocked.

## API reference

- `InfinoVectorStore(connection, table_name, *, dim, metric="cosine",
  text_column="text", vector_column="embedding", node_id_column="node_id",
  ref_doc_id_column="ref_doc_id", metadata_columns=(), n_cent=64,
  filter_oversample=10)` — also available as
  `InfinoVectorStore.from_params(connection, table_name, *, dim, **kwargs)`.
- `add(nodes) -> list[str]` — upsert by `node_id` (delete-then-append; not
  atomic).
- `delete(ref_doc_id)` — drop all chunks of a source document.
- `delete_nodes(node_ids=None, filters=None)` — combined predicate delete.
- `get_nodes(node_ids=None, filters=None) -> list[BaseNode]` — `node_ids`
  alone prune via `exact_match`.
- `clear()` — drop and recreate the table.
- `count() -> int` — total row count.
- `optimize()` / `gc(grace_secs)` — compact superfiles / reclaim storage.
- `query(VectorStoreQuery, *, filter_query=None, filter_column=None,
  filter_mode=None, embed_model=None) -> VectorStoreQueryResult`
- `search_by_sql(sql) -> VectorStoreQueryResult`
- Async siblings: `async_add`, `aquery`, `adelete`, `adelete_nodes`,
  `aget_nodes`, `aclear`.

`metric` is `"cosine"` (default), `"l2sq"` / `"l2"`, or `"negdot"` / `"dot"`.
See [Infino](https://github.com/infino-ai/infino) for engine internals.

## Development

```sh
make install      # pip install -e ".[test,lint]"
make unit         # unit tests (no engine)
make integration  # integration + end-to-end tests on a real Infino temp dir
make lint type    # ruff + mypy
make build        # build sdist + wheel into dist/
make smoke        # build the wheel, install it in a clean venv, run the smoke test
make clean        # remove build artifacts and caches
```

## License

Apache-2.0.
