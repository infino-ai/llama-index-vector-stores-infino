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

Requires Python 3.10+. `infino`, `llama-index-core`, and `pyarrow` are
installed as dependencies. Bring your own embeddings provider separately
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
  Both are full-text indexed, so lookups and deletes by id skip files that
  cannot hold the id.
- **Metadata, two tiers** — keys you name in `metadata_columns=` become real
  columns (scalar or list) you can filter on; everything else round-trips
  losslessly in the serialized node but isn't filterable. The schema is fixed
  at table creation — adding a new filterable key means recreating the table.
- **Storage** — the store needs durable storage: a local path, object
  storage, or Infino Cloud. A `memory://` connection cannot delete, so `add`
  (an upsert) raises on it. `dim` must be in `[16, 4096]`.
- **Scores** — `VectorStoreQueryResult.similarities` is always "higher is
  better": cosine similarity (`1 - distance`, in `[-1, 1]`) for `cosine`,
  `1 / (1 + distance)` for `l2` / `l2sq`, the dot product for `dot` /
  `negdot`, and the raw BM25 or RRF score for `TEXT_SEARCH` and `HYBRID`.

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
`MetadataFilters`. Every LlamaIndex operator is supported, with its LlamaIndex
meaning:

- scalar columns: `EQ` / `NE` / `GT` / `GTE` / `LT` / `LTE`, `IN` / `NIN`,
  `TEXT_MATCH` (substring, `LIKE`) / `TEXT_MATCH_INSENSITIVE` (`ILIKE`);
- list columns (e.g. `pa.list_(pa.large_utf8())`): `CONTAINS` (holds the
  value), `ANY` (holds any of the values), `ALL` (holds all of them);
- any column: `IS_EMPTY` (null, or an empty list);
- nested `AND` / `OR` / `NOT` groups.

An operator used on the wrong kind of column raises `ValueError`.

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

Filters run inside the search in every mode: the result is the top-k among
matching rows, however deep they rank, or every match when fewer than k exist.
A query's `doc_ids` and `node_ids` restrict the search the same way.

## Text-pushdown pre-filter

For a *text* predicate, the engine can prune to rows matching full-text terms
before the vector search ranks them.
Pass it via `vector_store_kwargs`:

```python
retriever = index.as_retriever(
    similarity_top_k=10,
    vector_store_kwargs={"filter_query": "billing", "filter_mode": "and"},
)
```

`filter_query` cannot be combined with `filters`, `doc_ids` or `node_ids`
in one call, and applies to vector search (`DEFAULT` and `MMR`) only.

## Not supported

These raise rather than being silently ignored:

- `VectorStoreQueryMode`s other than `DEFAULT`, `TEXT_SEARCH`, `HYBRID` and
  `MMR` (`NotImplementedError`).
- `alpha`: `HYBRID` fuses with reciprocal-rank fusion, which has no weight.
- `sparse_top_k`: `HYBRID` takes a single top-k (`hybrid_top_k`, else
  `similarity_top_k`).
- `output_fields`: results always carry whole nodes.
- `embedding_field` other than the store's vector column.

Two engine limits shape behaviour:

- Stored vectors cannot be read back, so `MMR` re-embeds its candidates and
  needs an `embed_model`.
- `add` replaces existing nodes by deleting then appending, which is not
  atomic: a failed append leaves the replaced nodes deleted.

## SQL escape hatch

The escape hatch for what the typed modes don't cover — joins, custom
`WHERE`, or the raw `vector_search` / `hybrid_search` table functions. The
result must include `text`, `_node_content` and `_node_type`; `score`, when
present, becomes the similarity as is:

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
  metadata_columns=())`. The table's own columns are `node_id`, `ref_doc_id`,
  `text`, `embedding`, `_node_content` and `_node_type`; metadata columns take
  other names, and not `score`.
- `add(nodes) -> list[str]` — upsert by `node_id` (delete-then-append; not
  atomic). A repeated `node_id` in one call raises.
- `delete(ref_doc_id)` — drop all chunks of a source document.
- `delete_nodes(node_ids=None, filters=None)` — combined predicate delete.
- `get_nodes(node_ids=None, filters=None) -> list[BaseNode]` — both given
  means both must match; an empty `node_ids` matches nothing.
- `clear()` — drop and recreate the table.
- `count() -> int` — total row count.
- `optimize()` / `gc(grace_secs)` — compact superfiles / reclaim storage.
- `query(VectorStoreQuery, *, filter_query=None, filter_mode=None,
  embed_model=None, mmr_fetch_k=20) -> VectorStoreQueryResult`
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
