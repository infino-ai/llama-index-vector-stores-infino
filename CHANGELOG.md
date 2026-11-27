# CHANGELOG — llama-index-vector-stores-infino

## [0.4.0]

- Requires `infino>=0.13,<0.14`; drops the `numpy` dependency.
- Constructor: `InfinoVectorStore(connection, table_name, *, dim, metric,
  metadata_columns)`. The table's column names are fixed; `from_params`,
  `n_cent`, `filter_oversample`, the four `*_column` arguments and the
  `filter_column` query option are gone.
- Filters, `doc_ids` and `node_ids` run inside the engine's search in every
  mode and return the top-k among matching rows in one round trip.
- Every LlamaIndex filter operator has its LlamaIndex meaning: `CONTAINS`,
  `ANY` and `ALL` on list columns, `IS_EMPTY` as null or an empty list, and
  `ValueError` for an operator on the wrong kind of column or a non-finite value.
- `alpha`, `sparse_top_k`, `output_fields` and a foreign `embedding_field`
  raise, as does a repeated `node_id` in one `add`.
- An empty `node_ids` in `get_nodes` / `delete_nodes` matches nothing.
- Similarities are cosine similarity in `[-1, 1]`, `1 / (1 + distance)` for
  L2, and the dot product for dot metrics.
- MMR re-embeds candidates with the text LlamaIndex embedded at ingest.
- Table and column names are quoted in SQL.
- `search_by_sql` names the columns its result is missing.

## [0.3.0]

- `MetadataFilters` now apply in `TEXT_SEARCH`, `HYBRID` and `MMR` modes; they
  were silently ignored, returning unfiltered results.
- `filter_query` is honoured in `MMR`, and raises in `TEXT_SEARCH` and `HYBRID`
  instead of being silently dropped.
- Opening an existing table with a different `dim` or `metadata_columns` raises
  instead of silently keeping the stored schema.
- A selective `MetadataFilters` no longer under-returns: the candidate pool
  widens until the top-k is filled or the table is exhausted. `filter_oversample`
  now sets the starting pool, not a hard ceiling.

## [0.2.0]

- Requires `infino>=0.7,<0.8`.
- Connect documentation for local, object-storage, and Infino Cloud targets.
- Release-triggered PyPI publish.

## [0.1.0]

- Initial release: `InfinoVectorStore`, a `BasePydanticVectorStore` backed by
  a single Infino table.
- Query modes: `DEFAULT` (vector), `TEXT_SEARCH` (BM25), `HYBRID` (RRF), `MMR`.
- Structured `MetadataFilters` compiled to SQL `WHERE`, a `filter_query`
  text-pushdown pre-filter, and a `search_by_sql` escape hatch.
- CRUD: `add` (upsert by `node_id`), `delete` (by `ref_doc_id`),
  `delete_nodes`, `get_nodes`, `clear`, plus `count` / `optimize` / `gc`.
- Async siblings for all mutating and query methods.
- Requires `infino>=0.1.5`.
