# CHANGELOG — llama-index-vector-stores-infino

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
