# CHANGELOG — llama-index-vector-stores-infino

## [0.1.0]

- Initial release: `InfinoVectorStore`, a `BasePydanticVectorStore` backed by
  a single Infino table.
- Query modes: `DEFAULT` (vector), `TEXT_SEARCH` (BM25), `HYBRID` (RRF), `MMR`.
- Structured `MetadataFilters` compiled to SQL `WHERE`, a `filter_query`
  text-pushdown pre-filter, and a `search_by_sql` escape hatch.
- CRUD: `add` (upsert by `node_id`), `delete` (by `ref_doc_id`),
  `delete_nodes`, `get_nodes`, `clear`, plus `count` / `optimize` / `gc`.
- Async siblings for all mutating and query methods.
- Requires `infino>=0.1.4`.
