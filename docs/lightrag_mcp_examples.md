# LightRAG MCP Tool Examples

These are representative request and response shapes for the tools registered by
`create_lightrag_mcp`. IDs, paths, metrics, and graph payloads are dynamic. Normal
operations return the common wrapper `{"status":"success","response":...}`;
uncaught failures return `{"status":"error","error":"..."}`.

## 1. `query_document`

Request:

```json
{"query":"Project: LightRAG\nProject path: /repo/LightRAG\nRepository: https://github.com/urm8/LightRAG\nQuestion: How is MCP mounted?","mode":"mix","top_k":36,"chunk_top_k":24,"only_need_context":true}
```

Response (content-first; one retrieval call):

```json
{"status":"success","response":{"matches":[{"content":"LightRAG serves FastMCP at /mcp from the live in-process runtime.","file_path":"mcp-memory/mount-note.txt","reference_id":"1","chunk_id":"chunk-abc"}],"response":"Retrieved 1 matching context chunk(s).","references":[{"reference_id":"1","file_path":"mcp-memory/mount-note.txt"}],"history_turns":10}}
```

With `only_need_context=false`, `matches` is unchanged and `response` contains the
generated answer. With `only_need_context=true`, the raw graph prompt is omitted.
`top_k` controls graph candidates while `chunk_top_k` independently controls text
chunks. `conversation_history` affects retrieval when
`use_history_for_retrieval=true`.

### Specialized query tools

- `query_text`: direct chunks for symbols, paths, errors, quotes, and config keys.
  Example: `{"query":"RecoveryAnchorMissingError","chunk_top_k":20}`.
- `query_graph`: entities and relationships. Example:
  `{"query":"How do purge anchors protect graph attribution?","scope":"hybrid","top_k":40,"chunk_top_k":20,"hl_keywords":["graph attribution"],"ll_keywords":["kg_purge","full_entities","full_relations"]}`.
- `query_mixed`: default coding/chat retrieval. Example:
  `{"query":"Why does this retry remain FAILED?","top_k":36,"chunk_top_k":24,"only_need_context":true}`.
- `query_tagged`: evidence only from documents containing every tag. Example:
  `{"query":"Find the verified MCP deployment procedure","required_tags":["skill","agentic-development"],"mode":"mix"}`.

## 2. `insert_document`

Request:

```json
{"text":"Project: LightRAG\nProject entity: Project|LightRAG\nProject path: /repo/LightRAG\nRepository: https://github.com/urm8/LightRAG\nFinding: MCP returns source excerpts.","tags":["agentic development","workflow"]}
```

Response:

```json
{"status":"success","response":{"status":"success","message":"Text accepted for background processing","track_id":"mcp-<uuid>","tags":["agentic-development","workflow"]}}
```

## 3. `get_document_content`

Request the complete body only after `query_document` returns a relevant
`document_id`:

```json
{"document_id":"doc-abc"}
```

```json
{"status":"success","response":{"document_id":"doc-abc","file_path":"mcp-memory/mount-note.txt","content":"Project: LightRAG\n...","metadata":{"tags":["agentic-development"]}}}
```

## 4. `save_skill`

```json
{"name":"verify-mcp-deployment","description":"Verify a deployed MCP route end to end.","applicability":"After changing the MCP server or deployment chart.","procedure":"Check health, initialize MCP, list tools, and call a read-only tool.","verification":"The MCP initialize response contains serverInfo.","failure_pattern":"The core API is healthy while /mcp returns 404.","ruled_out":["A health-only probe does not prove the MCP sub-application is mounted."],"references":["https://gofastmcp.com/"],"project_name":"LightRAG","project_path":"/repo/LightRAG","repository":"https://github.com/urm8/LightRAG","scope":"project"}
```

The response contains the normal background-ingestion `track_id` plus
`skill_name`, `scope`, and normalized skill tags.

## 5. `search_skills`

```json
{"query":"How should an MCP deployment be verified?","project_name":"LightRAG","project_path":"/repo/LightRAG","repository":"https://github.com/urm8/LightRAG","limit":3}
```

```json
{"status":"success","response":{"skills":[{"content":"Verification: The MCP initialize response contains serverInfo.","document_id":"doc-abc","tags":["skill","agentic-development"],"is_excerpt":true}],"response":"Found 1 related reusable skill(s).","references":[{"reference_id":"1","file_path":"mcp-memory/skill.txt"}]}}
```

## 6. `upload_document`

`file_path` is resolved on the LightRAG server, not on the remote MCP client's
filesystem. The file is copied into the configured input directory before enqueue.

Request:

```json
{"file_path":"/data/imports/design.md"}
```

Response:

```json
{"status":"success","response":{"status":"success","track_id":"mcp-file-<uuid>","file_path":"/app/inputs/design.md"}}
```

## 7. `insert_file`

`file_path` must already exist on the LightRAG server.

Request:

```json
{"file_path":"/app/inputs/runbook.md"}
```

Response:

```json
{"status":"success","response":{"status":"success","track_id":"mcp-file-<uuid>","file_path":"/app/inputs/runbook.md"}}
```

## 8. `insert_batch`

Request:

```json
{"directory_path":"/app/inputs/project","recursive":true,"depth":2,"include_only":["\\.(md|txt)$"],"ignore_files":[],"ignore_directories":["^node_modules$"]}
```

Response:

```json
{"status":"success","response":{"status":"success","track_id":"mcp-batch-<uuid>","accepted":12,"failed":0,"failures":[]}}
```

The inner status becomes `partial_success` when individual files fail.

## 9. `scan_for_new_documents`

Request:

```json
{}
```

Response:

```json
{"status":"success","response":{"status":"scanning_started","track_id":"mcp-scan-<uuid>","accepted":3}}
```

## 10. `get_documents`

Request:

```json
{"tags":["agentic-development","workflow"]}
```

Response:

```json
{"status":"success","response":{"statuses":{"pending":[],"processing":[],"processed":[{"id":"doc-abc","content_summary":"Project: LightRAG...","content_length":420,"status":"processed","file_path":"mcp-memory/note.txt","metadata":{"tags":["agentic-development","workflow"]}}],"failed":[]}}}
```

Every requested tag must be present. The exact status keys follow `DocStatus`.

## 11. `get_pipeline_status`

Request:

```json
{}
```

Response:

```json
{"status":"success","response":{"busy":false,"destructive_busy":false,"scanning":false,"scanning_exclusive":false,"pending_enqueues":0,"job_name":null,"history_messages":[]}}
```

## 12. `get_graph_labels`

Request:

```json
{}
```

Response:

```json
{"status":"success","response":["LightRAG","FastMCP","PostgreSQL","WebUI"]}
```

## 13. `check_lightrag_health`

Use only after an MCP timeout, connection failure, or invalid response.

Request:

```json
{}
```

Response:

```json
{"status":"success","response":{"status":"healthy","workspace":"default","pipeline_busy":false,"configuration":{"kv_storage":"PGKVStorage","doc_status_storage":"PGDocStatusStorage","graph_storage":"PGGraphStorage","vector_storage":"PGVectorStorage","llm_model":"deepseek-v4-flash","embedding_model":"text-embedding-3-small"}}}
```

## 14. `check_memory_pressure`

Request:

```json
{"top_process_limit":3}
```

Response:

```json
{"status":"success","response":{"platform":"Linux-...","memory_pressure":{"level":"normal","available_ratio":0.61,"total_mb":16384,"available_mb":9994,"used_mb":6390,"percent":39.0},"swap":{"total_mb":0,"used_mb":0,"free_mb":0,"percent":0},"current_process":{"pid":42,"rss_mb":512},"top_processes":[{"pid":42,"name":"python","rss_mb":512,"cmdline":"lightrag-server"}]}}
```

## 15. `merge_entities`

Request:

```json
{"source_entities":["Light Rag","LightRAG Framework"],"target_entity":"LightRAG","merge_strategy":{"description":"join_unique","entity_type":"keep_first"}}
```

Response:

```json
{"status":"success","response":{"entity_name":"LightRAG","entity_type":"Project","description":"Graph-based RAG framework","source_id":"chunk-a<SEP>chunk-b"}}
```

## 16. `create_entities`

Request:

```json
{"entities":[{"entity_name":"Query Document Tool","entity_type":"Tool","description":"Returns matching LightRAG excerpts.","source_id":"chunk-abc"}]}
```

Response:

```json
{"status":"success","response":{"total":1,"successful":1,"failed":0,"results":[{"entity_name":"Query Document Tool","status":"success","result":{"entity_name":"Query Document Tool","entity_type":"Tool","description":"Returns matching LightRAG excerpts.","source_id":"chunk-abc"}}]}}
```

## 17. `delete_by_entities`

Request:

```json
{"entity_names":["Obsolete Entity"]}
```

Response:

```json
{"status":"success","response":{"total":1,"successful":1,"failed":0,"results":[{"entity_name":"Obsolete Entity","status":"success","result":{"status":"success","doc_id":"Obsolete Entity","message":"Entity deleted"}}]}}
```

## 18. `delete_by_doc_ids`

Request:

```json
{"doc_ids":["doc-abc"]}
```

Response:

```json
{"status":"success","response":{"total":1,"successful":1,"failed":0,"results":[{"doc_id":"doc-abc","status":"success","result":{"status":"success","doc_id":"doc-abc","message":"Document deleted"}}]}}
```

## 19. `edit_entities`

Request:

```json
{"entities":[{"entity_name":"Query Document Tool","entity_type":"Tool","description":"Returns content-first excerpts with provenance.","source_id":"chunk-abc"}]}
```

Response:

```json
{"status":"success","response":{"total":1,"successful":1,"failed":0,"results":[{"entity_name":"Query Document Tool","status":"success","result":{"entity_name":"Query Document Tool","entity_type":"Tool","description":"Returns content-first excerpts with provenance.","source_id":"chunk-abc"}}]}}
```

## 20. `create_relations`

Request:

```json
{"relations":[{"source":"LightRAG","target":"Query Document Tool","description":"LightRAG exposes the tool through FastMCP.","keywords":"IMPLEMENTS","source_id":"chunk-abc","weight":1.0}]}
```

Response:

```json
{"status":"success","response":{"total":1,"successful":1,"failed":0,"results":[{"relation":"LightRAG -> Query Document Tool","status":"success","result":{"src_id":"LightRAG","tgt_id":"Query Document Tool","description":"LightRAG exposes the tool through FastMCP.","keywords":"IMPLEMENTS","source_id":"chunk-abc","weight":1.0}}]}}
```

## 21. `edit_relations`

Request:

```json
{"relations":[{"source":"LightRAG","target":"Query Document Tool","description":"LightRAG exposes content-first MCP retrieval.","keywords":"IMPLEMENTS,IMPROVES","relation_type":"IMPLEMENTS","source_id":"chunk-abc","weight":1.0}]}
```

Response:

```json
{"status":"success","response":{"total":1,"successful":1,"failed":0,"results":[{"relation":"LightRAG -> Query Document Tool","status":"success","result":{"src_id":"LightRAG","tgt_id":"Query Document Tool","description":"LightRAG exposes content-first MCP retrieval.","keywords":"IMPLEMENTS,IMPROVES","relation_type":"IMPLEMENTS","source_id":"chunk-abc","weight":1.0}}]}}
```

## Per-item and operation errors

Bulk graph tools keep the outer operation successful while reporting individual
failures in `results`:

```json
{"status":"success","response":{"total":1,"successful":0,"failed":1,"results":[{"entity_name":"unknown","status":"error","error":"Missing required fields"}]}}
```

An operation-level exception uses:

```json
{"status":"error","error":"LightRAG runtime is not initialized"}
```
