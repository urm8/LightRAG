---
name: lightrag-agentic-development
description: "Use when doing agentic software work with the local LightRAG MCP endpoint: analytics, management, QA, coding, design, planning, debugging, deployment, incident investigation, or durable project-memory updates. Requires every LightRAG query/write to cite the active project and save it as a Project entity anchor."
---

# LightRAG Agentic Development

Use the integrated MCP endpoint as the project memory layer:

```text
http://127.0.0.1:9621/mcp
```

## Core Workflow

1. Identify the current project before every LightRAG query or write. Use repo name, absolute path, and git remote when available.
2. Start non-trivial work directly with `query_document`; do not preflight with `check_lightrag_health`.
3. Include the current project in every `query_document` request, for example: `Project: LightRAG (/Users/max/projs/github.com/HKUDS/LightRAG). Question: ...`.
4. Use `query_document` with `only_need_context=true` for small relevant excerpts. Call `get_document_content(document_id)` only when an excerpt identifies a relevant source and the full document is needed.
5. Use normal repository tools for source edits, tests, grep, logs, and runtime checks.
6. Persist durable findings with `insert_document` after meaningful debugging, QA, planning, design, deploy, or coding work.
7. Check `get_pipeline_status` before assuming newly inserted memory is searchable.

## Tool Selection

Use only tools exposed by the active MCP session. Current core tools:

- `check_lightrag_health`: use only after an MCP timeout, connection failure, or invalid/unavailable response.
- `query_document`: retrieve bounded source excerpts and document IDs. Use `only_need_context=true` before edits.
- `get_document_content`: deliberately expand one relevant result by its `document_id`; never use it for speculative bulk reads.
- `insert_document`: save durable notes, decisions, incidents, validation results.
- `insert_file`: index important repo artifacts or logs.
- `scan_for_new_documents`: ingest newly added input docs.
- `get_pipeline_status`: check indexing state and memory search readiness.

Do not invent unavailable graph mutation tools. If graph curation tools are not exposed, save a project-anchored correction note with `insert_document`.

## Query Defaults

Use `mode="mix"` for general project recall. Use `top_k=10-40` depending on scope.

For code changes, prefer:

```json
{
  "mode": "mix",
  "only_need_context": true,
  "top_k": 10
}
```

Write the query text with a project prefix:

```text
Project: LightRAG
Project path: /Users/max/projs/github.com/HKUDS/LightRAG
Question: What prior decisions affect this change?
```

## Durable Memory

Insert memory only for facts worth reusing:

- architecture decisions
- non-obvious bugs and root causes
- validation results
- runtime quirks
- deployment steps
- prompt/model/settings changes
- graph cleanup decisions

Every `insert_document` payload must cite the current project and make it extractable as a `Project` entity. Put the project header first:

```text
Project: LightRAG
Project entity: Project|LightRAG
Project path: /Users/max/projs/github.com/HKUDS/LightRAG
Repository: HKUDS/LightRAG
Date: YYYY-MM-DD
Scope: debugging|coding|planning|qa|design|deploy|management|analytics

Finding: ...
Files: ...
Commands: ...
Outcome: ...
Relations:
- LightRAG IMPLEMENTS ...
- LightRAG USES ...
- LightRAG FAILS_WITH ...
```

Keep inserted notes short, dated, and specific. Include file paths, commands, ports, model names, and exact outcomes when relevant.

## MCP Failure Fallback

If MCP calls time out or service is busy:

1. Check HTTP health or pipeline status when available.
2. Continue with repo-local evidence if the task cannot wait.
3. Queue durable memory through the HTTP document insert endpoint or retry `insert_document` later.
4. Mention in final output that MCP memory write/search was blocked by timeout.

## Project Entity Anchoring

Agentic workflow memory should connect back to the active project. When saving or curating graph data:

- Ensure the active project is represented as a `Project` entity.
- Prefer `PART_OF`, `USES`, `IMPLEMENTS`, `DEPENDS_ON`, `FAILS_WITH`, `IMPROVES`, and `RELATED_TO` relations from the project to workflows, tools, services, issues, files, and decisions.
- Do not save orphan findings without the project name/path unless the user explicitly asks for cross-project notes.
- For multi-repo work, name each project explicitly and connect shared facts to the correct project.

## Graph Maintenance

Use graph mutation tools only when you are correcting known bad graph data or curating explicit project facts. Prefer `merge_entities` before delete operations when duplicate names refer to the same concept.

For relation curation, prefer canonical verbs already used in this repo: `USES`, `DEPENDS_ON`, `IMPLEMENTS`, `CALLS`, `CONNECTS_TO`, `RUNS_ON`, `STORES_IN`, `DEPLOYS_TO`, `AUTHORED_BY`, `PART_OF`, `RELATED_TO`, `CAUSES`, `MEASURES`, `TRACKS`, `FAILS_WITH`, `IMPROVES`, `REPLACES`, `COMPETES_WITH`.
