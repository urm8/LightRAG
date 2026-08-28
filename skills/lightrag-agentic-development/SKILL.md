---
name: lightrag-agentic-development
description: "Use LightRAG MCP during software work to retrieve source-grounded project knowledge, route queries across text, graph, mixed, and tagged search, and maintain verified reusable memory. Every project query and write identifies the active Project entity."
---

# LightRAG Agentic Development

Use the configured `lightrag` MCP server as shared project reference memory.

## Workflow

1. Put the project name, absolute path, and repository remote in every query and write.
2. For a recurring workflow or hard problem, call `search_skills` before deriving a new procedure. Reuse a relevant result; fetch its full body with `get_document_content(document_id)` only when needed.
3. Before broad source reads, select the narrowest query tool using the routing rules below; do not preflight health.
4. Verify retrieved claims against the repository, which remains authoritative.
5. Use `insert_document` only when verified current project knowledge is missing or stale. Use `get_pipeline_status` only before relying on a new insertion.
6. Use `save_skill` only for a repeatable procedure with a passing check, named failure pattern, and at least one ruled-out approach. Include applicability, constraints, and applicable source URLs, documentation, standards, or libraries; never include secrets.
7. After an MCP timeout, connection failure, or invalid response, call `check_lightrag_health`; if unavailable, continue from repository evidence and report the failed query or write.

## Query Tool Routing

- `query_text`: direct vector search over chunks without the graph. Use for code
  symbols, paths, errors, quotations, configuration keys, or other exact source
  evidence. Default `chunk_top_k=20`; retry once around 30-40 if evidence is thin.
- `query_graph`: graph-focused retrieval. Use `scope=local` for a named entity,
  `global` for broad themes or relationships, and `hybrid` for connections among
  multiple entities. Put themes in `hl_keywords` and exact names in `ll_keywords`.
- `query_mixed`: default for coding, architecture, debugging, chat, and uncertain
  retrieval shape. It combines graph and chunk retrieval; use the defaults
  (`top_k=36`, `chunk_top_k=24`) before increasing either independently.
- `query_tagged`: use when every returned source must contain all `required_tags`.
  It returns evidence only because tags are enforced after candidate retrieval;
  increase candidate limits when recall is low, never treat missing results as
  proof that tagged documents do not contain the answer.
- `query_document`: advanced fallback for custom budgets, answer format,
  `user_prompt`, reranking, tag scope, or history behavior. Prefer the specialized
  tools otherwise. `semantic` and `keyword` remain compatibility aliases; choose
  `query_text` and `query_graph` in new agent instructions.

For dependent chat questions, pass `conversation_history` to `query_mixed` or
`query_document`; `history_turns` selects recent turn pairs and the MCP server
adds them to retrieval when `use_history_for_retrieval=true`. Use
`only_need_context=true` for coding, evaluation, and high-stakes answers so the
agent can inspect sources itself. If retrieval is weak, change the query or mode
once before raising limits; usually keep each limit at or below 60.

## Memory Contract

LightRAG is a project encyclopedia, not a task log or backlog. Store concise,
present-tense, source-anchored facts about entities and roles, relationships,
architecture, processes and data flow, dependencies and rationale, lifecycle
and persistence, design tradeoffs, invariants, rules, exceptions, and
reproducible failure conditions. Reusable solution patterns are factual
reference entries, not session reports: describe the problem class, when the
approach applies, how it works, tradeoffs, verification criteria, and primary
references.

Reusable skills are tagged reference documents, not local executable skill
installations. Search before saving to avoid duplicates. Prefer project scope
unless the procedure is genuinely portable across repositories; use the
returned `document_id` to retrieve the complete procedure for reuse.

Never store intentions, TODOs, progress, task descriptions, debugging
chronology, command logs, validation diaries, change narratives, secrets, or
unrelated personal data. Express useful bug knowledge as a current constraint
or failure condition. Correct, replace, or delete stale entries instead of
appending history.

Begin every insertion with:

```text
Project: <name>
Project entity: Project|<name>
Project path: <absolute path>
Repository: <remote>
Reference topic: <entity, process, architecture area, or rule>
```

Then state the definition, responsibilities or process, design rationale,
invariants, source file or symbol anchors, factual relations, and applicable
reference URLs, documentation, standards, or library names. For multi-repo work,
anchor each fact to the correct project.

## Graph Curation

Mutate graph data only to correct known errors or record explicit project facts;
merge duplicate entities before deleting them. Reuse concise relations such as
`PART_OF`, `USES`, `IMPLEMENTS`, `DEPENDS_ON`, `CALLS`, `CONNECTS_TO`,
`RUNS_ON`, `STORES_IN`, and `RELATED_TO`.
