# Cursor Global Rule: Prefer LightRAG Project Memory

Copy this into Cursor **User Rules** when you want agents to consistently use
LightRAG as their project memory layer across repositories.

```text
When a LightRAG MCP/toolset is available, treat it as the primary project memory
system before relying on unstated memory or guessing prior project decisions.

Before non-trivial planning, debugging, code changes, QA, deployment work, or
architecture answers:
- Query LightRAG without a health preflight; use health only after a timeout,
  connection failure, or invalid response.
- Choose `query_text` for code symbols, paths, errors, quotes, and config keys;
  `query_graph` for entity/relationship questions; `query_mixed` for coding,
  architecture, debugging, chat, or uncertain retrieval; and `query_tagged`
  when every returned source must have all requested tags.
- Use advanced `query_document` only for custom token budgets, answer prompts,
  reranking, or history behavior not exposed by the specialized tools.
- Include the active project name, absolute project path, and repository remote
  in every LightRAG query. Example:
  "Project: <repo name>
   Project path: <absolute path>
   Repository: <remote>
   Question: <specific question>"
- Prefer raw/context-only retrieval when you need evidence before editing code.

Use LightRAG as memory, not as a replacement for repository inspection:
- Still read files, grep code, inspect logs, run tests, and verify behavior
  directly in the workspace.
- Prefer project-local instructions such as AGENTS.md, README files, Makefile
  targets, and test scripts when they conflict with generic habits.
- Pass `conversation_history` for dependent follow-up questions. Tune graph
  `top_k` and text `chunk_top_k` independently instead of increasing both.
- Do not invent unavailable LightRAG tools; inspect the active MCP schema when
  the server version does not expose the specialized names above.

After meaningful work, persist durable knowledge back into LightRAG:
- Save architecture decisions, root causes, fixes, validation results, runtime
  quirks, model/config changes, and operational procedures worth reusing.
- Start every saved note with a project anchor:
  "Project: <repo name>
   Project entity: Project|<repo name>
   Project path: <absolute path>
   Repository: <remote>
   Date: <YYYY-MM-DD>
   Scope: <debugging|coding|qa|deploy|planning|design>"
- Add explicit relation hints when useful, for example:
  "<Project> USES <tool/service>"
  "<Project> IMPLEMENTS <feature>"
  "<Project> FAILS_WITH <issue>"
  "<Project> IMPROVES <workflow>"
- Check pipeline/indexing status before assuming a newly saved note is
  immediately searchable.

If LightRAG is unavailable, slow, or returns no relevant context:
- Continue from repository-local evidence instead of blocking indefinitely.
- Mention the LightRAG limitation in the final response.
- Do not expose secrets, API keys, private tokens, or sensitive user data in
  LightRAG memory writes.
```
