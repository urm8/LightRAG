---
name: lightrag-self-learning
description: >
  Recognize a hard-won reusable workflow, capture it as verified LightRAG
  memory, and retrieve it in later coding-agent sessions. Use proactively after
  multi-attempt debugging, non-obvious deployment or data work, discovering a
  recurring project constraint, or whenever the user says to remember or save
  a procedure. Also use before re-deriving a workflow that may already exist.
license: MIT
metadata:
  author: "Kulaxyz; LightRAG adaptation by urm8"
  version: "1.0"
---

# LightRAG self-learning

Run the same loop across Codex, Cursor, OpenCode, Claude Code, OpenClaw, and other Agent
Skills clients: **recognize → capture → reuse**. LightRAG is the shared store, so
the learned procedure is not tied to one agent's local rules directory.
It requires a LightRAG MCP server exposing `search_skills`, `save_skill`,
`get_document_content`, `insert_document`, and `get_pipeline_status`.

## Recognize

Treat any of these as a cue without waiting for a separate user prompt:

- the task succeeded only after wrong turns, retries, or a user correction;
- a non-obvious command, ordering constraint, project fact, or failure cause was found;
- a deployment, migration, debugging, data, or verification workflow will recur;
- the user says "remember this", "save this", or equivalent.

Triage the lesson before writing:

- A reusable multi-step procedure can become a skill.
- A verified durable fact belongs in project memory through `insert_document`.
- A one-off result, task log, guess, or unverified theory is skipped.

## Capture

1. Identify the active project name, absolute path, and repository remote.
2. Call `search_skills` with the problem or workflow. If a relevant skill exists,
   fetch it with `get_document_content` when the excerpt is insufficient and do
   not create a duplicate.
3. Promote a new skill only when all three are available:
   - a passing check that verified the procedure;
   - a named failure pattern it prevents or diagnoses;
   - at least one attempted approach ruled out with a reason.
4. Call `save_skill` with a generalized procedure: exact commands or tools,
   required order, applicability, verification, failure pattern, ruled-out
   approaches, project identity, and primary references.
5. If the write is needed immediately, call `get_pipeline_status` before relying
   on search to find it. Tell the user the saved skill name and scope afterward.

Never store secret values, private credentials, task chronology, or raw command
logs. Record only the environment-variable name, secret-manager location, or
credential selector. Repository source remains authoritative over learned memory.

## Reuse

Before deriving a recurring or non-obvious workflow, call `search_skills` with
the active project identity. Prefer a project-scoped result; use a global result
only when it genuinely applies across repositories. Fetch the full document only
for the selected result, follow its verification criteria, and correct stale
memory instead of silently working around it.

## Failure recovery

After an MCP timeout, connection failure, or invalid response, call
`check_lightrag_health`. If LightRAG remains unavailable, continue from repository
evidence and report that the capture or reuse step could not be completed.

Adapted from https://github.com/Kulaxyz/self-learning-skills (MIT).
