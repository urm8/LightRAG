# Local LightRAG MCP FAQ

This FAQ documents the local LightRAG API service, the LightRAG MCP bridge, and client setup for Codex and VS Code on this machine.

## What runs where?

- LightRAG API and WebUI: `http://127.0.0.1:9621`
- LightRAG MCP stdio: optional legacy wrapper, started on demand by Codex or VS Code
- LightRAG MCP HTTP: `http://127.0.0.1:9621/mcp`
- LightRAG repo: `/Users/max/projs/github.com/HKUDS/LightRAG`
- MCP submodule: `/Users/max/projs/github.com/HKUDS/LightRAG/third_party/lightrag-mcp`
- Local config: `/Users/max/projs/github.com/HKUDS/LightRAG/.env`
- Local data: `/Users/max/projs/github.com/HKUDS/LightRAG/inputs` and `/Users/max/projs/github.com/HKUDS/LightRAG/rag_storage`
- Local MCP wrappers: `/Users/max/projs/github.com/HKUDS/LightRAG/data/bin/lightrag-mcp-stdio` and `/Users/max/projs/github.com/HKUDS/LightRAG/data/bin/lightrag-mcp-http` are legacy sidecar helpers.

## What is the current setup?

- LightRAG runs through launchd as `com.local.lightrag` on `127.0.0.1:9621`.
- LLM inference currently uses a local mlx OpenAI-compatible endpoint at `http://127.0.0.1:11436/v1` with `granite-4.1-3b` (query) and `http://127.0.0.1:11438/v1` with `granite-4.1-3b-abliterated` (extraction).

- The extraction model (`granite4.1-abliterated`) has no safety filter, so no bypass logic is needed.

- The query model uses a standard 8192-token context window.
- Embeddings use local Ollama at `http://localhost:11434` with `bge-m3:latest`, `EMBEDDING_DIM=1024`, and `EMBEDDING_TOKEN_LIMIT=8192`.
- Storage uses local PostgreSQL database `lightrag` with `PGKVStorage`, `PGDocStatusStorage`, `PGVectorStorage`, and `PGGraphStorage`.
- PostgreSQL extensions are `vector` for pgvector and Apache AGE `age` for graph storage.
- Processing concurrency uses `MAX_ASYNC=4` and `MAX_PARALLEL_INSERT=2`. The VS Code bridge throttles above four active requests with `429 throttled (active=4, max=4)`, so keep LightRAG LLM concurrency at or below four for this endpoint.
- Entity extraction includes coding-project metadata types such as `Workspace`, `Project`, `Repository`, `Directory`, `File`, `ProgrammingLanguage`, `TechnologyStack`, `Service`, `Deployment`, `Environment`, `Configuration`, `Command`, `APIEndpoint`, `Database`, and `StorageBackend`.
- MCP HTTP is mounted inside the main LightRAG service at `http://127.0.0.1:9621/mcp`.

## How do I start LightRAG?

Foreground debug run:

```bash
cd /Users/max/projs/github.com/HKUDS/LightRAG
source .venv/bin/activate
lightrag-server --log-level DEBUG
```

Background service with launchd:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.local.lightrag.plist
launchctl kickstart -k gui/$(id -u)/com.local.lightrag
```

Docker Compose:

```bash
cd /Users/max/projs/github.com/HKUDS/LightRAG
docker compose up -d
```

## How do I stop or restart LightRAG?

launchd:

```bash
launchctl kickstart -k gui/$(id -u)/com.local.lightrag
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.local.lightrag.plist
```

Docker Compose:

```bash
cd /Users/max/projs/github.com/HKUDS/LightRAG
docker compose restart lightrag
docker compose down
```

Foreground run: press `Ctrl-C`.

## How do I debug LightRAG?

Check health:

```bash
curl -sS http://127.0.0.1:9621/health
```

Check an API-key-protected request:

```bash
set -a; source /Users/max/projs/github.com/HKUDS/LightRAG/.env; set +a
curl -sS -H "X-API-Key: $LIGHTRAG_API_KEY" http://127.0.0.1:9621/health
```

launchd logs:

```bash
tail -f ~/Library/Logs/lightrag/lightrag.out.log
tail -f ~/Library/Logs/lightrag/lightrag.err.log
launchctl print gui/$(id -u)/com.local.lightrag
```

Docker logs:

```bash
cd /Users/max/projs/github.com/HKUDS/LightRAG
docker compose logs -f lightrag
```

## How do I switch models?

Edit `/Users/max/projs/github.com/HKUDS/LightRAG/.env`, then restart LightRAG.

For LLM changes, update:

```dotenv
LLM_BINDING=openai
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_MODEL=gpt-5-mini
LLM_BINDING_API_KEY=...
```

For the local VS Code Copilot Bridge, use the OpenAI-compatible bridge endpoint. This endpoint requires the VS Code bridge token in `LLM_BINDING_API_KEY`.

```dotenv
LLM_BINDING=openai
LLM_BINDING_HOST=http://127.0.0.1:8989/v1
LLM_MODEL=oswe-vscode
LLM_BINDING_API_KEY=<bridge token from VS Code settings>
MAX_ASYNC=4
MAX_PARALLEL_INSERT=2
```

For the local extraction service:

An abliterated Granite model serves at port 11438 with an 8192-token context window, suitable for document extraction tasks.

For a local MLX-backed AgentCPM service managed alongside LightRAG:

```dotenv
LLM_BINDING=openai
LLM_BINDING_HOST=http://127.0.0.1:11436/v1
LLM_BINDING_API_KEY=dummy
LLM_MODEL=openbmb/AgentCPM-Explore
```

The repo includes local targets for this path:

```bash
cd /Users/max/projs/github.com/HKUDS/LightRAG
uv sync --extra api --extra offline-storage --extra offline-llm --extra apple-mlx
make mlx-agentcpm-convert
make mlx-agentcpm-install
make mlx-agentcpm-restart
make mlx-agentcpm-health
make use-mlx-agentcpm
make lightrag-restart
```

The MLX model is stored under `/Users/max/projs/github.com/HKUDS/LightRAG/models/agentcpm-explore-mlx-4bit`, and the launchd service label is `com.local.mlx-agentcpm`.

## How do I validate prompts before indexing?

Run the extraction promptfoo gate:

This exports the active LightRAG prompts, builds `evals/promptfooconfig.generated.yaml`, sends fixture chunks to the extraction LLM, and fails if output would trigger parser warnings such as `entity_field_count`, `relation_field_count`, `delimiter_record_separator`, `entity_invalid_type`, empty descriptions, or same-source-target relations.

When LightRAG sees bad extraction output during indexing, it appends the failed prompt context to `evals/captured/lightrag_prompt_warnings.jsonl` and logs `PROMPTFOO_PROMPT_SUGGESTION` with the prompt file, prompt key, and suggested edit. The next `make test-prompt` run automatically includes those captured warning cases, so prompt changes can be tested against real failures.

The bridge token is stored in VS Code user settings as `bridge.token`. LightRAG uses the bridge only for chat completions in the current setup; embeddings are served by Ollama.

For the local Copilot API service, use the OpenAI-compatible endpoint for chat completions:

```dotenv
LLM_BINDING=openai
LLM_BINDING_HOST=http://localhost:4141/v1
LLM_MODEL=gpt-5-mini
LLM_BINDING_API_KEY=dummy
```

The Copilot API service is managed by launchd as `com.local.copilot-api`. Its GitHub auth token is stored by `copilot-api` under `~/.local/share/copilot-api`, not in LightRAG `.env`.

Do not set `OPENAI_LLM_REASONING_EFFORT` for document indexing. Reasoning models are slower and consume more provider capacity during extraction. Use stronger or reasoning-capable models at query time when needed, not for bulk indexing.

## How do I tune coding project extraction?

LightRAG reads extraction categories from `ENTITY_TYPES` in `.env`. The current local setup keeps the default entity types and adds coding-project metadata:

```dotenv
ENTITY_TYPES=["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject","Workspace","Project","Repository","Directory","File","ProgrammingLanguage","TechnologyStack","Framework","Library","Runtime","Service","Deployment","Environment","Configuration","Command","APIEndpoint","Database","StorageBackend"]
```

Restart LightRAG after changing this value. Existing indexed documents keep whatever entities were extracted at indexing time, so re-index documents when the entity schema materially changes.

For the current local Ollama embedding setup:

```dotenv
EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
EMBEDDING_TOKEN_LIMIT=8192
EMBEDDING_BINDING_API_KEY=
```

Do not change the embedding model or dimension after indexing documents unless you are ready to drop the Postgres vector tables and re-index. With `PGVectorStorage`, vector table names and dimensions are created from the embedding model and dimension.

Current bge-m3 vector tables:

```text
lightrag_vdb_chunks_bge_m3_latest_1024d
lightrag_vdb_entity_bge_m3_latest_1024d
lightrag_vdb_relation_bge_m3_latest_1024d
```

## How do I start the MCP service?

HTTP MCP is part of the main LightRAG service. Start or restart `com.local.lightrag`; no separate HTTP MCP launchd service is required.

Manual stdio command:

```bash
/Users/max/projs/github.com/HKUDS/LightRAG/data/bin/lightrag-mcp-stdio
```

Clients should connect to:

```text
http://127.0.0.1:9621/mcp
```

## How does Codex connect?

Codex uses `~/.codex/config.toml`. The configured `lightrag` MCP server runs stdio by default and reads the local LightRAG API key from `.env`.

Check configured MCP servers:

```bash
codex mcp list
```

Restart Codex after editing `~/.codex/config.toml`.

## How does VS Code connect?

VS Code uses `/Users/max/projs/github.com/HKUDS/LightRAG/.vscode/mcp.json`.

Use Command Palette:

```text
MCP: List Servers
```

Start `lightrag` for stdio mode, or use `lightragHttp` for the integrated HTTP MCP endpoint.

## Which key is which?

- OpenAI API key: used by LightRAG for LLM and embedding calls.
- LightRAG API key: generated locally in `.env`; used by MCP and API clients through `X-API-Key`.

Do not commit either key.
