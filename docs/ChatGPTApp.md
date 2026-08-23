# ChatGPT LightRAG Memory

LightRAG exposes a least-privilege, OAuth-protected MCP endpoint for ChatGPT:

```text
https://rag.urm8.org/chatgpt/mcp
```

Enable ChatGPT developer mode, add the endpoint as a plugin connection, then sign
in with a configured LightRAG `AUTH_ACCOUNTS` account. ChatGPT receives only:

- `search_memory`: bounded source excerpts with document IDs and references.
- `get_memory_source`: an explicit full-document read after an excerpt is relevant.
- `save_memory`: persistent memory insertion, requiring `memory:write`.

The existing `/mcp` endpoint remains the API-key-protected administration surface.
OAuth uses authorization code + PKCE, dynamic client registration, one-hour access
tokens, rotating refresh tokens, and `memory:read` / `memory:write` scopes.

Configuration:

```dotenv
AUTH_ACCOUNTS='admin:{bcrypt}$2b$...'
TOKEN_SECRET='a-private-random-secret'
LIGHTRAG_CHATGPT_APP_ENABLED=true
LIGHTRAG_CHATGPT_BASE_URL=https://rag.urm8.org/chatgpt
```

Never commit account hashes, token secrets, or private Helm values.
