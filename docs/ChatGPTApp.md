# Enable LightRAG in ChatGPT

LightRAG exposes a focused, OAuth-protected MCP app for ChatGPT at:

```text
https://rag.urm8.org/chatgpt/mcp
```

The app can search shared memory, open a selected source, and save durable
knowledge. It does not expose the administrative LightRAG MCP tools.

## Prerequisites

Before connecting, confirm that:

- Your ChatGPT account or workspace allows Developer mode.
- You have a LightRAG WebUI username and password.
- `https://rag.urm8.org/health` returns a successful response.
- You use the ChatGPT endpoint above, not the administrative `/mcp` endpoint.

## 1. Enable Developer Mode

1. Open ChatGPT.
2. Open **Settings**.
3. Select **Security and login**.
4. Turn on **Developer mode**.

Developer mode may be unavailable when disabled by an account or workspace
policy.

## 2. Add the LightRAG Connection

1. Open [ChatGPT Plugins](https://chatgpt.com/plugins).
2. Select the plus button.
3. Enter a name, for example `LightRAG Memory`.
4. Enter a description, for example `Shared searchable project memory`.
5. Under **Connection**, choose the public MCP endpoint option.
6. Enter this exact URL:

   ```text
   https://rag.urm8.org/chatgpt/mcp
   ```

7. Create the connection.
8. Review the discovered tools and continue to authentication.

## 3. Authenticate

1. ChatGPT redirects to the LightRAG login page.
2. Sign in with the same username and password used for the LightRAG WebUI.
3. Complete the authorization flow.
4. Return to ChatGPT and confirm that the connection is shown as connected.

The login uses OAuth authorization code flow with PKCE. Do not enter a
LightRAG API key in ChatGPT; API keys belong to the separate administrative
MCP endpoint.

## 4. Enable the App in a Chat

1. Start a new ChatGPT conversation.
2. Open the chat tools menu.
3. Add or enable the `LightRAG Memory` connection.
4. Ask ChatGPT to search or save memory.

The connection exposes exactly these tools:

- `search_memory`: returns up to six bounded, relevant source excerpts with
  document IDs and citations.
- `get_memory_source`: reads a full document only after a relevant excerpt has
  identified its document ID.
- `save_memory`: inserts durable knowledge into LightRAG.

## 5. Verify the Connection

Run these prompts in a new conversation with the app enabled:

```text
Search LightRAG memory for the current LightRAG MCP retrieval behavior and cite
the matching sources.
```

Expected behavior: ChatGPT calls `search_memory`, returns compact excerpts, and
cites the returned references.

```text
Open the full source for the most relevant result.
```

Expected behavior: ChatGPT reuses the returned `document_id` and calls
`get_memory_source` once. It should not open every matching document.

```text
Save this durable memory: Project LightRAG uses bounded MCP excerpts to reduce
agent context usage.
```

Expected behavior: ChatGPT calls `save_memory`. Save only information that is
intended to persist across chats.

## Troubleshooting

### Developer mode is missing

Developer mode availability depends on the ChatGPT account and workspace
policy. Ask the workspace administrator to allow it.

### ChatGPT cannot connect

1. Confirm the endpoint is exactly
   `https://rag.urm8.org/chatgpt/mcp`.
2. Confirm `https://rag.urm8.org/health` is reachable.
3. Remove the failed connection and add it again.
4. If the server metadata changed, open the connection in ChatGPT Plugins and
   select **Refresh**.

### Authentication fails

1. Verify the credentials by logging into the LightRAG WebUI.
2. Reconnect the ChatGPT app to restart OAuth authorization.
3. Ask the LightRAG operator to verify `AUTH_ACCOUNTS`, `TOKEN_SECRET`, and the
   configured public base URL without sharing those values.

### Tools are outdated

1. Deploy or restart LightRAG.
2. Open [ChatGPT Plugins](https://chatgpt.com/plugins).
3. Open the LightRAG connection and select **Refresh**.
4. Start a new conversation before retesting.

## Server Configuration

The deployment must provide:

```dotenv
AUTH_ACCOUNTS='admin:{bcrypt}$2b$...'
TOKEN_SECRET='a-private-random-secret'
LIGHTRAG_CHATGPT_APP_ENABLED=true
LIGHTRAG_CHATGPT_BASE_URL=https://rag.urm8.org/chatgpt
```

Never commit account hashes, token secrets, API keys, or private Helm values.
The API-key-protected administrative MCP endpoint remains separate at `/mcp`.

## References

- [OpenAI: Connect and test your plugin](https://developers.openai.com/plugins/deploy/connect-chatgpt)
- [ChatGPT Plugins](https://chatgpt.com/plugins)
