from types import SimpleNamespace

import pytest

from lightrag.api.mcp_server import (
    AGENTIC_TOOL_DESCRIPTIONS,
    create_lightrag_mcp,
    create_lightrag_mcp_http_app,
    mount_lightrag_mcp_http_app,
    _configure_lightrag_client_auth,
    _get_lifespan_context,
    _ensure_lightrag_mcp_submodule_importable,
    _mcp_query_profile,
)
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient


EXPECTED_TOOL_NAMES = {
    "query_document",
    "insert_document",
    "upload_document",
    "insert_file",
    "insert_batch",
    "scan_for_new_documents",
    "get_documents",
    "get_pipeline_status",
    "get_graph_labels",
    "check_lightrag_health",
    "check_memory_pressure",
    "merge_entities",
    "create_entities",
    "delete_by_entities",
    "delete_by_doc_ids",
    "edit_entities",
    "create_relations",
    "edit_relations",
}


@pytest.mark.asyncio
async def test_integrated_lightrag_mcp_exposes_sidecar_tool_names():
    mcp = create_lightrag_mcp(SimpleNamespace(port=9621), api_key="test-key")

    tools = await mcp.get_tools()
    tool_names = set(tools)

    assert EXPECTED_TOOL_NAMES <= tool_names


@pytest.mark.asyncio
async def test_integrated_lightrag_mcp_tool_descriptions_are_agentic():
    mcp = create_lightrag_mcp(SimpleNamespace(port=9621), api_key="test-key")

    tools = await mcp.get_tools()

    assert tools["query_document"].description == AGENTIC_TOOL_DESCRIPTIONS["query_document"]
    assert "analytics" in tools["query_document"].description
    assert "debugging" in tools["query_document"].description
    assert "deployment" in tools["query_document"].description
    assert "durable agent memory" in tools["insert_document"].description
    assert "memory pressure" in tools["check_memory_pressure"].description


def test_lightrag_mcp_http_app_mounts_at_root_for_fastapi_prefix():
    mcp_app = create_lightrag_mcp_http_app(SimpleNamespace(port=9621), api_key="test-key")

    assert hasattr(mcp_app, "lifespan")
    assert any(getattr(route, "path", None) == "/" for route in mcp_app.routes)
    assert mcp_app.state.path == "/"


def test_lightrag_mcp_mount_accepts_path_without_redirect():
    async def child_endpoint(request):
        return PlainTextResponse("ok")

    host_app = Starlette()
    mcp_app = Starlette(
        routes=[Route("/", child_endpoint, methods=["GET", "POST", "DELETE"])]
    )
    mount_lightrag_mcp_http_app(host_app, mcp_app, "/mcp")
    client = TestClient(host_app, follow_redirects=False)

    assert client.post("/mcp").status_code == 200
    assert client.post("/mcp/").status_code == 200


def test_lightrag_mcp_configures_x_api_key_header():
    class GeneratedClient:
        def __init__(self):
            self.headers = {}
            self.auth_header_name = "Authorization"
            self.prefix = "Bearer"

        def with_headers(self, headers):
            self.headers.update(headers)
            return self

    class Client:
        def __init__(self):
            self.client = GeneratedClient()

    client = Client()

    _configure_lightrag_client_auth(client, "secret-key")

    assert client.client.headers == {"X-API-Key": "secret-key"}
    assert client.client.auth_header_name == "X-API-Key"
    assert client.client.prefix == ""


def test_lightrag_mcp_context_accessor_supports_fastmcp_2_request_context():
    class RequestContext:
        lifespan_context = {"lightrag_client": object()}

    class Context:
        request_context = RequestContext()

    assert "lightrag_client" in _get_lifespan_context(Context())


def test_lightrag_mcp_query_profile_defaults_to_granite(monkeypatch):
    monkeypatch.delenv("LIGHTRAG_MCP_QUERY_PROFILE", raising=False)

    assert _mcp_query_profile() == "granite"


@pytest.mark.asyncio
async def test_lightrag_client_query_sends_query_model_profile(monkeypatch):
    _ensure_lightrag_mcp_submodule_importable()
    from lightrag_mcp.lightrag_client import LightRAGClient

    client = LightRAGClient(base_url="http://127.0.0.1:9621", api_key="test-key")
    captured = {}

    async def fake_call_api(**kwargs):
        captured["body"] = kwargs["body"]
        return None

    monkeypatch.setattr(client, "_call_api", fake_call_api)

    await client.query("Project: LightRAG. Test query", query_model="granite")

    assert captured["body"].additional_properties["query_model"] == "granite"
