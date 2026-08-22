import asyncio
from types import SimpleNamespace

import pytest

from lightrag.api.mcp_server import (
    AGENTIC_TOOL_DESCRIPTIONS,
    LightRAGMCPRuntime,
    create_lightrag_mcp,
    create_lightrag_mcp_http_app,
    mount_lightrag_mcp_http_app,
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


def _args():
    return SimpleNamespace(
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        graph_storage="NetworkXStorage",
        vector_storage="NanoVectorDBStorage",
        llm_model="test-llm",
        embedding_model="test-embedding",
    )


def _mcp():
    return create_lightrag_mcp(
        rag_provider=lambda: SimpleNamespace(),
        doc_manager=SimpleNamespace(),
        args=_args(),
    )


@pytest.mark.asyncio
async def test_integrated_lightrag_mcp_exposes_expected_tool_names():
    mcp = _mcp()

    tools = await mcp.get_tools()
    tool_names = set(tools)

    assert EXPECTED_TOOL_NAMES <= tool_names


@pytest.mark.asyncio
async def test_integrated_lightrag_mcp_tool_descriptions_are_agentic():
    mcp = _mcp()

    tools = await mcp.get_tools()

    assert tools["query_document"].description == AGENTIC_TOOL_DESCRIPTIONS["query_document"]
    assert "analytics" in tools["query_document"].description
    assert "debugging" in tools["query_document"].description
    assert "deployment" in tools["query_document"].description
    assert "durable agent memory" in tools["insert_document"].description
    assert "memory pressure" in tools["check_memory_pressure"].description


def test_lightrag_mcp_http_app_mounts_at_root_for_fastapi_prefix():
    mcp_app = create_lightrag_mcp_http_app(
        rag_provider=lambda: SimpleNamespace(),
        doc_manager=SimpleNamespace(),
        args=_args(),
    )

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


def test_lightrag_mcp_mount_requires_configured_api_key():
    async def child_endpoint(request):
        return PlainTextResponse("ok")

    host_app = Starlette()
    mcp_app = Starlette(
        routes=[Route("/", child_endpoint, methods=["GET", "POST", "DELETE"])]
    )
    mount_lightrag_mcp_http_app(host_app, mcp_app, "/mcp", api_key="test-key")
    client = TestClient(host_app, follow_redirects=False)

    assert client.post("/mcp").status_code == 403
    assert client.post("/mcp", headers={"X-API-Key": "wrong"}).status_code == 403
    assert (
        client.post("/mcp", headers={"X-API-Key": "test-key"}).status_code
        == 200
    )
    assert (
        client.post("/mcp/", headers={"X-API-Key": "test-key"}).status_code
        == 200
    )


@pytest.mark.asyncio
async def test_runtime_query_calls_live_lightrag_directly():
    class Rag:
        async def aquery_llm(self, query, param):
            assert query == "Project: LightRAG. Test query"
            assert param.mode == "mix"
            assert param.max_total_tokens == 3000
            return {
                "llm_response": {"content": "direct response"},
                "data": {
                    "chunks": [
                        {
                            "content": "matching source context",
                            "file_path": "a.md",
                            "reference_id": "1",
                            "chunk_id": "chunk-1",
                        }
                    ],
                    "references": [{"reference_id": "1", "file_path": "a.md"}],
                },
            }

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())

    result = await runtime.query(
        query="Project: LightRAG. Test query",
        mode="mix",
        top_k=10,
        only_need_context=False,
        only_need_prompt=False,
        response_type="Multiple Paragraphs",
        max_token_for_text_unit=1000,
        max_token_for_global_context=1000,
        max_token_for_local_context=1000,
        hl_keywords=[],
        ll_keywords=[],
        history_turns=0,
    )

    assert result["response"] == "direct response"
    assert result["matches"] == [
        {
            "content": "matching source context",
            "file_path": "a.md",
            "reference_id": "1",
            "chunk_id": "chunk-1",
        }
    ]
    assert result["references"][0]["file_path"] == "a.md"


@pytest.mark.asyncio
async def test_runtime_context_query_returns_matches_instead_of_raw_prompt():
    class Rag:
        async def aquery_llm(self, query, param):
            assert param.only_need_context is True
            return {
                "llm_response": {"content": "very large graph prompt"},
                "data": {
                    "chunks": [
                        {
                            "content": "the useful matching excerpt",
                            "file_path": "memory.md",
                            "reference_id": "7",
                            "chunk_id": "chunk-7",
                        }
                    ],
                    "references": [
                        {"reference_id": "7", "file_path": "memory.md"}
                    ],
                },
            }

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())

    result = await runtime.query(
        query="Project: LightRAG. Find the implementation decision",
        mode="mix",
        top_k=10,
        only_need_context=True,
        only_need_prompt=False,
        response_type="Multiple Paragraphs",
        max_token_for_text_unit=1000,
        max_token_for_global_context=1000,
        max_token_for_local_context=1000,
        hl_keywords=[],
        ll_keywords=[],
        history_turns=0,
    )

    assert result["response"] == "Retrieved 1 matching context chunk(s)."
    assert result["matches"][0]["content"] == "the useful matching excerpt"
    assert "very large graph prompt" not in str(result)


@pytest.mark.asyncio
async def test_runtime_insert_schedules_live_lightrag_ingestion():
    inserted = []

    class Rag:
        async def ainsert(self, texts, *, file_paths, track_id, tags):
            inserted.append((texts, file_paths, track_id, tags))

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())
    result = await runtime.insert_text(
        "Project: LightRAG. Durable finding.", ["Skill", "agentic development"]
    )
    await asyncio.gather(*runtime.background_tasks)

    assert result["status"] == "success"
    assert inserted[0][0] == ["Project: LightRAG. Durable finding."]
    assert inserted[0][3] == ["skill", "agentic-development"]
    assert result["tags"] == ["skill", "agentic-development"]
    assert inserted[0][2] == result["track_id"]


@pytest.mark.asyncio
async def test_runtime_documents_filters_by_all_tags():
    from lightrag.base import DocProcessingStatus, DocStatus

    docs = {
        DocStatus.PROCESSED: {
            "skill-doc": DocProcessingStatus(
                content_summary="skill",
                content_length=5,
                status=DocStatus.PROCESSED,
                created_at="2026-08-16T00:00:00Z",
                updated_at="2026-08-16T00:00:00Z",
                file_path="skill.md",
                metadata={"tags": ["skill", "agentic-development"]},
            ),
            "workflow-doc": DocProcessingStatus(
                content_summary="workflow",
                content_length=8,
                status=DocStatus.PROCESSED,
                created_at="2026-08-16T00:00:00Z",
                updated_at="2026-08-16T00:00:00Z",
                file_path="workflow.md",
                metadata={"tags": ["workflow"]},
            ),
        }
    }

    class Rag:
        async def get_docs_by_status(self, status):
            return docs.get(status, {})

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())
    result = await runtime.documents(["Skill", "agentic development"])

    processed = result["statuses"][DocStatus.PROCESSED.value]
    assert [doc["id"] for doc in processed] == ["skill-doc"]
