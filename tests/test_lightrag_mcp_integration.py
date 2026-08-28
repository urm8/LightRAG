import asyncio
from types import SimpleNamespace

import pytest

from lightrag.api.mcp_server import (
    AGENTIC_SERVER_INSTRUCTIONS,
    AGENTIC_TOOL_DESCRIPTIONS,
    LightRAGMCPRuntime,
    _matching_excerpt,
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
    "query_text",
    "query_graph",
    "query_mixed",
    "query_tagged",
    "insert_document",
    "save_skill",
    "search_skills",
    "upload_document",
    "insert_file",
    "insert_batch",
    "scan_for_new_documents",
    "get_documents",
    "get_document_content",
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

    assert (
        tools["query_document"].description
        == AGENTIC_TOOL_DESCRIPTIONS["query_document"]
    )
    assert "analytics" in tools["query_document"].description
    assert "debugging" in tools["query_document"].description
    assert "deployment" in tools["query_document"].description
    assert "factual project reference" in tools["insert_document"].description
    assert "passing check" in tools["save_skill"].description
    assert "only reusable agent skills" in tools["search_skills"].description
    assert "memory pressure" in tools["check_memory_pressure"].description
    assert tools["query_text"].description == AGENTIC_TOOL_DESCRIPTIONS["query_text"]
    assert "scope=local" in tools["query_graph"].description
    assert "conversation_history" in tools["query_mixed"].description
    assert "every required tag" in tools["query_tagged"].description
    assert mcp.instructions == AGENTIC_SERVER_INSTRUCTIONS


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
    assert client.post("/mcp", headers={"X-API-Key": "test-key"}).status_code == 200
    assert client.post("/mcp/", headers={"X-API-Key": "test-key"}).status_code == 200


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
    assert result["matches"][0]["content"] == "matching source context"
    assert result["matches"][0]["chunk_id"] == "chunk-1"
    assert result["matches"][0]["is_excerpt"] is False
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
                    "references": [{"reference_id": "7", "file_path": "memory.md"}],
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

    assert result["response"] == "Retrieved 1 bounded source excerpt(s)."
    assert result["matches"][0]["content"] == "the useful matching excerpt"
    assert "very large graph prompt" not in str(result)


@pytest.mark.asyncio
async def test_runtime_query_supports_independent_limits_and_retrieval_history():
    class Rag:
        async def aquery_llm(self, query, param):
            assert "user: Which query tool handles code?" in query
            assert query.endswith("Current query:\nWhat about errors?")
            assert param.top_k == 40
            assert param.chunk_top_k == 12
            assert param.conversation_history == [
                {"role": "user", "content": "Which query tool handles code?"},
                {"role": "assistant", "content": "Use query_text."},
            ]
            assert param.user_prompt == "Cite the implementation."
            assert param.enable_rerank is False
            return {"llm_response": {"content": "answer"}, "data": {}}

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())
    result = await runtime.query(
        query="What about errors?",
        mode="mix",
        top_k=40,
        chunk_top_k=12,
        only_need_context=False,
        only_need_prompt=False,
        response_type="Multiple Paragraphs",
        max_token_for_text_unit=1000,
        max_token_for_global_context=1000,
        max_token_for_local_context=1000,
        hl_keywords=[],
        ll_keywords=[],
        history_turns=1,
        conversation_history=[
            {"role": "user", "content": "Which query tool handles code?"},
            {"role": "assistant", "content": "Use query_text."},
        ],
        user_prompt="Cite the implementation.",
        enable_rerank=False,
    )

    assert result["retrieval"] == {
        "mode": "mix",
        "top_k": 40,
        "chunk_top_k": 12,
        "history_messages_used": 2,
        "history_used_for_retrieval": True,
        "required_tags": [],
        "enable_rerank": False,
    }


@pytest.mark.asyncio
async def test_runtime_tag_filter_forces_context_only_output():
    class TextChunks:
        async def get_by_ids(self, chunk_ids):
            return [{"full_doc_id": "doc-1"}]

    class DocStatus:
        async def get_by_id(self, document_id):
            return {"metadata": {"tags": ["project", "decision"]}}

    class Rag:
        text_chunks = TextChunks()
        doc_status = DocStatus()

        async def aquery_llm(self, query, param):
            assert param.only_need_context is True
            return {
                "llm_response": {"content": "must not escape"},
                "data": {
                    "chunks": [
                        {
                            "content": "Tagged decision",
                            "reference_id": "1",
                            "chunk_id": "chunk-1",
                        }
                    ],
                    "references": [{"reference_id": "1"}],
                },
            }

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())
    result = await runtime.query(
        query="Find the decision",
        mode="mix",
        top_k=20,
        chunk_top_k=10,
        only_need_context=False,
        only_need_prompt=False,
        response_type="Multiple Paragraphs",
        max_token_for_text_unit=1000,
        max_token_for_global_context=1000,
        max_token_for_local_context=1000,
        hl_keywords=[],
        ll_keywords=[],
        history_turns=0,
        required_tags=["Decision"],
    )

    assert result["response"] == "Retrieved 1 bounded source excerpt(s)."
    assert result["matches"][0]["tags"] == ["project", "decision"]


def test_matching_excerpt_keeps_small_lookaround_around_query_terms():
    content = "irrelevant " * 100 + "OAuth PKCE validation decision" + " tail" * 100

    excerpt, matched_terms = _matching_excerpt(content, ["oauth", "pkce"])

    assert len(excerpt) <= 706
    assert "OAuth PKCE validation decision" in excerpt
    assert matched_terms == ["oauth", "pkce"]
    assert excerpt.startswith("...")
    assert excerpt.endswith("...")


@pytest.mark.asyncio
async def test_runtime_query_limits_matches_and_resolves_document_ids():
    chunks = [
        {
            "content": f"result {index} about OAuth",
            "file_path": f"{index}.md",
            "reference_id": str(index),
            "chunk_id": f"chunk-{index}",
        }
        for index in range(10)
    ]

    class TextChunks:
        async def get_by_ids(self, chunk_ids):
            return [
                {"full_doc_id": f"doc-{chunk_id.removeprefix('chunk-')}"}
                for chunk_id in chunk_ids
            ]

    class Rag:
        text_chunks = TextChunks()

        async def aquery_llm(self, query, param):
            return {
                "llm_response": {"content": "unused"},
                "data": {
                    "chunks": chunks,
                    "references": [
                        {"reference_id": str(index), "file_path": f"{index}.md"}
                        for index in range(10)
                    ],
                },
            }

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())
    result = await runtime.query(
        query="Find OAuth decisions",
        mode="mix",
        top_k=60,
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

    assert len(result["matches"]) == 6
    assert len(result["references"]) == 6
    assert result["matches"][0]["document_id"] == "doc-0"


@pytest.mark.asyncio
async def test_runtime_document_content_fetches_full_body_explicitly():
    class Store:
        def __init__(self, value):
            self.value = value

        async def get_by_id(self, key):
            return self.value

    rag = SimpleNamespace(
        full_docs=Store({"content": "complete document body"}),
        doc_status=Store(
            {"file_path": "docs/design.md", "metadata": {"tags": ["design"]}}
        ),
    )
    runtime = LightRAGMCPRuntime(lambda: rag, SimpleNamespace(), _args())

    result = await runtime.document_content("doc-1")

    assert result == {
        "document_id": "doc-1",
        "file_path": "docs/design.md",
        "content": "complete document body",
        "metadata": {"tags": ["design"]},
    }


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
async def test_runtime_save_skill_stores_verified_tagged_reference():
    inserted = []

    class Rag:
        async def ainsert(self, texts, *, file_paths, track_id, tags):
            inserted.append((texts, file_paths, track_id, tags))

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())
    result = await runtime.save_skill(
        name="verify-mcp-tools",
        description="Verify every integrated MCP tool through FastMCP.",
        applicability="An integrated MCP route or tool changes.",
        procedure="List tools, call each read-only tool, then inspect structured output.",
        verification="The focused MCP integration test passes.",
        failure_pattern="A mounted MCP route advertises stale or unusable tools.",
        ruled_out=["Health-only checks do not exercise tool schemas or handlers."],
        references=["https://gofastmcp.com/"],
        project_name="LightRAG",
        project_path="/work/LightRAG",
        repository="https://github.com/HKUDS/LightRAG",
        scope="project",
    )
    await asyncio.gather(*runtime.background_tasks)

    stored_text = inserted[0][0][0]
    assert "Project entity: Project|LightRAG" in stored_text
    assert "Skill entity: Workflow|verify-mcp-tools" in stored_text
    assert "Verification: The focused MCP integration test passes." in stored_text
    assert "https://gofastmcp.com/" in stored_text
    assert inserted[0][3] == [
        "skill",
        "agentic-development",
        "reusable-solution",
        "skill-verify-mcp-tools",
        "project",
    ]
    assert result["skill_name"] == "verify-mcp-tools"


@pytest.mark.asyncio
async def test_runtime_save_skill_enforces_promotion_gate():
    runtime = LightRAGMCPRuntime(lambda: SimpleNamespace(), SimpleNamespace(), _args())

    with pytest.raises(ValueError, match="ruled_out"):
        await runtime.save_skill(
            name="unverified-skill",
            description="A procedure.",
            applicability="A condition.",
            procedure="Do the thing.",
            verification="A check passed.",
            failure_pattern="The thing fails.",
            ruled_out=[],
            references=[],
            project_name="LightRAG",
            project_path="/work/LightRAG",
            repository="https://github.com/HKUDS/LightRAG",
            scope="project",
        )


@pytest.mark.asyncio
async def test_runtime_search_skills_filters_before_match_limit_and_deduplicates():
    chunks = [
        {
            "content": f"Unrelated architecture note {index}",
            "file_path": f"note-{index}.md",
            "reference_id": str(index),
            "chunk_id": f"chunk-{index}",
        }
        for index in range(8)
    ] + [
        {
            "content": "Reusable procedure for verifying FastMCP tools",
            "file_path": "mcp-memory/skill.txt",
            "reference_id": "8",
            "chunk_id": "chunk-8",
        },
        {
            "content": "FastMCP verification commands and constraints",
            "file_path": "mcp-memory/skill.txt",
            "reference_id": "9",
            "chunk_id": "chunk-9",
        },
    ]

    class TextChunks:
        async def get_by_ids(self, chunk_ids):
            return [
                {
                    "full_doc_id": "skill-doc"
                    if chunk_id in {"chunk-8", "chunk-9"}
                    else f"note-doc-{chunk_id}"
                }
                for chunk_id in chunk_ids
            ]

    class DocStatus:
        async def get_by_id(self, document_id):
            tags = (
                ["skill", "agentic-development"]
                if document_id == "skill-doc"
                else ["note"]
            )
            return {"metadata": {"tags": tags}}

    class Rag:
        text_chunks = TextChunks()
        doc_status = DocStatus()

        async def aquery_llm(self, query, param):
            assert "Reusable agent skill related to" in query
            return {
                "llm_response": {"content": "unused"},
                "data": {
                    "chunks": chunks,
                    "references": [
                        {"reference_id": str(index), "file_path": f"{index}.md"}
                        for index in range(10)
                    ],
                },
            }

    runtime = LightRAGMCPRuntime(lambda: Rag(), SimpleNamespace(), _args())
    result = await runtime.search_skills(
        query="verify MCP tools",
        project_name="LightRAG",
        project_path="/work/LightRAG",
        repository="https://github.com/HKUDS/LightRAG",
        limit=3,
    )

    assert len(result["skills"]) == 1
    assert result["skills"][0]["document_id"] == "skill-doc"
    assert result["skills"][0]["tags"] == ["skill", "agentic-development"]


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
