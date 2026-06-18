import sys


def _query_request_cls(monkeypatch):
    # The API config parser consumes sys.argv during router import.
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lightrag.api.routers.query_routes import QueryRequest

    return QueryRequest


def test_query_request_clamps_to_local_retrieval_context(monkeypatch):
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_CONTEXT_LENGTH", "8192")
    monkeypatch.setenv("OPENAI_LLM_MAX_COMPLETION_TOKENS", "1024")

    QueryRequest = _query_request_cls(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated")
    monkeypatch.setenv("LLM_BINDING_HOST", "http://127.0.0.1:11436/v1")
    param = QueryRequest(
        query="what's new on hackernews",
        mode="global",
        top_k=40,
        chunk_top_k=20,
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        max_total_tokens=30000,
    ).to_query_params(is_stream=True)

    assert param.max_total_tokens == int(8192 * 0.8)
    assert param.max_entity_tokens + param.max_relation_tokens <= int(
        param.max_total_tokens * 0.6
    )
    assert param.stream is True


def test_query_request_keeps_small_budget(monkeypatch):
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_CONTEXT_LENGTH", "8192")
    monkeypatch.setenv("OPENAI_LLM_MAX_COMPLETION_TOKENS", "1024")

    QueryRequest = _query_request_cls(monkeypatch)
    monkeypatch.setenv("LLM_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated")
    monkeypatch.setenv("LLM_BINDING_HOST", "http://127.0.0.1:11436/v1")
    param = QueryRequest(
        query="what's new on hackernews",
        mode="global",
        max_entity_tokens=500,
        max_relation_tokens=700,
        max_total_tokens=3000,
    ).to_query_params(is_stream=False)

    assert param.max_total_tokens == 3000
    assert param.max_entity_tokens == 500
    assert param.max_relation_tokens == 700
    assert param.stream is False


def test_query_request_apfel_fast_defaults_fit_4k_window(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "apple-foundationmodel")
    monkeypatch.setenv("LLM_BINDING_HOST", "http://127.0.0.1:11435/v1")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_CONTEXT_LENGTH", "8192")
    monkeypatch.setenv("OPENAI_LLM_MAX_COMPLETION_TOKENS", "1024")

    QueryRequest = _query_request_cls(monkeypatch)
    param = QueryRequest(
        query="what changed in LightRAG query routing",
        mode="mix",
        top_k=60,
        chunk_top_k=20,
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        max_total_tokens=30000,
        conversation_history=[
            {"role": "user", "content": "old question" * 500},
            {"role": "assistant", "content": "old answer" * 500},
        ],
        enable_rerank=True,
    ).to_query_params(is_stream=True)

    assert param.top_k == 8
    assert param.chunk_top_k == 4
    assert param.max_total_tokens == 2600
    assert param.max_entity_tokens == 350
    assert param.max_relation_tokens == 450
    assert param.response_type == "Short bullet list"
    assert param.conversation_history == []
    assert param.enable_rerank is False
    assert param.enable_agent_tools is False
    assert param.max_completion_tokens == 384


def test_query_request_granite_profile_bypasses_apfel_fast_defaults(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "apple-foundationmodel")
    monkeypatch.setenv("LLM_BINDING_HOST", "http://127.0.0.1:11435/v1")
    monkeypatch.setenv("LIGHTRAG_GRANITE_QUERY_MODEL", "granite-query")
    monkeypatch.setenv("LIGHTRAG_GRANITE_QUERY_BINDING_HOST", "http://127.0.0.1:11436/v1")
    monkeypatch.setenv("WEBUI_QUERY_ENRICHMENT_TOP_K", "24")
    monkeypatch.setenv("WEBUI_QUERY_ENRICHMENT_CHUNK_TOP_K", "8")
    monkeypatch.setenv("WEBUI_QUERY_ENRICHMENT_MAX_TOTAL_TOKENS", "6500")
    monkeypatch.setenv("WEBUI_QUERY_ENRICHMENT_MAX_ENTITY_TOKENS", "1600")
    monkeypatch.setenv("WEBUI_QUERY_ENRICHMENT_MAX_RELATION_TOKENS", "2200")
    monkeypatch.setenv("WEBUI_QUERY_ENRICHMENT_MAX_COMPLETION_TOKENS", "1024")
    monkeypatch.setenv("WEBUI_QUERY_ENRICHMENT_ENABLE_RERANK", "true")
    monkeypatch.setenv("LIGHTRAG_RERANK_ENABLED", "true")

    QueryRequest = _query_request_cls(monkeypatch)
    param = QueryRequest(
        query="what changed in LightRAG query routing",
        query_model="granite",
        mode="mix",
        top_k=60,
        chunk_top_k=20,
        max_entity_tokens=6000,
        max_relation_tokens=8000,
        max_total_tokens=30000,
        conversation_history=[{"role": "user", "content": "keep this"}],
        enable_rerank=False,
    ).to_query_params(is_stream=False)

    assert param.model_func is not None
    assert param.top_k == 24
    assert param.chunk_top_k == 8
    assert param.max_total_tokens == 6500
    assert param.max_entity_tokens == 1600
    assert param.max_relation_tokens == 2200
    assert param.conversation_history == [{"role": "user", "content": "keep this"}]
    assert param.response_type == "Multiple Paragraphs"
    assert param.enable_rerank is True
    assert param.max_completion_tokens == 1024
