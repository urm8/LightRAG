import importlib
import sys
from types import SimpleNamespace

from lightrag.types import ExtractionStructuredOutput


def _load_lightrag_server(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    return importlib.import_module("lightrag.api.lightrag_server")


def test_structured_extraction_response_format_is_detected(monkeypatch):
    server = _load_lightrag_server(monkeypatch)

    assert server._is_structured_extraction_response_format(ExtractionStructuredOutput)
    assert not server._is_structured_extraction_response_format(dict)
    assert not server._is_structured_extraction_response_format(None)
    assert server._is_extraction_request(ExtractionStructuredOutput) is True
    assert server._is_extraction_request(None, forced_extraction_request=True) is True
    assert server._is_extraction_request(None) is False


def test_extraction_openai_override_uses_extraction_env(monkeypatch):
    server = _load_lightrag_server(monkeypatch)
    monkeypatch.setenv("EXTRACTION_LLM_MODEL", "granite-extract")
    monkeypatch.setenv("EXTRACTION_LLM_BINDING_HOST", "http://127.0.0.1:11436/v1")
    monkeypatch.setenv("EXTRACTION_LLM_BINDING_API_KEY", "extract-key")
    args = SimpleNamespace(
        llm_model="apple-foundationmodel",
        llm_binding_host="http://127.0.0.1:11435/v1",
        llm_binding_api_key="query-key",
    )

    model, base_url, api_key = server._get_extraction_openai_override(args)

    assert model == "granite-extract"
    assert base_url == "http://127.0.0.1:11436/v1"
    assert api_key == "extract-key"


def test_extraction_max_completion_tokens_prefers_extraction_env(monkeypatch):
    server = _load_lightrag_server(monkeypatch)
    monkeypatch.setenv("OPENAI_LLM_MAX_COMPLETION_TOKENS", "384")
    monkeypatch.setenv("MLX_OPENAI_SERVER_EXTRACTION_MAX_TOKENS", "1024")
    monkeypatch.setenv("EXTRACTION_OPENAI_LLM_MAX_COMPLETION_TOKENS", "2048")

    assert server._get_extraction_max_completion_tokens() == 2048


def test_extraction_max_async_prefers_extraction_env(monkeypatch):
    server = _load_lightrag_server(monkeypatch)
    monkeypatch.setenv("EXTRACTION_MAX_ASYNC", "1")

    assert server._get_extraction_max_async(4) == 1


def test_extraction_max_async_falls_back_to_default(monkeypatch):
    server = _load_lightrag_server(monkeypatch)
    monkeypatch.delenv("EXTRACTION_MAX_ASYNC", raising=False)

    assert server._get_extraction_max_async(4) == 4
