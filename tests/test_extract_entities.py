"""Tests for entity extraction gleaning token limit guard."""

from unittest.mock import AsyncMock, patch

import pytest

from lightrag.types import ExtractionStructuredOutput
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Simple 1:1 character-to-token mapping for testing."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_global_config(
    max_extract_input_tokens: int = 20480,
    entity_extract_max_gleaning: int = 1,
) -> dict:
    """Build a minimal global_config dict for extract_entities."""
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "llm_model_func": AsyncMock(return_value=""),
        "entity_extract_max_gleaning": entity_extract_max_gleaning,
        "addon_params": {},
        "tokenizer": tokenizer,
        "max_extract_input_tokens": max_extract_input_tokens,
        "llm_model_max_async": 1,
    }


# Minimal valid extraction result that _process_extraction_result can parse
_EXTRACTION_RESULT = (
    "(entity<|#|>TEST_ENTITY<|#|>CONCEPT<|#|>A test entity)<|COMPLETE|>"
)

_STRUCTURED_EXTRACTION_RESULT = """
{
  "entities": [
    {
      "entity_name": "LightRAG",
      "entity_type": "Project",
      "entity_description": "LightRAG is a graph RAG project."
    },
    {
      "entity_name": "FastAPI",
      "entity_type": "Framework",
      "entity_description": "FastAPI is used by LightRAG."
    }
  ],
  "relations": [
    {
      "source_entity": "LightRAG",
      "target_entity": "FastAPI",
      "relationship_keywords": "USES",
      "relationship_description": "LightRAG uses FastAPI."
    }
  ]
}
"""


def _make_chunks(content: str = "Test content.") -> dict[str, dict]:
    return {
        "chunk-001": {
            "tokens": len(content),
            "content": content,
            "full_doc_id": "doc-001",
            "chunk_order_index": 0,
        }
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_skipped_when_tokens_exceed_limit():
    """Gleaning should be skipped when estimated tokens exceed max_extract_input_tokens."""
    from lightrag.operate import extract_entities

    # Use a very small token limit so the gleaning context will exceed it
    global_config = _make_global_config(
        max_extract_input_tokens=10,
        entity_extract_max_gleaning=1,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger") as mock_logger:
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # LLM should be called exactly once (initial extraction only, no gleaning)
    assert llm_func.await_count == 1
    # Warning should be logged about skipping gleaning
    mock_logger.warning.assert_called_once()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert "Gleaning stopped" in warning_msg
    assert "exceeded limit" in warning_msg


@pytest.mark.offline
@pytest.mark.asyncio
async def test_gleaning_proceeds_when_tokens_within_limit():
    """Gleaning should proceed when estimated tokens are within max_extract_input_tokens."""
    from lightrag.operate import extract_entities

    # Use a very large token limit so gleaning will proceed
    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=1,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # LLM should be called twice (initial extraction + gleaning)
    assert llm_func.await_count == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_gleaning_when_max_gleaning_zero():
    """No gleaning when entity_extract_max_gleaning is 0, regardless of token limit."""
    from lightrag.operate import extract_entities

    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=0,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _EXTRACTION_RESULT

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    # LLM should be called exactly once (initial extraction only)
    assert llm_func.await_count == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_process_extraction_result_parses_structured_json_first():
    from lightrag.operate import _process_extraction_result

    nodes, edges = await _process_extraction_result(
        _STRUCTURED_EXTRACTION_RESULT,
        "chunk-001",
        123,
        "test.md",
    )

    assert set(nodes.keys()) == {"LightRAG", "FastAPI"}
    assert ("LightRAG", "FastAPI") in edges
    assert edges[("LightRAG", "FastAPI")][0]["keywords"] == "USES"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_requests_structured_output_contract():
    from lightrag.operate import extract_entities

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("LIGHTRAG_MANAGE_MLX_OPENAI_SERVER", "false")
    monkeypatch.setenv("EXTRACTION_LLM_BINDING_HOST", "https://api.openai.com/v1")

    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=0,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _STRUCTURED_EXTRACTION_RESULT

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    monkeypatch.undo()

    assert llm_func.await_count == 1
    assert llm_func.await_args.kwargs["response_format"] is ExtractionStructuredOutput


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_skips_api_response_format_for_managed_local_mlx(
    monkeypatch,
):
    from lightrag.operate import extract_entities

    monkeypatch.setenv("LIGHTRAG_MANAGE_MLX_OPENAI_SERVER", "true")
    monkeypatch.setenv("MLX_OPENAI_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MLX_OPENAI_SERVER_PORT", "11436")
    monkeypatch.setenv("EXTRACTION_LLM_BINDING_HOST", "http://127.0.0.1:11436/v1")

    global_config = _make_global_config(
        max_extract_input_tokens=999999,
        entity_extract_max_gleaning=0,
    )

    llm_func = global_config["llm_model_func"]
    llm_func.return_value = _STRUCTURED_EXTRACTION_RESULT

    with patch("lightrag.operate.logger"):
        await extract_entities(
            chunks=_make_chunks(),
            global_config=global_config,
        )

    assert llm_func.await_count == 1
    assert "response_format" not in llm_func.await_args.kwargs
