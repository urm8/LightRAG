from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from lightrag.utils import use_llm_func_with_cache


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache_for_entity_extract": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_partitions_cache_by_response_format():
    cache = _FakeKVStorage()
    llm_func = AsyncMock(side_effect=["plain-text", '{"answer":"json"}'])

    plain_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
    )
    json_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        response_format={"type": "json_object"},
    )

    assert plain_result == "plain-text"
    assert json_result == '{"answer":"json"}'
    assert llm_func.await_count == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_accepts_typed_response_format_classes():
    class _StructuredAnswer(BaseModel):
        answer: str

    cache = _FakeKVStorage()
    llm_func = AsyncMock(return_value='{"answer":"ok"}')

    first_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        response_format=_StructuredAnswer,
    )
    second_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        response_format=_StructuredAnswer,
    )

    assert first_result == '{"answer":"ok"}'
    assert second_result == '{"answer":"ok"}'
    assert llm_func.await_count == 1
    assert len(cache._store) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_partitions_cache_by_llm_identity():
    cache = _FakeKVStorage()
    llm_func = AsyncMock(side_effect=["model-a", "model-b"])

    first_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        llm_cache_identity={
            "role": "query",
            "binding": "openai",
            "model": "model-a",
            "host": "https://api.example.com/v1",
        },
    )
    second_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        llm_cache_identity={
            "role": "query",
            "binding": "openai",
            "model": "model-b",
            "host": "https://api.example.com/v1",
        },
    )

    assert first_result == "model-a"
    assert second_result == "model-b"
    assert llm_func.await_count == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_rejects_json_schema_response_format():
    llm_func = AsyncMock()

    with pytest.raises(ValueError, match="json_schema"):
        await use_llm_func_with_cache(
            "same prompt",
            llm_func,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_payload",
                    "schema": {"type": "object"},
                },
            },
        )

    llm_func.assert_not_awaited()
