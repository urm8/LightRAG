import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag.prompt import CAVEMAN_SYSTEM_PROMPT, PROMPTS
from lightrag.prompt_capture import (
    build_prompt_text,
    extraction_prompt_warnings,
    record_prompt_attempt,
)


@pytest.mark.parametrize(
    "prompt_key",
    [
        "entity_extraction_json_system_prompt",
        "entity_extraction_user_prompt",
        "rag_response",
        "naive_rag_response",
    ],
)
def test_caveman_rules_are_in_requested_prompts(prompt_key: str) -> None:
    assert PROMPTS[prompt_key].startswith(CAVEMAN_SYSTEM_PROMPT)


def test_extraction_warnings_classify_invalid_json_and_sparse_output() -> None:
    assert extraction_prompt_warnings(
        "not json",
        use_json=True,
        entity_count=0,
        relation_count=0,
        tuple_delimiter="<|>",
        completion_delimiter="<END>",
        truncated=True,
    ) == [
        "token_limit_truncation",
        "sparse_entities",
        "sparse_relations",
        "invalid_json",
    ]


def test_prompt_bundle_includes_conversation_history() -> None:
    prompt = build_prompt_text(
        "system", "user", history_messages=[{"role": "user", "content": "before"}]
    )

    assert '---Conversation History---\n[{"role": "user", "content": "before"}]' in prompt


@pytest.mark.asyncio
async def test_record_prompt_attempt_forwards_stable_prompt_bundle() -> None:
    recorder = AsyncMock()
    config = {
        "workspace": "project-a",
        "_prompt_capture_db": SimpleNamespace(record_prompt_attempt=recorder),
    }

    await record_prompt_attempt(
        config,
        kind="query",
        prompt_key="rag_response",
        system_prompt="system",
        user_prompt="user",
        input_text="question",
        output={"answer": "yes"},
        warnings=["empty_response", "empty_response"],
        metadata={"mode": "mix"},
    )

    recorder.assert_awaited_once_with(
        kind="query",
        workspace="project-a",
        prompt_key="rag_response",
        prompt_text=build_prompt_text("system", "user"),
        input_text="question",
        output_text=json.dumps({"answer": "yes"}, ensure_ascii=False),
        warnings=["empty_response"],
        metadata={"mode": "mix"},
    )
