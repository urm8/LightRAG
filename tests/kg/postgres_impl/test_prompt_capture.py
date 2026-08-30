import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.postgres_impl import TABLES, PostgreSQLDB


def test_prompt_attempt_tables_have_prompt_foreign_keys() -> None:
    query_ddl = TABLES["LIGHTRAG_QUERY_ATTEMPT"]["ddl"]
    extraction_ddl = TABLES["LIGHTRAG_EXTRACTION_ATTEMPT"]["ddl"]

    assert "REFERENCES LIGHTRAG_QUERY_PROMPT(workspace, id)" in query_ddl
    assert (
        "REFERENCES LIGHTRAG_EXTRACTION_PROMPT(workspace, id)" in extraction_ddl
    )
    table_order = list(TABLES)
    assert table_order.index("LIGHTRAG_QUERY_PROMPT") < table_order.index(
        "LIGHTRAG_QUERY_ATTEMPT"
    )
    assert table_order.index("LIGHTRAG_EXTRACTION_PROMPT") < table_order.index(
        "LIGHTRAG_EXTRACTION_ATTEMPT"
    )


@pytest.mark.asyncio
async def test_record_prompt_attempt_deduplicates_prompt_by_sha256() -> None:
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    connection = MagicMock()
    connection.execute = AsyncMock()
    transaction = AsyncMock()
    connection.transaction.return_value = transaction

    async def run_with_retry(operation, **_kwargs):
        await operation(connection)

    db._run_with_retry = run_with_retry

    await db.record_prompt_attempt(
        kind="query",
        workspace="project-a",
        prompt_key="rag_response",
        prompt_text="stable prompt",
        input_text="question",
        output_text="answer",
        warnings=["warning"],
        metadata={"mode": "mix"},
    )

    prompt_call, attempt_call = connection.execute.await_args_list
    expected_prompt_id = hashlib.sha256(b"stable prompt").hexdigest()
    assert "ON CONFLICT (workspace, id) DO NOTHING" in prompt_call.args[0]
    assert prompt_call.args[2] == expected_prompt_id
    assert attempt_call.args[3] == expected_prompt_id
    assert json.loads(attempt_call.args[6]) == ["warning"]
    assert json.loads(attempt_call.args[7]) == {"mode": "mix"}
    transaction.__aenter__.assert_awaited_once()
    transaction.__aexit__.assert_awaited_once()
