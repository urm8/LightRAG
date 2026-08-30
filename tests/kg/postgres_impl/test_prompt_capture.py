import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.postgres_impl import TABLES, PostgreSQLDB


def _db_with_connection(connection: MagicMock) -> PostgreSQLDB:
    db = PostgreSQLDB.__new__(PostgreSQLDB)

    async def run_with_retry(operation, **_kwargs):
        await operation(connection)

    db._run_with_retry = run_with_retry
    return db


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
    connection = MagicMock()
    connection.execute = AsyncMock()
    transaction = AsyncMock()
    connection.transaction.return_value = transaction
    db = _db_with_connection(connection)

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


@pytest.mark.asyncio
async def test_table_creation_skips_read_only_age_schema() -> None:
    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value="public")
    connection.execute = AsyncMock()
    transaction = AsyncMock()
    connection.transaction.return_value = transaction
    db = _db_with_connection(connection)
    ddl = "CREATE TABLE LIGHTRAG_QUERY_PROMPT (id TEXT)"

    await db._create_table_in_writable_schema(ddl)

    connection.fetchval.assert_awaited_once()
    assert "has_schema_privilege" in connection.fetchval.await_args.args[0]
    assert connection.execute.await_args_list[0].args == (
        "SELECT set_config('search_path', quote_ident($1) || ', pg_catalog', true)",
        "public",
    )
    assert connection.execute.await_args_list[1].args == (ddl,)
    transaction.__aenter__.assert_awaited_once()
    transaction.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_table_creation_reports_missing_writable_schema() -> None:
    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value=None)
    connection.execute = AsyncMock()
    connection.transaction.return_value = AsyncMock()
    db = _db_with_connection(connection)

    with pytest.raises(PermissionError, match="no schema"):
        await db._create_table_in_writable_schema("CREATE TABLE broken (id TEXT)")

    connection.execute.assert_not_awaited()
