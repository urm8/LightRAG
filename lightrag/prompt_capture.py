"""Best-effort PostgreSQL capture for prompt tuning."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Literal

from lightrag.utils import logger


PromptKind = Literal["query", "extraction"]


def build_prompt_text(
    system_prompt: str,
    user_prompt: str,
    history_messages: list[dict[str, Any]] | None = None,
) -> str:
    """Build one stable prompt body for hashing and storage."""

    parts = []
    if system_prompt:
        parts.append(f"---System Prompt---\n{system_prompt}")
    if history_messages:
        parts.append(
            "---Conversation History---\n"
            + json.dumps(history_messages, ensure_ascii=False, default=str)
        )
    if user_prompt:
        parts.append(f"---User Prompt---\n{user_prompt}")
    return "\n\n".join(parts)


def extraction_prompt_warnings(
    output: Any,
    *,
    use_json: bool,
    entity_count: int,
    relation_count: int,
    tuple_delimiter: str,
    completion_delimiter: str,
    truncated: bool,
) -> list[str]:
    """Return compact warning classes used by prompt tuning."""

    warnings = []
    text = output if isinstance(output, str) else json.dumps(output, default=str)
    if truncated:
        warnings.append("token_limit_truncation")
    if entity_count == 0:
        warnings.append("sparse_entities")
    if relation_count == 0:
        warnings.append("sparse_relations")
    if use_json:
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            warnings.append("invalid_json")
        else:
            if not isinstance(parsed, dict):
                warnings.append("invalid_json")
        return warnings

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[-1] != completion_delimiter:
        warnings.append("completion_missing")
    for line in lines:
        if line == completion_delimiter:
            continue
        fields = line.split(tuple_delimiter)
        if fields[0] == "entity" and len(fields) != 4:
            warnings.append("entity_field_count")
        elif fields[0] == "relation":
            if len(fields) != 5:
                warnings.append("relation_field_count")
            elif not fields[3].strip():
                warnings.append("relation_missing_keyword")
    return list(dict.fromkeys(warnings))


async def record_prompt_attempt(
    global_config: dict[str, Any],
    *,
    kind: PromptKind,
    prompt_key: str,
    system_prompt: str = "",
    user_prompt: str,
    input_text: str,
    output: Any = None,
    warnings: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
    history_messages: list[dict[str, Any]] | None = None,
) -> None:
    """Persist one attempt when PostgreSQL storage is active; never break work."""

    db = global_config.get("_prompt_capture_db")
    recorder = getattr(db, "record_prompt_attempt", None)
    if not callable(recorder):
        return

    output_text = output if isinstance(output, str) else None
    if output is not None and output_text is None:
        try:
            output_text = json.dumps(output, ensure_ascii=False, default=str)
        except Exception:
            output_text = str(output)

    try:
        await recorder(
            kind=kind,
            workspace=global_config.get("workspace") or "default",
            prompt_key=prompt_key,
            prompt_text=build_prompt_text(
                system_prompt, user_prompt, history_messages=history_messages
            ),
            input_text=input_text,
            output_text=output_text,
            warnings=list(dict.fromkeys(warnings)),
            metadata=metadata or {},
        )
    except Exception as exc:
        logger.warning(
            "Prompt attempt capture failed kind=%s error=%s", kind, type(exc).__name__
        )
