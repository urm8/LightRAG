from __future__ import annotations

import asyncio
import json
import os
import sys

from lightrag.operate import (
    _parse_structured_extraction_output,
    _process_extraction_result,
)
from lightrag.prompt import (
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_TUPLE_DELIMITER,
    ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_JSON_USER_PROMPT,
)
from lightrag.utils import TiktokenTokenizer


DEFAULT_ENTITY_TYPES = (
    "Person, Organization, Location, Event, Concept, Method, Content, Data, Artifact, "
    "Workspace, Project, Repository, Directory, File, ProgrammingLanguage, "
    "TechnologyStack, Framework, Library, Runtime, Service, Deployment, Environment, "
    "Configuration, Command, APIEndpoint, Database, StorageBackend, Other"
)


def fill(template: str, variables: dict) -> str:
    return (
        template.replace("{entity_types}", variables.get("entity_types") or DEFAULT_ENTITY_TYPES)
        .replace("{language}", variables.get("language") or "English")
        .replace("{tuple_delimiter}", DEFAULT_TUPLE_DELIMITER)
        .replace("{completion_delimiter}", DEFAULT_COMPLETION_DELIMITER)
        .replace("{input_text}", variables.get("input_text") or "")
        .replace("{draft_extraction}", variables.get("draft_extraction") or "")
    )


def manual_errors(
    output: str,
    parsed_output,
    node_count: int,
    edge_count: int,
    variables: dict,
) -> list[str]:
    errors: list[str] = []
    stripped = output.strip()

    if parsed_output is None:
        errors.append("structured_parse_failed")
        if stripped.startswith("{") and not stripped.endswith("}"):
            errors.append("length_truncated_suspected")
    if variables.get("disallow_legacy_fallback") and DEFAULT_TUPLE_DELIMITER in output:
        errors.append("legacy_output_detected")

    expected_min_entities = int(variables.get("expected_min_entities") or 0)
    expected_min_relations = int(variables.get("expected_min_relations") or 0)
    if expected_min_entities and node_count < expected_min_entities:
        errors.append(
            f"entity_count_below_expected:{node_count}<{expected_min_entities}"
        )
    if expected_min_relations and edge_count < expected_min_relations:
        errors.append(
            f"relation_count_below_expected:{edge_count}<{expected_min_relations}"
        )

    return errors


async def main() -> None:
    payload = json.load(sys.stdin)
    output = payload.get("output") or ""
    variables = payload.get("vars") or {}

    system_prompt = fill(ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT, variables)
    user_prompt = fill(ENTITY_EXTRACTION_JSON_USER_PROMPT, variables)

    tokenizer = TiktokenTokenizer()
    input_tokens = len(tokenizer.encode(system_prompt + user_prompt)) + 10
    input_budget = int(
        variables.get("input_budget")
        or os.getenv("OPENAI_LLM_INPUT_TOKEN_BUDGET")
        or 3072
    )

    parsed_output = _parse_structured_extraction_output(output)
    nodes, edges = await _process_extraction_result(
        parsed_output or output,
        "promptfoo",
        0,
        "promptfoo",
        tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
        completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
    )

    result = {
        "input_tokens": input_tokens,
        "input_budget": input_budget,
        "token_budget_ok": input_tokens <= input_budget,
        "warning_classes": [],
        "warnings": [],
        "manual_errors": manual_errors(
            output,
            parsed_output,
            len(nodes),
            len(edges),
            variables,
        ),
        "structured_parse_ok": parsed_output is not None,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
