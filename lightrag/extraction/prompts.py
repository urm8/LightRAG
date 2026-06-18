from __future__ import annotations

from lightrag.extraction.context import ExtractionPromptContext


ENTITY_EXTRACTION_SYSTEM_PROMPT = """You extract graph-ready entities and relations for a local RAG system.

You must call the `submit_extraction` tool. Do not answer with prose.

Rules:
1. Extract stable, reusable entities from the provided chunk only.
2. Use only these entity types: {entity_types}.
3. Prefer these canonical relation labels: {relation_labels}. If none fits, use RELATED_TO.
4. Keep descriptions short, factual, and explicit. Never omit descriptions.
5. Avoid duplicates, self-relations, speculation, and cross-chunk assumptions.
6. If no useful graph data exists, return empty entities and relations arrays.
7. Output language: {language}.
8. Keep the result within {max_total_records} total records and {max_entity_records} entities."""


ENTITY_EXTRACTION_USER_PROMPT = """Extract entities and relations from this chunk.

Chunk text:
```text
{input_text}
```"""


ENTITY_EXTRACTION_CONTINUE_USER_PROMPT = """Review the same chunk again and use `submit_extraction` to return only entities or relations that were missed previously. If nothing is missing, return empty arrays."""


def build_system_prompt(context: ExtractionPromptContext) -> str:
    return ENTITY_EXTRACTION_SYSTEM_PROMPT.format(**context.prompt_vars())


def build_initial_user_prompt(
    context: ExtractionPromptContext,
    *,
    input_text: str,
) -> str:
    return ENTITY_EXTRACTION_USER_PROMPT.format(
        **context.prompt_vars(),
        input_text=input_text,
    )


def build_continue_prompt(context: ExtractionPromptContext) -> str:
    return ENTITY_EXTRACTION_CONTINUE_USER_PROMPT.format(**context.prompt_vars())
