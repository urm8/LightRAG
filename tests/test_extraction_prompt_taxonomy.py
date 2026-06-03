from lightrag.constants import DEFAULT_ENTITY_TYPES, DEFAULT_RELATION_LABELS
from lightrag.prompt import (
    ENTITY_CONTINUE_EXTRACTION_USER_PROMPT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_USER_PROMPT,
    PROMPTS,
)


def test_default_entity_taxonomy_matches_practical_graph_types() -> None:
    assert DEFAULT_ENTITY_TYPES == [
        "Person",
        "Organization",
        "Project",
        "Repository",
        "Library",
        "Framework",
        "Model",
        "Agent",
        "Tool",
        "API",
        "Service",
        "Database",
        "File",
        "Document",
        "ProgrammingLanguage",
        "Technology",
        "Method",
        "Workflow",
        "Task",
        "Concept",
        "Issue",
        "Event",
        "Metric",
        "Company",
        "Location",
    ]


def test_default_relation_labels_match_canonical_runtime_verbs() -> None:
    assert DEFAULT_RELATION_LABELS == [
        "USES",
        "DEPENDS_ON",
        "IMPLEMENTS",
        "CALLS",
        "CONNECTS_TO",
        "RUNS_ON",
        "STORES_IN",
        "DEPLOYS_TO",
        "AUTHORED_BY",
        "PART_OF",
        "RELATED_TO",
        "CAUSES",
        "MEASURES",
        "TRACKS",
        "FAILS_WITH",
        "IMPROVES",
        "REPLACES",
        "COMPETES_WITH",
    ]


def test_extraction_prompts_require_canonical_relation_labels() -> None:
    assert "Prefer exact uppercase relation labels from this set" in ENTITY_EXTRACTION_SYSTEM_PROMPT
    assert "{relation_labels}" in ENTITY_EXTRACTION_SYSTEM_PROMPT
    assert "prefer canonical relation labels" in ENTITY_EXTRACTION_USER_PROMPT
    assert "prefer canonical relation labels" in ENTITY_CONTINUE_EXTRACTION_USER_PROMPT


def test_extraction_examples_are_aligned_to_new_taxonomy() -> None:
    examples = "\n".join(PROMPTS["entity_extraction_examples"])
    assert "Project Atlas" in examples
    assert "DEPLOYS_TO" in examples
    assert "FAILS_WITH" in examples
