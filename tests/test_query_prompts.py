from lightrag.prompt import (
    AGENT_TOOL_PROTOCOL_QUERY,
    NAIVE_RAG_RESPONSE,
    RAG_RESPONSE,
)


def test_query_prompt_forces_exact_source_links() -> None:
    for prompt in (RAG_RESPONSE, NAIVE_RAG_RESPONSE):
        assert "Every grounded answer MUST include at least one reference entry." in prompt
        assert "copied from the matching entry in the `Reference Document List` exactly as written" in prompt
        assert "render it as a Markdown link using the same URL for both label and destination" in prompt
        assert "Never replace a URL with prose like \"Document Title\" or \"Source Article\"." in prompt


def test_query_tool_protocol_mentions_references_section() -> None:
    assert "always end with a `### References` section" in AGENT_TOOL_PROTOCOL_QUERY
