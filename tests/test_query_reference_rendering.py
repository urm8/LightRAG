import sys


def _query_routes_module(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    from lightrag.api.routers import query_routes

    return query_routes


def test_canonical_references_replace_model_mangled_sources(monkeypatch):
    query_routes = _query_routes_module(monkeypatch)

    response = """**What’s New on Hacker News?**

1. Item one

### References

- [1] [arstechnica.com](https://news.ycombinator.com/from?site=arstechnica.com)
"""
    references = [
        {
            "reference_id": "2",
            "file_path": "https://news.ycombinator.com/item?id=48199314",
        }
    ]

    updated = query_routes._ensure_response_has_canonical_references(
        response, references
    )

    assert "from?site=arstechnica.com" not in updated
    assert (
        "[https://news.ycombinator.com/item?id=48199314]"
        "(https://news.ycombinator.com/item?id=48199314)" in updated
    )
    assert updated.count("### References") == 1


def test_canonical_references_append_when_missing(monkeypatch):
    query_routes = _query_routes_module(monkeypatch)

    response = "Short answer."
    references = [
        {"reference_id": "1", "file_path": "https://example.com/a"},
        {"reference_id": "2", "file_path": "/tmp/local-note.md"},
    ]

    updated = query_routes._ensure_response_has_canonical_references(
        response, references
    )

    assert updated.startswith("Short answer.")
    assert "### References" in updated
    assert "- [1] [https://example.com/a](https://example.com/a)" in updated
    assert "- [2] /tmp/local-note.md" in updated


def test_canonical_references_replace_inline_mangled_section(monkeypatch):
    query_routes = _query_routes_module(monkeypatch)

    response = "### Answer\nok\n\n### References - [n] [url](url)"
    references = [{"reference_id": "1", "file_path": "https://example.com/source"}]

    updated = query_routes._ensure_response_has_canonical_references(
        response, references
    )

    assert "[url](url)" not in updated
    assert "- [1] [https://example.com/source](https://example.com/source)" in updated


def test_references_from_entities_and_relationships(monkeypatch):
    query_routes = _query_routes_module(monkeypatch)

    references = query_routes._references_from_query_data(
        {
            "references": [],
            "chunks": [],
            "entities": [
                {
                    "entity_name": "LightRAG",
                    "file_path": "https://github.com/HKUDS/LightRAG<SEP>unknown_source",
                }
            ],
            "relationships": [
                {
                    "src_id": "LightRAG",
                    "tgt_id": "Apfel",
                    "file_path": "codex-memory/query-split.md",
                }
            ],
        }
    )

    assert references == [
        {"reference_id": "1", "file_path": "https://github.com/HKUDS/LightRAG"},
        {"reference_id": "2", "file_path": "codex-memory/query-split.md"},
    ]


def test_references_prefer_final_chunks(monkeypatch):
    query_routes = _query_routes_module(monkeypatch)

    references = query_routes._references_from_query_data(
        {
            "references": [
                {"reference_id": "1", "file_path": "https://irrelevant.example.com"}
            ],
            "chunks": [
                {"reference_id": "7", "file_path": "https://relevant.example.com"}
            ],
            "entities": [
                {"entity_name": "Other", "file_path": "https://entity.example.com"}
            ],
        }
    )

    assert references == [
        {"reference_id": "7", "file_path": "https://relevant.example.com"}
    ]
