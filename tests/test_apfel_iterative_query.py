import pytest

from lightrag.query.apfel import (
    _run_apfel_iterative_answer_generator,
)
from lightrag.base import QueryParam
from lightrag.query.apfel import _build_apfel_iterative_portions, _parse_apfel_iterative_response


class WordTokenizer:
    def encode(self, content: str):
        return (content or "").split()

    def decode(self, tokens):
        return " ".join(tokens)


def _raw_query_data():
    return {
        "data": {
            "entities": [
                {
                    "entity_name": "LightRAG",
                    "entity_type": "Project",
                    "description": "Local graph RAG service",
                    "source_id": "1",
                    "file_path": "project.md",
                },
                {
                    "entity_name": "Apfel",
                    "entity_type": "Model",
                    "description": "Fast Apple local query model",
                    "source_id": "2",
                    "file_path": "apfel.md",
                },
            ],
            "relationships": [
                {
                    "src_id": "LightRAG",
                    "tgt_id": "Apfel",
                    "keywords": "USES",
                    "description": "LightRAG uses Apfel for fast query synthesis",
                    "source_id": "1",
                    "file_path": "project.md",
                }
            ],
            "chunks": [
                {
                    "reference_id": "1",
                    "file_path": "project.md",
                    "chunk_id": "chunk-a",
                    "content": "LightRAG streams Apfel query answer deltas.",
                },
                {
                    "reference_id": "2",
                    "file_path": "apfel.md",
                    "chunk_id": "chunk-b",
                    "content": "Apfel requires small context windows and compact prompts.",
                },
            ],
            "references": [
                {"reference_id": "1", "file_path": "project.md"},
                {"reference_id": "2", "file_path": "apfel.md"},
            ],
        },
        "metadata": {},
    }


def test_apfel_iterative_portions_cover_all_query_data():
    portions = _build_apfel_iterative_portions(
        _raw_query_data(), WordTokenizer(), max_total_tokens=80
    )

    joined = "\n".join(portion["context"] for portion in portions)

    assert "CHUNK ref=[1]" in joined
    assert "CHUNK ref=[2]" in joined
    assert "RELATION LightRAG -> Apfel" in joined
    assert "ENTITY LightRAG" in joined
    assert "ENTITY Apfel" in joined
    assert sum(portion["counts"]["chunks"] for portion in portions) == 2
    assert sum(portion["counts"]["relationships"] for portion in portions) == 1
    assert sum(portion["counts"]["entities"] for portion in portions) == 2
    assert all(portion["token_estimate"] <= int(80 * 0.72) for portion in portions)


def test_apfel_iterative_response_parser_removes_carry_and_references():
    answer, carry = _parse_apfel_iterative_response(
        """### Answer Delta
- LightRAG uses Apfel for fast answers.

### Carry Summary
LightRAG uses Apfel.

### References
- [1] wrong"""
    )

    assert answer == "- LightRAG uses Apfel for fast answers."
    assert carry == "LightRAG uses Apfel."
    assert "References" not in answer


def test_apfel_iterative_response_parser_drops_empty_filler():
    answer, carry = _parse_apfel_iterative_response(
        "### Answer Delta\n* Nothing new on Hacker News.\n### Carry Summary\nsame"
    )

    assert answer == ""
    assert carry == "same"


@pytest.mark.asyncio
async def test_apfel_iterative_generator_streams_deltas_and_final():
    portions = [
        {
            "context": "Document Chunks:\nCHUNK ref=[1]\nfirst",
            "token_estimate": 8,
            "counts": {"chunks": 1, "relationships": 0, "entities": 0},
        },
        {
            "context": "Document Chunks:\nCHUNK ref=[2]\nsecond",
            "token_estimate": 8,
            "counts": {"chunks": 1, "relationships": 0, "entities": 0},
        },
    ]
    responses = iter(
        [
            "### Answer Delta\n- First portion.\n### Carry Summary\nfirst",
            "### Answer Delta\n- Second portion.\n### Carry Summary\nsecond",
            "### Answer Delta\nCombined answer.\n### Carry Summary\ndone",
        ]
    )

    async def model_func(*args, **kwargs):
        return next(responses)

    param = QueryParam(stream=True)
    param.max_completion_tokens = 64
    raw_data = {"metadata": {}}
    chunks = [
        chunk
        async for chunk in _run_apfel_iterative_answer_generator(
            query="what changed",
            response_type="Short bullet list",
            query_param=param,
            use_model_func=model_func,
            tokenizer=WordTokenizer(),
            portions=portions,
            raw_data=raw_data,
        )
    ]

    assert chunks == [
        "- First portion.\n\n",
        "- Second portion.\n\n",
        "### Final\n\nCombined answer.",
    ]
    assert raw_data["metadata"]["apfel_iterative"]["current_portion"] == 2


@pytest.mark.asyncio
async def test_apfel_iterative_generator_falls_back_to_context_excerpt():
    portions = [
        {
            "context": "Document Chunks:\n\nCHUNK ref=[1] source=note.md\nUseful retrieved fact.",
            "token_estimate": 8,
            "counts": {"chunks": 1, "relationships": 0, "entities": 0},
        }
    ]

    async def model_func(*args, **kwargs):
        return "### Answer Delta\nNothing new.\n### Carry Summary\nnone"

    param = QueryParam(stream=True)
    param.max_completion_tokens = 64
    chunks = [
        chunk
        async for chunk in _run_apfel_iterative_answer_generator(
            query="what changed",
            response_type="Short bullet list",
            query_param=param,
            use_model_func=model_func,
            tokenizer=WordTokenizer(),
            portions=portions,
            raw_data={"metadata": {}},
        )
    ]

    assert chunks == ["Useful retrieved fact."]
