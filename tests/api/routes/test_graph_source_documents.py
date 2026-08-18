"""Graph provenance routes resolve source chunks to viewable documents."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline


def test_entity_source_documents_resolve_chunks_to_document():
    rag = SimpleNamespace(
        chunk_entity_relation_graph=SimpleNamespace(
            get_node=AsyncMock(return_value={"source_id": "chunk-a"})
        ),
        entity_chunks=SimpleNamespace(
            get_by_id=AsyncMock(return_value={"chunk_ids": ["chunk-a", "chunk-b"]})
        ),
        text_chunks=SimpleNamespace(
            get_by_id=AsyncMock(
                side_effect=[
                    {"full_doc_id": "doc-1", "content": "first excerpt"},
                    {"full_doc_id": "doc-1", "content": "second excerpt"},
                ]
            )
        ),
        doc_status=SimpleNamespace(
            get_by_id=AsyncMock(
                return_value=SimpleNamespace(
                    file_path="skill.md",
                    status="processed",
                    metadata={"tags": ["skill", "agentic-development"]},
                )
            )
        ),
    )
    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag))

    response = TestClient(app).get(
        "/graph/entity/source-documents", params={"name": "LightRAG"}
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "id": "doc-1",
            "file_path": "skill.md",
            "status": "processed",
            "tags": ["skill", "agentic-development"],
            "chunk_ids": ["chunk-a", "chunk-b"],
            "excerpts": ["first excerpt", "second excerpt"],
        }
    ]


def test_entity_source_documents_returns_404_for_missing_entity():
    rag = SimpleNamespace(
        chunk_entity_relation_graph=SimpleNamespace(get_node=AsyncMock(return_value=None))
    )
    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag))

    response = TestClient(app).get(
        "/graph/entity/source-documents", params={"name": "missing"}
    )

    assert response.status_code == 404


def test_document_content_is_loaded_on_demand():
    rag = SimpleNamespace(
        full_docs=SimpleNamespace(
            get_by_id=AsyncMock(return_value={"content": "complete skill document"})
        ),
        doc_status=SimpleNamespace(
            get_by_id=AsyncMock(
                return_value=SimpleNamespace(
                    file_path="skill.md", metadata={"tags": ["skill"]}
                )
            )
        ),
    )
    app = FastAPI()
    app.include_router(
        document_routes.create_document_routes(rag, SimpleNamespace())
    )

    response = TestClient(app).get("/documents/doc-1/content")

    assert response.status_code == 200
    assert response.json() == {
        "id": "doc-1",
        "file_path": "skill.md",
        "content": "complete skill document",
        "metadata": {"tags": ["skill"]},
    }


def test_document_content_falls_back_to_ordered_chunks():
    rag = SimpleNamespace(
        full_docs=SimpleNamespace(get_by_id=AsyncMock(return_value={"content": ""})),
        text_chunks=SimpleNamespace(
            get_by_ids=AsyncMock(
                return_value=[
                    {"content": "second", "chunk_order_index": 1},
                    {"content": "first", "chunk_order_index": 0},
                ]
            )
        ),
        doc_status=SimpleNamespace(
            get_by_id=AsyncMock(
                return_value=SimpleNamespace(
                    file_path="workflow.md",
                    metadata={"tags": ["workflow"]},
                    chunks_list=["chunk-2", "chunk-1"],
                )
            )
        ),
    )
    app = FastAPI()
    app.include_router(
        document_routes.create_document_routes(rag, SimpleNamespace())
    )

    response = TestClient(app).get("/documents/doc-2/content")

    assert response.status_code == 200
    assert response.json()["content"] == "first\n\nsecond"
