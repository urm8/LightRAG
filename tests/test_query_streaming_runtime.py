from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.offline
@pytest.mark.asyncio
async def test_run_tool_loop_emits_stream_events():
    from lightrag.base import QueryParam
    from lightrag.operate import _run_tool_loop

    events: list[dict] = []

    responses = iter(
        [
            '<tool_call>{"tool":"search_entities","args":{"query":"LightRAG","top_k":1}}</tool_call>',
            "Final grounded answer.",
        ]
    )

    async def use_model_func(*_args, **_kwargs):
        return next(responses)

    class _EntitiesVDB:
        async def query(self, query: str, top_k: int = 10):
            return [{"entity_name": "LightRAG", "description": f"query={query}", "top_k": top_k}]

    result = await _run_tool_loop(
        query="What is LightRAG?",
        system_prompt="tool prompt",
        use_model_func=use_model_func,
        query_param=QueryParam(stream=True, stream_event_callback=events.append),
        entities_vdb=_EntitiesVDB(),
        relationships_vdb=None,
        chunks_vdb=None,
        knowledge_graph_inst=None,
    )

    assert result == "Final grounded answer."
    assert [event["phase"] for event in events] == ["tool_call", "tool_result"]
    assert events[0]["tool"] == "search_entities"
    assert "LightRAG" in events[1]["output"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pick_by_vector_similarity_keeps_available_vectors():
    from lightrag.utils import pick_by_vector_similarity

    class _ChunksVDB:
        async def get_vectors_by_ids(self, _ids):
            return {
                "chunk-a": np.array([1.0, 0.0], dtype=np.float32),
                "chunk-c": np.array([0.5, 0.0], dtype=np.float32),
            }

    async def embedding_func(_texts, context=None):
        assert context == "query"
        return np.array([[1.0, 0.0]], dtype=np.float32)

    selected = await pick_by_vector_similarity(
        query="employment status",
        text_chunks_storage=None,
        chunks_vdb=_ChunksVDB(),
        num_of_chunks=2,
        entity_info=[
            {"sorted_chunks": ["chunk-a", "chunk-b"]},
            {"sorted_chunks": ["chunk-c"]},
        ],
        embedding_func=embedding_func,
    )

    assert selected == ["chunk-a", "chunk-c"]
