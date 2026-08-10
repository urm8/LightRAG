from types import SimpleNamespace

import pytest

from lightrag.llm.anthropic import anthropic_complete_if_cache


@pytest.mark.asyncio
async def test_anthropic_uses_text_after_thinking_block(monkeypatch):
    captured = {}

    class FakeClient:
        async def close(self):
            return None

        class messages:
            @staticmethod
            async def create(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(thinking="internal", type="thinking"),
                        SimpleNamespace(text="answer", type="text"),
                    ]
                )

    monkeypatch.setenv("ANTHROPIC_THINKING_MODE", "disabled")
    monkeypatch.setattr("lightrag.llm.anthropic.AsyncAnthropic", lambda **_: FakeClient())

    response = await anthropic_complete_if_cache(
        "deepseek-v4-flash", "test", api_key="test", base_url="https://example.test"
    )

    assert response == "answer"
    assert captured["thinking"] == {"type": "disabled"}
