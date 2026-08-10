from pathlib import Path

import pytest

from scripts.switch_query_llm_profile import switch_profile


def _env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "LLM_BINDING=anthropic\n"
        "LLM_BINDING_HOST=https://api.openmodel.ai\n"
        "LLM_BINDING_API_KEY=secret\n"
        "LLM_MODEL=deepseek-v4-flash\n"
        "KEYWORD_LLM_MODEL=old-keyword\n"
        "QUERY_LLM_MODEL=old-query\n"
    )
    return path


def test_switch_profile_round_trips_deepseek_credentials(tmp_path: Path):
    path = _env_file(tmp_path)

    switch_profile(path, "mlx", "http://127.0.0.1:11436/v1", "local-gemma")
    mlx_text = path.read_text()
    assert "LLM_BINDING=openai" in mlx_text
    assert "LLM_MODEL=local-gemma" in mlx_text
    assert "LIGHTRAG_DEEPSEEK_API_KEY=secret" in mlx_text
    assert "KEYWORD_LLM_MODEL" not in mlx_text
    assert "QUERY_LLM_MODEL" not in mlx_text

    switch_profile(path, "deepseek", "unused", "unused")
    deepseek_text = path.read_text()
    assert "LLM_BINDING=anthropic" in deepseek_text
    assert "LLM_BINDING_API_KEY=secret" in deepseek_text
    assert "LLM_MODEL=deepseek-v4-flash" in deepseek_text


def test_switch_profile_requires_saved_deepseek_key(tmp_path: Path):
    path = tmp_path / ".env"
    path.write_text("LLM_BINDING=openai\n")

    with pytest.raises(ValueError, match="LIGHTRAG_DEEPSEEK_API_KEY"):
        switch_profile(path, "deepseek", "unused", "unused")
