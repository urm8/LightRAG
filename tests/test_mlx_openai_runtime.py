from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from lightrag.llm import openai as openai_llm


class _StructuredSmokeOutput(BaseModel):
    status: str


@pytest.fixture(autouse=True)
def _disable_native_swift_sidecars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIGHTRAG_MANAGE_SWIFT_LM", "false")
    monkeypatch.setenv("LIGHTRAG_MANAGE_SWIFT_EMBEDDINGS", "false")


@pytest.mark.offline
def test_build_mlx_openai_server_config_contains_expected_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("MLX_OPENAI_SERVER_QUEUE_TIMEOUT_S", "")
    from lightrag.api.lightrag_server import _build_mlx_openai_server_config

    retrieval_path = tmp_path / "agentcpm"
    extraction_path = tmp_path / "granite"
    embedding_path = tmp_path / "bge"
    retrieval_path.mkdir()
    extraction_path.mkdir()
    embedding_path.mkdir()

    monkeypatch.setenv("MLX_OPENAI_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MLX_OPENAI_SERVER_PORT", "11436")
    monkeypatch.setenv("MLX_AGENTCPM_MODEL", "openbmb/AgentCPM-Explore")
    monkeypatch.setenv("MLX_AGENTCPM_MODEL_PATH", str(retrieval_path))
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_CONTEXT_LENGTH", "12288")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_DISABLE_BATCHING", "true")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_DECODE_CONCURRENCY", "4")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_PROMPT_CONCURRENCY", "1")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_PREFILL_STEP_SIZE", "2048")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_PROMPT_CACHE_SIZE", "2")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_PROMPT_CACHE_MAX_BYTES", "4294967296")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_TEMPERATURE", "0.1")
    monkeypatch.setenv(
        "MLX_EXTRACTION_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )
    monkeypatch.setenv("MLX_EXTRACTION_MODEL_PATH", str(extraction_path))
    monkeypatch.setenv("MLX_OPENAI_SERVER_EXTRACTION_TEMPERATURE", "0.0")
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL", "mlx-community/bge-m3-mlx-4bit")
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL_PATH", str(embedding_path))
    monkeypatch.setenv("MLX_OPENAI_SERVER_IDLE_TIMEOUT_S", "180")
    monkeypatch.setenv("MLX_OPENAI_SERVER_EMBEDDINGS_IDLE_TIMEOUT_S", "240")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_ON_DEMAND", "true")
    monkeypatch.setenv("MLX_OPENAI_SERVER_EXTRACTION_ON_DEMAND", "true")
    monkeypatch.setenv("MLX_OPENAI_SERVER_EMBEDDINGS_ON_DEMAND", "true")
    monkeypatch.setenv("LLM_TIMEOUT", "900")
    monkeypatch.setenv("EMBEDDING_TIMEOUT", "180")

    args = SimpleNamespace(
        llm_model="openbmb/AgentCPM-Explore",
        embedding_model="mlx-community/bge-m3-mlx-4bit",
    )
    extraction_args = SimpleNamespace(
        llm_model="huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )

    config = _build_mlx_openai_server_config(args, extraction_args)

    assert config["server"] == {
        "host": "127.0.0.1",
        "port": 11436,
        "log_level": "INFO",
    }
    models = {
        model["served_model_name"]: model
        for model in config["models"]  # type: ignore[index]
    }
    assert set(models) == {
        "openbmb/AgentCPM-Explore",
        "huihui-ai/Huihui-granite-4.1-3b-abliterated",
        "mlx-community/bge-m3-mlx-4bit",
    }
    retrieval = models["openbmb/AgentCPM-Explore"]
    extraction = models["huihui-ai/Huihui-granite-4.1-3b-abliterated"]
    assert retrieval["model_type"] == "lm"
    assert retrieval["context_length"] == 12288
    assert retrieval["batch_completion_size"] == 4
    assert retrieval["batch_prefill_size"] == 1
    assert retrieval["batch_prefill_step_size"] == 2048
    assert retrieval["prompt_cache_size"] == 2
    assert retrieval["prompt_cache_max_bytes"] == 4294967296
    assert retrieval["default_temperature"] == 0.1
    assert retrieval["disable_batching"] is True
    assert retrieval["on_demand"] is True
    assert retrieval["queue_timeout"] == 900
    assert models["huihui-ai/Huihui-granite-4.1-3b-abliterated"][
        "default_max_tokens"
    ] == 2048
    assert extraction["default_temperature"] == 0.0
    assert extraction["on_demand"] is True
    assert extraction["queue_timeout"] == 900
    assert models["mlx-community/bge-m3-mlx-4bit"]["model_type"] == "embeddings"
    assert models["mlx-community/bge-m3-mlx-4bit"]["on_demand"] is True
    assert models["mlx-community/bge-m3-mlx-4bit"]["queue_timeout"] == 180
    assert retrieval["on_demand_idle_timeout"] == 180
    assert extraction["on_demand_idle_timeout"] == 180
    assert models["mlx-community/bge-m3-mlx-4bit"]["on_demand_idle_timeout"] == 240


@pytest.mark.offline
def test_build_mlx_openai_server_config_prefers_role_specific_queue_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _build_mlx_openai_server_config

    shared_path = tmp_path / "qwen"
    embedding_path = tmp_path / "bge"
    shared_path.mkdir()
    embedding_path.mkdir()

    monkeypatch.setenv(
        "MLX_AGENTCPM_MODEL",
        "osmapi/Qwen3.6-27B-Claude-Opus-Reasoning-Distill-v2-abliterated-OptiQ-3.7bpw-mlx",
    )
    monkeypatch.setenv("MLX_AGENTCPM_MODEL_PATH", str(shared_path))
    monkeypatch.setenv(
        "MLX_EXTRACTION_MODEL",
        "osmapi/Qwen3.6-27B-Claude-Opus-Reasoning-Distill-v2-abliterated-OptiQ-3.7bpw-mlx",
    )
    monkeypatch.setenv("MLX_EXTRACTION_MODEL_PATH", str(shared_path))
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL", "mlx-community/bge-m3-mlx-4bit")
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL_PATH", str(embedding_path))
    monkeypatch.setenv("MLX_OPENAI_SERVER_QUEUE_TIMEOUT_S", "600")
    monkeypatch.setenv("MLX_OPENAI_SERVER_EXTRACTION_QUEUE_TIMEOUT_S", "1200")

    args = SimpleNamespace(
        llm_model="osmapi/Qwen3.6-27B-Claude-Opus-Reasoning-Distill-v2-abliterated-OptiQ-3.7bpw-mlx",
        embedding_model="mlx-community/bge-m3-mlx-4bit",
    )
    extraction_args = SimpleNamespace(
        llm_model="osmapi/Qwen3.6-27B-Claude-Opus-Reasoning-Distill-v2-abliterated-OptiQ-3.7bpw-mlx"
    )

    config = _build_mlx_openai_server_config(args, extraction_args)
    models = {
        model["served_model_name"]: model
        for model in config["models"]  # type: ignore[index]
    }

    assert models[
        "osmapi/Qwen3.6-27B-Claude-Opus-Reasoning-Distill-v2-abliterated-OptiQ-3.7bpw-mlx"
    ]["queue_timeout"] == 1200
    assert models["mlx-community/bge-m3-mlx-4bit"]["queue_timeout"] == 600


@pytest.mark.offline
def test_find_managed_mlx_openai_processes_matches_wrapper_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _find_managed_mlx_openai_processes

    config_path = tmp_path / "mlx-openai-server-config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    config_marker = str(config_path)
    snapshot = {
        101: {
            "pid": 101,
            "ppid": 1,
            "command": (
                "/tmp/python -m lightrag.api.mlx_openai_server_wrapper launch "
                f"--config {config_marker}"
            ),
        },
        102: {
            "pid": 102,
            "ppid": 1,
            "command": (
                "uv run --with mlx-openai-server==1.8.1 python -m "
                f"lightrag.api.mlx_openai_server_wrapper launch --config {config_marker}"
            ),
        },
        103: {
            "pid": 103,
            "ppid": 1,
            "command": f"/tmp/python other_server.py --config {config_marker}",
        },
    }

    monkeypatch.setattr(
        "lightrag.api.lightrag_server._collect_process_snapshot",
        lambda: snapshot,
    )

    matches = _find_managed_mlx_openai_processes(config_path)

    assert [proc["pid"] for proc in matches] == [101, 102]


@pytest.mark.offline
def test_runtime_data_dir_honors_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    runtime_dir = tmp_path / "runtime-data"
    monkeypatch.setenv("LIGHTRAG_RUNTIME_DATA_DIR", str(runtime_dir))

    from lightrag.api.lightrag_server import _runtime_data_dir

    assert _runtime_data_dir() == runtime_dir
    assert runtime_dir.exists()


@pytest.mark.offline
def test_build_mlx_openai_server_config_respects_role_specific_on_demand_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _build_mlx_openai_server_config

    retrieval_path = tmp_path / "agentcpm"
    extraction_path = tmp_path / "granite"
    embedding_path = tmp_path / "bge"
    retrieval_path.mkdir()
    extraction_path.mkdir()
    embedding_path.mkdir()

    monkeypatch.setenv("MLX_AGENTCPM_MODEL", "openbmb/AgentCPM-Explore")
    monkeypatch.setenv("MLX_AGENTCPM_MODEL_PATH", str(retrieval_path))
    monkeypatch.setenv(
        "MLX_EXTRACTION_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )
    monkeypatch.setenv("MLX_EXTRACTION_MODEL_PATH", str(extraction_path))
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL", "mlx-community/bge-m3-mlx-4bit")
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL_PATH", str(embedding_path))
    monkeypatch.setenv("MLX_OPENAI_SERVER_ON_DEMAND", "true")
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_ON_DEMAND", "false")
    monkeypatch.setenv("MLX_OPENAI_SERVER_EXTRACTION_ON_DEMAND", "false")

    args = SimpleNamespace(
        llm_model="openbmb/AgentCPM-Explore",
        embedding_model="mlx-community/bge-m3-mlx-4bit",
    )
    extraction_args = SimpleNamespace(
        llm_model="huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )

    config = _build_mlx_openai_server_config(args, extraction_args)
    models = {
        model["served_model_name"]: model
        for model in config["models"]  # type: ignore[index]
    }

    assert models["openbmb/AgentCPM-Explore"]["on_demand"] is False
    assert models["huihui-ai/Huihui-granite-4.1-3b-abliterated"]["on_demand"] is False
    assert models["mlx-community/bge-m3-mlx-4bit"]["on_demand"] is True


@pytest.mark.offline
def test_unified_request_kind_uses_model_name(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIGHTRAG_MANAGE_MLX_OPENAI_SERVER", "true")
    monkeypatch.setenv("MLX_OPENAI_SERVER_HOST", "127.0.0.1")
    monkeypatch.setenv("MLX_OPENAI_SERVER_PORT", "11436")
    monkeypatch.setenv(
        "MLX_EXTRACTION_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )

    base_url = "http://127.0.0.1:11436/v1"

    assert (
        openai_llm._get_local_mlx_request_kind(
            base_url, "huihui-ai/Huihui-granite-4.1-3b-abliterated"
        )
        == "extraction"
    )
    assert (
        openai_llm._get_local_mlx_request_kind(base_url, "openbmb/AgentCPM-Explore")
        == "chat"
    )
    assert openai_llm._is_local_mlx_embeddings_base_url(base_url) is True


@pytest.mark.offline
def test_unified_request_recycle_limit_defaults_to_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LIGHTRAG_MANAGE_MLX_OPENAI_SERVER", "true")
    monkeypatch.delenv("MLX_OPENAI_SERVER_MAX_REQUESTS_BEFORE_RECYCLE", raising=False)

    assert openai_llm._get_local_mlx_recycle_limit("chat") == 0
    assert openai_llm._get_local_mlx_recycle_limit("extraction") == 0
    assert openai_llm._get_local_mlx_recycle_limit("embedding") == 0


@pytest.mark.offline
def test_build_mlx_openai_server_config_omits_kv_quantization_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _build_mlx_openai_server_config

    retrieval_path = tmp_path / "rag-distiller"
    extraction_path = tmp_path / "granite"
    embedding_path = tmp_path / "bge"
    retrieval_path.mkdir()
    extraction_path.mkdir()
    embedding_path.mkdir()

    monkeypatch.setenv("MLX_AGENTCPM_MODEL", "lokeshjothiram/rag-distiller-v1:4b")
    monkeypatch.setenv("MLX_AGENTCPM_MODEL_PATH", str(retrieval_path))
    monkeypatch.setenv("MLX_OPENAI_SERVER_RETRIEVAL_KV_BITS", "0")
    monkeypatch.setenv(
        "MLX_EXTRACTION_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )
    monkeypatch.setenv("MLX_EXTRACTION_MODEL_PATH", str(extraction_path))
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL", "mlx-community/bge-m3-mlx-4bit")
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL_PATH", str(embedding_path))

    args = SimpleNamespace(
        llm_model="lokeshjothiram/rag-distiller-v1:4b",
        embedding_model="mlx-community/bge-m3-mlx-4bit",
    )
    extraction_args = SimpleNamespace(
        llm_model="huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )

    config = _build_mlx_openai_server_config(args, extraction_args)
    models = {
        model["served_model_name"]: model
        for model in config["models"]  # type: ignore[index]
    }
    retrieval = models["lokeshjothiram/rag-distiller-v1:4b"]

    assert "kv_bits" not in retrieval
    assert "kv_group_size" not in retrieval
    assert "quantized_kv_start" not in retrieval


@pytest.mark.offline
def test_managed_mlx_rerank_enabled_is_decoupled_from_embedding_host(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _managed_mlx_rerank_enabled

    monkeypatch.setenv("LIGHTRAG_RERANK_ENABLED", "true")
    monkeypatch.setenv("MLX_RERANK_HOST", "127.0.0.1")
    monkeypatch.setenv("MLX_RERANK_PORT", "11437")
    args = SimpleNamespace(
        enable_rerank=True,
        rerank_binding="jina",
        rerank_binding_host="http://127.0.0.1:11437/v1/rerank",
    )

    assert _managed_mlx_rerank_enabled(args) is True


@pytest.mark.offline
def test_build_mlx_openai_server_config_deduplicates_shared_lm_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _build_mlx_openai_server_config

    shared_path = tmp_path / "huihui"
    embedding_path = tmp_path / "bge"
    shared_path.mkdir()
    embedding_path.mkdir()

    monkeypatch.setenv(
        "MLX_AGENTCPM_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )
    monkeypatch.setenv("MLX_AGENTCPM_MODEL_PATH", str(shared_path))
    monkeypatch.setenv(
        "MLX_EXTRACTION_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )
    monkeypatch.setenv("MLX_EXTRACTION_MODEL_PATH", str(shared_path))
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL", "mlx-community/bge-m3-mlx-4bit")
    monkeypatch.setenv("MLX_EMBEDDINGS_MODEL_PATH", str(embedding_path))

    args = SimpleNamespace(
        llm_model="huihui-ai/Huihui-granite-4.1-3b-abliterated",
        embedding_model="mlx-community/bge-m3-mlx-4bit",
    )
    extraction_args = SimpleNamespace(
        llm_model="huihui-ai/Huihui-granite-4.1-3b-abliterated"
    )

    config = _build_mlx_openai_server_config(args, extraction_args)
    served_names = [model["served_model_name"] for model in config["models"]]

    assert served_names == [
        "huihui-ai/Huihui-granite-4.1-3b-abliterated",
        "mlx-community/bge-m3-mlx-4bit",
    ]


@pytest.mark.offline
def test_managed_mlx_openai_launch_command_pins_mlx_lm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _managed_mlx_openai_launch_command

    config_path = tmp_path / "mlx-openai-server-config.yaml"
    monkeypatch.setenv("MLX_OPENAI_SERVER_PACKAGE_VERSION", "1.8.1")
    monkeypatch.setenv("MLX_LM_PACKAGE_VERSION", "0.31.3")

    command = _managed_mlx_openai_launch_command(config_path)

    assert command == [
        "uv",
        "run",
        "--with",
        "mlx-openai-server==1.8.1",
        "--with",
        "mlx-lm==0.31.3",
        "python",
        "-m",
        "lightrag.api.mlx_openai_server_wrapper",
        "launch",
        "--config",
        str(config_path),
    ]


@pytest.mark.offline
def test_install_dill_batch_setitems_compatibility_patch_handles_python314_signature():
    from lightrag.api.mlx_openai_server_wrapper import (
        _install_dill_batch_setitems_compatibility_patch,
    )

    class ParentPickler:
        calls: list[tuple[list[tuple[object, object]], object | None]] = []

        def _batch_setitems(self, items, obj=None):
            self.calls.append((list(items), obj))

    class FakeDill:
        Pickler = ParentPickler

    class FakeHasher:
        @staticmethod
        def hash(value: object) -> str:
            return str(value)

    class ChildPickler(ParentPickler):
        _legacy_no_dict_keys_sorting = False

        def _batch_setitems(self, items):
            raise AssertionError("patch not installed")

    class FakeAppDill:
        Pickler = ChildPickler
        Hasher = FakeHasher

    assert _install_dill_batch_setitems_compatibility_patch(FakeDill, FakeAppDill) is True

    pickler = ChildPickler()
    pickler._batch_setitems([("b", 2), ("a", 1)], object())

    assert pickler.calls == [([("a", 1), ("b", 2)], pickler.calls[0][1])]


@pytest.mark.offline
def test_build_mlx_openai_server_config_omits_embeddings_for_swift_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _build_mlx_openai_server_config

    lm_path = tmp_path / "gemma"
    embedding_path = tmp_path / "nomic"
    lm_path.mkdir()
    embedding_path.mkdir()

    monkeypatch.setenv("MLX_AGENTCPM_MODEL", "local/gemma")
    monkeypatch.setenv("MLX_AGENTCPM_MODEL_PATH", str(lm_path))
    monkeypatch.setenv("MLX_EXTRACTION_MODEL", "local/gemma")
    monkeypatch.setenv("MLX_EXTRACTION_MODEL_PATH", str(lm_path))
    monkeypatch.setenv("LIGHTRAG_MANAGE_SWIFT_EMBEDDINGS", "true")
    monkeypatch.setenv("SWIFT_EMBEDDINGS_MODEL_PATH", str(embedding_path))

    config = _build_mlx_openai_server_config(
        SimpleNamespace(
            llm_model="local/gemma",
            embedding_model="nomic-ai/nomic-embed-text-v2-moe",
        )
    )

    assert [model["model_type"] for model in config["models"]] == ["lm"]


@pytest.mark.offline
def test_managed_swift_lm_launch_command_uses_memory_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _managed_swift_lm_launch_command

    binary = tmp_path / "SwiftLM"
    model_path = tmp_path / "granite"
    binary.touch()
    model_path.mkdir()
    monkeypatch.setenv("SWIFT_LM_BINARY", str(binary))
    monkeypatch.setenv("SWIFT_LM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SWIFT_LM_CONTEXT_SIZE", "8192")
    monkeypatch.setenv("SWIFT_LM_MAX_TOKENS", "2048")
    monkeypatch.setenv("SWIFT_LM_PARALLEL", "1")
    monkeypatch.setenv("SWIFT_LM_MEM_LIMIT_MB", "16384")
    monkeypatch.setenv("SWIFT_LM_PREFILL_SIZE", "512")

    command = _managed_swift_lm_launch_command()

    assert command[:3] == [str(binary), "--model", str(model_path)]
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert command[command.index("--max-tokens") + 1] == "2048"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--mem-limit") + 1] == "16384"
    assert command[command.index("--prefill-size") + 1] == "512"


@pytest.mark.offline
def test_managed_swift_lm_launch_command_supports_acceleration_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _managed_swift_lm_launch_command

    binary = tmp_path / "SwiftLM"
    model_path = tmp_path / "gemma"
    draft_model_path = tmp_path / "draft"
    binary.touch()
    model_path.mkdir()
    draft_model_path.mkdir()
    monkeypatch.setenv("SWIFT_LM_BINARY", str(binary))
    monkeypatch.setenv("SWIFT_LM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SWIFT_LM_GPU_LAYERS", "auto")
    monkeypatch.setenv("SWIFT_LM_TURBO_KV", "true")
    monkeypatch.setenv("SWIFT_LM_DRAFT_MODEL_PATH", str(draft_model_path))
    monkeypatch.setenv("SWIFT_LM_NUM_DRAFT_TOKENS", "1")
    monkeypatch.setenv("SWIFT_LM_MTP", "true")
    monkeypatch.setenv("SWIFT_LM_NUM_MTP_TOKENS", "2")

    command = _managed_swift_lm_launch_command()

    assert command[command.index("--gpu-layers") + 1] == "auto"
    assert "--turbo-kv" in command
    assert command[command.index("--draft-model") + 1] == str(draft_model_path)
    assert command[command.index("--num-draft-tokens") + 1] == "1"
    assert "--mtp" in command
    assert command[command.index("--num-mtp-tokens") + 1] == "2"


@pytest.mark.offline
def test_managed_swift_lm_launch_command_forces_ssd_prefetch_with_stream_experts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _managed_swift_lm_launch_command

    binary = tmp_path / "SwiftLM"
    model_path = tmp_path / "gemma"
    binary.touch()
    model_path.mkdir()
    monkeypatch.setenv("SWIFT_LM_BINARY", str(binary))
    monkeypatch.setenv("SWIFT_LM_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SWIFT_LM_STREAM_EXPERTS", "false")
    monkeypatch.setenv("SWIFT_LM_SSD_PREFETCH", "false")

    command = _managed_swift_lm_launch_command(force_ssd_prefetch=True)

    assert "--stream-experts" in command
    assert "--ssd-prefetch" in command


@pytest.mark.offline
def test_managed_swift_embeddings_launch_command_uses_native_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    from lightrag.api.lightrag_server import _managed_swift_embeddings_launch_command

    binary = tmp_path / "NomicEmbeddingsServer"
    model_path = tmp_path / "nomic"
    binary.touch()
    model_path.mkdir()
    monkeypatch.setenv("SWIFT_EMBEDDINGS_SERVER_BINARY", str(binary))
    monkeypatch.setenv("SWIFT_EMBEDDINGS_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SWIFT_EMBEDDINGS_HOST", "127.0.0.1")
    monkeypatch.setenv("SWIFT_EMBEDDINGS_PORT", "11439")
    monkeypatch.setenv("SWIFT_EMBEDDINGS_MAX_TOKENS", "2048")
    monkeypatch.setenv("SWIFT_EMBEDDINGS_IDLE_TIMEOUT_S", "300")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    assert _managed_swift_embeddings_launch_command() == [
        str(binary),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        "11439",
        "--max-tokens",
        "2048",
        "--served-model-name",
        "BAAI/bge-m3",
        "--idle-timeout-s",
        "300",
    ]


@pytest.mark.offline
def test_swift_lm_response_format_uses_json_object_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LIGHTRAG_MANAGE_SWIFT_LM", "true")
    monkeypatch.setenv("SWIFT_LM_HOST", "127.0.0.1")
    monkeypatch.setenv("SWIFT_LM_PORT", "11436")

    assert openai_llm._is_managed_swift_lm_base_url("http://127.0.0.1:11436/v1")
    assert openai_llm._response_format_to_schema(
        _StructuredSmokeOutput,
        json_object_only=True,
    ) == {"type": "json_object"}


@pytest.mark.offline
def test_validate_openai_response_format_accepts_pydantic_type():
    openai_llm._validate_openai_response_format(_StructuredSmokeOutput)


@pytest.mark.offline
def test_validate_openai_response_format_rejects_non_schema_object():
    with pytest.raises(TypeError):
        openai_llm._validate_openai_response_format(_StructuredSmokeOutput(status="ok"))


@pytest.mark.offline
async def test_openai_complete_if_cache_strips_private_lightrag_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    class _FakeCompletions:
        async def create(self, *, model, messages, **kwargs):
            captured["model"] = model
            captured["messages"] = messages
            captured["kwargs"] = dict(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

        async def close(self):
            return None

    monkeypatch.setattr(
        openai_llm,
        "create_openai_async_client",
        lambda **_: _FakeClient(),
    )

    result = await openai_llm.openai_complete_if_cache(
        "test-model",
        "hello",
        api_key="test-key",
        base_url="http://127.0.0.1:9999/v1",
        _lightrag_request_kind="extraction",
        _lightrag_extraction_request=True,
        timeout=12,
    )

    assert result == "ok"
    assert captured["model"] == "test-model"
    assert captured["kwargs"] == {"timeout": 12}


@pytest.mark.offline
async def test_openai_complete_if_cache_returns_tool_call_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeCompletions:
        async def create(self, *, model, messages, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="",
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        arguments='{"entities":[],"relations":[]}'
                                    )
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

        async def close(self):
            return None

    monkeypatch.setattr(
        openai_llm,
        "create_openai_async_client",
        lambda **_: _FakeClient(),
    )

    result = await openai_llm.openai_complete_if_cache(
        "test-model",
        "hello",
        api_key="test-key",
        base_url="http://127.0.0.1:9999/v1",
        tools=[{"type": "function", "function": {"name": "submit_extraction"}}],
        tool_choice={"type": "function", "function": {"name": "submit_extraction"}},
    )

    assert result == '{"entities":[],"relations":[]}'


@pytest.mark.offline
def test_resolve_non_stream_ttft_uses_prompt_ms_when_present():
    response = SimpleNamespace(timings={"prompt_ms": 245.0, "predicted_ms": 900.0})

    assert openai_llm._resolve_non_stream_ttft_s(response, 1.4) == pytest.approx(0.245)


@pytest.mark.offline
def test_resolve_non_stream_ttft_derives_from_total_minus_predicted_ms():
    response = SimpleNamespace(timings={"predicted_ms": 1250.0, "predicted_per_second": 42.0})

    assert openai_llm._resolve_non_stream_ttft_s(response, 2.0) == pytest.approx(0.75)


@pytest.mark.offline
def test_resolve_non_stream_ttft_skips_invalid_predicted_ms():
    response = SimpleNamespace(timings={"predicted_ms": 2500.0})

    assert openai_llm._resolve_non_stream_ttft_s(response, 1.0) is None
