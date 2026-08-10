from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from scripts import install_mlx_embeddings_launchd, install_mlx_model_launchd


def _repo_layout(tmp_path: Path, script_name: str) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / ".venv" / "bin").mkdir(parents=True)
    script_path = repo_root / "scripts" / script_name
    script_path.write_text("# test", encoding="utf-8")
    return repo_root, script_path


@pytest.mark.offline
def test_install_mlx_model_launchd_uses_swift_lm_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo_root, script_path = _repo_layout(tmp_path, "install_mlx_model_launchd.py")
    home_dir = tmp_path / "home"
    binary = repo_root / "data" / "SwiftLM" / ".build" / "release" / "SwiftLM"
    model = repo_root / "models" / "gemma-4-e4b-it-4bit"
    binary.parent.mkdir(parents=True)
    model.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(install_mlx_model_launchd.Path, "home", lambda: home_dir)
    monkeypatch.setattr(install_mlx_model_launchd, "__file__", str(script_path))
    monkeypatch.setenv("MLX_MODEL_LAUNCHD_LABEL", "com.local.test-mlx-model")
    monkeypatch.setenv("SWIFT_LM_BINARY", str(binary))
    monkeypatch.setenv("SWIFT_LM_MODEL_PATH", str(model))
    monkeypatch.setenv("SWIFT_LM_HOST", "127.0.0.1")
    monkeypatch.setenv("SWIFT_LM_PORT", "11436")
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    install_mlx_model_launchd.main()

    plist_path = home_dir / "Library" / "LaunchAgents" / "com.local.test-mlx-model.plist"
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)

    assert plist["Label"] == "com.local.test-mlx-model"
    assert plist["ProgramArguments"][:4] == [str(binary), "--model", str(model), "--host"]
    assert "11436" in plist["ProgramArguments"]
    assert plist["EnvironmentVariables"]["TOKENIZERS_PARALLELISM"] == "false"


@pytest.mark.offline
def test_install_mlx_embeddings_launchd_uses_swift_embeddings_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo_root, script_path = _repo_layout(tmp_path, "install_mlx_embeddings_launchd.py")
    home_dir = tmp_path / "home"
    binary = (
        repo_root
        / "data"
        / "SwiftLM"
        / ".build"
        / "release"
        / "NomicEmbeddingsServer"
    )
    model = repo_root / "models" / "bge-m3-mlx-4bit"
    binary.parent.mkdir(parents=True)
    model.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(install_mlx_embeddings_launchd.Path, "home", lambda: home_dir)
    monkeypatch.setattr(install_mlx_embeddings_launchd, "__file__", str(script_path))
    monkeypatch.setenv("MLX_EMBEDDINGS_LAUNCHD_LABEL", "com.local.test-mlx-embeddings")
    monkeypatch.setenv("SWIFT_EMBEDDINGS_SERVER_BINARY", str(binary))
    monkeypatch.setenv("SWIFT_EMBEDDINGS_MODEL_PATH", str(model))
    monkeypatch.setenv("SWIFT_EMBEDDINGS_HOST", "127.0.0.1")
    monkeypatch.setenv("SWIFT_EMBEDDINGS_PORT", "11439")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")

    install_mlx_embeddings_launchd.main()

    plist_path = (
        home_dir / "Library" / "LaunchAgents" / "com.local.test-mlx-embeddings.plist"
    )
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)

    assert plist["Label"] == "com.local.test-mlx-embeddings"
    assert plist["ProgramArguments"][:4] == [str(binary), "--model", str(model), "--host"]
    assert "11439" in plist["ProgramArguments"]
    assert "BAAI/bge-m3" in plist["ProgramArguments"]
