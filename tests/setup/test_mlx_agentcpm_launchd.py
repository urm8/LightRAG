from __future__ import annotations

import plistlib
from pathlib import Path

import pytest

from scripts import install_mlx_agentcpm_launchd


@pytest.mark.offline
def test_install_mlx_agentcpm_launchd_sets_tokenizers_parallelism(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo_root = tmp_path / "repo"
    venv_bin = repo_root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)

    script_path = repo_root / "scripts" / "install_mlx_agentcpm_launchd.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# test", encoding="utf-8")

    home_dir = tmp_path / "home"
    monkeypatch.setattr(install_mlx_agentcpm_launchd.Path, "home", lambda: home_dir)
    monkeypatch.setattr(
        install_mlx_agentcpm_launchd,
        "__file__",
        str(script_path),
    )
    monkeypatch.setenv("MLX_AGENTCPM_LAUNCHD_LABEL", "com.local.mlx-agentcpm")
    monkeypatch.setenv("MLX_AGENTCPM_HOST", "127.0.0.1")
    monkeypatch.setenv("MLX_AGENTCPM_PORT", "11436")
    monkeypatch.setenv(
        "MLX_AGENTCPM_MODEL_PATH",
        str(repo_root / "models" / "agentcpm-explore-mlx-4bit"),
    )
    monkeypatch.setenv("MLX_AGENTCPM_CHAT_TEMPLATE_ARGS", "{}")
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)

    install_mlx_agentcpm_launchd.main()

    plist_path = home_dir / "Library" / "LaunchAgents" / "com.local.mlx-agentcpm.plist"
    with plist_path.open("rb") as handle:
        plist = plistlib.load(handle)

    assert (
        plist["EnvironmentVariables"]["TOKENIZERS_PARALLELISM"] == "false"
    )
