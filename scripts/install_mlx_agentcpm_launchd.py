#!/usr/bin/env python3
from __future__ import annotations

import plistlib
from pathlib import Path

from scripts.local_env import env_str


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    log_dir = Path.home() / "Library" / "Logs" / "lightrag"
    launch_agents_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    label = env_str("MLX_AGENTCPM_LAUNCHD_LABEL", "com.local.mlx-agentcpm")
    host = env_str("MLX_AGENTCPM_HOST", "127.0.0.1")
    port = env_str("MLX_AGENTCPM_PORT", "11436")
    model_path = env_str("MLX_AGENTCPM_MODEL_PATH") or str(
        repo_root / "models" / "agentcpm-explore-mlx-4bit"
    )
    chat_template_args = env_str("MLX_AGENTCPM_CHAT_TEMPLATE_ARGS", "{}")
    plist_path = launch_agents_dir / f"{label}.plist"
    out_log = log_dir / "mlx-agentcpm.out.log"
    err_log = log_dir / "mlx-agentcpm.err.log"

    program_arguments = [
        str(repo_root / ".venv" / "bin" / "python"),
        "-m",
        "mlx_lm",
        "server",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        port,
        "--trust-remote-code",
        "--chat-template-args",
        chat_template_args,
    ]

    plist = {
        "Label": label,
        "ProgramArguments": program_arguments,
        "WorkingDirectory": str(repo_root),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(out_log),
        "StandardErrorPath": str(err_log),
        "EnvironmentVariables": {
            "PATH": (
                f"{repo_root / '.venv' / 'bin'}:"
                "/Users/max/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
            ),
            "PYTHONUNBUFFERED": "1",
        },
    }

    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)

    print(plist_path)


if __name__ == "__main__":
    main()
