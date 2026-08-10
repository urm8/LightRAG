#!/usr/bin/env python3
from __future__ import annotations

import os
import plistlib
from pathlib import Path

from scripts.local_env import env_int, env_str


def _require_path(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"{name} is required")
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"{name} does not exist: {path}")
    return str(path)


def _append_optional_value(command: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    rendered = str(value).strip()
    if rendered:
        command.extend([flag, rendered])


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
    log_dir = Path.home() / "Library" / "Logs" / "lightrag"
    launch_agents_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    label = os.environ.get("MLX_EMBEDDINGS_LAUNCHD_LABEL", "com.local.mlx-embeddings")
    plist_path = launch_agents_dir / f"{label}.plist"
    out_log = log_dir / "mlx-embeddings.out.log"
    err_log = log_dir / "mlx-embeddings.err.log"

    command = [
        _require_path(
            env_str("SWIFT_EMBEDDINGS_SERVER_BINARY"),
            "SWIFT_EMBEDDINGS_SERVER_BINARY",
        ),
        "--model",
        _require_path(env_str("SWIFT_EMBEDDINGS_MODEL_PATH"), "SWIFT_EMBEDDINGS_MODEL_PATH"),
        "--host",
        env_str("SWIFT_EMBEDDINGS_HOST", "127.0.0.1"),
        "--port",
        str(env_int("SWIFT_EMBEDDINGS_PORT", 11439)),
        "--max-tokens",
        str(env_int("SWIFT_EMBEDDINGS_MAX_TOKENS", 8192)),
    ]
    _append_optional_value(
        command,
        "--served-model-name",
        env_str("EMBEDDING_MODEL") or Path(command[2]).name,
    )
    _append_optional_value(
        command,
        "--idle-timeout-s",
        env_str("SWIFT_EMBEDDINGS_IDLE_TIMEOUT_S"),
    )

    plist = {
        "Label": label,
        "ProgramArguments": command,
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
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        },
    }

    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)

    print(plist_path)


if __name__ == "__main__":
    main()
