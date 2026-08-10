#!/usr/bin/env python3
from __future__ import annotations

import os
import plistlib
from pathlib import Path
from typing import Any

from scripts.local_env import env_bool, env_int, env_str


def _require_path(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"{name} is required")
    path = Path(value).expanduser()
    if not path.exists():
        raise SystemExit(f"{name} does not exist: {path}")
    return str(path)


def _append_optional_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _append_optional_value(command: list[str], flag: str, value: Any) -> None:
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

    label = os.environ.get("MLX_MODEL_LAUNCHD_LABEL", "com.local.mlx-model")
    plist_path = launch_agents_dir / f"{label}.plist"
    out_log = log_dir / "mlx-model.out.log"
    err_log = log_dir / "mlx-model.err.log"

    command = [
        _require_path(env_str("SWIFT_LM_BINARY"), "SWIFT_LM_BINARY"),
        "--model",
        _require_path(env_str("SWIFT_LM_MODEL_PATH"), "SWIFT_LM_MODEL_PATH"),
        "--host",
        env_str("SWIFT_LM_HOST", "127.0.0.1"),
        "--port",
        str(env_int("SWIFT_LM_PORT", 11436)),
        "--ctx-size",
        str(env_int("SWIFT_LM_CONTEXT_SIZE", 16384)),
        "--max-tokens",
        str(env_int("SWIFT_LM_MAX_TOKENS", 2048)),
        "--temp",
        "0",
        "--parallel",
        str(env_int("SWIFT_LM_PARALLEL", 1)),
        "--mem-limit",
        str(env_int("SWIFT_LM_MEM_LIMIT_MB", 0)),
        "--prefill-size",
        str(env_int("SWIFT_LM_PREFILL_SIZE", 512)),
    ]
    _append_optional_value(command, "--gpu-layers", env_str("SWIFT_LM_GPU_LAYERS"))
    _append_optional_flag(command, "--stream-experts", env_bool("SWIFT_LM_STREAM_EXPERTS"))
    _append_optional_flag(command, "--ssd-prefetch", env_bool("SWIFT_LM_SSD_PREFETCH"))
    _append_optional_flag(command, "--turbo-kv", env_bool("SWIFT_LM_TURBO_KV"))
    _append_optional_value(command, "--draft-model", env_str("SWIFT_LM_DRAFT_MODEL_PATH"))
    _append_optional_value(command, "--num-draft-tokens", env_str("SWIFT_LM_NUM_DRAFT_TOKENS"))
    _append_optional_flag(command, "--mtp", env_bool("SWIFT_LM_MTP"))
    _append_optional_value(command, "--num-mtp-tokens", env_str("SWIFT_LM_NUM_MTP_TOKENS"))

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
