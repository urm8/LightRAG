#!/usr/bin/env python3
"""Switch the active query LLM profile without leaving role overrides behind."""

from __future__ import annotations

import argparse
from pathlib import Path


ROLE_OVERRIDE_PREFIXES = ("KEYWORD_LLM_", "QUERY_LLM_")


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def _rewrite_env(path: Path, updates: dict[str, str], remove_prefixes: tuple[str, ...]) -> None:
    lines = path.read_text().splitlines()
    seen: set[str] = set()
    rewritten: list[str] = []

    for line in lines:
        if not line or line.lstrip().startswith("#") or "=" not in line:
            rewritten.append(line)
            continue

        key, _ = line.split("=", 1)
        if key.startswith(remove_prefixes):
            continue
        if key in updates:
            rewritten.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            rewritten.append(line)

    for key, value in updates.items():
        if key not in seen:
            rewritten.append(f"{key}={value}")

    path.write_text("\n".join(rewritten) + "\n")


def switch_profile(path: Path, profile: str, mlx_host: str, mlx_model: str) -> None:
    values = _read_env(path)
    if profile == "mlx":
        deepseek_api_key = values.get("LLM_BINDING_API_KEY", "")
        if values.get("LLM_BINDING") == "anthropic" and deepseek_api_key:
            values["LIGHTRAG_DEEPSEEK_API_KEY"] = deepseek_api_key
        updates = {
            "LIGHTRAG_QUERY_LLM_PROFILE": "mlx",
            "LLM_BINDING": "openai",
            "LLM_BINDING_HOST": mlx_host,
            "LLM_BINDING_API_KEY": "dummy",
            "LLM_MODEL": mlx_model,
        }
        if values.get("LIGHTRAG_DEEPSEEK_API_KEY"):
            updates["LIGHTRAG_DEEPSEEK_API_KEY"] = values["LIGHTRAG_DEEPSEEK_API_KEY"]
    else:
        deepseek_api_key = values.get("LIGHTRAG_DEEPSEEK_API_KEY")
        if not deepseek_api_key and values.get("LLM_BINDING") == "anthropic":
            deepseek_api_key = values.get("LLM_BINDING_API_KEY")
        if not deepseek_api_key:
            raise ValueError(
                "LIGHTRAG_DEEPSEEK_API_KEY is required before switching to DeepSeek; "
                "run make use-mlx from an active DeepSeek profile first."
            )
        updates = {
            "LIGHTRAG_QUERY_LLM_PROFILE": "deepseek",
            "LLM_BINDING": "anthropic",
            "LLM_BINDING_HOST": "https://api.openmodel.ai",
            "LLM_BINDING_API_KEY": deepseek_api_key,
            "LLM_MODEL": "deepseek-v4-flash",
            "LIGHTRAG_DEEPSEEK_API_KEY": deepseek_api_key,
        }

    _rewrite_env(path, updates, ROLE_OVERRIDE_PREFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("deepseek", "mlx"))
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--mlx-host", default="http://127.0.0.1:11436/v1")
    parser.add_argument("--mlx-model", default="mlx-community/gemma-4-e4b-it-4bit")
    args = parser.parse_args()
    switch_profile(args.env, args.profile, args.mlx_host, args.mlx_model)


if __name__ == "__main__":
    main()
