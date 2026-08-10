#!/usr/bin/env python3
"""Replay the ORANGE-IBIS-742 answer request through the OpenAI SDK only.

The script intentionally does not import LightRAG or call the LightRAG API.
It expects a prompt JSON file with:

{
  "system": "...full system prompt...",
  "user": "...user query..."
}
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from openai import OpenAI


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _base_url(raw_host: str) -> str:
    host = raw_host.rstrip("/")
    return host if host.endswith("/v1") else f"{host}/v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--disable-thinking", action="store_true")
    args = parser.parse_args()

    env_file = _load_dotenv(Path(args.env_file))
    api_key = args.api_key or env_file.get("OPENAI_API_KEY") or env_file.get("LLM_BINDING_API_KEY")
    base_url = args.base_url or env_file.get("OPENAI_BASE_URL") or env_file.get("LLM_BINDING_HOST")
    model = args.model or env_file.get("OPENAI_MODEL") or env_file.get("LLM_MODEL")

    if not api_key:
        raise SystemExit("Missing API key: set OPENAI_API_KEY or LLM_BINDING_API_KEY")
    if not base_url:
        raise SystemExit("Missing base URL: set OPENAI_BASE_URL or LLM_BINDING_HOST")
    if not model:
        raise SystemExit("Missing model: set OPENAI_MODEL or LLM_MODEL")

    prompt = json.loads(Path(args.prompt_json).read_text(encoding="utf-8"))
    system_prompt = prompt["system"]
    user_query = prompt["user"]

    client = OpenAI(api_key=api_key, base_url=_base_url(base_url))
    print(
        json.dumps(
            {
                "model": model,
                "base_url": str(client.base_url),
                "runs": args.runs,
                "system_chars": len(system_prompt),
                "user_chars": len(user_query),
                "max_tokens": args.max_tokens,
                "disable_thinking": args.disable_thinking,
            },
            ensure_ascii=False,
        )
    )

    timings_ms: list[float] = []
    for idx in range(1, args.runs + 1):
        kwargs = {}
        if args.disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            max_tokens=args.max_tokens,
            stream=False,
            **kwargs,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings_ms.append(elapsed_ms)

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage
        print(
            json.dumps(
                {
                    "run": idx,
                    "wall_ms": round(elapsed_ms, 1),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "finish_reason": choice.finish_reason,
                    "response_chars": len(text),
                    "contains_answer": "Lighthouse Cache" in text and "Redis" in text,
                    "preview": text[:300].replace("\n", " "),
                },
                ensure_ascii=False,
            )
        )

    print(
        json.dumps(
            {
                "median_ms": round(statistics.median(timings_ms), 1),
                "min_ms": round(min(timings_ms), 1),
                "max_ms": round(max(timings_ms), 1),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
