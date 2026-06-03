from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/promptfoo-results.json")
    if not path.exists():
        print(f"No promptfoo result file found at {path}")
        return 1

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Promptfoo result file is not valid JSON: {path}: {exc}")
        return 1
    results = data.get("results", {}).get("results", [])
    total = len(results)
    successes = [item for item in results if item.get("success")]
    failures = [item for item in results if not item.get("success")]
    errors = [item for item in failures if item.get("error")]

    print(f"\n{'=' * 60}")
    print(f"Promptfoo results: {len(successes)} passed, {len(errors)} errors, {len(results)} total")
    print(f"{'=' * 60}")

    if not failures:
        print("\nAll tests passed.")
        return 0

    for index, item in enumerate(failures, start=1):
        test_case = item.get("testCase") or {}
        metadata = test_case.get("metadata") or {}
        response = item.get("response") or {}
        output = response.get("output") or ""
        error = item.get("error") or item.get("failureReason") or ""
        prompt = item.get("prompt") or {}
        test_vars = test_case.get("vars") or {}

        print("\n" + "=" * 80)
        print(f"FAIL {index}. {test_case.get('description', 'unnamed test')}")
        if error:
            print("\nReason:")
            print(str(error)[:2000])

        failed_prompt = test_vars.get("input_text") or prompt.get("raw", "")
        if failed_prompt:
            print("\nFailed prompt (input_text):")
            print(failed_prompt[:3000])
            if len(failed_prompt) > 3000:
                print("... [truncated]")

        provider_error = metadata.get("provider_error")
        if provider_error:
            print("\nCaptured provider error:")
            print(str(provider_error)[:2000])
        provider_error_info = metadata.get("provider_error_info")
        if provider_error_info:
            print("\nCaptured provider error info:")
            print(json.dumps(provider_error_info, ensure_ascii=False, indent=2)[:2000])
        suggestions = metadata.get("prompt_suggestions") or []
        if suggestions:
            print("\nPrompt suggestions:")
            for suggestion in suggestions:
                print(
                    "- "
                    f"{suggestion.get('prompt_file')}:{suggestion.get('prompt_key')} "
                    f"- {suggestion.get('suggestion')}"
                )
        if output:
            print("\nModel output:")
            print(output[:3000])

    print(f"\nFull results: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
