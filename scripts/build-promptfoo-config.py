from __future__ import annotations

import json
import runpy
from pathlib import Path

import yaml


BASE_CONFIG = Path("promptfooconfig.yaml")
CAPTURED_CASES = Path("evals/captured/lightrag_prompt_warnings.jsonl")
EXTRACTION_CAPTURE_CASES = Path("evals/captured/extraction_attempts.jsonl")
RECENT_LOG_CASES = Path("evals/captured/recent_log_chunks.jsonl")
OUTPUT_CONFIG = Path("evals/promptfooconfig.generated.yaml")

MAX_CAPTURED_CASES = 50
MAX_EXTRACTION_CAPTURE_CASES = 120
MAX_RECENT_LOG_CASES = 40
MAX_EVAL_PROMPT_BASELINES = 20


def _iter_jsonl_cases(path: Path) -> list[dict]:
    if not path.exists():
        return []

    cases_by_key: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        input_text = (item.get("input_text") or "").strip()
        if not input_text:
            continue

        warning_classes = item.get("warning_classes") or []
        prompt_suggestions = item.get("prompt_suggestions") or []
        provider_error = item.get("provider_error")
        provider_error_info = item.get("provider_error_info")
        metadata = item.get("metadata") or {}
        key = item.get("capture_key") or item.get("chunk_key") or input_text[:120]
        cases_by_key[key] = {
            "description": item.get("description")
            or f"captured {key} {' '.join(warning_classes)}",
            "vars": {
                "input_text": input_text,
                "language": "English",
            },
            "metadata": {
                "source": "captured-warning",
                "chunk_key": item.get("chunk_key"),
                "file_path": item.get("file_path"),
                "warning_classes": warning_classes,
                "prompt_suggestions": prompt_suggestions,
                "provider_error": provider_error,
                "provider_error_info": provider_error_info,
                **metadata,
            },
        }

    return list(cases_by_key.values())


def iter_captured_cases() -> list[dict]:
    return _iter_jsonl_cases(CAPTURED_CASES)[-MAX_CAPTURED_CASES:]


def iter_recent_log_cases() -> list[dict]:
    return _iter_jsonl_cases(RECENT_LOG_CASES)[-MAX_RECENT_LOG_CASES:]


def iter_extraction_capture_cases() -> list[dict]:
    if not EXTRACTION_CAPTURE_CASES.exists():
        return []

    cases_by_key: dict[str, dict] = {}
    for line in EXTRACTION_CAPTURE_CASES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        input_text = (item.get("input_text") or "").strip()
        if not input_text:
            continue

        tags = item.get("tags") or []
        parsed_output = item.get("parsed_output") or {}
        entity_count = int(item.get("entity_count") or 0)
        relation_count = int(item.get("relation_count") or 0)
        capture_key = item.get("capture_key") or item.get("chunk_key") or input_text[:120]

        for focus in ("entity", "relation"):
            focus_key = f"{capture_key}:{focus}"
            vars_payload = {
                "input_text": input_text,
                "language": "English",
                "expected_min_entities": min(entity_count, 2) if focus == "entity" and "sparse_entities" not in tags else 0,
                "expected_min_relations": min(relation_count, 1) if focus == "relation" and "sparse_relations" not in tags else 0,
                "disallow_legacy_fallback": True,
                "disallow_length_truncated": True,
            }
            cases_by_key[focus_key] = {
                "description": f"{item.get('phase', 'capture')} {focus} {item.get('chunk_key') or capture_key[-12:]}",
                "vars": vars_payload,
                "metadata": {
                    "source": "extraction-capture",
                    "focus": focus,
                    "chunk_key": item.get("chunk_key"),
                    "file_path": item.get("file_path"),
                    "phase": item.get("phase"),
                    "tags": tags,
                    "captured_output": item.get("raw_output"),
                    "captured_parsed_output": parsed_output,
                    "entity_count": entity_count,
                    "relation_count": relation_count,
                },
            }

    return list(cases_by_key.values())[-MAX_EXTRACTION_CAPTURE_CASES:]


def iter_eval_prompt_baselines() -> list[dict]:
    try:
        namespace = runpy.run_path("scripts/eval_prompts.py")
    except Exception:
        return []

    tests = namespace.get("TESTS") or []
    cases: list[dict] = []
    for index, text in enumerate(tests[:MAX_EVAL_PROMPT_BASELINES], start=1):
        cases.append(
            {
                "description": f"eval_prompts baseline #{index}",
                "vars": {
                    "input_text": str(text).strip(),
                    "language": "English",
                    "disallow_legacy_fallback": True,
                    "disallow_length_truncated": True,
                },
                "metadata": {
                    "source": "eval-prompts-baseline",
                },
            }
        )
    return cases


def main() -> None:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config.setdefault("tests", [])
    config["providers"] = [{"id": "file://extraction-provider.cjs"}]
    for assertion in config.get("defaultTest", {}).get("assert", []):
        if assertion.get("value") == "file://evals/assert-lightrag-format.cjs":
            assertion["value"] = "file://assert-lightrag-format.cjs"

    captured = iter_captured_cases()
    extraction_captures = iter_extraction_capture_cases()
    recent_log_cases = iter_recent_log_cases()
    eval_prompt_baselines = iter_eval_prompt_baselines()
    config["tests"].extend(captured)
    config["tests"].extend(extraction_captures)
    config["tests"].extend(recent_log_cases)
    config["tests"].extend(eval_prompt_baselines)
    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CONFIG.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        f"Wrote {OUTPUT_CONFIG} with {len(config['tests'])} tests "
        f"({len(captured)} captured, {len(extraction_captures)} extraction, "
        f"{len(recent_log_cases)} recent-log, {len(eval_prompt_baselines)} baseline)"
    )


if __name__ == "__main__":
    main()
