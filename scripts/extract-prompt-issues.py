"""
extract-prompt-issues.py

Stage 1 of the prompt enhancement pipeline.
Aggregates failure patterns from all captured JSONL evaluation data
into a structured issues report grouped by warning class.

Output: evals/prompt_issues.aggregated.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


CAPTURED_WARNINGS = Path("evals/captured/lightrag_prompt_warnings.jsonl")
EXTRACTION_ATTEMPTS = Path("evals/captured/extraction_attempts.jsonl")
QUERY_ATTEMPTS = Path("evals/captured/query_attempts.jsonl")
RECENT_LOG_CHUNKS = Path("evals/captured/recent_log_chunks.jsonl")
OUTPUT = Path("evals/prompt_issues.aggregated.json")

# Maps lowercase suggestion prompt_key references to actual constants in prompt.py
# Warning classes to exclude — provider-level issues (old Apple safety filter), not prompt quality issues
EXCLUDED_WARNING_CLASSES: set[str] = {
    "bad_request_400",
    "content_policy_violation",
}

PROMPT_KEY_MAP: dict[str, str] = {
    "entity_extraction_system_prompt": "ENTITY_EXTRACTION_SYSTEM_PROMPT",
    "entity_extraction_user_prompt": "ENTITY_EXTRACTION_USER_PROMPT",
    "entity_continue_extraction_user_prompt": "ENTITY_CONTINUE_EXTRACTION_USER_PROMPT",
    "entity_extraction_json_system_prompt": "ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT",
    "entity_extraction_json_user_prompt": "ENTITY_EXTRACTION_JSON_USER_PROMPT",
    "entity_continue_extraction_json_user_prompt": "ENTITY_CONTINUE_EXTRACTION_JSON_USER_PROMPT",
    "keywords_extraction": "KEYWORDS_EXTRACTION",
    "rag_response": "RAG_RESPONSE",
    "naive_rag_response": "NAIVE_RAG_RESPONSE",
    "summarize_entity_descriptions": "SUMMARIZE_ENTITY_DESCRIPTIONS",
    "kg_query_context": "KG_QUERY_CONTEXT",
    "naive_query_context": "NAIVE_QUERY_CONTEXT",
    "agent_tool_protocol_query": "AGENT_TOOL_PROTOCOL_QUERY",
    "agent_tool_protocol_extract": "AGENT_TOOL_PROTOCOL_EXTRACT",
}


def _iter_jsonl(path: Path) -> list[dict]:
    """Read JSONL file, returning list of parsed records."""
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _normalize_prompt_key(key: str) -> str:
    """Normalize a prompt key reference to the constant name in prompt.py."""
    lower = key.strip().lower()
    if lower in PROMPT_KEY_MAP:
        return PROMPT_KEY_MAP[lower]
    # Try to find by suffix match (e.g. "system_prompt" -> ENTITY_EXTRACTION_SYSTEM_PROMPT)
    for alias, constant in PROMPT_KEY_MAP.items():
        if lower.endswith(alias) or alias.endswith(lower):
            return constant
    return key


def _aggregate_warnings(records: list[dict]) -> dict:
    """Aggregate warning_classes and prompt_suggestions from lightrag_prompt_warnings.jsonl."""
    by_class: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "prompt_suggestions": Counter(),
        "example_inputs": [],
        "file_paths": Counter(),
        "max_examples": 5,
    })

    for rec in records:
        warning_classes = rec.get("warning_classes") or []
        if not warning_classes:
            continue

        input_text = (rec.get("input_text") or "")[:200]
        file_path = rec.get("file_path") or "unknown"
        prompt_suggestions = rec.get("prompt_suggestions") or []

        for wc in warning_classes:
            if wc in EXCLUDED_WARNING_CLASSES:
                continue
            info = by_class[wc]
            info["count"] += 1
            info["file_paths"][file_path] += 1

            # Track which prompt keys this class targets
            for suggestion in prompt_suggestions:
                prompt_key = _normalize_prompt_key(
                    suggestion.get("prompt_key", "unknown")
                )
                suggestion_text = suggestion.get("suggestion", "")
                key = f"{prompt_key}:::{suggestion_text}"
                info["prompt_suggestions"][key] += 1

            # Collect example inputs (deduplicate by first 120 chars)
            short = input_text[:120]
            if short and short not in {e[:120] for e in info["example_inputs"]}:
                info["example_inputs"].append(input_text)
                # Keep only the max_examples most recent
                if len(info["example_inputs"]) > info["max_examples"] * 2:
                    info["example_inputs"] = info["example_inputs"][-info["max_examples"]:]

    # Sort and trim
    result: dict[str, dict] = {}
    for wc in sorted(by_class.keys()):
        info = by_class[wc]
        top_suggestions = [
            {"prompt_key": s.split(":::")[0], "suggestion": s.split(":::")[1], "count": c}
            for s, c in info["prompt_suggestions"].most_common(8)
        ]
        result[wc] = {
            "count": info["count"],
            "trend": "unknown",
            "top_prompt_suggestions": top_suggestions,
            "example_inputs": info["example_inputs"][-info["max_examples"]:],
            "top_file_paths": [
                {"path": p, "count": c}
                for p, c in info["file_paths"].most_common(5)
            ],
        }
    return result


def _aggregate_extraction_tags(records: list[dict]) -> dict:
    """Aggregate tags (sparse_entities, sparse_relations, etc.) from extraction_attempts.jsonl."""
    tag_counts: Counter = Counter()
    phase_counts: Counter = Counter()
    by_tag: dict[str, list[dict]] = defaultdict(list)

    for rec in records:
        tags = rec.get("tags") or []
        phase = rec.get("phase") or "unknown"
        phase_counts[phase] += 1
        for tag in tags:
            tag_counts[tag] += 1
            if len(by_tag[tag]) < 5:
                by_tag[tag].append({
                    "input_text": (rec.get("input_text") or "")[:200],
                    "entity_count": rec.get("entity_count", 0),
                    "relation_count": rec.get("relation_count", 0),
                    "phase": phase,
                    "file_path": rec.get("file_path"),
                })

    return {
        "tag_counts": dict(tag_counts.most_common()),
        "phase_counts": dict(phase_counts.most_common()),
        "by_tag": {
            tag: examples for tag, examples in by_tag.items()
        },
    }


def _aggregate_query_issues(records: list[dict]) -> dict:
    """Analyze query_attempts.jsonl for keyword extraction issues."""
    phases: Counter = Counter()
    modes: Counter = Counter()
    for rec in records:
        phases[rec.get("phase", "unknown")] += 1
        modes[rec.get("mode", "unknown")] += 1

    return {
        "total_query_attempts": len(records),
        "phase_distribution": dict(phases.most_common()),
        "mode_distribution": dict(modes.most_common()),
    }


def _aggregate_log_chunks(records: list[dict]) -> dict:
    """Aggregate recent log chunk data for context on what's being processed."""
    warning_classes: Counter = Counter()
    total = len(records)
    for rec in records:
        for wc in rec.get("warning_classes") or []:
            warning_classes[wc] += 1

    return {
        "total_chunk_cases": total,
        "warning_distribution": dict(warning_classes.most_common()),
    }


def main() -> None:
    warning_records = _iter_jsonl(CAPTURED_WARNINGS)
    extraction_records = _iter_jsonl(EXTRACTION_ATTEMPTS)
    query_records = _iter_jsonl(QUERY_ATTEMPTS)
    log_records = _iter_jsonl(RECENT_LOG_CHUNKS)

    aggregated = {
        "aggregated_at": __import__("datetime").datetime.now().isoformat(),
        "source_files": {
            "lightrag_prompt_warnings.jsonl": len(warning_records),
            "extraction_attempts.jsonl": len(extraction_records),
            "query_attempts.jsonl": len(query_records),
            "recent_log_chunks.jsonl": len(log_records),
        },
        "warning_classes": _aggregate_warnings(warning_records),
        "extraction_tags": _aggregate_extraction_tags(extraction_records),
        "query_issues": _aggregate_query_issues(query_records),
        "log_chunks": _aggregate_log_chunks(log_records),
    }

    # Assign trends by comparing with previous aggregate (if it exists)
    if OUTPUT.exists():
        try:
            prev = json.loads(OUTPUT.read_text(encoding="utf-8"))
            prev_classes = prev.get("warning_classes", {})
            for wc, info in aggregated["warning_classes"].items():
                prev_count = prev_classes.get(wc, {}).get("count", 0)
                curr_count = info["count"]
                if curr_count > prev_count * 1.2:
                    info["trend"] = "growing"
                elif curr_count < prev_count * 0.8:
                    info["trend"] = "declining"
                else:
                    info["trend"] = "stable"
        except (json.JSONDecodeError, KeyError):
            pass

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(aggregated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    class_counts = aggregated["warning_classes"]
    total_issues = sum(c["count"] for c in class_counts.values())
    tag_counts = aggregated["extraction_tags"].get("tag_counts", {})
    print(
        f"Aggregated {len(class_counts)} warning classes "
        f"({total_issues} total issues), "
        f"{len(tag_counts)} extraction tags "
        f"from {len(warning_records)} warning + {len(extraction_records)} extraction records"
    )
    if class_counts:
        print("\nWarning classes:")
        for wc, info in sorted(class_counts.items(), key=lambda x: -x[1]["count"]):
            trend_mark = {"growing": "↑", "declining": "↓", "stable": "→", "unknown": "?"}
            print(
                f"  {trend_mark.get(info['trend'], '?')} {wc}: {info['count']} "
                f"(top prompt: {info['top_prompt_suggestions'][0]['prompt_key'] if info['top_prompt_suggestions'] else 'n/a'})"
            )


if __name__ == "__main__":
    main()
