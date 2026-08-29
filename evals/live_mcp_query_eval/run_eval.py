#!/usr/bin/env python3
"""Run read-only use-case evaluations against the deployed LightRAG MCP server."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ENDPOINT = "https://rag.urm8.org/mcp"
DEFAULT_CASES_PATH = Path(__file__).with_name("qa.jsonl")


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("qa.jsonl case ids must be unique")
    return cases


SAFE_AUXILIARY_TOOLS = [
    ("check_lightrag_health", {}),
    ("get_pipeline_status", {}),
    ("get_graph_labels", {}),
    ("get_documents", {"tags": ["skill", "agentic-development"]}),
    ("check_memory_pressure", {"top_process_limit": 3}),
]

MUTATING_TOOLS = {
    "insert_document",
    "save_skill",
    "upload_document",
    "insert_file",
    "insert_batch",
    "scan_for_new_documents",
    "merge_entities",
    "create_entities",
    "delete_by_entities",
    "delete_by_doc_ids",
    "edit_entities",
    "create_relations",
    "edit_relations",
}


class MCPClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.request_id = 0

    def request(
        self, method: str, params: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        self.request_id += 1
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params,
            }
        ).encode()
        request = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={
                "X-API-Key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        data_lines = [
            line[6:] for line in body.splitlines() if line.startswith("data: ")
        ]
        decoded = json.loads(data_lines[-1] if data_lines else body)
        return decoded, latency_ms

    def list_tools(self) -> tuple[list[dict[str, Any]], float]:
        response, latency = self.request("tools/list", {})
        return response["result"]["tools"], latency

    def call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        response, latency = self.request(
            "tools/call", {"name": name, "arguments": arguments}
        )
        if response.get("error"):
            raise RuntimeError(json.dumps(response["error"], ensure_ascii=False))
        result = response.get("result", {})
        if result.get("isError"):
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        return result.get("structuredContent", {}), latency


def query_payload(structured: dict[str, Any]) -> dict[str, Any]:
    response = structured.get("response", {})
    return response if isinstance(response, dict) else {}


def evaluate_case(case: dict[str, Any], structured: dict[str, Any]) -> dict[str, Any]:
    payload = query_payload(structured)
    matches = payload.get("matches") or payload.get("skills") or []
    answer = payload.get("response", "")
    searchable = "\n".join(
        [str(answer), *(str(item.get("content", "")) for item in matches)]
    ).casefold()
    expected_terms = case.get("expected_terms", [])
    term_hits = [term for term in expected_terms if term.casefold() in searchable]
    required_tags = {tag.casefold() for tag in case.get("required_tags", [])}
    tag_compliance = None
    if required_tags:
        tag_compliance = all(
            required_tags.issubset(
                {str(tag).casefold() for tag in match.get("tags", [])}
            )
            for match in matches
        )
    expected_retrieval = case.get("expected_retrieval", {})
    actual_retrieval = payload.get("retrieval", {})
    retrieval_checks = {
        key: actual_retrieval.get(key) == value
        for key, value in expected_retrieval.items()
    }
    zero_match_check = len(matches) == 0 if case.get("expect_zero_matches") else None
    return {
        "match_count": len(matches),
        "reference_count": len(payload.get("references", [])),
        "expected_term_hits": term_hits,
        "expected_term_recall": (
            round(len(term_hits) / len(expected_terms), 4) if expected_terms else None
        ),
        "tag_compliance": tag_compliance,
        "retrieval_parameter_checks": retrieval_checks,
        "retrieval_parameter_compliance": (
            all(retrieval_checks.values()) if retrieval_checks else None
        ),
        "zero_match_check": zero_match_check,
        "response_nonempty": bool(answer),
    }


def run(output: Path, cases_path: Path) -> None:
    api_key = os.environ.get("LIGHTRAG_API_KEY")
    if not api_key:
        raise SystemExit("LIGHTRAG_API_KEY is required")
    client = MCPClient(api_key)
    cases = load_cases(cases_path)
    tools, list_latency = client.list_tools()
    results: list[dict[str, Any]] = []

    for case in cases:
        record = {**case, "status": "pending"}
        try:
            structured, latency = client.call(case["tool"], case["arguments"])
            record.update(
                status="success",
                latency_ms=latency,
                result=structured,
                deterministic=evaluate_case(case, structured),
            )
        except Exception as exc:  # continue the evaluation matrix
            record.update(status="error", error=str(exc), latency_ms=None)
        results.append(record)

    auxiliary_results = []
    for tool, arguments in SAFE_AUXILIARY_TOOLS:
        try:
            structured, latency = client.call(tool, arguments)
            auxiliary_results.append(
                {
                    "tool": tool,
                    "arguments": arguments,
                    "status": "success",
                    "latency_ms": latency,
                    "result": structured,
                }
            )
        except Exception as exc:
            auxiliary_results.append(
                {
                    "tool": tool,
                    "arguments": arguments,
                    "status": "error",
                    "error": str(exc),
                }
            )

    skill_case = next(
        (item for item in results if item["id"] == "reusable-skill-search"), None
    )
    skills = (
        query_payload(skill_case.get("result", {})).get("skills", [])
        if skill_case
        else []
    )
    if skills and skills[0].get("document_id"):
        document_id = skills[0]["document_id"]
        try:
            structured, latency = client.call(
                "get_document_content", {"document_id": document_id}
            )
            auxiliary_results.append(
                {
                    "tool": "get_document_content",
                    "arguments": {"document_id": document_id},
                    "status": "success",
                    "latency_ms": latency,
                    "result": structured,
                }
            )
        except Exception as exc:
            auxiliary_results.append(
                {"tool": "get_document_content", "status": "error", "error": str(exc)}
            )

    executed_tools = {case["tool"] for case in cases} | {
        item["tool"] for item in auxiliary_results
    }
    method_coverage = []
    for tool in tools:
        name = tool["name"]
        if name in executed_tools:
            execution = "executed_read_only"
        elif name in MUTATING_TOOLS:
            execution = "schema_validated_execution_skipped_mutation"
        else:
            execution = "schema_validated_not_applicable"
        method_coverage.append(
            {
                "tool": name,
                "execution": execution,
                "description": tool.get("description"),
                "required": tool.get("inputSchema", {}).get("required", []),
                "parameters": sorted(tool.get("inputSchema", {}).get("properties", {})),
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "endpoint": ENDPOINT,
        "project": {
            "name": "LightRAG",
            "path": "/Users/max/projs/github.com/HKUDS/LightRAG",
            "repository": "https://github.com/urm8/LightRAG",
        },
        "policy": {
            "production_safe": True,
            "mutating_tools_executed": False,
            "secret_recorded": False,
            "judge": "Aurora API",
        },
        "tool_schema": {"latency_ms": list_latency, "tool_count": len(tools)},
        "method_coverage": method_coverage,
        "cases": results,
        "auxiliary_checks": auxiliary_results,
        "aurora": {"status": "pending", "ratings": []},
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def merge_ratings(report_path: Path, ratings_path: Path) -> None:
    report = json.loads(report_path.read_text())
    raw_ratings = ratings_path.read_text().strip()
    try:
        rating_document = json.loads(raw_ratings)
    except json.JSONDecodeError:
        start, end = raw_ratings.find("{"), raw_ratings.rfind("}")
        if start < 0 or end <= start:
            raise SystemExit("Aurora output does not contain a JSON object")
        rating_document = json.loads(raw_ratings[start : end + 1])
    if not isinstance(rating_document, dict):
        raise SystemExit("Aurora output must be a JSON object")
    if rating_document.get("verdict") != "SATISFIED":
        raise SystemExit("Aurora did not return verdict: SATISFIED")
    ratings = rating_document.get("ratings", [])
    by_id = {rating["case_id"]: rating for rating in ratings}
    missing = [case["id"] for case in report["cases"] if case["id"] not in by_id]
    if missing:
        raise SystemExit(f"Aurora ratings missing cases: {', '.join(missing)}")
    for case in report["cases"]:
        case["aurora_rating"] = by_id[case["id"]]
    report["aurora"] = {
        "status": "complete",
        "rated_at": datetime.now(UTC).isoformat(),
        "verdict": rating_document["verdict"],
        "summary": rating_document.get("summary", ""),
        "ratings": ratings,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def print_judge_prompt(report_path: Path) -> None:
    report = json.loads(report_path.read_text())
    judge_cases = []
    for case in report["cases"]:
        payload = query_payload(case.get("result", {}))
        matches = payload.get("matches") or payload.get("skills") or []
        judge_cases.append(
            {
                "case_id": case["id"],
                "use_case": case["use_case"],
                "tool": case["tool"],
                "query": case["arguments"].get("query"),
                "expected_terms": case.get("expected_terms", []),
                "required_tags": case.get("required_tags", []),
                "expect_zero_matches": case.get("expect_zero_matches", False),
                "status": case["status"],
                "deterministic": case.get("deterministic", {}),
                "answer": str(payload.get("response", ""))[:1200],
                "matches": [
                    {
                        "index": index,
                        "content": str(match.get("content", ""))[:900],
                        "file_path": match.get("file_path"),
                        "tags": match.get("tags", []),
                    }
                    for index, match in enumerate(matches)
                ],
            }
        )
    rubric = {
        "instruction": (
            "Judge every LightRAG MCP case below. Return one JSON object only, "
            "without Markdown. Rate every case exactly once. relevant_match_indices "
            "are zero-based indices of matches that directly help the use case; use an "
            "empty list when none do. Judge only supplied evidence. Set top-level "
            "verdict to SATISFIED only when all case IDs are present and ratings are "
            "internally consistent."
        ),
        "output_schema": {
            "verdict": "SATISFIED",
            "summary": "brief overall finding",
            "ratings": [
                {
                    "case_id": "exact case id",
                    "tool_fit": "integer 0-5",
                    "retrieval_relevance": "integer 0-5",
                    "context_precision": "integer 0-5",
                    "completeness": "integer 0-5",
                    "groundedness": "integer 0-5",
                    "relevant_match_indices": [0],
                    "verdict": "PASS|PARTIAL|FAIL",
                    "reason": "brief evidence-based reason",
                }
            ],
        },
        "cases": judge_cases,
    }
    print(json.dumps(rubric, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("output", type=Path)
    run_parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    merge_parser = subparsers.add_parser("merge-aurora")
    merge_parser.add_argument("report", type=Path)
    merge_parser.add_argument("ratings", type=Path)
    judge_parser = subparsers.add_parser("judge-prompt")
    judge_parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.cases)
    elif args.command == "merge-aurora":
        merge_ratings(args.report, args.ratings)
    else:
        print_judge_prompt(args.report)


if __name__ == "__main__":
    main()
