"""
improve-prompts.py

RL (Reinforcement Learning) prompt enhancement pipeline.

Iterative loop:
  1. Extract issues from captured eval data
  2. Run promptfoo eval → baseline pass rate (first iteration)
  3. Call LLM ("subagent") to suggest prompt improvements targeting failure patterns
  4. Apply suggested changes
  5. Run promptfoo eval → new pass rate (reward signal)
  6. If pass rate improved → KEEP changes, log success
  7. If pass rate degraded → REVERT changes, log failure
  8. Repeat until convergence or max iterations

Usage:
    python scripts/improve-prompts.py                          # full RL loop
    python scripts/improve-prompts.py --iters 10               # max 10 iterations
    python scripts/improve-prompts.py --target ENTITY_EXTRACTION_SYSTEM_PROMPT  # single prompt
    python scripts/improve-prompts.py --dry-run                # preview without changes
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Ensure the project root is on sys.path so that "import lightrag" works
# even when the script is run as "python scripts/improve-prompts.py"
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from lightrag.config import settings

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

AGGREGATED_ISSUES = Path("evals/prompt_issues.aggregated.json")
PROMPT_FILE = Path("lightrag/prompt.py")
ENHANCEMENT_LOG = Path("evals/prompt_enhancement_log.jsonl")
PROMPT_DIFF_DIR = Path("evals/prompt_diffs")
PROMPTFOO_CONFIG = Path("evals/promptfooconfig.generated.yaml")
PROMPTFOO_RESULTS = Path("evals/promptfoo-results.json")
ENHANCEMENT_RESULTS = Path("evals/prompt_enhancement_results.jsonl")

# Warning classes to exclude — provider-level issues (old Apple safety filter), not prompt quality issues
EXCLUDED_WARNING_CLASSES: set[str] = {
    "bad_request_400",
    "content_policy_violation",
}

# Default iterations
MAX_ITERATIONS = settings.improve_max_iterations
# Convergence: if pass rate improves by less than this, consider converged
CONVERGENCE_THRESHOLD = settings.improve_convergence_threshold
# Target prompts per iteration (top N most-issue-heavy)
TARGETS_PER_ITERATION = settings.improve_targets_per_iteration

# LLM config (the "subagent" that suggests prompt improvements)
LLM_BINDING_HOST = (
    settings.improve_llm_binding_host
    or settings.llm_binding_host
    or "http://127.0.0.1:11436/v1"
)
LLM_API_KEY = (
    settings.improve_llm_api_key
    or settings.llm_binding_api_key
    or "dummy"
)
LLM_MODEL = (
    settings.improve_llm_model
    or settings.llm_model
    or "default"
)

# Prompt constants we are allowed to modify (safety allowlist)
ALLOWED_PROMPT_KEYS: set[str] = {
    "ENTITY_EXTRACTION_SYSTEM_PROMPT",
    "ENTITY_EXTRACTION_USER_PROMPT",
    "ENTITY_CONTINUE_EXTRACTION_USER_PROMPT",
    "ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT",
    "ENTITY_EXTRACTION_JSON_USER_PROMPT",
    "ENTITY_CONTINUE_EXTRACTION_JSON_USER_PROMPT",
    "KEYWORDS_EXTRACTION",
    "RAG_RESPONSE",
    "NAIVE_RAG_RESPONSE",
    "SUMMARIZE_ENTITY_DESCRIPTIONS",
    "KG_QUERY_CONTEXT",
    "NAIVE_QUERY_CONTEXT",
    "AGENT_TOOL_PROTOCOL_QUERY",
    "AGENT_TOOL_PROTOCOL_EXTRACT",
}

PROMPT_KEY_ALIASES: dict[str, str] = {
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


def resolve_prompt_key(key: str) -> str | None:
    k = key.strip()
    # Some LLM tokenizers split on underscores, yielding "SYSTEM_ PROMPT" etc.
    # Normalise by removing all internal whitespace.
    k = re.sub(r"\s+", "", k)
    if k in ALLOWED_PROMPT_KEYS:
        return k
    return PROMPT_KEY_ALIASES.get(k.lower())


# ---------------------------------------------------------------------------
# LLM subagent
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Call an OpenAI-compatible LLM API — this is the 'subagent' for each RL step."""
    import httpx

    base = LLM_BINDING_HOST.rstrip("/")
    if "chat/completions" in base:
        url = base
    elif "/v1" in base:
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 8192,
    }

    with httpx.Client(timeout=600.0) as client:
        resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Prompt file manipulation
# ---------------------------------------------------------------------------

# Reverse map: CONSTANT_NAME -> PROMPTS dict key
_PROMPT_CONST_TO_KEY: dict[str, str] = {v: k for k, v in PROMPT_KEY_ALIASES.items()}


def get_prompt_range(source: str, const_name: str) -> tuple[int, int] | None:
    """Return (start, end) of the full const definition in source."""
    # Try PATTERN 1: STANDALONE_CONSTANT: str = """..."""
    decl = re.compile(rf'^{const_name}\s*:\s*str\s*=\s*"""', re.MULTILINE)
    m = decl.search(source)
    if m:
        start = m.start()
        content_start = m.end()
        close = re.compile(r'^"""\s*$', re.MULTILINE)
        cm = close.search(source, content_start)
        if cm:
            return (start, cm.end())
        close2 = re.compile(r'"""\s*\n', re.MULTILINE)
        cm2 = close2.search(source, content_start)
        if cm2:
            return (start, cm2.end())
        return None

    # Try PATTERN 2: PROMPTS["key_name"] = """..."""
    dict_key = _PROMPT_CONST_TO_KEY.get(const_name)
    if dict_key:
        decl2 = re.compile(
            rf'^PROMPTS\["{re.escape(dict_key)}"\]\s*=\s*"""', re.MULTILINE
        )
        m2 = decl2.search(source)
        if m2:
            start = m2.start()
            content_start = m2.end()
            close = re.compile(r'^"""\s*$', re.MULTILINE)
            cm = close.search(source, content_start)
            if cm:
                return (start, cm.end())
            close2 = re.compile(r'"""\s*\n', re.MULTILINE)
            cm2 = close2.search(source, content_start)
            if cm2:
                return (start, cm2.end())
    return None
    start = m.start()
    content_start = m.end()
    close = re.compile(r'^"""\s*$', re.MULTILINE)
    cm = close.search(source, content_start)
    if cm:
        return (start, cm.end())
    close2 = re.compile(r'"""\s*\n', re.MULTILINE)
    cm2 = close2.search(source, content_start)
    if cm2:
        return (start, cm2.end())
    return None


def replace_prompt_in_source(source: str, const_name: str, new_content: str) -> str:
    """Return source with the given prompt constant's content replaced."""
    r = get_prompt_range(source, const_name)
    if r is None:
        raise ValueError(f"Cannot locate '{const_name}' in prompt.py")
    start, end = r
    prefix_end = source.index('"""', start) + 3
    decl_line = source[start:prefix_end]
    replacement = f'{decl_line}{new_content}"""'
    return source[:start] + replacement + source[end:]


def read_current_prompts() -> dict[str, str]:
    """Reload and return current prompt constants from lightrag.prompt."""
    mod = importlib.import_module("lightrag.prompt")
    mod = importlib.reload(mod)
    prompts: dict[str, str] = {}
    for name in dir(mod):
        if name in ALLOWED_PROMPT_KEYS:
            val = getattr(mod, name)
            if isinstance(val, str) and len(val) > 50:
                prompts[name] = val
    return prompts


# ---------------------------------------------------------------------------
# Promptfoo evaluation
# ---------------------------------------------------------------------------

def run_promptfoo_eval(label: str = "") -> dict:
    """Run make test-prompt and return pass/fail stats.

    Returns:
        {
            "pass_rate": float,
            "total": int,
            "passed": int,
            "failures": int,
            "failure_summary": {warning_class: count},
            "failures_detail": [list of failure test info],
        }
    """
    print(f"  └─ Running promptfoo eval{f' ({label})' if label else ''}...")
    sys.stdout.flush()

    # Remove previous results
    if PROMPTFOO_RESULTS.exists():
        PROMPTFOO_RESULTS.unlink()

    # Run the eval via make test-prompt
    # We replicate the env setup from the Makefile
    env = settings.snapshot()

    # Source .env if it exists
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k, v)

    # Set extraction LLM vars for the provider
    env.setdefault("EXTRACTION_LLM_BINDING_HOST",
                    env.get("LLM_BINDING_HOST", "http://127.0.0.1:11438/v1"))
    env.setdefault("EXTRACTION_LLM_BINDING_API_KEY",
                    env.get("LLM_BINDING_API_KEY", "dummy"))
    env.setdefault("EXTRACTION_LLM_MODEL",
                    env.get("LLM_MODEL", "huihui-ai/Huihui-granite-4.1-3b-abliterated"))
    env.setdefault("EXTRACTION_LLM_MAX_TOKENS",
                    env.get("EXTRACTION_OPENAI_LLM_MAX_COMPLETION_TOKENS", "2048"))
    # Also set legacy APFEL_* names for backward compat with older provider versions
    env.setdefault("APFEL_OPENAI_BASE_URL", env["EXTRACTION_LLM_BINDING_HOST"])
    env.setdefault("APFEL_OPENAI_API_KEY", env["EXTRACTION_LLM_BINDING_API_KEY"])
    env.setdefault("APFEL_MODEL", env["EXTRACTION_LLM_MODEL"])
    env.setdefault("APFEL_MAX_TOKENS", env["EXTRACTION_LLM_MAX_TOKENS"])
    env.setdefault("OPENAI_LLM_INPUT_TOKEN_BUDGET",
                    env.get("MAX_EXTRACT_INPUT_TOKENS", "3072"))

    max_concurrency = settings.promptfoo_max_concurrency
    eval_timeout = settings.promptfoo_eval_timeout
    cmd = [
        "npx", "--yes", "promptfoo@latest", "eval",
        "-c", "evals/promptfooconfig.generated.yaml",
        "--max-concurrency", str(max_concurrency),
        "--no-cache",
        "-o", "evals/promptfoo-results.json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=eval_timeout,
            env=env,
        )
        if result.returncode != 0:
            print(f"    promptfoo stderr: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print(f"    promptfoo eval timed out ({eval_timeout}s)")
    except Exception as e:
        print(f"    promptfoo eval failed: {e}")

    # Parse results
    if not PROMPTFOO_RESULTS.exists():
        print("    No results file produced")
        return {"pass_rate": 0.0, "total": 0, "passed": 0, "failures": 0,
                "failure_summary": {}, "failures_detail": []}

    try:
        data = json.loads(PROMPTFOO_RESULTS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return {"pass_rate": 0.0, "total": 0, "passed": 0, "failures": 0,
                "failure_summary": {}, "failures_detail": []}

    results_raw = data.get("results")
    if results_raw is None:
        return {"pass_rate": 0.0, "total": 0, "passed": 0, "failures": 0,
                "failure_summary": {}, "failures_detail": []}
    results = results_raw.get("results", [])
    total = len(results)
    passed = sum(1 for r in results if r.get("success"))
    failures = total - passed

    # Classify failures by reading the assertion reason and testCase metadata
    failure_summary: Counter = Counter()
    failures_detail: list[dict] = []
    for r in results:
        if r.get("success"):
            continue
        reason = ""
        grading = r.get("gradingResult")
        if not grading or not isinstance(grading, dict):
            grading = {}
        component_results = grading.get("componentResults", [])
        for cr in component_results:
            if not cr.get("pass"):
                reason = cr.get("reason", "")
                break
        if not reason:
            reason = grading.get("reason", "")

        # Extract warning classes from reason string
        warning_classes = [wc for wc in extract_warning_classes(reason)
                           if wc not in EXCLUDED_WARNING_CLASSES]
        for wc in warning_classes:
            failure_summary[wc] += 1

        # Get test case metadata for richer classification
        test_case = r.get("testCase", {})
        metadata = test_case.get("metadata", {})
        captured_wc = [wc for wc in metadata.get("warning_classes", [])
                       if wc not in EXCLUDED_WARNING_CLASSES]
        for wc in captured_wc:
            failure_summary[wc] += 1

        failures_detail.append({
            "description": test_case.get("description", "?"),
            "reason": reason[:200],
            "warning_classes": warning_classes,
            "source": metadata.get("source", "unknown"),
            "input_preview": (test_case.get("vars", {}).get("input_text", "") or "")[:120],
        })

    return {
        "pass_rate": passed / max(total, 1),
        "total": total,
        "passed": passed,
        "failures": failures,
        "failure_summary": dict(failure_summary.most_common()),
        "failures_detail": failures_detail[:30],  # keep top 30 for context
    }


WARNING_PATTERNS: list[tuple[str, type]] = [
    (r"completion_missing", str),
    (r"relation_field_count", str),
    (r"relation_missing_keyword", str),
    (r"entity_invalid_type", str),
    (r"bad_request_400", str),
    (r"content_policy_violation", str),
    (r"sparse_entities?", str),
    (r"sparse_relations?", str),
    (r"token_budget", str),
    (r"legacy_fallback", str),
    (r"length_truncat", str),
    (r"parse_fail", str),
    (r"min_entit", str),
    (r"min_relat", str),
]


def extract_warning_classes(reason: str) -> list[str]:
    """Parse warning class names from an assertion reason string."""
    found: list[str] = []
    for pattern, _ in WARNING_PATTERNS:
        m = re.search(pattern, reason, re.IGNORECASE)
        if m:
            found.append(m.group(0).lower())
    return found


# ---------------------------------------------------------------------------
# Issue extraction from captured data
# ---------------------------------------------------------------------------

def run_extract_issues() -> bool:
    """Run the extract-prompt-issues.py script."""
    result = subprocess.run(
        [sys.executable, "scripts/extract-prompt-issues.py"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"extract-prompt-issues failed: {result.stderr[:500]}")
        return False
    print(f"  {result.stdout.strip()}")
    return True


# ---------------------------------------------------------------------------
# RL core: build targets from aggregated issues
# ---------------------------------------------------------------------------

def build_targets(issues: dict) -> list[dict]:
    """Rank prompt targets by issue frequency."""
    warning_classes = issues.get("warning_classes", {})
    extraction_tags = issues.get("extraction_tags", {})

    prompt_issue_map: dict[str, dict] = {}

    for wc_name, wc_info in warning_classes.items():
        if wc_name in EXCLUDED_WARNING_CLASSES:
            continue
        for suggestion in wc_info.get("top_prompt_suggestions", []):
            key = resolve_prompt_key(suggestion["prompt_key"])
            if key is None:
                continue
            if key not in prompt_issue_map:
                prompt_issue_map[key] = {"issues": [], "suggestions": set()}
            prompt_issue_map[key]["issues"].append(
                {"warning_class": wc_name, "count": wc_info["count"]}
            )
            prompt_issue_map[key]["suggestions"].add(suggestion["suggestion"])

    # Add sparse_entity/sparse_relation as extraction prompt issues
    tag_counts = extraction_tags.get("tag_counts", {})
    if tag_counts.get("sparse_entities", 0) > 3:
        k = "ENTITY_EXTRACTION_SYSTEM_PROMPT"
        if k not in prompt_issue_map:
            prompt_issue_map[k] = {"issues": [], "suggestions": set()}
        prompt_issue_map[k]["issues"].append(
            {"warning_class": "sparse_entities", "count": tag_counts["sparse_entities"]}
        )
        prompt_issue_map[k]["suggestions"].add(
            "Strengthen recall: instruct model to extract EVERY named entity, even seemingly minor ones."
        )

    if tag_counts.get("sparse_relations", 0) > 3:
        k = "ENTITY_EXTRACTION_SYSTEM_PROMPT"
        if k not in prompt_issue_map:
            prompt_issue_map[k] = {"issues": [], "suggestions": set()}
        prompt_issue_map[k]["issues"].append(
            {"warning_class": "sparse_relations", "count": tag_counts["sparse_relations"]}
        )
        prompt_issue_map[k]["suggestions"].add(
            "Encourage relation density: every entity should participate in at least one relation."
        )

    ranked = sorted(
        prompt_issue_map.items(),
        key=lambda kv: sum(i["count"] for i in kv[1]["issues"]),
        reverse=True,
    )

    current = read_current_prompts()
    targets = []
    for prompt_key, info in ranked:
        if prompt_key not in ALLOWED_PROMPT_KEYS:
            continue
        targets.append({
            "prompt_key": prompt_key,
            "current_content": current.get(prompt_key, ""),
            "issues": info["issues"],
            "suggestions": sorted(info["suggestions"]),
            "total_issue_count": sum(i["count"] for i in info["issues"]),
        })
    return targets


# ---------------------------------------------------------------------------
# LLM prompt templates for the subagent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE: str = (
    "You are a prompt engineer improving prompts for LightRAG, a knowledge-graph RAG system.\n\n"
    "You will receive:\n"
    "1. Current prompt text from lightrag/prompt.py\n"
    "2. Evaluation failure patterns from real model runs (warning classes, suggestions, example inputs)\n\n"
    "Rules:\n"
    "- Preserve ALL format placeholders: {tuple_delimiter}, {completion_delimiter}, {language}, "
    "{entity_types}, {input_text}, {draft_extraction}, {query}, {examples}, {context_data}, etc.\n"
    "- Return the COMPLETE new prompt text, not a diff\n"
    '- Do NOT use triple quotes (\\"\\"\\") inside new_content — the enclosing Python string uses them\n'
    "- Targeted fixes only — do not rewrite prompts that work fine\n"
    "- For completion_missing: reinforce that output must end with exactly `{completion_delimiter}` "
    "and no trailing text\n"
    "- For relation_field_count: tighten schema enforcement, say \"exactly 5 fields or skip the relation\"\n"
    "- For relation_missing_keyword: make field 4 (relationship_keywords) mandatory\n"
    "- For entity_invalid_type: constrain type to one plain label from the allowed list only\n"
    "- For sparse_entities: strengthen recall instruction\n"
    "- For sparse_relations: encourage connection density\n\n"
    "Output ONLY valid JSON:\n"
    '{\n'
    '  "changes": [\n'
    '    {\n'
    '      "prompt_key": "ENTITY_EXTRACTION_SYSTEM_PROMPT",\n'
    '      "reason": "Targets completion_missing (12 occurrences) — reinforced completion delimiter requirement",\n'
    '      "new_content": "The COMPLETE replacement prompt text..."\n'
    '    }\n'
    "  ]\n"
    "}"
)


def build_subagent_prompt(targets: list[dict], iteration: int, prev_pass_rate: float,
                           failure_summary: dict, failures_detail: list[dict]) -> str:
    """Build the prompt sent to the LLM subagent for this RL iteration."""
    sections = []
    sections.append(f"=== RL Iteration {iteration} ===")
    sections.append(f"Previous pass rate: {prev_pass_rate:.1%}")
    sections.append(f"Failure summary: {json.dumps(failure_summary, indent=2)}")

    # Top failures
    if failures_detail:
        sections.append("\nRecent failures (sample):\n")
        for fd in failures_detail[:8]:
            sections.append(f"  - [{fd['source']}] {fd['description']}")
            sections.append(f"    Reason: {fd['reason'][:150]}")
            if fd["warning_classes"]:
                sections.append(f"    Classes: {', '.join(fd['warning_classes'])}")

    sections.append("")
    for t in targets:
        issues_str = "\n".join(
            f"  - {i['warning_class']}: {i['count']} occurrences"
            for i in t["issues"]
        )
        suggestions_str = "\n".join(
            f"  - \"{s}\"" for s in t["suggestions"][:8]
        )
        sections.append(f"""--- Prompt: {t["prompt_key"]} ---

Current content:
```python
{t["current_content"]}
```

Issues ({t["total_issue_count"]} total):
{issues_str}

Top suggestions:
{suggestions_str}
""")

    sections.append("Analyze these issues and return improved prompt versions as JSON.")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# RL Loop
# ---------------------------------------------------------------------------

def main() -> int:
    # Parse --iters, --target, --dry-run, --skip-eval
    dry_run = "--dry-run" in sys.argv
    skip_eval = dry_run or "--skip-eval" in sys.argv
    max_iters = MAX_ITERATIONS
    target_filter = None
    for arg in sys.argv:
        if arg.startswith("--iters"):
            if "=" in arg:
                max_iters = int(arg.split("=", 1)[1])
            else:
                # handle --iters N (space-separated, need to look at next arg)
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                    max_iters = int(sys.argv[idx + 1])
        elif arg.startswith("--target"):
            if "=" in arg:
                target_filter = arg.split("=", 1)[1]
            else:
                idx = sys.argv.index(arg)
                if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                    target_filter = sys.argv[idx + 1]
        elif arg == "--skip-eval":
            skip_eval = True

    print(f"{"=" * 60}")
    print(f"LightRAG RL Prompt Enhancement")
    print(f"Max iterations: {max_iters} | Targets per iteration: {TARGETS_PER_ITERATION}")
    print(f"LLM subagent: {LLM_MODEL} @ {LLM_BINDING_HOST}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD:.1%}")
    print(f"{"=" * 60}\n")

    # Step 0: Run initial eval for baseline
    if not PROMPTFOO_CONFIG.exists():
        print("Building promptfoo config first...")
        subprocess.run(
            [sys.executable, "scripts/build-promptfoo-config.py"],
            capture_output=True, text=True, timeout=30,
        )

    # Backup original prompts
    original_source = PROMPT_FILE.read_text(encoding="utf-8")

    # Track best state
    best_pass_rate = 0.0
    best_source = original_source
    history: list[dict] = []

    # Pre-seed the enhancement log header
    PROMPT_DIFF_DIR.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, max_iters + 1):
        print(f"\n--- RL Iteration {iteration}/{max_iters} ---")

        # 1. Extract issues from captured data
        print(" [1/5] Extracting issues from captured data...")
        if not run_extract_issues():
            print("  Skipping iteration — extract failed")
            continue
        if not AGGREGATED_ISSUES.exists():
            print("  No issues file produced")
            continue
        issues = json.loads(AGGREGATED_ISSUES.read_text(encoding="utf-8"))

        # 2. Build targets from issues
        print(" [2/5] Building improvement targets from aggregated data...")
        targets = build_targets(issues)

        # Filter if user specified a specific target
        if target_filter:
            resolved = resolve_prompt_key(target_filter)
            if resolved:
                targets = [t for t in targets if t["prompt_key"] == resolved]
            if not targets:
                print(f"  No issues found for target '{target_filter}'")
                continue

        targets = targets[:TARGETS_PER_ITERATION]

        if not targets:
            print("  No actionable targets. Checking convergence...")
            break

        for t in targets:
            wc_names = ", ".join(sorted(set(i["warning_class"] for i in t["issues"])))
            print(f"  - {t['prompt_key']}: {t['total_issue_count']} issues [{wc_names}]")

        # 3. Run promptfoo eval for baseline
        print(" [3/5] Evaluating current prompts...")
        if skip_eval:
            print("  (skipped --dry-run or --skip-eval)")
            # Use failure data from captured issues as a proxy
            failure_summary_from_issues = {}
            for t in targets:
                for i in t["issues"]:
                    wc = i["warning_class"]
                    failure_summary_from_issues[wc] = failure_summary_from_issues.get(wc, 0) + i["count"]
            eval_result = {
                "pass_rate": 0.0,
                "total": 0,
                "passed": 0,
                "failures": 0,
                "failure_summary": failure_summary_from_issues,
                "failures_detail": [],
            }
        else:
            eval_result = run_promptfoo_eval(f"iteration {iteration} baseline")
        current_pass_rate = eval_result["pass_rate"]
        current_failure_summary = eval_result["failure_summary"]
        current_failures_detail = eval_result["failures_detail"]

        if eval_result["total"] > 0:
            print(f"  Pass rate: {current_pass_rate:.1%} "
                  f"({eval_result['passed']}/{eval_result['total']})")
        if current_failure_summary:
            print(f"  Failure summary: {dict(current_failure_summary)}")

        # 4. Call LLM subagent to suggest improvements
        print(" [4/5] Calling LLM subagent for prompt improvements...")
        subagent_prompt = build_subagent_prompt(
            targets, iteration, current_pass_rate,
            current_failure_summary, current_failures_detail,
        )

        if dry_run:
            print("  [Dry run] Would call LLM now. Skipping apply.")
            print(f"  Subagent prompt ({len(subagent_prompt)} chars):")
            print(subagent_prompt[:1500])
            if len(subagent_prompt) > 1500:
                print("  ... (truncated)")
            continue

        sys.stdout.flush()
        try:
            raw_response = call_llm(SYSTEM_PROMPT_TEMPLATE, subagent_prompt)
        except Exception as e:
            print(f"  LLM call failed: {e}")
            print("  Skipping iteration.")
            history.append({"iteration": iteration, "status": "llm_error", "error": str(e)})
            continue

        # Parse response JSON
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            brace_start = raw_response.find("{")
            brace_end = raw_response.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                json_str = raw_response[brace_start:brace_end + 1]
            else:
                print(f"  Could not find JSON in LLM response")
                print(f"  Response preview: {raw_response[:500]}")
                history.append({"iteration": iteration, "status": "parse_error"})
                continue

        try:
            llm_result = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            history.append({"iteration": iteration, "status": "json_error"})
            continue

        changes = llm_result.get("changes", [])
        if not changes:
            print("  LLM returned no changes.")
            history.append({"iteration": iteration, "status": "no_changes"})
            continue

        # Resolve prompt keys
        resolved_changes = []
        for c in changes:
            pk = resolve_prompt_key(c.get("prompt_key", ""))
            if pk is None:
                print(f"  Skipping unknown key: {c.get('prompt_key')}")
                continue
            c["prompt_key"] = pk
            resolved_changes.append(c)

        # 5. Apply changes
        print(" [5/5] Applying changes and re-evaluating...")
        source_before = PROMPT_FILE.read_text(encoding="utf-8")
        source = source_before
        applied = []
        for change in resolved_changes:
            pk = change["prompt_key"]
            new_content = change.get("new_content", "")
            reason = change.get("reason", "")
            if not new_content:
                continue
            try:
                source = replace_prompt_in_source(source, pk, new_content)
                applied.append({"prompt_key": pk, "reason": reason})
                print(f"  ✓ {pk}: {reason[:100]}")
            except ValueError as e:
                print(f"  ✗ {pk}: {e}")

        if not applied:
            print("  No changes applied.")
            history.append({"iteration": iteration, "status": "apply_failed"})
            continue

        # Write changes
        PROMPT_FILE.write_text(source, encoding="utf-8")

        # Sync exported prompts
        subprocess.run(
            [sys.executable, "scripts/export-prompts.py"],
            capture_output=True, text=True, timeout=15,
        )

        # Re-evaluate with new prompts
        new_eval = run_promptfoo_eval(f"iteration {iteration} after changes")
        new_pass_rate = new_eval["pass_rate"]

        print(f"\n  Reward signal: {current_pass_rate:.1%} → {new_pass_rate:.1%}")
        print(f"  Best so far:   {best_pass_rate:.1%}")

        # Save diff regardless
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        diff_path = PROMPT_DIFF_DIR / f"rl_iter{iteration}_{ts}.diff"
        diff_content = (
            f"# RL Iteration {iteration}\n"
            f"# Baseline pass rate: {current_pass_rate:.1%}\n"
            f"# Post-change pass rate: {new_pass_rate:.1%}\n"
            f"# Best pass rate: {best_pass_rate:.1%}\n"
            f"# Changes: {json.dumps([a['prompt_key'] for a in applied])}\n"
            f"# Failure summary before: {json.dumps(current_failure_summary)}\n"
            f"# Failure summary after: {json.dumps(new_eval['failure_summary'])}\n\n"
        )
        for a in applied:
            pk = a["prompt_key"]
            r_before = get_prompt_range(source_before, pk)
            r_after = get_prompt_range(source, pk)
            if r_before and r_after:
                diff_content += f"### {pk}\n# Reason: {a['reason']}\n"
                diff_content += "--- before\n" + source_before[r_before[0]:r_before[1]]
                diff_content += "\n+++ after\n" + source[r_after[0]:r_after[1]] + "\n\n"
        diff_path.write_text(diff_content, encoding="utf-8")
        print(f"  Diff saved: {diff_path}")

        # RL Decision: KEEP or REVERT
        reward = new_pass_rate - current_pass_rate
        if new_pass_rate >= best_pass_rate:
            best_pass_rate = new_pass_rate
            best_source = source
            status = "accepted"
            print(f"  ✅ KEPT — pass rate improved by {reward:+.1%}")
        else:
            # Revert
            PROMPT_FILE.write_text(source_before, encoding="utf-8")
            status = "reverted"
            print(f"  ❌ REVERTED — pass rate dropped by {reward:.1%}")

        # Log to enhancement results
        log_entry = {
            "timestamp": ts,
            "iteration": iteration,
            "diff_file": diff_path.name,
            "status": status,
            "baseline_pass_rate": round(current_pass_rate, 4),
            "post_change_pass_rate": round(new_pass_rate, 4),
            "best_pass_rate": round(best_pass_rate, 4),
            "reward": round(reward, 4),
            "changes_applied": applied,
            "failure_summary_before": current_failure_summary,
            "failure_summary_after": new_eval["failure_summary"],
            "target_prompts": [t["prompt_key"] for t in targets],
            "llm_model": LLM_MODEL,
        }
        with ENHANCEMENT_RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        history.append(log_entry)

        # Check convergence
        if reward < CONVERGENCE_THRESHOLD and new_pass_rate >= best_pass_rate:
            print(f"\n  Pass rate improvement ({reward:.1%}) below threshold "
                  f"({CONVERGENCE_THRESHOLD:.1%}). Converged!")
            break

    # Restore best state if last change was a revert
    current_source = PROMPT_FILE.read_text(encoding="utf-8")
    if current_source != best_source and best_pass_rate > 0:
        PROMPT_FILE.write_text(best_source, encoding="utf-8")
        subprocess.run(
            [sys.executable, "scripts/export-prompts.py"],
            capture_output=True, text=True, timeout=15,
        )
        print(f"\nRestored best prompts (pass rate: {best_pass_rate:.1%})")

    # Final report
    print(f"\n{'=' * 60}")
    print(f"RL Prompt Enhancement Complete")
    print(f"Iterations: {len(history)}")
    if history:
        accepted = sum(1 for h in history if h.get("status") == "accepted")
        reverted = sum(1 for h in history if h.get("status") == "reverted")
        best = max(h.get("best_pass_rate", 0) for h in history)
        print(f"Accepted: {accepted} | Reverted: {reverted} | Best pass rate: {best:.1%}")
    print(f"Results logged: {ENHANCEMENT_RESULTS}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
