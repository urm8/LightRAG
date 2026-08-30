# LightRAG Prompt Enhancement Pipeline

## Problem

Extraction, query, and enrichment prompts in `lightrag/prompt.py` degrade over time as:
- New document types trigger format violations not covered by examples
- Models may mishandle delimiter-based output or trigger safety filters
- Failure patterns repeat across capture runs with no systematic feedback into the prompt source

The `promptfooconfig.yaml` + captured JSONL data provides *detection* but not *correction*. This pipeline bridges that gap.

---

## Current State

PostgreSQL-backed runs now capture the live source data in four tables:

- `LIGHTRAG_QUERY_PROMPT` and `LIGHTRAG_EXTRACTION_PROMPT` store the exact
  system/user prompt bundle once per workspace. The SHA-256 digest of the full
  text is the prompt ID.
- `LIGHTRAG_QUERY_ATTEMPT` and `LIGHTRAG_EXTRACTION_ATTEMPT` store every LLM
  action, its input/output, warning classes, and phase metadata. Each attempt
  has a foreign key to the prompt version used.
- Capture is best-effort: a telemetry write failure is logged but never fails
  ingestion or query execution. Non-PostgreSQL deployments remain unchanged.

The JSONL files below remain portable evaluation exports. They are no longer
the only possible capture source.

```
captured/extraction_attempts.jsonl  ─┐
captured/lightrag_prompt_warnings.jsonl ─┤
captured/recent_log_chunks.jsonl    ─┤
  ┌──────────────────────────────────────┘
  ▼
build-promptfoo-config.py  ──► promptfooconfig.generated.yaml
                                       │
                                       ▼
                                 npx promptfoo eval
                                       │
                                       ▼
                                 promptfoo-results.json
                                       │
                                       ▼
                                 show-promptfoo-failures.py
                                       │
                                       ▼
                                 Terminal report (human reads it)
```

**Gap:** The report is human-consumed only. Nothing writes back to `lightrag/prompt.py`.

---

## Proposed Pipeline

```
                         ┌──────────────────────────────────────┐
                         │  CAPTURE & CATEGORIZE                │
                         │  ───────────────────────────────────  │
                         │  extraction_attempts.jsonl            │
                         │  + lightrag_prompt_warnings.jsonl     │
                         │  + query_attempts.jsonl               │
                         │  + recent_log_chunks.jsonl            │
                         └──────────────┬───────────────────────┘
                                        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 1: AGGREGATE PATTERNS  (scripts/aggregate-prompt-issues.py) │
  │  ─────────────────────────────────────────────────────────────  │
  │  Reads ALL captured JSONL files → produces:                    │
  │    evals/prompt_issues.aggregated.json                          │
  │  with warning class counts, per-class input examples,          │
  │  and per-class success/error rates over the last N-run window.  │
  └──────────────────────────────┬──────────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 2: SYNTHESIZE PROMPT FIXES  (scripts/synthesize-prompt-fixes.py) │
  │  ─────────────────────────────────────────────────────────────  │
  │  Reads aggregated issues → applies rule-based transformations   │
  │  to ENTITY_EXTRACTION_*_PROMPT / KEYWORDS_EXTRACTION / etc.     │
  │  in lightrag/prompt.py, then tests the diff with:               │
  │    npx promptfoo eval -c promptfoo.patch.yaml                    │
  │  Output: evals/prompt_diffs/proposed_YYYYMMDD_HHMMSS.diff       │
  └──────────────────────────────┬──────────────────────────────────┘
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 3: VALIDATE  (scripts/validate-prompt-diff.py)            │
  │  ─────────────────────────────────────────────────────────────  │
  │  Runs full promptfoo eval on the patched prompts, compares      │
  │  pass rate vs baseline, refuses to apply if pass rate drops.    │
  │  Produces: evals/prompt_diffs/validation_YYYYMMDD_HHMMSS.json    │
  └──────────────┬──────────────────────────────────────────────────┘
                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STAGE 4: APPLY & COMMIT  (scripts/apply-prompt-enhancement.sh)   │
  │  ─────────────────────────────────────────────────────────────  │
  │  Applies the validated diff to lightrag/prompt.py, updates       │
  │  evals/prompt_enhancement_log.jsonl with:                        │
  │    - timestamp, warning class, prompt key, before/after,        │
  │    - validation pass rate, number of impacted tests.             │
  │  Then runs export-prompts.py to sync evals/prompts.active.json.  │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1 — Aggregate Patterns

**File:** `scripts/aggregate-prompt-issues.py`

Reads all four captured JSONL files and produces a single aggregated report.

**Inputs:**
- `evals/captured/lightrag_prompt_warnings.jsonl`
- `evals/captured/extraction_attempts.jsonl`
- `evals/captured/query_attempts.jsonl`
- `evals/captured/recent_log_chunks.jsonl`

**Output:** `evals/prompt_issues.aggregated.json`

```json
{
  "aggregated_at": "2026-05-24T12:00:00",
  "window": "last 7 runs",
  "classes": {
    "completion_missing": {
      "count": 12,
      "trend": "stable",
      "top_prompt_suggestions": [
        {"prompt_key": "entity_extraction_system_prompt", "count": 12},
        {"prompt_key": "entity_extraction_user_prompt", "count": 3}
      ],
      "example_inputs": [
        "SECURITY.md: first 200 chars...",
        "AGENTS.md: first 200 chars..."
      ],
      "current_pass_rate": 0.68
    },
    "relation_field_count": {
      "count": 28,
      "trend": "growing",
      ...
    },
    "relation_missing_keyword": {
      "count": 22,
      ...
    },
    "entity_invalid_type": {
      "count": 15,
      ...
    },
    "bad_request_400": {
        # (excluded — provider-level Apple safety filter, not prompt quality)

    "content_policy_violation": {
        # (excluded — provider-level Apple safety filter, not prompt quality)
      "count": 8,
      ...
    },
    "sparse_entities": {
      "count": 45,
      "trend": "declining",
      ...
    },
    "sparse_relations": {
      "count": 38,
      ...
    }
  },
  "total_tests": 324,
  "overall_pass_rate": 0.72
}
```

**Implementation approach:**
- Parse all captured JSONL files
- Group by `warning_classes` (each record can have multiple)
- Track per-class: count, count trend (vs. previous aggregate), most common `prompt_suggestions`, example input excerpts
- Write to `evals/prompt_issues.aggregated.json`

### Stage 2 — Synthesize Prompt Fixes

**File:** `scripts/synthesize-prompt-fixes.py`

This is the core engine. It maps detected warning classes to concrete transformations of `lightrag/prompt.py`.

**Transformation Rules** (extensible table):

| Warning Class | Prompt Key | Transformation |
|---|---|---|
| `completion_missing` | `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Add sentence: "The final line MUST be exactly `{completion_delimiter}` with no trailing whitespace or text after it." |
| `relation_field_count` | `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Strengthen schema enforcement: add "Each relation record MUST have exactly 5 fields separated by `{tuple_delimiter}`. If you cannot fill all 5 fields, skip the relation entirely." |
| `relation_missing_keyword` | `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Add: "Field 4 (relationship_keywords) is MANDATORY - a comma-separated list of 1-4 short keywords describing the relationship type." |
| `entity_invalid_type` | `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Add: "Entity type MUST be one plain label from the allowed list only. Do NOT use quotes, brackets, pipes, slashes, angle brackets, or multi-word phrases as the type." |
| `bad_request_400` / `content_policy_violation` | `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Excluded — these are Apple safety filter rejections, not prompt quality issues. The model switch to granite4.1-abliterated (no filter) eliminates this class. |
| `sparse_entities` | `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Strengthen recall instruction: add "Be exhaustive — extract EVERY named entity, even seemingly minor ones. Under-extraction is worse than over-extraction." |
| `sparse_relations` | `ENTITY_EXTRACTION_SYSTEM_PROMPT` | Add "Every entity should participate in at least one relation. Connect related entities explicitly." |

Each rule application:
1. Reads `lightrag/prompt.py` as AST or line-by-line
2. Locates the target prompt constant
3. Appends/embeds the fix text while preserving Python string quoting
4. Writes a unified diff to `evals/prompt_diffs/proposed_*.diff`
5. Optionally runs a dry-run promptfoo eval on a minimal test subset to check the fix doesn't break existing passes

### Stage 3 — Validate

**File:** `scripts/validate-prompt-diff.py`

Ensures quality gate before writing to `lightrag/prompt.py`.

**Logic:**
1. Run `npx promptfoo eval -c evals/promptfooconfig.generated.yaml --max-concurrency 1 --no-cache` with **current** prompts → baseline pass rate
2. Apply the patch to `lightrag/prompt.py` (in-memory copy)
3. Run `export-prompts.py` to update `evals/prompts.active.json`
4. Run the same promptfoo eval with patched prompts → new pass rate
5. If new pass rate >= baseline pass rate - 0.02 (allow 2% tolerance), mark as VALIDATED
6. If new pass rate drops more than 2%, reject the diff and suggest the user inspect manually

**Output:** `evals/prompt_diffs/validation_*.json`

```json
{
  "diff_file": "proposed_20260524_120000.diff",
  "baseline_pass_rate": 0.72,
  "patched_pass_rate": 0.78,
  "delta": 0.06,
  "status": "VALIDATED",
  "tests_improved": ["relation_field_count", "completion_missing"],
  "tests_regressed": [],
  "suggested": true
}
```

### Stage 4 — Apply & Commit

**File:** `scripts/apply-prompt-enhancement.sh`

The last stage, meant to be run after validation passes.

**Steps:**
1. Apply the validated diff to `lightrag/prompt.py`
2. Run `scripts/export-prompts.py` to sync `evals/prompts.active.json`
3. Append a record to `evals/prompt_enhancement_log.jsonl`:

```json
{
  "timestamp": "2026-05-24T12:00:00",
  "diff_file": "proposed_20260524_120000.diff",
  "classes_targeted": ["completion_missing", "relation_field_count"],
  "prompt_keys_modified": ["ENTITY_EXTRACTION_SYSTEM_PROMPT"],
  "baseline_pass_rate": 0.72,
  "patched_pass_rate": 0.78,
  "impacted_test_count": 47
}
```

4. Print a summary for the developer

---

## Makefile Integration

New target to run the entire pipeline:

```makefile
.PHONY: enhance-prompts
enhance-prompts: ## Run full prompt enhancement pipeline
enhance-prompts: test-prompt aggregate-issues synthesize-fixes validate-fixes apply-fixes

.PHONY: aggregate-issues
aggregate-issues: ## Stage 1: aggregate warning patterns from captured data
	@python scripts/aggregate-prompt-issues.py

.PHONY: synthesize-fixes
synthesize-fixes: ## Stage 2: generate prompt diffs from aggregated issues
	@python scripts/synthesize-prompt-fixes.py

.PHONY: validate-fixes
validate-fixes: ## Stage 3: validate prompt diffs with promptfoo eval
	@python scripts/validate-prompt-diff.py

.PHONY: apply-fixes
apply-fixes: ## Stage 4: apply validated prompt diffs
	@bash scripts/apply-prompt-enhancement.sh
```

Also add a `make prompt-enhancement-report` target that prints the latest logs:

```makefile
.PHONY: prompt-enhancement-report
prompt-enhancement-report: ## Show prompt enhancement history
	@python -c "
import json
for line in open('evals/prompt_enhancement_log.jsonl'):
    r = json.loads(line)
    print(f\"{r['timestamp']} | {', '.join(r['classes_targeted'])} | {r['baseline_pass_rate']:.0%}→{r['patched_pass_rate']:.0%} | {r['diff_file']}\")
" 2>/dev/null || echo "No enhancement history yet."
```

---

## New Files Summary

| File | Purpose |
|---|---|
| `scripts/aggregate-prompt-issues.py` | Stage 1: aggregate failure patterns across captured JSONL |
| `scripts/synthesize-prompt-fixes.py` | Stage 2: map warning classes → prompt transformations |
| `scripts/validate-prompt-diff.py` | Stage 3: run promptfoo on patched prompts, gate on pass rate |
| `scripts/apply-prompt-enhancement.sh` | Stage 4: apply validated diffs, log results |
| `evals/prompt_issues.aggregated.json` | Stage 1 output: aggregated warning class stats |
| `evals/prompt_diffs/` | Stage 2 output: generated unified diffs |
| `evals/prompt_diffs/validation_*.json` | Stage 3 output: validation results |
| `evals/prompt_enhancement_log.jsonl` | Stage 4 output: permanent log of applied enhancements |

---

## Integration with Existing Captured Data

The existing `lightrag_prompt_warnings.jsonl` already contains `prompt_suggestions` — a field with `{prompt_file, prompt_key, suggestion}` objects. Stage 2 will consume these directly as the primary signal for what to fix. The rules table above serves as a safety net for when the LLM-generated suggestions are missing or incomplete.

The `extraction_attempts.jsonl` already contains parsed output with entity/relation counts and tags (`sparse_entities`, `sparse_relations`). Stage 1 will aggregate these tags to identify recall gaps.

The `query_attempts.jsonl` currently tracks keyword extraction and RAG answer prompts. Stage 1 will include query-phase issues in the aggregate.

---

## Guardrails

1. **Never degrade pass rate** — Stage 3 refuses to apply if pass rate drops more than 2%
2. **Human in the loop** — Stage 4 only applies the diff; it does not `git commit` automatically. The developer reviews before committing.
3. **Rollback capability** — `evals/prompt_enhancement_log.jsonl` records which diff was applied when. Reverting is `git checkout -- lightrag/prompt.py`.
4. **Per-diff granularity** — Each warning class gets its own diff. If one fix fails validation, it doesn't block others.
5. **Log-based evaluation window** — The pipeline considers the last N runs of data (configurable via `MAX_HISTORY_RUNS`), not all history, so trends reflect recent model/provider behavior.
