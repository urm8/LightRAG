SHELL := /bin/bash
SETUP_SCRIPT := scripts/setup/setup.sh
SETUP_BASH ?= $(or $(firstword $(wildcard /opt/homebrew/bin/bash /usr/local/bin/bash /opt/local/bin/bash)),$(shell command -v bash 2>/dev/null),bash)
SETUP_OPTS ?=
APFEL_LAUNCHD_LABEL ?= homebrew.mxcl.apfel
APFEL_LAUNCHD_PLIST ?= /Users/max/Library/LaunchAgents/homebrew.mxcl.apfel.plist
PROMPTFOO ?= npx --yes promptfoo@latest
PROMPTFOO_RESULTS ?= evals/promptfoo-results.json

LIGHTRAG_LAUNCHD_LABEL ?= com.local.lightrag
LIGHTRAG_LAUNCHD_PLIST ?= $(HOME)/Library/LaunchAgents/$(LIGHTRAG_LAUNCHD_LABEL).plist
LIGHTRAG_LOG_DIR ?= $(HOME)/Library/Logs/lightrag
COPILOT_API_LAUNCHD_LABEL ?= com.local.copilot-api
COPILOT_API_LAUNCHD_PLIST ?= $(HOME)/Library/LaunchAgents/$(COPILOT_API_LAUNCHD_LABEL).plist
COPILOT_API_LOG_DIR ?= $(HOME)/Library/Logs/copilot-api
VSCODE_BRIDGE_URL ?= http://localhost:8989/v1/models
COPILOT_API_URL ?= http://localhost:4141/v1/models
MLX_OPENAI_SERVER_HOST ?= 127.0.0.1
MLX_OPENAI_SERVER_PORT ?= 11436
MLX_OPENAI_SERVER_URL ?= http://$(MLX_OPENAI_SERVER_HOST):$(MLX_OPENAI_SERVER_PORT)/v1/models
MLX_OPENAI_SERVER_LOG_DIR ?= $(HOME)/Library/Logs/lightrag
MLX_MODEL_LAUNCHD_LABEL ?= com.local.mlx-model
MLX_MODEL_LAUNCHD_PLIST ?= $(HOME)/Library/LaunchAgents/$(MLX_MODEL_LAUNCHD_LABEL).plist
MLX_EMBEDDINGS_LAUNCHD_LABEL ?= com.local.mlx-embeddings
MLX_EMBEDDINGS_LAUNCHD_PLIST ?= $(HOME)/Library/LaunchAgents/$(MLX_EMBEDDINGS_LAUNCHD_LABEL).plist
MLX_EMBEDDINGS_HOST ?= 127.0.0.1
MLX_EMBEDDINGS_PORT ?= 11439
MLX_EMBEDDINGS_URL ?= http://$(MLX_EMBEDDINGS_HOST):$(MLX_EMBEDDINGS_PORT)/v1/models
MLX_AGENTCPM_MODEL ?= huihui-ai/Huihui-granite-4.1-3b-abliterated
MLX_AGENTCPM_MODEL_DIR ?= $(CURDIR)/models/huihui-granite-4.1-3b-abliterated-mlx-4bit
QUERY_MLX_MODEL ?= mlx-community/gemma-4-e4b-it-4bit
QUERY_MLX_HOST ?= http://$(MLX_OPENAI_SERVER_HOST):$(MLX_OPENAI_SERVER_PORT)/v1
MLX_AGENTCPM_MAX_KV_SIZE ?= 4096
MLX_AGENTCPM_CHAT_TEMPLATE_ARGS ?= {}
MLX_CHAT_URL ?= http://$(MLX_OPENAI_SERVER_HOST):$(MLX_OPENAI_SERVER_PORT)/v1/chat/completions
MLX_CHAT_MODEL ?= $(MLX_AGENTCPM_MODEL)
MLX_CHAT_PROMPT ?= Say ok.
MLX_CHAT_MAX_TOKENS ?= 16
MLX_CHAT_TIMEOUT ?= 300
MLX_CHAT_SESSION_MAX_TOKENS ?= 512
HN_FRONT_PAGE_LOAD_ARGS ?= --no-wait
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_BLUE := \033[34m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m

ifeq ($(NO_COLOR),1)
COLOR_RESET :=
COLOR_BOLD :=
COLOR_BLUE :=
COLOR_GREEN :=
COLOR_YELLOW :=
endif

.PHONY: help dev test-prompt test-prompt-results promptfoo-apfel promptfoo-capture-logs extract-prompt-issues improve-prompts rl-enhance-prompts prompt-enhancement-report lightrag-start lightrag-restart lightrag-stop lightrag-status lightrag-logs lightrag-clear-db clear-db lightrag-reindex reindex lightrag-rebuild-vdb vscode-bridge-health copilot-api-health bridge-health copilot-api-start copilot-api-restart copilot-api-stop copilot-api-status copilot-api-logs mlx-openai-health mlx-openai-logs mlx-model-install mlx-model-start mlx-model-restart mlx-model-stop mlx-model-status mlx-model-logs mlx-model-health mlx-embeddings-install mlx-embeddings-start mlx-embeddings-restart mlx-embeddings-stop mlx-embeddings-status mlx-embeddings-logs mlx-embeddings-health swiftlm-install mlx-agentcpm-install mlx-agentcpm-convert mlx-agentcpm-start mlx-agentcpm-restart mlx-agentcpm-stop mlx-agentcpm-status mlx-agentcpm-logs mlx-chat mlx-chat-native mlx-chat-test hn-front-page-load embedding-candidates-bench modernbert-embed-bench use-deepseek use-mlx configure env-base env-storage env-server env-validate env-backup env-security-check env-base-rewrite env-storage-rewrite env base storage server validate backup security security-check base-rewrite storage-rewrite

help:
	@printf "$(COLOR_BOLD)Interactive setup targets$(COLOR_RESET)\n"
	@printf "  $(COLOR_GREEN)make dev$(COLOR_RESET)                    Bootstrap local dev+test+offline env with uv + bun\n"
	@printf "  $(COLOR_GREEN)make test-prompt$(COLOR_RESET)            Evaluate extraction prompts against local apfel\n"
	@printf "  $(COLOR_GREEN)make test-prompt-results$(COLOR_RESET)    Show saved promptfoo test results\n"
	@printf "  $(COLOR_GREEN)make promptfoo-capture-logs$(COLOR_RESET) Import recent extraction chunks from logs into promptfoo cases\n"
	@printf "  $(COLOR_GREEN)make extract-prompt-issues$(COLOR_RESET)  Aggregate eval failure patterns from captured data\n"
	@printf "  $(COLOR_GREEN)make improve-prompts$(COLOR_RESET)        Single RL iteration: extract issues → LLM subagent → apply\n"
	@printf "  $(COLOR_GREEN)make rl-enhance-prompts$(COLOR_RESET)     Full RL loop: iteratively improve prompts until convergence\n"
	@printf "  $(COLOR_GREEN)make prompt-enhancement-report$(COLOR_RESET) Show RL enhancement history\n"
	@printf "  $(COLOR_GREEN)make lightrag-start$(COLOR_RESET)         Start the local launchd LightRAG service\n"
	@printf "  $(COLOR_GREEN)make lightrag-restart$(COLOR_RESET)       Restart the local launchd LightRAG service\n"
	@printf "  $(COLOR_GREEN)make lightrag-stop$(COLOR_RESET)          Stop the local launchd LightRAG service\n"
	@printf "  $(COLOR_GREEN)make lightrag-status$(COLOR_RESET)        Print local launchd LightRAG service status\n"
	@printf "  $(COLOR_GREEN)make lightrag-logs$(COLOR_RESET)          Tail local launchd LightRAG logs\n"
	@printf "  $(COLOR_GREEN)make lightrag-clear-db$(COLOR_RESET)      Delete all ingested documents, chunks, graph, and vectors\n"
	@printf "  $(COLOR_GREEN)make lightrag-rebuild-vdb$(COLOR_RESET)   Rebuild vectors after stopping LightRAG\n"
	@printf "  $(COLOR_GREEN)make vscode-bridge-health$(COLOR_RESET)   Check VS Code Copilot bridge /v1/models\n"
	@printf "  $(COLOR_GREEN)make copilot-api-health$(COLOR_RESET)     Check copilot-api /v1/models\n"
	@printf "  $(COLOR_GREEN)make bridge-health$(COLOR_RESET)          Check both local Copilot bridges\n"
	@printf "  $(COLOR_GREEN)make copilot-api-start$(COLOR_RESET)      Start the local launchd Copilot API service\n"
	@printf "  $(COLOR_GREEN)make copilot-api-restart$(COLOR_RESET)    Restart the local launchd Copilot API service\n"
	@printf "  $(COLOR_GREEN)make copilot-api-stop$(COLOR_RESET)       Stop the local launchd Copilot API service\n"
	@printf "  $(COLOR_GREEN)make copilot-api-status$(COLOR_RESET)     Print local launchd Copilot API service status\n"
	@printf "  $(COLOR_GREEN)make copilot-api-logs$(COLOR_RESET)       Tail local launchd Copilot API logs\n"
	@printf "  $(COLOR_GREEN)make mlx-model-restart$(COLOR_RESET)      Restart the local launchd MLX model service\n"
	@printf "  $(COLOR_GREEN)make mlx-model-health$(COLOR_RESET)       Check the local MLX model service\n"
	@printf "  $(COLOR_GREEN)make mlx-embeddings-restart$(COLOR_RESET) Restart the local launchd MLX embeddings service\n"
	@printf "  $(COLOR_GREEN)make mlx-embeddings-health$(COLOR_RESET)  Check the local MLX embeddings service\n"
	@printf "  $(COLOR_GREEN)make swiftlm-install$(COLOR_RESET)        Build native SwiftLM and the local BGE-M3 embedding runtime\n"
	@printf "  $(COLOR_GREEN)make mlx-agentcpm-convert$(COLOR_RESET)   Convert AgentCPM-Explore to local MLX 4-bit model\n"
	@printf "  $(COLOR_GREEN)make mlx-chat$(COLOR_RESET)               Open an interactive chat session with the local MLX model\n"
	@printf "  $(COLOR_GREEN)make mlx-chat-native$(COLOR_RESET)        Open native mlx_lm.chat against the local MLX model files\n"
	@printf "  $(COLOR_GREEN)make mlx-chat-test$(COLOR_RESET)          Send a chat smoke test to the local MLX chat endpoint\n"
	@printf "  $(COLOR_GREEN)make embedding-candidates-bench$(COLOR_RESET) Benchmark isolated embedding candidates\n"
	@printf "  $(COLOR_GREEN)make modernbert-embed-bench$(COLOR_RESET) Benchmark isolated ModernBERT embeddings\n"
	@printf "  $(COLOR_GREEN)make hn-front-page-load$(COLOR_RESET)     Load Hacker News front-page posts into LightRAG via uv\n"
	@printf "  $(COLOR_GREEN)make use-deepseek$(COLOR_RESET)           Switch query roles to DeepSeek and restart LightRAG\n"
	@printf "  $(COLOR_GREEN)make use-mlx$(COLOR_RESET)                Switch query roles to local MLX and restart LightRAG\n"
	@printf "  $(COLOR_GREEN)make env-base$(COLOR_RESET)               Configure LLM, embedding, and reranker (run first)\n"
	@printf "  $(COLOR_GREEN)make env-storage$(COLOR_RESET)            Configure storage backends and databases\n"
	@printf "  $(COLOR_GREEN)make env-server$(COLOR_RESET)             Configure server, security, and SSL\n"
	@printf "  $(COLOR_GREEN)make env-validate$(COLOR_RESET)           Validate existing .env\n"
	@printf "  $(COLOR_GREEN)make env-security-check$(COLOR_RESET)     Audit existing .env for security risks\n"
	@printf "  $(COLOR_GREEN)make env-backup$(COLOR_RESET)             Backup current .env\n"
	@printf "  $(COLOR_GREEN)make env-base-rewrite$(COLOR_RESET)       Force-regenerate wizard-managed compose services during base setup\n"
	@printf "  $(COLOR_GREEN)make env-storage-rewrite$(COLOR_RESET)    Force-regenerate wizard-managed compose services during storage setup\n"
	@printf "  $(COLOR_GREEN)make base$(COLOR_RESET)                   Short form of make env-base (all env prefix can be stripped)\n"
	@printf "\n"
	@printf "$(COLOR_BOLD)Typical workflow$(COLOR_RESET)\n"
	@printf "  1. make dev            # install backend/test deps and build frontend\n"
	@printf "  2. make env-base       # set LLM/embedding/reranker\n"
	@printf "  3. make env-storage    # set storage backends (optional)\n"
	@printf "  4. make env-server     # set port/security/SSL (optional)\n\n"
	@printf "$(COLOR_BOLD)Examples$(COLOR_RESET)\n"
	@printf "  make dev\n"
	@printf "  make test-prompt\n"
	@printf "  make test-prompt-results\n"
	@printf "  make copilot-api-restart\n"
	@printf "  make mlx-model-health\n"
	@printf "  make mlx-embeddings-health\n"
	@printf "  make lightrag-restart\n"
	@printf "  make lightrag-logs\n"
	@printf "  make env-base\n"
	@printf "  make env-storage SETUP_OPTS=--debug\n"
	@printf "  make env-server\n\n"
	@printf "  make env-storage-rewrite\n\n"
	@printf "  make env-security-check\n\n"
	@printf "$(COLOR_BOLD)Compose Output$(COLOR_RESET)\n"
	@printf "  Bundled service images are defined in scripts/setup/templates/*.yml.\n"
	@printf "  Compose file output: docker-compose.final.yml\n"

dev:
	@if ! command -v uv >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)uv is required for make dev.$(COLOR_RESET)\n"; \
		printf "Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n"; \
		printf "Unix/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh\n"; \
		printf "Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n"; \
		exit 1; \
	fi
	@if ! command -v bun >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)bun is required for make dev.$(COLOR_RESET)\n"; \
		printf "Install Bun first: https://bun.sh/docs/installation\n"; \
		printf "macOS/Linux: curl -fsSL https://bun.sh/install | bash\n"; \
		printf "Windows: powershell -c \"irm bun.sh/install.ps1 | iex\"\n"; \
		exit 1; \
	fi
	@printf "$(COLOR_BLUE)Syncing backend and test dependencies with uv...$(COLOR_RESET)\n"
	@uv sync --extra test --extra offline
	@printf "$(COLOR_BLUE)Installing frontend dependencies with Bun...$(COLOR_RESET)\n"
	@cd lightrag_webui && bun install --frozen-lockfile
	@printf "$(COLOR_BLUE)Building frontend assets...$(COLOR_RESET)\n"
	@cd lightrag_webui && bun run build
	@printf "$(COLOR_GREEN)Development environment is ready.$(COLOR_RESET)\n"
	@printf "Next steps:\n"
	@printf "  source .venv/bin/activate\n"
	@printf "  make env-base\n"
	@printf "  lightrag-server\n"

promptfoo-capture-logs:
	@.venv/bin/python scripts/capture_recent_log_chunks.py

test-prompt:
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	APFEL_OPENAI_BASE_URL="$${APFEL_OPENAI_BASE_URL:-$${EXTRACTION_LLM_BINDING_HOST:-$${LLM_BINDING_HOST:-http://127.0.0.1:11435/v1}}}" \
	APFEL_OPENAI_API_KEY="$${APFEL_OPENAI_API_KEY:-$${EXTRACTION_LLM_BINDING_API_KEY:-$${LLM_BINDING_API_KEY:-dummy}}}" \
	APFEL_MODEL="$${APFEL_MODEL:-$${EXTRACTION_LLM_MODEL:-$${LLM_MODEL:-apple-foundationmodel}}}" \
	APFEL_MAX_TOKENS="$${APFEL_MAX_TOKENS:-$${EXTRACTION_OPENAI_LLM_MAX_COMPLETION_TOKENS:-2048}}" \
	OPENAI_LLM_INPUT_TOKEN_BUDGET="$${OPENAI_LLM_INPUT_TOKEN_BUDGET:-$${MAX_EXTRACT_INPUT_TOKENS:-3072}}" \
	.venv/bin/python scripts/capture_recent_log_chunks.py && \
	.venv/bin/python scripts/export-prompts.py && \
	.venv/bin/python scripts/build-promptfoo-config.py && \
	.venv/bin/python -c 'from pathlib import Path; Path("$(PROMPTFOO_RESULTS)").unlink(missing_ok=True)' && \
	APFEL_OPENAI_BASE_URL="$${APFEL_OPENAI_BASE_URL:-$${EXTRACTION_LLM_BINDING_HOST:-$${LLM_BINDING_HOST:-http://127.0.0.1:11435/v1}}}" \
	APFEL_OPENAI_API_KEY="$${APFEL_OPENAI_API_KEY:-$${EXTRACTION_LLM_BINDING_API_KEY:-$${LLM_BINDING_API_KEY:-dummy}}}" \
	APFEL_MODEL="$${APFEL_MODEL:-$${EXTRACTION_LLM_MODEL:-$${LLM_MODEL:-apple-foundationmodel}}}" \
	APFEL_MAX_TOKENS="$${APFEL_MAX_TOKENS:-$${EXTRACTION_OPENAI_LLM_MAX_COMPLETION_TOKENS:-2048}}" \
	APFEL_SYSTEM_PROMPT="$$APFEL_SYSTEM_PROMPT" \
	OPENAI_LLM_INPUT_TOKEN_BUDGET="$${OPENAI_LLM_INPUT_TOKEN_BUDGET:-$${MAX_EXTRACT_INPUT_TOKENS:-3072}}" \
	PYTHON="$${PYTHON:-.venv/bin/python}" \
	$(PROMPTFOO) eval -c evals/promptfooconfig.generated.yaml --max-concurrency 1 --no-cache -o "$(PROMPTFOO_RESULTS)"; \
	eval_status=$$?; \
	.venv/bin/python scripts/show-promptfoo-failures.py "$(PROMPTFOO_RESULTS)"; \
	exit $$eval_status

test-prompt-results:
	@.venv/bin/python scripts/show-promptfoo-failures.py "$(PROMPTFOO_RESULTS)"

# ---------------------------------------------------------------------------
# RL Prompt Enhancement Pipeline
# ---------------------------------------------------------------------------

extract-prompt-issues:
	@.venv/bin/python scripts/extract-prompt-issues.py

improve-prompts:
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	.venv/bin/python scripts/extract-prompt-issues.py && \
	.venv/bin/python scripts/improve-prompts.py --iters 1

rl-enhance-prompts:
	@echo "$(COLOR_BOLD)Starting RL Prompt Enhancement Loop$(COLOR_RESET)"
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	.venv/bin/python scripts/extract-prompt-issues.py && \
	.venv/bin/python scripts/improve-prompts.py

prompt-enhancement-report:
	@printf "$(COLOR_BOLD)RL Prompt Enhancement History$(COLOR_RESET)\n"
	@if [ -f evals/prompt_enhancement_results.jsonl ]; then \
		python3 -c "import json; from pathlib import Path; lines = Path('evals/prompt_enhancement_results.jsonl').read_text().splitlines(); [print(f\"{'✅' if r.get('status') == 'accepted' else '❌' if r.get('status') == 'reverted' else '⚠️'} Iter {r.get('iteration', '?')}: {r.get('baseline_pass_rate', 0)*100:.0f}%→{r.get('post_change_pass_rate', 0)*100:.0f}% (best: {r.get('best_pass_rate', 0)*100:.0f}%) | {', '.join(c['prompt_key'] for c in r.get('changes_applied', []))[:80]}\") for r in (json.loads(line) for line in lines)]"; \
	else \
		echo "No enhancement history yet. Run 'make rl-enhance-prompts' first."; \
	fi
	@printf "\nEnhancement results: $(COLOR_GREEN)evals/prompt_enhancement_results.jsonl$(COLOR_RESET)\n"
	@printf "Prompt diffs:        $(COLOR_GREEN)evals/prompt_diffs/$(COLOR_RESET)\n"

promptfoo-apfel: test-prompt

apfel-start:
	brew services start apfel --file="/Users/max/.config/launchd/apfel-11435.plist"
	@printf "$(COLOR_GREEN)Started $(APFEL_LAUNCHD_LABEL).$(COLOR_RESET)\n"
	@printf $$(curl http://127.0.0.1:11435/health)

apfel-stop:
	brew services stop apfel 
	@printf "$(COLOR_GREEN)Stopped $(APFEL_LAUNCHD_LABEL).$(COLOR_RESET)\n"

apfel-restart: 
	apfel-stop
	apfel-start
	@printf "$(COLOR_GREEN)Restarted $(APFEL_LAUNCHD_LABEL).$(COLOR_RESET)\n"

apfel-logs:
	@tail -f "/opt/homebrew/var/log/apfel.log" "/opt/homebrew/var/log/apfel.err.log"

lightrag-start:
	@launchctl bootstrap gui/$$(id -u) "$(LIGHTRAG_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl kickstart gui/$$(id -u)/$(LIGHTRAG_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Started $(LIGHTRAG_LAUNCHD_LABEL).$(COLOR_RESET)\n"

lightrag-restart:
	@launchctl bootout gui/$$(id -u) "$(LIGHTRAG_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl bootstrap gui/$$(id -u) "$(LIGHTRAG_LAUNCHD_PLIST)"
	@launchctl kickstart gui/$$(id -u)/$(LIGHTRAG_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Restarted $(LIGHTRAG_LAUNCHD_LABEL).$(COLOR_RESET)\n"

lightrag-stop:
	@launchctl bootout gui/$$(id -u) "$(LIGHTRAG_LAUNCHD_PLIST)"
	@printf "$(COLOR_GREEN)Stopped $(LIGHTRAG_LAUNCHD_LABEL).$(COLOR_RESET)\n"

lightrag-status:
	@launchctl print gui/$$(id -u)/$(LIGHTRAG_LAUNCHD_LABEL)

lightrag-logs:
	@tail -f "$(LIGHTRAG_LOG_DIR)/lightrag.out.log" "$(LIGHTRAG_LOG_DIR)/lightrag.err.log"

lightrag-clear-db clear-db:
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	base_url="http://$${HOST:-127.0.0.1}:$${PORT:-9621}"; \
	auth_header=""; \
	if [ -n "$${LIGHTRAG_API_KEY:-}" ]; then \
		auth_header="-H X-API-Key:$${LIGHTRAG_API_KEY}"; \
	fi; \
	code=$$(curl -sS -o /tmp/lightrag-clear-db.json -w '%{http_code}' -X DELETE $$auth_header "$$base_url/documents" 2>/tmp/lightrag-clear-db.err || true); \
	if [ "$$code" = "200" ]; then \
		printf "$(COLOR_GREEN)LightRAG cleared$(COLOR_RESET) %s\n" "$$base_url/documents"; \
		cat /tmp/lightrag-clear-db.json; \
		printf "\n"; \
	else \
		printf "$(COLOR_YELLOW)LightRAG clear failed$(COLOR_RESET) %s (http=%s)\n" "$$base_url/documents" "$${code:-000}"; \
		[ ! -s /tmp/lightrag-clear-db.err ] || sed 's/^/  /' /tmp/lightrag-clear-db.err; \
		[ ! -s /tmp/lightrag-clear-db.json ] || sed 's/^/  /' /tmp/lightrag-clear-db.json; \
		exit 1; \
	fi

lightrag-reindex reindex: lightrag-clear-db
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	base_url="http://$${HOST:-127.0.0.1}:$${PORT:-9621}"; \
	auth_header=""; \
	if [ -n "$${LIGHTRAG_API_KEY:-}" ]; then \
		auth_header="-H X-API-Key:$${LIGHTRAG_API_KEY}"; \
	fi; \
	input_dir="$${INPUT_DIR:-inputs}"; \
	enqueued_dir="$${input_dir}/__enqueued__"; \
	if [ -d "$$enqueued_dir" ]; then \
		count=$$(ls -1 "$$enqueued_dir" 2>/dev/null | wc -l | tr -d ' '); \
		if [ "$$count" -gt 0 ]; then \
			printf "$(COLOR_BOLD)Moving %d files from __enqueued__ back to input directory...$(COLOR_RESET)\n" "$$count"; \
			mv "$$enqueued_dir"/* "$$input_dir"/ 2>/dev/null || true; \
		else \
			printf "$(COLOR_YELLOW)No files in __enqueued__ to restore.$(COLOR_RESET)\n"; \
		fi; \
	else \
		printf "$(COLOR_YELLOW)No __enqueued__ directory found.$(COLOR_RESET)\n"; \
	fi; \
	printf "$(COLOR_BOLD)Triggering document scan...$(COLOR_RESET)\n"; \
	code=$$(curl -sS -o /tmp/lightrag-reindex.json -w '%{http_code}' -X POST $$auth_header "$$base_url/documents/scan" 2>/tmp/lightrag-reindex.err || true); \
	if [ "$$code" = "200" ]; then \
		printf "$(COLOR_GREEN)Reindex triggered$(COLOR_RESET) %s\n" "$$base_url/documents/scan"; \
		cat /tmp/lightrag-reindex.json; \
		printf "\n$(COLOR_BOLD)Done. Run 'make lightrag-logs' to monitor progress.$(COLOR_RESET)\n"; \
	else \
		printf "$(COLOR_YELLOW)Reindex trigger failed$(COLOR_RESET) %s (http=%s)\n" "$$base_url/documents/scan" "$${code:-000}"; \
		[ ! -s /tmp/lightrag-reindex.err ] || sed 's/^/  /' /tmp/lightrag-reindex.err; \
		[ ! -s /tmp/lightrag-reindex.json ] || sed 's/^/  /' /tmp/lightrag-reindex.json; \
		exit 1; \
	fi

embedding-candidates-bench:
	@.venv/bin/python scripts/benchmark_embedding_candidates.py

modernbert-embed-bench:
	@.venv/bin/python scripts/benchmark_embedding_candidates.py --profiles modernbert

lightrag-rebuild-vdb:
	@printf "$(COLOR_YELLOW)Stop LightRAG first with 'make lightrag-stop'; this tool can drop and rebuild vector storages.$(COLOR_RESET)\n"
	@.venv/bin/python -m lightrag.tools.rebuild_vdb

vscode-bridge-health:
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	code=$$(curl -sS -o /tmp/vscode-bridge-health.json -w '%{http_code}' \
		-H "Authorization: Bearer $$LLM_BINDING_API_KEY" "$(VSCODE_BRIDGE_URL)" 2>/tmp/vscode-bridge-health.err || true); \
	if [ "$$code" = "200" ]; then \
		count=$$(python3 -c 'import json; print(len(json.load(open("/tmp/vscode-bridge-health.json")).get("data", [])))' 2>/dev/null || printf "?"); \
		printf "$(COLOR_GREEN)VS Code bridge OK$(COLOR_RESET) %s (%s models)\n" "$(VSCODE_BRIDGE_URL)" "$$count"; \
	else \
		printf "$(COLOR_YELLOW)VS Code bridge unavailable$(COLOR_RESET) %s (http=%s)\n" "$(VSCODE_BRIDGE_URL)" "$${code:-000}"; \
		[ ! -s /tmp/vscode-bridge-health.err ] || sed 's/^/  /' /tmp/vscode-bridge-health.err; \
		exit 1; \
	fi

copilot-api-health:
	@code=$$(curl -sS -o /tmp/copilot-api-health.json -w '%{http_code}' \
		-H "Authorization: Bearer dummy" "$(COPILOT_API_URL)" 2>/tmp/copilot-api-health.err || true); \
	if [ "$$code" = "200" ]; then \
		count=$$(python3 -c 'import json; print(len(json.load(open("/tmp/copilot-api-health.json")).get("data", [])))' 2>/dev/null || printf "?"); \
		printf "$(COLOR_GREEN)copilot-api OK$(COLOR_RESET) %s (%s models)\n" "$(COPILOT_API_URL)" "$$count"; \
	else \
		printf "$(COLOR_YELLOW)copilot-api unavailable$(COLOR_RESET) %s (http=%s)\n" "$(COPILOT_API_URL)" "$${code:-000}"; \
		[ ! -s /tmp/copilot-api-health.err ] || sed 's/^/  /' /tmp/copilot-api-health.err; \
		exit 1; \
	fi

bridge-health: vscode-bridge-health copilot-api-health

copilot-api-start:
	@launchctl bootstrap gui/$$(id -u) "$(COPILOT_API_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl kickstart gui/$$(id -u)/$(COPILOT_API_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Started $(COPILOT_API_LAUNCHD_LABEL).$(COLOR_RESET)\n"

copilot-api-restart:
	@launchctl bootout gui/$$(id -u) "$(COPILOT_API_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl bootstrap gui/$$(id -u) "$(COPILOT_API_LAUNCHD_PLIST)"
	@launchctl kickstart gui/$$(id -u)/$(COPILOT_API_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Restarted $(COPILOT_API_LAUNCHD_LABEL).$(COLOR_RESET)\n"

copilot-api-stop:
	@launchctl bootout gui/$$(id -u) "$(COPILOT_API_LAUNCHD_PLIST)"
	@printf "$(COLOR_GREEN)Stopped $(COPILOT_API_LAUNCHD_LABEL).$(COLOR_RESET)\n"

copilot-api-status:
	@launchctl print gui/$$(id -u)/$(COPILOT_API_LAUNCHD_LABEL)

copilot-api-logs:
	@tail -f "$(COPILOT_API_LOG_DIR)/copilot-api.out.log" "$(COPILOT_API_LOG_DIR)/copilot-api.err.log"

mlx-model-install:
	@MLX_MODEL_LAUNCHD_LABEL="$(MLX_MODEL_LAUNCHD_LABEL)" .venv/bin/python scripts/install_mlx_model_launchd.py

mlx-model-start: mlx-model-install
	@launchctl bootstrap gui/$$(id -u) "$(MLX_MODEL_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl kickstart -k gui/$$(id -u)/$(MLX_MODEL_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Started $(MLX_MODEL_LAUNCHD_LABEL).$(COLOR_RESET)\n"

mlx-model-restart: mlx-model-install
	@launchctl bootout gui/$$(id -u) "$(MLX_MODEL_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl bootstrap gui/$$(id -u) "$(MLX_MODEL_LAUNCHD_PLIST)"
	@launchctl kickstart -k gui/$$(id -u)/$(MLX_MODEL_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Restarted $(MLX_MODEL_LAUNCHD_LABEL).$(COLOR_RESET)\n"

mlx-model-stop:
	@launchctl bootout gui/$$(id -u) "$(MLX_MODEL_LAUNCHD_PLIST)"
	@printf "$(COLOR_GREEN)Stopped $(MLX_MODEL_LAUNCHD_LABEL).$(COLOR_RESET)\n"

mlx-model-status:
	@launchctl print gui/$$(id -u)/$(MLX_MODEL_LAUNCHD_LABEL)

mlx-model-logs:
	@tail -f "$(MLX_OPENAI_SERVER_LOG_DIR)/mlx-model.out.log" "$(MLX_OPENAI_SERVER_LOG_DIR)/mlx-model.err.log"

mlx-model-health:
	@code=$$(curl -sS -o /tmp/mlx-openai-health.json -w '%{http_code}' \
		-H "Authorization: Bearer dummy" "$(MLX_OPENAI_SERVER_URL)" 2>/tmp/mlx-openai-health.err || true); \
	if [ "$$code" = "200" ]; then \
		count=$$(python3 -c 'import json; print(len(json.load(open("/tmp/mlx-openai-health.json")).get("data", [])))' 2>/dev/null || printf "?"); \
		printf "$(COLOR_GREEN)MLX model OK$(COLOR_RESET) %s (%s models)\n" "$(MLX_OPENAI_SERVER_URL)" "$$count"; \
	else \
		printf "$(COLOR_YELLOW)MLX model unavailable$(COLOR_RESET) %s (http=%s)\n" "$(MLX_OPENAI_SERVER_URL)" "$${code:-000}"; \
		[ ! -s /tmp/mlx-openai-health.err ] || sed 's/^/  /' /tmp/mlx-openai-health.err; \
		exit 1; \
	fi

mlx-embeddings-install:
	@MLX_EMBEDDINGS_LAUNCHD_LABEL="$(MLX_EMBEDDINGS_LAUNCHD_LABEL)" .venv/bin/python scripts/install_mlx_embeddings_launchd.py

mlx-embeddings-start: mlx-embeddings-install
	@launchctl bootstrap gui/$$(id -u) "$(MLX_EMBEDDINGS_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl kickstart -k gui/$$(id -u)/$(MLX_EMBEDDINGS_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Started $(MLX_EMBEDDINGS_LAUNCHD_LABEL).$(COLOR_RESET)\n"

mlx-embeddings-restart: mlx-embeddings-install
	@launchctl bootout gui/$$(id -u) "$(MLX_EMBEDDINGS_LAUNCHD_PLIST)" 2>/dev/null || true
	@launchctl bootstrap gui/$$(id -u) "$(MLX_EMBEDDINGS_LAUNCHD_PLIST)"
	@launchctl kickstart -k gui/$$(id -u)/$(MLX_EMBEDDINGS_LAUNCHD_LABEL)
	@printf "$(COLOR_GREEN)Restarted $(MLX_EMBEDDINGS_LAUNCHD_LABEL).$(COLOR_RESET)\n"

mlx-embeddings-stop:
	@launchctl bootout gui/$$(id -u) "$(MLX_EMBEDDINGS_LAUNCHD_PLIST)"
	@printf "$(COLOR_GREEN)Stopped $(MLX_EMBEDDINGS_LAUNCHD_LABEL).$(COLOR_RESET)\n"

mlx-embeddings-status:
	@launchctl print gui/$$(id -u)/$(MLX_EMBEDDINGS_LAUNCHD_LABEL)

mlx-embeddings-logs:
	@tail -f "$(MLX_OPENAI_SERVER_LOG_DIR)/mlx-embeddings.out.log" "$(MLX_OPENAI_SERVER_LOG_DIR)/mlx-embeddings.err.log"

mlx-embeddings-health:
	@code=$$(curl -sS -o /tmp/mlx-embeddings-health.json -w '%{http_code}' \
		-H "Authorization: Bearer dummy" "$(MLX_EMBEDDINGS_URL)" 2>/tmp/mlx-embeddings-health.err || true); \
	if [ "$$code" = "200" ]; then \
		count=$$(python3 -c 'import json; print(len(json.load(open("/tmp/mlx-embeddings-health.json")).get("data", [])))' 2>/dev/null || printf "?"); \
		printf "$(COLOR_GREEN)MLX embeddings OK$(COLOR_RESET) %s (%s models)\n" "$(MLX_EMBEDDINGS_URL)" "$$count"; \
	else \
		printf "$(COLOR_YELLOW)MLX embeddings unavailable$(COLOR_RESET) %s (http=%s)\n" "$(MLX_EMBEDDINGS_URL)" "$${code:-000}"; \
		[ ! -s /tmp/mlx-embeddings-health.err ] || sed 's/^/  /' /tmp/mlx-embeddings-health.err; \
		exit 1; \
	fi

mlx-openai-health: mlx-model-health

mlx-openai-logs: mlx-model-logs

swiftlm-install:
	@./scripts/install_swiftlm_runtime.sh

mlx-agentcpm-install:
	@printf "$(COLOR_YELLOW)mlx-agentcpm-install is deprecated.$(COLOR_RESET)\n"
	@printf "Use make swiftlm-install, then make mlx-model-restart.\n"

mlx-agentcpm-convert:
	@mkdir -p "$(dir $(MLX_AGENTCPM_MODEL_DIR))"
	@if printf '%s' "$(MLX_AGENTCPM_MODEL)" | grep -qi 'mlx'; then \
		.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="$(MLX_AGENTCPM_MODEL)", local_dir="$(MLX_AGENTCPM_MODEL_DIR)", local_dir_use_symlinks=False)'; \
	else \
		.venv/bin/python -m mlx_lm convert \
			--hf-path "$(MLX_AGENTCPM_MODEL)" \
			--mlx-path "$(MLX_AGENTCPM_MODEL_DIR)" \
			--quantize \
			--q-bits 4 \
			--q-group-size 64 \
			--trust-remote-code; \
	fi

mlx-agentcpm-start:
	@printf "$(COLOR_YELLOW)mlx-agentcpm-start is deprecated.$(COLOR_RESET)\n"
	@printf "Use make mlx-model-start.\n"

mlx-agentcpm-restart:
	@printf "$(COLOR_YELLOW)mlx-agentcpm-restart is deprecated.$(COLOR_RESET)\n"
	@printf "Use make mlx-model-restart.\n"

mlx-agentcpm-stop:
	@printf "$(COLOR_YELLOW)mlx-agentcpm-stop is deprecated.$(COLOR_RESET)\n"
	@printf "Use make mlx-model-stop.\n"

mlx-agentcpm-status:
	@printf "$(COLOR_YELLOW)mlx-agentcpm-status is deprecated.$(COLOR_RESET)\n"
	@printf "Use make mlx-model-status.\n"

mlx-agentcpm-logs: mlx-model-logs

mlx-agentcpm-health: mlx-model-health

mlx-chat:
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	MLX_CHAT_URL="$(MLX_CHAT_URL)" MLX_CHAT_MODEL="$(MLX_CHAT_MODEL)" \
	MLX_CHAT_MAX_TOKENS="$(MLX_CHAT_SESSION_MAX_TOKENS)" MLX_CHAT_TIMEOUT="$(MLX_CHAT_TIMEOUT)" \
	.venv/bin/python scripts/mlx_chat_cli.py

mlx-chat-native:
	@.venv/bin/python -m mlx_lm chat --model "$(MLX_AGENTCPM_MODEL_DIR)" --trust-remote-code

mlx-chat-test:
	@set -a; [ ! -f .env ] || . ./.env; set +a; \
	MLX_CHAT_URL="$(MLX_CHAT_URL)" MLX_CHAT_MODEL="$(MLX_CHAT_MODEL)" MLX_CHAT_PROMPT="$(MLX_CHAT_PROMPT)" \
	MLX_CHAT_MAX_TOKENS="$(MLX_CHAT_MAX_TOKENS)" MLX_CHAT_TIMEOUT="$(MLX_CHAT_TIMEOUT)" \
	.venv/bin/python scripts/mlx_chat_test.py

hn-front-page-load:
	@if ! command -v uv >/dev/null 2>&1; then \
		printf "$(COLOR_YELLOW)uv is required for make hn-front-page-load.$(COLOR_RESET)\n"; \
		printf "Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n"; \
		exit 1; \
	fi
	@uv run playwright install webkit
	@uv run python scripts/load_hn_front_page.py $(HN_FRONT_PAGE_LOAD_ARGS)

# DEPRECATED: use `make use-mlx`; the old target only rewrote legacy values.
# use-mlx-agentcpm:

use-deepseek:
	@.venv/bin/python scripts/switch_query_llm_profile.py deepseek
	@$(MAKE) lightrag-restart

use-mlx:
	@.venv/bin/python scripts/switch_query_llm_profile.py mlx --mlx-host "$(QUERY_MLX_HOST)" --mlx-model "$(QUERY_MLX_MODEL)"
	@$(MAKE) lightrag-restart

env-base env base configure:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base $(SETUP_OPTS)

env-storage storage:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --storage $(SETUP_OPTS)

env-base-rewrite base-rewrite:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --base --rewrite-compose $(SETUP_OPTS)

env-storage-rewrite storage-rewrite:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --storage --rewrite-compose $(SETUP_OPTS)

env-server server:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --server $(SETUP_OPTS)

env-validate validate:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --validate $(SETUP_OPTS)

env-security-check security security-check:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --security-check $(SETUP_OPTS)

env-backup backup:
	@$(SETUP_BASH) $(SETUP_SCRIPT) --backup $(SETUP_OPTS)
