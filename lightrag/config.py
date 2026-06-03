from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from dotenv import dotenv_values
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from lightrag.constants import (
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_EMBEDDING_TIMEOUT,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_KG_CHUNK_PICK_METHOD,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_LOG_BACKUP_COUNT,
    DEFAULT_LOG_MAX_BYTES,
    DEFAULT_MAX_ASYNC,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
    DEFAULT_MAX_FILE_PATHS,
    DEFAULT_MAX_GLEANING,
    DEFAULT_MAX_GRAPH_NODES,
    DEFAULT_MAX_PARALLEL_INSERT,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
    DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    DEFAULT_RERANK_BINDING,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_RELATION_LABELS,
    DEFAULT_SOURCE_IDS_LIMIT_METHOD,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_WOKERS,
)

logger = logging.getLogger("lightrag")

_TRUE_VALUES = {"1", "true", "yes", "t", "on"}
NO_PREFIX_SENTINEL = "NO_PREFIX"


def _resolve_env_file() -> Path:
    env_file = os.environ.get("LIGHTRAG_ENV_FILE", "").strip() or ".env"
    return Path(env_file).expanduser()


def _load_raw_env() -> dict[str, str]:
    env_file = _resolve_env_file()
    file_values = {
        key: str(value)
        for key, value in dotenv_values(env_file).items()
        if value is not None
    }
    merged = file_values
    merged.update({key: str(value) for key, value in os.environ.items()})
    return merged


def _environment_signature() -> tuple[tuple[tuple[str, str], ...], str, int | None]:
    env_file = _resolve_env_file()
    env_items = tuple(sorted((str(key), str(value)) for key, value in os.environ.items()))
    env_mtime_ns = env_file.stat().st_mtime_ns if env_file.exists() else None
    return env_items, str(env_file), env_mtime_ns


@dataclass(frozen=True)
class MLXOpenAIServerLMRoleConfig:
    context_length: int
    max_tokens: int
    queue_timeout_s: int
    idle_timeout_s: int
    on_demand: bool
    decode_concurrency: int
    prompt_concurrency: int
    prefill_step_size: int
    temperature: float
    prompt_cache_size: int
    prompt_cache_max_bytes: int
    disable_batching: bool
    prompt_cache_dir: str | None
    draft_model_path: str | None
    num_draft_tokens: int
    kv_bits: int
    kv_group_size: int
    quantized_kv_start: int


@dataclass(frozen=True)
class MLXOpenAIServerEmbeddingsConfig:
    queue_timeout_s: int
    idle_timeout_s: int
    on_demand: bool


class LightRAGSettings(BaseSettings):
    raw_env: dict[str, str] = Field(default_factory=_load_raw_env)

    model_config = SettingsConfigDict(extra="ignore", validate_default=True)

    # Core runtime settings should be declared as typed attributes.
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=9621, validation_alias="PORT")
    working_dir: str = Field(default="./rag_storage", validation_alias="WORKING_DIR")
    input_dir: str = Field(default="./inputs", validation_alias="INPUT_DIR")
    lightrag_api_key: str | None = Field(default=None, validation_alias="LIGHTRAG_API_KEY")
    lightrag_api_url: str = Field(
        default="http://localhost:9621",
        validation_alias="LIGHTRAG_API_URL",
    )

    llm_binding_host: str | None = Field(default=None, validation_alias="LLM_BINDING_HOST")
    llm_binding_api_key: str | None = Field(default=None, validation_alias="LLM_BINDING_API_KEY")
    embedding_binding_api_key: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_BINDING_API_KEY",
    )
    llm_model: str = Field(default="mistral-nemo:latest", validation_alias="LLM_MODEL")
    embedding_model: str | None = Field(default=None, validation_alias="EMBEDDING_MODEL")
    embedding_token_limit: int | None = Field(
        default=None,
        validation_alias="EMBEDDING_TOKEN_LIMIT",
    )

    rerank_model: str | None = Field(default=None, validation_alias="RERANK_MODEL")
    rerank_binding_api_key: str | None = Field(
        default=None,
        validation_alias="RERANK_BINDING_API_KEY",
    )
    lightrag_rerank_enabled: bool = Field(
        default=True,
        validation_alias="LIGHTRAG_RERANK_ENABLED",
    )

    mlx_embeddings_model: str | None = Field(
        default=None,
        validation_alias="MLX_EMBEDDINGS_MODEL",
    )
    mlx_embeddings_model_path: str | None = Field(
        default=None,
        validation_alias="MLX_EMBEDDINGS_MODEL_PATH",
    )
    mlx_rerank_server_max_rss_mb: int = Field(
        default=8192,
        validation_alias="MLX_RERANK_SERVER_MAX_RSS_MB",
    )
    mlx_rerank_model_path: str | None = Field(
        default=None,
        validation_alias="MLX_RERANK_MODEL_PATH",
    )
    mlx_rerank_model: str = Field(
        default="soichisumi/bge-reranker-v2-m3-mlx-affine8",
        validation_alias="MLX_RERANK_MODEL",
    )
    mlx_rerank_batch_size: int = Field(default=4, validation_alias="MLX_RERANK_BATCH_SIZE")
    mlx_rerank_cache_max_mb: int | None = Field(
        default=None,
        validation_alias="MLX_RERANK_CACHE_MAX_MB",
    )

    mlx_chat_url: str | None = Field(default=None, validation_alias="MLX_CHAT_URL")
    mlx_chat_model: str | None = Field(default=None, validation_alias="MLX_CHAT_MODEL")
    mlx_chat_prompt: str = Field(default="Say ok.", validation_alias="MLX_CHAT_PROMPT")
    mlx_chat_max_tokens: int = Field(default=16, validation_alias="MLX_CHAT_MAX_TOKENS")
    mlx_chat_timeout: int = Field(default=300, validation_alias="MLX_CHAT_TIMEOUT")

    improve_max_iterations: int = Field(
        default=5,
        validation_alias="IMPROVE_MAX_ITERATIONS",
    )
    improve_convergence_threshold: float = Field(
        default=0.02,
        validation_alias="IMPROVE_CONVERGENCE_THRESHOLD",
    )
    improve_targets_per_iteration: int = Field(
        default=2,
        validation_alias="IMPROVE_TARGETS_PER_ITERATION",
    )
    improve_llm_binding_host: str | None = Field(
        default=None,
        validation_alias="IMPROVE_LLM_BINDING_HOST",
    )
    improve_llm_api_key: str | None = Field(
        default=None,
        validation_alias="IMPROVE_LLM_API_KEY",
    )
    improve_llm_model: str | None = Field(
        default=None,
        validation_alias="IMPROVE_LLM_MODEL",
    )
    promptfoo_max_concurrency: int = Field(
        default=2,
        validation_alias="PROMPTFOO_MAX_CONCURRENCY",
    )
    promptfoo_eval_timeout: int = Field(
        default=900,
        validation_alias="PROMPTFOO_EVAL_TIMEOUT",
    )

    lightrag_log_path: str = Field(
        default="/Users/max/Library/Logs/lightrag/lightrag.err.log",
        validation_alias="LIGHTRAG_LOG_PATH",
    )
    lightrag_promptfoo_log_capture_file: str = Field(
        default="evals/captured/recent_log_chunks.jsonl",
        validation_alias="LIGHTRAG_PROMPTFOO_LOG_CAPTURE_FILE",
    )
    lightrag_promptfoo_log_capture_max_blocks: int = Field(
        default=20,
        validation_alias="LIGHTRAG_PROMPTFOO_LOG_CAPTURE_MAX_BLOCKS",
    )

    mlx_agentcpm_launchd_label: str = Field(
        default="com.local.mlx-agentcpm",
        validation_alias="MLX_AGENTCPM_LAUNCHD_LABEL",
    )
    mlx_agentcpm_host: str = Field(default="127.0.0.1", validation_alias="MLX_AGENTCPM_HOST")
    mlx_agentcpm_port: str = Field(default="11436", validation_alias="MLX_AGENTCPM_PORT")
    mlx_agentcpm_chat_template_args: str = Field(
        default='{"enable_thinking": false}',
        validation_alias="MLX_AGENTCPM_CHAT_TEMPLATE_ARGS",
    )
    hn_loader_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.5 Safari/605.1.15"
        ),
        validation_alias="HN_LOADER_USER_AGENT",
    )

    lightrag_keep_artifacts: bool = Field(
        default=False,
        validation_alias="LIGHTRAG_KEEP_ARTIFACTS",
    )
    lightrag_stress_test: bool = Field(default=False, validation_alias="LIGHTRAG_STRESS_TEST")
    lightrag_test_workers: int = Field(default=3, validation_alias="LIGHTRAG_TEST_WORKERS")
    lightrag_run_integration: bool = Field(
        default=False,
        validation_alias="LIGHTRAG_RUN_INTEGRATION",
    )

    _lock: RLock = PrivateAttr(default_factory=RLock)
    _signature: tuple[tuple[tuple[str, str], ...], str, int | None] = PrivateAttr(
        default_factory=_environment_signature
    )

    def _reload_from_environment(self) -> None:
        fresh = type(self)()
        for field_name in type(self).model_fields:
            setattr(self, field_name, getattr(fresh, field_name))
        self._signature = _environment_signature()

    def _ensure_current(self) -> None:
        current_signature = _environment_signature()
        if current_signature == self._signature:
            return

        with self._lock:
            current_signature = _environment_signature()
            if current_signature == self._signature:
                return
            self._reload_from_environment()

    def refresh(self, *, force: bool = False) -> None:
        if force:
            with self._lock:
                self._reload_from_environment()
            return
        self._ensure_current()

    def _raw(self, env_key: str) -> str | None:
        self._ensure_current()
        return self.raw_env.get(env_key)

    def _has(self, env_key: str) -> bool:
        self._ensure_current()
        return env_key in self.raw_env

    def _coerce_bool(self, value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if not normalized:
            return default
        return normalized in _TRUE_VALUES

    def _str(
        self,
        env_key: str,
        default: str | None = None,
        *,
        special_none: bool = False,
    ) -> str | None:
        value = self._raw(env_key)
        if value is None:
            return default
        if special_none and value == "None":
            return None
        return value

    def _int(
        self,
        env_key: str,
        default: int | None = None,
        *,
        special_none: bool = False,
        minimum: int | None = None,
    ) -> int | None:
        value = self._raw(env_key)
        if value is None:
            return default
        if special_none and value == "None":
            return None
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid integer %s=%r", env_key, value)
            return default
        if minimum is not None:
            parsed = max(minimum, parsed)
        return parsed

    def _float(
        self,
        env_key: str,
        default: float | None = None,
        *,
        minimum: float | None = None,
    ) -> float | None:
        value = self._raw(env_key)
        if value is None:
            return default
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid float %s=%r", env_key, value)
            return default
        if minimum is not None:
            parsed = max(minimum, parsed)
        return parsed

    def _bool(self, env_key: str, default: bool = False) -> bool:
        return self._coerce_bool(self._raw(env_key), default)

    def _list(self, env_key: str, default: list[str] | Any = None) -> Any:
        value = self._raw(env_key)
        if value is None:
            return default
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to parse %s as JSON list: %s, using default",
                env_key,
                exc,
            )
            return default
        if not isinstance(parsed, list):
            logger.warning(
                "Environment variable %s is not a valid JSON list, using default",
                env_key,
            )
            return default
        return parsed

    def _dict(self, env_key: str, default: dict[str, Any] | Any = None) -> Any:
        value = self._raw(env_key)
        if value is None:
            return default
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to parse %s as JSON object: %s, using default",
                env_key,
                exc,
            )
            return default
        if not isinstance(parsed, dict):
            logger.warning(
                "Environment variable %s is not a valid JSON object, using default",
                env_key,
            )
            return default
        return parsed

    def __call__(
        self,
        env_key: str,
        default: Any = None,
        value_type: type = str,
        special_none: bool = False,
    ) -> Any:
        return self.get_typed(
            env_key,
            default=default,
            value_type=value_type,
            special_none=special_none,
        )

    def get(self, env_key: str, default: Any = None) -> Any:
        value = self._raw(env_key)
        return default if value is None else value

    def has(self, env_key: str) -> bool:
        return self._has(env_key)

    def get_bool(self, env_key: str, default: bool = False) -> bool:
        return self._bool(env_key, default)

    def get_typed(
        self,
        env_key: str,
        default: Any = None,
        value_type: type = str,
        special_none: bool = False,
    ) -> Any:
        if value_type is bool:
            return self._bool(env_key, default if isinstance(default, bool) else False)
        if value_type is int:
            return self._int(env_key, default, special_none=special_none)
        if value_type is float:
            return self._float(env_key, default)
        if value_type is list:
            return self._list(env_key, default)
        if value_type is dict:
            return self._dict(env_key, default)
        return self._str(env_key, default, special_none=special_none)

    def snapshot(self) -> dict[str, str]:
        self._ensure_current()
        return dict(self.raw_env)

    def binding_env_value(self, env_name: str, default: Any = None) -> Any:
        return self.get(env_name, default)

    def binding_env_bool(self, env_name: str, default: bool = False) -> bool:
        return self.get_bool(env_name, default)

    def binding_env_json_list(self, env_name: str, default: Any = None) -> Any:
        return self._list(env_name, default)

    def binding_env_json_dict(self, env_name: str, default: Any = None) -> Any:
        return self._dict(env_name, default)

    def binding_env_configured(self, env_name: str) -> bool:
        return self._has(env_name)

    def _embedding_prefix_config(self, env_key: str) -> tuple[str | None, bool]:
        if not self._has(env_key):
            return None, False
        value = self._str(env_key)
        if value == "None":
            return None, False
        if value == NO_PREFIX_SENTINEL:
            return "", True
        if value == "":
            raise ValueError(
                f"{env_key} is empty. Use {NO_PREFIX_SENTINEL} to explicitly request "
                "no prefix, or remove the variable to leave it unconfigured."
            )
        return value, True

    def default_host_for_binding(self, binding_type: str) -> str:
        default_hosts = {
            "ollama": self.llm_binding_host or "http://localhost:11434",
            "lollms": self.llm_binding_host or "http://localhost:9600",
            "azure_openai": self.azure_openai_endpoint or "https://api.openai.com/v1",
            "openai": self.llm_binding_host or "https://api.openai.com/v1",
            "gemini": self.llm_binding_host
            or "https://generativelanguage.googleapis.com",
        }
        return default_hosts.get(
            binding_type,
            self.llm_binding_host or "http://localhost:11434",
        )

    def effective_llm_binding_host(self, binding_type: str | None = None) -> str:
        return self.llm_binding_host or self.default_host_for_binding(
            binding_type or self.llm_binding
        )

    def effective_embedding_binding_host(
        self, binding_type: str | None = None
    ) -> str:
        return self.embedding_binding_host or self.default_host_for_binding(
            binding_type or self.embedding_binding
        )

    def mlx_managed_request_recycle_limit(
        self, kind: Literal["chat", "extraction", "embedding"]
    ) -> int:
        if self.lightrag_manage_mlx_openai_server:
            return self.mlx_openai_server_max_requests_before_recycle
        if kind == "chat":
            return self.mlx_llm_max_requests_before_recycle
        if kind == "extraction":
            return self.mlx_extraction_max_requests_before_recycle
        return self.mlx_embed_max_requests_before_recycle

    def mlx_openai_server_role_queue_timeout_s(
        self,
        role: Literal["retrieval", "extraction", "embeddings"],
    ) -> int:
        env_value = self._int(
            f"MLX_OPENAI_SERVER_{role.upper()}_QUEUE_TIMEOUT_S",
            minimum=30,
        )
        if env_value is not None:
            return env_value

        global_value = self.mlx_openai_server_queue_timeout_s
        if global_value is not None:
            return global_value

        if role == "retrieval":
            return max(30, self.llm_timeout)
        if role == "extraction":
            return max(30, self.extraction_llm_timeout or self.llm_timeout)
        return max(30, self.embedding_timeout or self.llm_timeout)

    def mlx_openai_server_role_idle_timeout_s(
        self,
        role: Literal["retrieval", "extraction", "embeddings"],
    ) -> int:
        role_value = self._int(
            f"MLX_OPENAI_SERVER_{role.upper()}_IDLE_TIMEOUT_S",
            minimum=30,
        )
        if role_value is not None:
            return role_value
        return self.mlx_openai_server_idle_timeout_s

    def mlx_openai_server_role_on_demand(
        self,
        role: Literal["retrieval", "extraction", "embeddings"],
        default: bool = True,
    ) -> bool:
        role_key = f"MLX_OPENAI_SERVER_{role.upper()}_ON_DEMAND"
        if self._has(role_key):
            return self._bool(role_key, default)
        if self._has("MLX_OPENAI_SERVER_ON_DEMAND"):
            return self.mlx_openai_server_on_demand
        return default

    def mlx_openai_server_lm_role_config(
        self,
        role: Literal["retrieval", "extraction"],
    ) -> MLXOpenAIServerLMRoleConfig:
        prefix = f"MLX_OPENAI_SERVER_{role.upper()}_"
        default_context_length = 12288 if role == "retrieval" else 8192
        default_max_tokens = 1024 if role == "retrieval" else 2048
        default_temperature = 0.1 if role == "retrieval" else 0.0
        default_prompt_cache_size = 2 if role == "retrieval" else 1
        prompt_cache_size = self._int(
            f"{prefix}PROMPT_CACHE_SIZE",
            default_prompt_cache_size,
            minimum=0,
        )
        resolved_prompt_cache_size: int = (
            default_prompt_cache_size
            if prompt_cache_size is None
            else prompt_cache_size
        )

        return MLXOpenAIServerLMRoleConfig(
            context_length=self._int(
                f"{prefix}CONTEXT_LENGTH",
                default_context_length,
                minimum=1,
            )
            or default_context_length,
            max_tokens=self._int(
                f"{prefix}MAX_TOKENS",
                default_max_tokens,
                minimum=1,
            )
            or default_max_tokens,
            queue_timeout_s=self.mlx_openai_server_role_queue_timeout_s(role),
            idle_timeout_s=self.mlx_openai_server_role_idle_timeout_s(role),
            on_demand=self.mlx_openai_server_role_on_demand(role),
            decode_concurrency=self._int(
                f"{prefix}DECODE_CONCURRENCY",
                4,
                minimum=1,
            )
            or 4,
            prompt_concurrency=self._int(
                f"{prefix}PROMPT_CONCURRENCY",
                1,
                minimum=1,
            )
            or 1,
            prefill_step_size=self._int(
                f"{prefix}PREFILL_STEP_SIZE",
                2048,
                minimum=1,
            )
            or 2048,
            temperature=self._float(
                f"{prefix}TEMPERATURE",
                default_temperature,
            )
            or default_temperature,
            prompt_cache_size=resolved_prompt_cache_size,
            prompt_cache_max_bytes=self._int(
                f"{prefix}PROMPT_CACHE_MAX_BYTES",
                8589934592,
                minimum=0,
            )
            or 8589934592,
            disable_batching=self._bool(f"{prefix}DISABLE_BATCHING", False),
            prompt_cache_dir=self._str(f"{prefix}PROMPT_CACHE_DIR") or None,
            draft_model_path=self._str(f"{prefix}DRAFT_MODEL_PATH") or None,
            num_draft_tokens=self._int(
                f"{prefix}NUM_DRAFT_TOKENS",
                2,
                minimum=1,
            )
            or 2,
            kv_bits=self._int(f"{prefix}KV_BITS", 4, minimum=0) or 0,
            kv_group_size=self._int(
                f"{prefix}KV_GROUP_SIZE",
                64,
                minimum=1,
            )
            or 64,
            quantized_kv_start=self._int(
                f"{prefix}QUANTIZED_KV_START",
                0,
                minimum=0,
            )
            or 0,
        )

    def mlx_openai_server_embeddings_config(self) -> MLXOpenAIServerEmbeddingsConfig:
        return MLXOpenAIServerEmbeddingsConfig(
            queue_timeout_s=self.mlx_openai_server_role_queue_timeout_s("embeddings"),
            idle_timeout_s=self.mlx_openai_server_role_idle_timeout_s("embeddings"),
            on_demand=self.mlx_openai_server_role_on_demand("embeddings"),
        )

    @property
    def lightrag_env_file(self) -> Path:
        return _resolve_env_file()

    @property
    def timeout(self) -> int | None:
        return self._int("TIMEOUT", DEFAULT_TIMEOUT, special_none=True)

    @property
    def max_async(self) -> int:
        return self._int("MAX_ASYNC", DEFAULT_MAX_ASYNC) or DEFAULT_MAX_ASYNC

    @property
    def summary_max_tokens(self) -> int:
        return self._int("SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS) or DEFAULT_SUMMARY_MAX_TOKENS

    @property
    def summary_context_size(self) -> int:
        return self._int("SUMMARY_CONTEXT_SIZE", DEFAULT_SUMMARY_CONTEXT_SIZE) or DEFAULT_SUMMARY_CONTEXT_SIZE

    @property
    def summary_length_recommended(self) -> int:
        return self._int(
            "SUMMARY_LENGTH_RECOMMENDED",
            DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
        ) or DEFAULT_SUMMARY_LENGTH_RECOMMENDED

    @property
    def log_level(self) -> str:
        return self._str("LOG_LEVEL", "INFO") or "INFO"

    @property
    def verbose(self) -> bool:
        return self._bool("VERBOSE", False)

    @property
    def ssl(self) -> bool:
        return self._bool("SSL", False)

    @property
    def ssl_certfile(self) -> str | None:
        return self._str("SSL_CERTFILE")

    @property
    def ssl_keyfile(self) -> str | None:
        return self._str("SSL_KEYFILE")

    @property
    def ollama_emulating_model_name(self) -> str:
        return self._str(
            "OLLAMA_EMULATING_MODEL_NAME",
            DEFAULT_OLLAMA_MODEL_NAME,
        ) or DEFAULT_OLLAMA_MODEL_NAME

    @property
    def ollama_emulating_model_tag(self) -> str:
        return self._str(
            "OLLAMA_EMULATING_MODEL_TAG",
            DEFAULT_OLLAMA_MODEL_TAG,
        ) or DEFAULT_OLLAMA_MODEL_TAG

    @property
    def workspace(self) -> str:
        return self._str("WORKSPACE", "") or ""

    @property
    def workers(self) -> int:
        return self._int("WORKERS", DEFAULT_WOKERS) or DEFAULT_WOKERS

    @property
    def llm_binding(self) -> str:
        return self._str("LLM_BINDING", "ollama") or "ollama"

    @property
    def embedding_binding(self) -> str:
        return self._str("EMBEDDING_BINDING", "ollama") or "ollama"

    @property
    def rerank_binding(self) -> str:
        return self._str("RERANK_BINDING", DEFAULT_RERANK_BINDING) or DEFAULT_RERANK_BINDING

    @property
    def keepalive(self) -> int:
        return self._int("KEEPALIVE", 5) or 5

    @property
    def embedding_binding_host(self) -> str | None:
        return self._str("EMBEDDING_BINDING_HOST")



    @property
    def llm_model_configured(self) -> str | None:
        return self._str("LLM_MODEL")

    @property
    def embedding_dim(self) -> int | None:
        return self._int("EMBEDDING_DIM", special_none=True)

    @property
    def embedding_send_dim(self) -> bool:
        return self._bool("EMBEDDING_SEND_DIM", False)

    @property
    def chunk_size(self) -> int:
        return self._int("CHUNK_SIZE", 1200) or 1200

    @property
    def chunk_overlap_size(self) -> int:
        return self._int("CHUNK_OVERLAP_SIZE", 100) or 100

    @property
    def enable_llm_cache_for_extract(self) -> bool:
        return self._bool("ENABLE_LLM_CACHE_FOR_EXTRACT", True)

    @property
    def enable_llm_cache(self) -> bool:
        return self._bool("ENABLE_LLM_CACHE", True)

    @property
    def document_loading_engine(self) -> str:
        return self._str("DOCUMENT_LOADING_ENGINE", "DEFAULT") or "DEFAULT"

    @property
    def pdf_decrypt_password(self) -> str | None:
        return self._str("PDF_DECRYPT_PASSWORD")

    @property
    def cors_origins(self) -> str:
        return self._str("CORS_ORIGINS", "*") or "*"

    @property
    def summary_language(self) -> str:
        return self._str("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE) or DEFAULT_SUMMARY_LANGUAGE

    @property
    def entity_types(self) -> list[str]:
        return self._list("ENTITY_TYPES", list(DEFAULT_ENTITY_TYPES))

    @property
    def relation_labels(self) -> list[str]:
        return self._list("RELATION_LABELS", list(DEFAULT_RELATION_LABELS))

    @property
    def whitelist_paths(self) -> str:
        return self._str("WHITELIST_PATHS", "/health,/api/*") or "/health,/api/*"

    @property
    def auth_accounts(self) -> str:
        return self._str("AUTH_ACCOUNTS", "") or ""

    @property
    def token_secret(self) -> str | None:
        return self._str("TOKEN_SECRET")

    @property
    def token_expire_hours(self) -> float:
        return self._float("TOKEN_EXPIRE_HOURS", 48.0) or 48.0

    @property
    def guest_token_expire_hours(self) -> float:
        return self._float("GUEST_TOKEN_EXPIRE_HOURS", 24.0) or 24.0

    @property
    def jwt_algorithm(self) -> str:
        return self._str("JWT_ALGORITHM", "HS256") or "HS256"

    @property
    def token_auto_renew(self) -> bool:
        return self._bool("TOKEN_AUTO_RENEW", True)

    @property
    def token_renew_threshold(self) -> float:
        return self._float("TOKEN_RENEW_THRESHOLD", 0.5) or 0.5

    @property
    def rerank_binding_host(self) -> str | None:
        return self._str("RERANK_BINDING_HOST")

    @property
    def history_turns(self) -> int:
        return self._int("HISTORY_TURNS", DEFAULT_HISTORY_TURNS) or DEFAULT_HISTORY_TURNS

    @property
    def top_k(self) -> int:
        return self._int("TOP_K", DEFAULT_TOP_K) or DEFAULT_TOP_K

    @property
    def chunk_top_k(self) -> int:
        return self._int("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K) or DEFAULT_CHUNK_TOP_K

    @property
    def max_entity_tokens(self) -> int:
        return self._int("MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS) or DEFAULT_MAX_ENTITY_TOKENS

    @property
    def max_relation_tokens(self) -> int:
        return self._int("MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS) or DEFAULT_MAX_RELATION_TOKENS

    @property
    def max_total_tokens(self) -> int:
        return self._int("MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS) or DEFAULT_MAX_TOTAL_TOKENS

    @property
    def cosine_threshold(self) -> float:
        return self._float("COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD) or DEFAULT_COSINE_THRESHOLD

    @property
    def related_chunk_number(self) -> int:
        return self._int("RELATED_CHUNK_NUMBER", DEFAULT_RELATED_CHUNK_NUMBER) or DEFAULT_RELATED_CHUNK_NUMBER

    @property
    def kg_chunk_pick_method(self) -> str:
        return self._str(
            "KG_CHUNK_PICK_METHOD",
            DEFAULT_KG_CHUNK_PICK_METHOD,
        ) or DEFAULT_KG_CHUNK_PICK_METHOD

    @property
    def max_gleaning(self) -> int:
        return self._int("MAX_GLEANING", DEFAULT_MAX_GLEANING) or DEFAULT_MAX_GLEANING

    @property
    def max_extract_input_tokens(self) -> int:
        return self._int(
            "MAX_EXTRACT_INPUT_TOKENS",
            DEFAULT_MAX_EXTRACT_INPUT_TOKENS,
        ) or DEFAULT_MAX_EXTRACT_INPUT_TOKENS

    @property
    def force_llm_summary_on_merge(self) -> int:
        return self._int(
            "FORCE_LLM_SUMMARY_ON_MERGE",
            DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
        ) or DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE

    @property
    def embedding_func_max_async(self) -> int:
        return self._int(
            "EMBEDDING_FUNC_MAX_ASYNC",
            DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
        ) or DEFAULT_EMBEDDING_FUNC_MAX_ASYNC

    @property
    def embedding_batch_num(self) -> int:
        return self._int(
            "EMBEDDING_BATCH_NUM",
            DEFAULT_EMBEDDING_BATCH_NUM,
        ) or DEFAULT_EMBEDDING_BATCH_NUM

    @property
    def embedding_timeout(self) -> int:
        return self._int("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT) or DEFAULT_EMBEDDING_TIMEOUT

    @property
    def max_upload_size(self) -> int | None:
        return self._int("MAX_UPLOAD_SIZE", 104857600, special_none=True)

    @property
    def llm_timeout(self) -> int:
        return self._int("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT) or DEFAULT_LLM_TIMEOUT

    @property
    def min_rerank_score(self) -> float:
        return self._float("MIN_RERANK_SCORE", DEFAULT_MIN_RERANK_SCORE) or DEFAULT_MIN_RERANK_SCORE

    @property
    def lightrag_agent_tools(self) -> bool:
        return self._bool("LIGHTRAG_AGENT_TOOLS", False)

    @property
    def max_parallel_insert(self) -> int:
        return self._int("MAX_PARALLEL_INSERT", DEFAULT_MAX_PARALLEL_INSERT) or DEFAULT_MAX_PARALLEL_INSERT

    @property
    def max_graph_nodes(self) -> int:
        return self._int("MAX_GRAPH_NODES", DEFAULT_MAX_GRAPH_NODES) or DEFAULT_MAX_GRAPH_NODES

    @property
    def max_source_ids_per_entity(self) -> int:
        return self._int(
            "MAX_SOURCE_IDS_PER_ENTITY",
            DEFAULT_MAX_SOURCE_IDS_PER_ENTITY,
        ) or DEFAULT_MAX_SOURCE_IDS_PER_ENTITY

    @property
    def max_source_ids_per_relation(self) -> int:
        return self._int(
            "MAX_SOURCE_IDS_PER_RELATION",
            DEFAULT_MAX_SOURCE_IDS_PER_RELATION,
        ) or DEFAULT_MAX_SOURCE_IDS_PER_RELATION

    @property
    def source_ids_limit_method(self) -> str:
        return self._str(
            "SOURCE_IDS_LIMIT_METHOD",
            DEFAULT_SOURCE_IDS_LIMIT_METHOD,
        ) or DEFAULT_SOURCE_IDS_LIMIT_METHOD

    @property
    def max_file_paths(self) -> int:
        return self._int("MAX_FILE_PATHS", DEFAULT_MAX_FILE_PATHS) or DEFAULT_MAX_FILE_PATHS

    @property
    def lightrag_kv_storage(self) -> str:
        return self._str("LIGHTRAG_KV_STORAGE", "JsonKVStorage") or "JsonKVStorage"

    @property
    def lightrag_doc_status_storage(self) -> str:
        return self._str(
            "LIGHTRAG_DOC_STATUS_STORAGE",
            "JsonDocStatusStorage",
        ) or "JsonDocStatusStorage"

    @property
    def lightrag_graph_storage(self) -> str:
        return self._str("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage") or "NetworkXStorage"

    @property
    def lightrag_vector_storage(self) -> str:
        return self._str("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage") or "NanoVectorDBStorage"

    @property
    def embedding_document_prefix(self) -> str | None:
        return self._embedding_prefix_config("EMBEDDING_DOCUMENT_PREFIX")[0]

    @property
    def embedding_document_prefix_configured(self) -> bool:
        return self._embedding_prefix_config("EMBEDDING_DOCUMENT_PREFIX")[1]

    @property
    def embedding_query_prefix(self) -> str | None:
        return self._embedding_prefix_config("EMBEDDING_QUERY_PREFIX")[0]

    @property
    def embedding_query_prefix_configured(self) -> bool:
        return self._embedding_prefix_config("EMBEDDING_QUERY_PREFIX")[1]

    @property
    def embedding_prefixes_configured(self) -> bool:
        return (
            self.embedding_document_prefix_configured
            or self.embedding_query_prefix_configured
        )

    @property
    def embedding_asymmetric(self) -> bool:
        return self._bool("EMBEDDING_ASYMMETRIC", False)

    @property
    def embedding_asymmetric_configured(self) -> bool:
        return self._has("EMBEDDING_ASYMMETRIC")

    @property
    def log_dir(self) -> str:
        return self._str("LOG_DIR", os.getcwd()) or os.getcwd()

    @property
    def log_max_bytes(self) -> int:
        return self._int("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES) or DEFAULT_LOG_MAX_BYTES

    @property
    def log_backup_count(self) -> int:
        return self._int("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT) or DEFAULT_LOG_BACKUP_COUNT

    @property
    def error_log(self) -> str | None:
        return self._str("ERROR_LOG")

    @property
    def access_log(self) -> str | None:
        return self._str("ACCESS_LOG")

    @property
    def lightrag_mcp_query_profile(self) -> str:
        profile = (self._str("LIGHTRAG_MCP_QUERY_PROFILE", "granite") or "granite").strip().lower()
        if profile not in {"default", "granite"}:
            logger.warning(
                "Ignoring invalid LIGHTRAG_MCP_QUERY_PROFILE=%r; using granite",
                profile,
            )
            return "granite"
        return profile

    @property
    def lightrag_mcp_api_base_url(self) -> str | None:
        value = self._str("LIGHTRAG_MCP_API_BASE_URL")
        if value is None:
            return None
        value = value.strip()
        return value.rstrip("/") if value else None

    @property
    def lightrag_mcp_api_key(self) -> str | None:
        return self._str("LIGHTRAG_MCP_API_KEY")

    @property
    def lightrag_mcp_name(self) -> str:
        return self._str("LIGHTRAG_MCP_NAME", "LightRAG MCP Server") or "LightRAG MCP Server"

    @property
    def lightrag_mcp_transport(self) -> Literal["http", "streamable-http", "sse"]:
        value = self._str("LIGHTRAG_MCP_TRANSPORT", "streamable-http") or "streamable-http"
        if value not in {"http", "streamable-http", "sse"}:
            logger.warning(
                "Ignoring invalid LIGHTRAG_MCP_TRANSPORT=%r; using streamable-http",
                value,
            )
            return "streamable-http"
        return cast(Literal["http", "streamable-http", "sse"], value)

    @property
    def lightrag_mcp_json_response(self) -> bool:
        return self._bool("LIGHTRAG_MCP_JSON_RESPONSE", False)

    @property
    def lightrag_mcp_stateless_http(self) -> bool:
        return self._bool("LIGHTRAG_MCP_STATELESS_HTTP", True)

    @property
    def lightrag_mcp_enabled(self) -> bool:
        return self._bool("LIGHTRAG_MCP_ENABLED", True)

    @property
    def lightrag_mcp_path(self) -> str:
        return self._str("LIGHTRAG_MCP_PATH", "/mcp") or "/mcp"

    @property
    def lightrag_query_fast_prompt(self) -> str | None:
        return self._str("LIGHTRAG_QUERY_FAST_PROMPT")

    @property
    def webui_query_enrichment_prompt(self) -> str | None:
        return self._str("WEBUI_QUERY_ENRICHMENT_PROMPT")

    @property
    def lightrag_query_fast_context_length(self) -> int | None:
        return self._int("LIGHTRAG_QUERY_FAST_CONTEXT_LENGTH")

    @property
    def apfel_context_length(self) -> int:
        return self._int("APFEL_CONTEXT_LENGTH", 4096) or 4096

    @property
    def llm_context_length(self) -> int | None:
        return self._int("LLM_CONTEXT_LENGTH")

    @property
    def lightrag_query_fast_history_turns(self) -> int | None:
        return self._int("LIGHTRAG_QUERY_FAST_HISTORY_TURNS")

    @property
    def lightrag_query_fast_history_max_chars(self) -> int:
        return self._int("LIGHTRAG_QUERY_FAST_HISTORY_MAX_CHARS", 1200) or 1200

    @property
    def lightrag_query_fast_top_k(self) -> int:
        return self._int("LIGHTRAG_QUERY_FAST_TOP_K", 8) or 8

    @property
    def lightrag_query_fast_chunk_top_k(self) -> int:
        return self._int("LIGHTRAG_QUERY_FAST_CHUNK_TOP_K", 4) or 4

    @property
    def lightrag_query_fast_max_total_tokens(self) -> int:
        return self._int("LIGHTRAG_QUERY_FAST_MAX_TOTAL_TOKENS", 2600) or 2600

    @property
    def lightrag_query_fast_max_entity_tokens(self) -> int:
        return self._int("LIGHTRAG_QUERY_FAST_MAX_ENTITY_TOKENS", 350) or 350

    @property
    def lightrag_query_fast_max_relation_tokens(self) -> int:
        return self._int("LIGHTRAG_QUERY_FAST_MAX_RELATION_TOKENS", 450) or 450

    @property
    def lightrag_query_fast_response_type(self) -> str:
        return self._str("LIGHTRAG_QUERY_FAST_RESPONSE_TYPE", "Short bullet list") or "Short bullet list"

    @property
    def lightrag_query_fast_enable_rerank(self) -> bool:
        return self._bool("LIGHTRAG_QUERY_FAST_ENABLE_RERANK", False)

    @property
    def lightrag_query_fast_max_completion_tokens(self) -> int:
        return self._int("LIGHTRAG_QUERY_FAST_MAX_COMPLETION_TOKENS", 384) or 384

    @property
    def openai_llm_max_completion_tokens(self) -> int | None:
        return self._int("OPENAI_LLM_MAX_COMPLETION_TOKENS")

    @property
    def webui_query_enrichment_model(self) -> str | None:
        return self._str("WEBUI_QUERY_ENRICHMENT_MODEL")

    @property
    def webui_query_enrichment_binding_host(self) -> str | None:
        return self._str("WEBUI_QUERY_ENRICHMENT_BINDING_HOST")

    @property
    def webui_query_enrichment_binding_api_key(self) -> str:
        return self._str("WEBUI_QUERY_ENRICHMENT_BINDING_API_KEY", "dummy") or "dummy"

    @property
    def webui_query_enrichment_enabled(self) -> bool:
        return self._bool("WEBUI_QUERY_ENRICHMENT_ENABLED", True)

    @property
    def webui_query_enrichment_timeout(self) -> int:
        return self._int("WEBUI_QUERY_ENRICHMENT_TIMEOUT", 900) or 900

    @property
    def webui_query_enrichment_top_k(self) -> int:
        return self._int("WEBUI_QUERY_ENRICHMENT_TOP_K", 24) or 24

    @property
    def webui_query_enrichment_chunk_top_k(self) -> int:
        return self._int("WEBUI_QUERY_ENRICHMENT_CHUNK_TOP_K", 8) or 8

    @property
    def webui_query_enrichment_max_total_tokens(self) -> int:
        return self._int("WEBUI_QUERY_ENRICHMENT_MAX_TOTAL_TOKENS", 6500) or 6500

    @property
    def webui_query_enrichment_max_entity_tokens(self) -> int:
        return self._int("WEBUI_QUERY_ENRICHMENT_MAX_ENTITY_TOKENS", 1600) or 1600

    @property
    def webui_query_enrichment_max_relation_tokens(self) -> int:
        return self._int("WEBUI_QUERY_ENRICHMENT_MAX_RELATION_TOKENS", 2200) or 2200

    @property
    def webui_query_enrichment_max_completion_tokens(self) -> int:
        return self._int("WEBUI_QUERY_ENRICHMENT_MAX_COMPLETION_TOKENS", 1024) or 1024

    @property
    def webui_query_enrichment_agent_tools(self) -> bool:
        return self._bool("WEBUI_QUERY_ENRICHMENT_AGENT_TOOLS", False)

    @property
    def webui_query_enrichment_enable_rerank(self) -> bool:
        return self._bool("WEBUI_QUERY_ENRICHMENT_ENABLE_RERANK", True)

    @property
    def lightrag_granite_query_model(self) -> str | None:
        return self._str("LIGHTRAG_GRANITE_QUERY_MODEL")

    @property
    def lightrag_granite_query_binding_host(self) -> str | None:
        return self._str("LIGHTRAG_GRANITE_QUERY_BINDING_HOST")

    @property
    def lightrag_granite_query_binding_api_key(self) -> str | None:
        return self._str("LIGHTRAG_GRANITE_QUERY_BINDING_API_KEY")

    @property
    def lightrag_granite_query_timeout(self) -> int | None:
        return self._int("LIGHTRAG_GRANITE_QUERY_TIMEOUT")

    @property
    def rerank_by_default(self) -> bool:
        return self._bool("RERANK_BY_DEFAULT", True)

    @property
    def cohere_api_key(self) -> str | None:
        return self._str("COHERE_API_KEY")

    @property
    def jina_api_key(self) -> str | None:
        return self._str("JINA_API_KEY")

    @property
    def dashscope_api_key(self) -> str | None:
        return self._str("DASHSCOPE_API_KEY")

    @property
    def langfuse_public_key(self) -> str | None:
        return self._str("LANGFUSE_PUBLIC_KEY")

    @property
    def langfuse_secret_key(self) -> str | None:
        return self._str("LANGFUSE_SECRET_KEY")

    @property
    def embedding_use_base64(self) -> bool:
        return self._bool("EMBEDDING_USE_BASE64", True)

    @property
    def mlx_openai_server_host(self) -> str:
        return self._str("MLX_OPENAI_SERVER_HOST", "127.0.0.1") or "127.0.0.1"

    @property
    def mlx_openai_server_port(self) -> int:
        return self._int("MLX_OPENAI_SERVER_PORT", 11436) or 11436

    @property
    def managed_mlx_openai_server_base_url(self) -> str:
        return f"http://{self.mlx_openai_server_host}:{self.mlx_openai_server_port}/v1"

    @property
    def mlx_embeddings_host(self) -> str:
        return self._str("MLX_EMBEDDINGS_HOST", "127.0.0.1") or "127.0.0.1"

    @property
    def mlx_embeddings_port(self) -> int:
        return self._int("MLX_EMBEDDINGS_PORT", 11437) or 11437

    @property
    def managed_mlx_embeddings_base_url(self) -> str:
        return f"http://{self.mlx_embeddings_host}:{self.mlx_embeddings_port}/v1"

    @property
    def lightrag_manage_mlx_openai_server(self) -> bool:
        return self._bool("LIGHTRAG_MANAGE_MLX_OPENAI_SERVER", False)

    @property
    def lightrag_manage_swift_lm(self) -> bool:
        return self._bool("LIGHTRAG_MANAGE_SWIFT_LM", False)

    @property
    def swift_lm_binary(self) -> str:
        return self._str("SWIFT_LM_BINARY", "SwiftLM") or "SwiftLM"

    @property
    def swift_lm_model_path(self) -> str | None:
        return self._str("SWIFT_LM_MODEL_PATH")

    @property
    def swift_lm_host(self) -> str:
        return self._str("SWIFT_LM_HOST", "127.0.0.1") or "127.0.0.1"

    @property
    def swift_lm_port(self) -> int:
        return self._int("SWIFT_LM_PORT", 11436) or 11436

    @property
    def managed_swift_lm_base_url(self) -> str:
        return f"http://{self.swift_lm_host}:{self.swift_lm_port}/v1"

    @property
    def swift_lm_context_size(self) -> int:
        return self._int("SWIFT_LM_CONTEXT_SIZE", 8192, minimum=1) or 8192

    @property
    def swift_lm_max_tokens(self) -> int:
        return self._int("SWIFT_LM_MAX_TOKENS", 2048, minimum=1) or 2048

    @property
    def swift_lm_parallel(self) -> int:
        return self._int("SWIFT_LM_PARALLEL", 2, minimum=1) or 2

    @property
    def swift_lm_mem_limit_mb(self) -> int:
        return self._int("SWIFT_LM_MEM_LIMIT_MB", 16384, minimum=1) or 16384

    @property
    def swift_lm_prefill_size(self) -> int:
        return self._int("SWIFT_LM_PREFILL_SIZE", 512, minimum=1) or 512

    @property
    def swift_lm_gpu_layers(self) -> str | None:
        return self._str("SWIFT_LM_GPU_LAYERS")

    @property
    def swift_lm_stream_experts(self) -> bool:
        return self._bool("SWIFT_LM_STREAM_EXPERTS", False)

    @property
    def swift_lm_ssd_prefetch(self) -> bool:
        return self._bool("SWIFT_LM_SSD_PREFETCH", False)

    @property
    def swift_lm_turbo_kv(self) -> bool:
        return self._bool("SWIFT_LM_TURBO_KV", False)

    @property
    def swift_lm_draft_model_path(self) -> str | None:
        return self._str("SWIFT_LM_DRAFT_MODEL_PATH")

    @property
    def swift_lm_num_draft_tokens(self) -> int | None:
        return self._int("SWIFT_LM_NUM_DRAFT_TOKENS", minimum=1)

    @property
    def swift_lm_mtp(self) -> bool:
        return self._bool("SWIFT_LM_MTP", False)

    @property
    def swift_lm_num_mtp_tokens(self) -> int | None:
        return self._int("SWIFT_LM_NUM_MTP_TOKENS", minimum=1)

    @property
    def lightrag_manage_swift_embeddings(self) -> bool:
        return self._bool("LIGHTRAG_MANAGE_SWIFT_EMBEDDINGS", False)

    @property
    def swift_embeddings_server_binary(self) -> str:
        return self._str("SWIFT_EMBEDDINGS_SERVER_BINARY", "NomicEmbeddingsServer") or "NomicEmbeddingsServer"

    @property
    def swift_embeddings_model_path(self) -> str | None:
        return self._str("SWIFT_EMBEDDINGS_MODEL_PATH")

    @property
    def swift_embeddings_host(self) -> str:
        return self._str("SWIFT_EMBEDDINGS_HOST", "127.0.0.1") or "127.0.0.1"

    @property
    def swift_embeddings_port(self) -> int:
        return self._int("SWIFT_EMBEDDINGS_PORT", 11439) or 11439

    @property
    def swift_embeddings_max_tokens(self) -> int:
        return self._int("SWIFT_EMBEDDINGS_MAX_TOKENS", 2048, minimum=1) or 2048

    @property
    def swift_embeddings_idle_timeout_s(self) -> int:
        return self._int("SWIFT_EMBEDDINGS_IDLE_TIMEOUT_S", 180, minimum=0) or 180

    @property
    def managed_swift_embeddings_base_url(self) -> str:
        return f"http://{self.swift_embeddings_host}:{self.swift_embeddings_port}/v1"

    @property
    def swift_embeddings_max_rss_mb(self) -> int:
        return self._int("SWIFT_EMBEDDINGS_MAX_RSS_MB", 16384, minimum=0) or 16384

    @property
    def swift_embeddings_watchdog_interval_s(self) -> int:
        return self._int("SWIFT_EMBEDDINGS_WATCHDOG_INTERVAL_S", 15, minimum=5) or 15

    @property
    def extraction_llm_model(self) -> str | None:
        return self._str("EXTRACTION_LLM_MODEL")

    @property
    def mlx_extraction_model(self) -> str | None:
        return self._str("MLX_EXTRACTION_MODEL")

    @property
    def extraction_llm_binding_host(self) -> str | None:
        return self._str("EXTRACTION_LLM_BINDING_HOST")

    @property
    def extraction_llm_binding_api_key(self) -> str | None:
        return self._str("EXTRACTION_LLM_BINDING_API_KEY")

    @property
    def extraction_llm_timeout(self) -> int | None:
        return self._int("EXTRACTION_LLM_TIMEOUT")

    @property
    def extraction_openai_llm_max_completion_tokens(self) -> int | None:
        return self._int("EXTRACTION_OPENAI_LLM_MAX_COMPLETION_TOKENS")

    @property
    def extraction_max_async(self) -> int | None:
        return self._int("EXTRACTION_MAX_ASYNC")

    @property
    def mlx_openai_server_max_requests_before_recycle(self) -> int:
        return self._int("MLX_OPENAI_SERVER_MAX_REQUESTS_BEFORE_RECYCLE", 0) or 0

    @property
    def mlx_llm_max_requests_before_recycle(self) -> int:
        return self._int("MLX_LLM_MAX_REQUESTS_BEFORE_RECYCLE", 0) or 0

    @property
    def mlx_extraction_max_requests_before_recycle(self) -> int:
        return self._int("MLX_EXTRACTION_MAX_REQUESTS_BEFORE_RECYCLE", 0) or 0

    @property
    def mlx_embed_max_requests_before_recycle(self) -> int:
        return self._int("MLX_EMBED_MAX_REQUESTS_BEFORE_RECYCLE", 0) or 0

    @property
    def lightrag_llm_request_metrics_enabled(self) -> bool:
        return self._bool("LIGHTRAG_LLM_REQUEST_METRICS_ENABLED", True)

    @property
    def azure_openai_api_key(self) -> str | None:
        return self._str("AZURE_OPENAI_API_KEY")

    @property
    def openai_api_key(self) -> str | None:
        return self._str("OPENAI_API_KEY")

    @property
    def openai_api_base(self) -> str:
        return self._str("OPENAI_API_BASE", "https://api.openai.com/v1") or "https://api.openai.com/v1"

    @property
    def azure_openai_deployment(self) -> str | None:
        return self._str("AZURE_OPENAI_DEPLOYMENT")

    @property
    def azure_openai_endpoint(self) -> str | None:
        return self._str("AZURE_OPENAI_ENDPOINT")

    @property
    def azure_openai_api_version(self) -> str | None:
        return self._str("AZURE_OPENAI_API_VERSION")

    @property
    def openai_api_version(self) -> str | None:
        return self._str("OPENAI_API_VERSION")

    @property
    def azure_embedding_deployment(self) -> str | None:
        return self._str("AZURE_EMBEDDING_DEPLOYMENT")

    @property
    def azure_embedding_endpoint(self) -> str | None:
        return self._str("AZURE_EMBEDDING_ENDPOINT")

    @property
    def azure_embedding_api_key(self) -> str | None:
        return self._str("AZURE_EMBEDDING_API_KEY")

    @property
    def azure_embedding_api_version(self) -> str | None:
        return self._str("AZURE_EMBEDDING_API_VERSION")

    @property
    def ollama_api_key(self) -> str | None:
        return self._str("OLLAMA_API_KEY")

    @property
    def google_genai_use_vertexai(self) -> bool:
        return self._bool("GOOGLE_GENAI_USE_VERTEXAI", False)

    @property
    def google_cloud_project(self) -> str | None:
        return self._str("GOOGLE_CLOUD_PROJECT")

    @property
    def google_cloud_location(self) -> str:
        return self._str("GOOGLE_CLOUD_LOCATION", "us-central1") or "us-central1"

    @property
    def gemini_api_key(self) -> str | None:
        return self._str("GEMINI_API_KEY")

    @property
    def voyage_api_key(self) -> str | None:
        return self._str("VOYAGE_API_KEY") or self._str("VOYAGEAI_API_KEY")

    @property
    def anthropic_api_key(self) -> str | None:
        return self._str("ANTHROPIC_API_KEY")

    @property
    def bedrock_llm_temperature(self) -> float:
        return self._float("BEDROCK_LLM_TEMPERATURE", 1.0) or 1.0

    @property
    def aws_access_key_id(self) -> str | None:
        return self._str("AWS_ACCESS_KEY_ID")

    @property
    def aws_secret_access_key(self) -> str | None:
        return self._str("AWS_SECRET_ACCESS_KEY")

    @property
    def aws_session_token(self) -> str | None:
        return self._str("AWS_SESSION_TOKEN")

    @property
    def aws_region(self) -> str | None:
        return self._str("AWS_REGION")

    @property
    def webui_title(self) -> str | None:
        return self._str("WEBUI_TITLE")

    @property
    def webui_description(self) -> str | None:
        return self._str("WEBUI_DESCRIPTION")

    @property
    def lightrag_runtime_data_dir(self) -> str | None:
        return self._str("LIGHTRAG_RUNTIME_DATA_DIR")

    @property
    def mlx_openai_server_log_level(self) -> str:
        return self._str("MLX_OPENAI_SERVER_LOG_LEVEL", "INFO") or "INFO"

    @property
    def mlx_agentcpm_model(self) -> str | None:
        return self._str("MLX_AGENTCPM_MODEL")

    @property
    def mlx_agentcpm_model_path(self) -> str | None:
        return self._str("MLX_AGENTCPM_MODEL_PATH")

    @property
    def mlx_extraction_model_path(self) -> str | None:
        return self._str("MLX_EXTRACTION_MODEL_PATH")

    @property
    def mlx_openai_server_queue_timeout_s(self) -> int | None:
        return self._int("MLX_OPENAI_SERVER_QUEUE_TIMEOUT_S", minimum=30)

    @property
    def mlx_openai_server_idle_timeout_s(self) -> int:
        return self._int("MLX_OPENAI_SERVER_IDLE_TIMEOUT_S", 180, minimum=30) or 180

    @property
    def mlx_openai_server_on_demand(self) -> bool:
        return self._bool("MLX_OPENAI_SERVER_ON_DEMAND", True)

    @property
    def mlx_openai_server_retrieval_context_length(self) -> int:
        return self._int("MLX_OPENAI_SERVER_RETRIEVAL_CONTEXT_LENGTH", 12288, minimum=1) or 12288

    @property
    def mlx_openai_server_retrieval_context_length_configured(self) -> int | None:
        return self._int("MLX_OPENAI_SERVER_RETRIEVAL_CONTEXT_LENGTH", minimum=1)

    @property
    def mlx_openai_server_retrieval_max_tokens(self) -> int:
        return self._int("MLX_OPENAI_SERVER_RETRIEVAL_MAX_TOKENS", 1024, minimum=1) or 1024

    @property
    def mlx_openai_server_retrieval_max_tokens_configured(self) -> int | None:
        return self._int("MLX_OPENAI_SERVER_RETRIEVAL_MAX_TOKENS", minimum=1)

    @property
    def mlx_openai_server_extraction_context_length(self) -> int:
        return self._int("MLX_OPENAI_SERVER_EXTRACTION_CONTEXT_LENGTH", 8192, minimum=1) or 8192

    @property
    def mlx_openai_server_extraction_context_length_configured(self) -> int | None:
        return self._int("MLX_OPENAI_SERVER_EXTRACTION_CONTEXT_LENGTH", minimum=1)

    @property
    def mlx_openai_server_extraction_max_tokens(self) -> int:
        return self._int("MLX_OPENAI_SERVER_EXTRACTION_MAX_TOKENS", 2048, minimum=1) or 2048

    @property
    def mlx_openai_server_extraction_max_tokens_configured(self) -> int | None:
        return self._int("MLX_OPENAI_SERVER_EXTRACTION_MAX_TOKENS", minimum=1)

    @property
    def mlx_rerank_host(self) -> str:
        return self._str("MLX_RERANK_HOST", self.mlx_embeddings_host) or self.mlx_embeddings_host

    @property
    def mlx_rerank_port(self) -> int:
        return self._int("MLX_RERANK_PORT", self.mlx_embeddings_port) or self.mlx_embeddings_port

    @property
    def mlx_rerank_watchdog_interval_s(self) -> int:
        return self._int("MLX_RERANK_WATCHDOG_INTERVAL_S", 15, minimum=5) or 15

    @property
    def mlx_openai_server_max_rss_mb(self) -> int:
        return self._int("MLX_OPENAI_SERVER_MAX_RSS_MB", 20480, minimum=0) or 20480

    @property
    def mlx_openai_server_watchdog_interval_s(self) -> int:
        return self._int("MLX_OPENAI_SERVER_WATCHDOG_INTERVAL_S", 15, minimum=5) or 15

    @property
    def pythonpath(self) -> str:
        return self._str("PYTHONPATH", "") or ""

    @property
    def mlx_openai_server_package_version(self) -> str:
        return self._str("MLX_OPENAI_SERVER_PACKAGE_VERSION", "1.8.1") or "1.8.1"

    @property
    def mlx_lm_package_version(self) -> str:
        return self._str("MLX_LM_PACKAGE_VERSION", "0.31.3") or "0.31.3"

    @property
    def lightrag_gunicorn_mode_configured(self) -> bool:
        return self._has("LIGHTRAG_GUNICORN_MODE")

    @property
    def gunicorn_cmd_args_configured(self) -> bool:
        return self._has("GUNICORN_CMD_ARGS")

    @property
    def rerank_enable_chunking(self) -> bool:
        return self._bool("RERANK_ENABLE_CHUNKING", False)

    @property
    def rerank_max_tokens_per_doc(self) -> int:
        return self._int("RERANK_MAX_TOKENS_PER_DOC", 4096, minimum=1) or 4096

    @property
    def objc_disable_initialize_fork_safety(self) -> str | None:
        return self._str("OBJC_DISABLE_INITIALIZE_FORK_SAFETY")

    @property
    def lightrag_performance_timing_logs(self) -> bool:
        return self._bool("LIGHTRAG_PERFORMANCE_TIMING_LOGS", False)

    def missing_env_names(self, env_names: list[str]) -> list[str]:
        self._ensure_current()
        return [env_name for env_name in env_names if env_name not in self.raw_env]

    def _optional_stripped(self, env_key: str) -> str | None:
        value = self._str(env_key)
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def _effective_workspace(
        self,
        override: str | None,
        workspace: str | None,
        *,
        default: str = "",
    ) -> str:
        if override is not None:
            return override
        candidate = str(workspace or "").strip()
        return candidate or default

    def workspace_for_kv_storage(self, storage_name: str) -> str:
        workspace_overrides = {
            "PGKVStorage": self.postgres_workspace_override,
            "MongoKVStorage": self.mongodb_workspace_override,
            "RedisKVStorage": self.redis_workspace_override,
            "OpenSearchKVStorage": self.opensearch_workspace_override,
        }
        return workspace_overrides.get(storage_name) or self.workspace

    @property
    def redis_max_connections(self) -> int:
        return self._int("REDIS_MAX_CONNECTIONS", 200, minimum=1) or 200

    @property
    def redis_socket_timeout(self) -> float:
        return self._float("REDIS_SOCKET_TIMEOUT", 30.0) or 30.0

    @property
    def redis_connect_timeout(self) -> float:
        return self._float("REDIS_CONNECT_TIMEOUT", 10.0) or 10.0

    @property
    def redis_retry_attempts(self) -> int:
        return self._int("REDIS_RETRY_ATTEMPTS", 3, minimum=1) or 3

    @property
    def redis_workspace_override(self) -> str | None:
        return self._optional_stripped("REDIS_WORKSPACE")

    def effective_redis_workspace(self, workspace: str | None) -> str:
        return self._effective_workspace(self.redis_workspace_override, workspace)

    def redis_uri(self, default: str | None = None) -> str | None:
        return self._str("REDIS_URI", default)

    @property
    def neo4j_workspace_override(self) -> str | None:
        return self._optional_stripped("NEO4J_WORKSPACE")

    def effective_neo4j_workspace(self, workspace: str | None) -> str:
        return self._effective_workspace(
            self.neo4j_workspace_override,
            workspace,
            default="base",
        )

    def neo4j_uri(self, default: str | None = None) -> str | None:
        return self._str("NEO4J_URI", default)

    def neo4j_username(self, default: str | None = None) -> str | None:
        return self._str("NEO4J_USERNAME", default)

    def neo4j_password(self, default: str | None = None) -> str | None:
        return self._str("NEO4J_PASSWORD", default)

    def neo4j_max_connection_pool_size(self, default: int = 100) -> int:
        return self._int("NEO4J_MAX_CONNECTION_POOL_SIZE", default, minimum=1) or default

    def neo4j_connection_timeout(self, default: float = 30.0) -> float:
        return self._float("NEO4J_CONNECTION_TIMEOUT", default, minimum=0.0) or default

    def neo4j_connection_acquisition_timeout(self, default: float = 30.0) -> float:
        return (
            self._float(
                "NEO4J_CONNECTION_ACQUISITION_TIMEOUT",
                default,
                minimum=0.0,
            )
            or default
        )

    def neo4j_max_transaction_retry_time(self, default: float = 30.0) -> float:
        return (
            self._float(
                "NEO4J_MAX_TRANSACTION_RETRY_TIME",
                default,
                minimum=0.0,
            )
            or default
        )

    def neo4j_max_connection_lifetime(self, default: float = 300.0) -> float:
        return self._float("NEO4J_MAX_CONNECTION_LIFETIME", default, minimum=0.0) or default

    def neo4j_liveness_check_timeout(self, default: float = 30.0) -> float:
        return self._float("NEO4J_LIVENESS_CHECK_TIMEOUT", default, minimum=0.0) or default

    def neo4j_keep_alive(self, default: bool = True) -> bool:
        if not self._has("NEO4J_KEEP_ALIVE"):
            return default
        return self._bool("NEO4J_KEEP_ALIVE", default)

    def neo4j_database(self, default: str | None = None) -> str | None:
        return self._str("NEO4J_DATABASE", default)

    @property
    def memgraph_workspace_override(self) -> str | None:
        return self._optional_stripped("MEMGRAPH_WORKSPACE")

    def effective_memgraph_workspace(self, workspace: str | None) -> str:
        return self._effective_workspace(
            self.memgraph_workspace_override,
            workspace,
            default="base",
        )

    def memgraph_uri(self, default: str | None = None) -> str | None:
        return self._str("MEMGRAPH_URI", default)

    def memgraph_username(self, default: str | None = None) -> str | None:
        return self._str("MEMGRAPH_USERNAME", default)

    def memgraph_password(self, default: str | None = None) -> str | None:
        return self._str("MEMGRAPH_PASSWORD", default)

    def memgraph_database(self, default: str | None = None) -> str | None:
        return self._str("MEMGRAPH_DATABASE", default)

    @property
    def milvus_index_type(self) -> str:
        return self._str("MILVUS_INDEX_TYPE", "AUTOINDEX") or "AUTOINDEX"

    @property
    def milvus_metric_type(self) -> str:
        return self._str("MILVUS_METRIC_TYPE", "COSINE") or "COSINE"

    @property
    def milvus_hnsw_m(self) -> int:
        return self._int("MILVUS_HNSW_M", 16, minimum=1) or 16

    @property
    def milvus_hnsw_ef_construction(self) -> int:
        return self._int("MILVUS_HNSW_EF_CONSTRUCTION", 360, minimum=1) or 360

    @property
    def milvus_hnsw_ef(self) -> int:
        return self._int("MILVUS_HNSW_EF", 200, minimum=1) or 200

    @property
    def milvus_hnsw_sq_type(self) -> str:
        return self._str("MILVUS_HNSW_SQ_TYPE", "SQ8") or "SQ8"

    @property
    def milvus_hnsw_sq_refine(self) -> bool:
        return self._bool("MILVUS_HNSW_SQ_REFINE", False)

    @property
    def milvus_hnsw_sq_refine_type(self) -> str:
        return self._str("MILVUS_HNSW_SQ_REFINE_TYPE", "FP32") or "FP32"

    @property
    def milvus_hnsw_sq_refine_k(self) -> int:
        return self._int("MILVUS_HNSW_SQ_REFINE_K", 10, minimum=1) or 10

    @property
    def milvus_ivf_nlist(self) -> int:
        return self._int("MILVUS_IVF_NLIST", 1024, minimum=1) or 1024

    @property
    def milvus_ivf_nprobe(self) -> int:
        return self._int("MILVUS_IVF_NPROBE", 16, minimum=1) or 16

    @property
    def milvus_workspace_override(self) -> str | None:
        return self._optional_stripped("MILVUS_WORKSPACE")

    def effective_milvus_workspace(self, workspace: str | None) -> str:
        return self._effective_workspace(self.milvus_workspace_override, workspace)

    def milvus_uri(self, default: str | None = None) -> str | None:
        return self._str("MILVUS_URI", default)

    def milvus_user(self, default: str | None = None) -> str | None:
        return self._str("MILVUS_USER", default)

    def milvus_password(self, default: str | None = None) -> str | None:
        return self._str("MILVUS_PASSWORD", default)

    def milvus_token(self, default: str | None = None) -> str | None:
        return self._str("MILVUS_TOKEN", default)

    def milvus_db_name(self, default: str | None = None) -> str | None:
        return self._str("MILVUS_DB_NAME", default)

    @property
    def postgres_workspace_override(self) -> str | None:
        return self._optional_stripped("POSTGRES_WORKSPACE")

    def postgres_host(self, default: str = "localhost") -> str:
        return self._str("POSTGRES_HOST", default) or default

    def postgres_port(self, default: int = 5432) -> int:
        return self._int("POSTGRES_PORT", default, minimum=1) or default

    def postgres_user(self, default: str = "postgres") -> str:
        return self._str("POSTGRES_USER", default) or default

    def postgres_password(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_PASSWORD", default)

    def postgres_database(self, default: str = "postgres") -> str:
        return self._str("POSTGRES_DATABASE", default) or default

    def postgres_workspace_name(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_WORKSPACE", default)

    def postgres_max_connections(self, default: int = 50) -> int:
        return self._int("POSTGRES_MAX_CONNECTIONS", default, minimum=1) or default

    def postgres_ssl_mode(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_SSL_MODE", default)

    def postgres_ssl_cert(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_SSL_CERT", default)

    def postgres_ssl_key(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_SSL_KEY", default)

    def postgres_ssl_root_cert(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_SSL_ROOT_CERT", default)

    def postgres_ssl_crl(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_SSL_CRL", default)

    def postgres_vector_index_type(self, default: str = "HNSW") -> str:
        return self._str("POSTGRES_VECTOR_INDEX_TYPE", default) or default

    def postgres_hnsw_m(self, default: int = 16) -> int:
        return self._int("POSTGRES_HNSW_M", default, minimum=1) or default

    def postgres_hnsw_ef(self, default: int = 64) -> int:
        return self._int("POSTGRES_HNSW_EF", default, minimum=1) or default

    def postgres_ivfflat_lists(self, default: int = 100) -> int:
        return self._int("POSTGRES_IVFFLAT_LISTS", default, minimum=1) or default

    def postgres_vchordrq_build_options(self, default: str = "") -> str:
        return self._str("POSTGRES_VCHORDRQ_BUILD_OPTIONS", default) or default

    def postgres_vchordrq_probes(self, default: str = "") -> str:
        return self._str("POSTGRES_VCHORDRQ_PROBES", default) or default

    def postgres_vchordrq_epsilon(self, default: float = 1.9) -> float:
        return self._float("POSTGRES_VCHORDRQ_EPSILON", default) or default

    def postgres_server_settings(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_SERVER_SETTINGS", default)

    def postgres_statement_cache_size(self, default: str | None = None) -> str | None:
        return self._str("POSTGRES_STATEMENT_CACHE_SIZE", default)

    def postgres_connection_retries(self, default: int = 10) -> int:
        return min(
            100,
            self._int("POSTGRES_CONNECTION_RETRIES", default, minimum=0) or default,
        )

    def postgres_connection_retry_backoff(self, default: float = 3.0) -> float:
        return min(
            300.0,
            self._float(
                "POSTGRES_CONNECTION_RETRY_BACKOFF",
                default,
                minimum=0.0,
            )
            or default,
        )

    def postgres_connection_retry_backoff_max(self, default: float = 30.0) -> float:
        return min(
            600.0,
            self._float(
                "POSTGRES_CONNECTION_RETRY_BACKOFF_MAX",
                default,
                minimum=0.0,
            )
            or default,
        )

    def postgres_pool_close_timeout(self, default: float = 5.0) -> float:
        return min(
            30.0,
            self._float("POSTGRES_POOL_CLOSE_TIMEOUT", default, minimum=0.0)
            or default,
        )

    def opensearch_number_of_shards(self, default: int = 1) -> int:
        return self._int("OPENSEARCH_NUMBER_OF_SHARDS", default, minimum=1) or default

    def opensearch_number_of_replicas(self, default: int = 0) -> int:
        return self._int("OPENSEARCH_NUMBER_OF_REPLICAS", default, minimum=0) or default

    @property
    def opensearch_workspace_override(self) -> str | None:
        return self._optional_stripped("OPENSEARCH_WORKSPACE")

    def effective_opensearch_workspace(self, workspace: str | None) -> str:
        return self._effective_workspace(self.opensearch_workspace_override, workspace)

    def opensearch_hosts(self, default: str = "localhost:9200") -> str:
        return self._str("OPENSEARCH_HOSTS", default) or default

    def opensearch_user(self, default: str = "admin") -> str:
        return self._str("OPENSEARCH_USER", default) or default

    def opensearch_password(self, default: str = "admin") -> str:
        return self._str("OPENSEARCH_PASSWORD", default) or default

    def opensearch_use_ssl(self, default: bool = True) -> bool:
        if not self._has("OPENSEARCH_USE_SSL"):
            return default
        return self._bool("OPENSEARCH_USE_SSL", default)

    def opensearch_verify_certs(self, default: bool = False) -> bool:
        if not self._has("OPENSEARCH_VERIFY_CERTS"):
            return default
        return self._bool("OPENSEARCH_VERIFY_CERTS", default)

    def opensearch_timeout(self, default: int = 30) -> int:
        return self._int("OPENSEARCH_TIMEOUT", default, minimum=1) or default

    def opensearch_max_retries(self, default: int = 3) -> int:
        return self._int("OPENSEARCH_MAX_RETRIES", default, minimum=0) or default

    @property
    def opensearch_use_ppl_graphlookup_override(self) -> bool | None:
        if not self._has("OPENSEARCH_USE_PPL_GRAPHLOOKUP"):
            return None
        return self._bool("OPENSEARCH_USE_PPL_GRAPHLOOKUP", False)

    @property
    def opensearch_knn_ef_construction(self) -> int:
        return self._int("OPENSEARCH_KNN_EF_CONSTRUCTION", 200, minimum=1) or 200

    @property
    def opensearch_knn_m(self) -> int:
        return self._int("OPENSEARCH_KNN_M", 16, minimum=1) or 16

    @property
    def opensearch_knn_ef_search(self) -> int:
        return self._int("OPENSEARCH_KNN_EF_SEARCH", 100, minimum=1) or 100

    @property
    def qdrant_workspace_override(self) -> str | None:
        return self._optional_stripped("QDRANT_WORKSPACE")

    def effective_qdrant_workspace(self, workspace: str | None) -> str:
        return self._effective_workspace(self.qdrant_workspace_override, workspace)

    def qdrant_url(self, default: str | None = None) -> str | None:
        return self._str("QDRANT_URL", default)

    def qdrant_api_key(self, default: str | None = None) -> str | None:
        return self._str("QDRANT_API_KEY", default)

    @property
    def qdrant_upsert_max_payload_bytes(self) -> int:
        return self._int("QDRANT_UPSERT_MAX_PAYLOAD_BYTES", 16 * 1024 * 1024) or 16 * 1024 * 1024

    @property
    def qdrant_upsert_max_points_per_batch(self) -> int:
        return self._int("QDRANT_UPSERT_MAX_POINTS_PER_BATCH", 128) or 128

    @property
    def mongodb_workspace_override(self) -> str | None:
        return self._optional_stripped("MONGODB_WORKSPACE")

    def effective_mongodb_workspace(self, workspace: str | None) -> str:
        return self._effective_workspace(self.mongodb_workspace_override, workspace)

    def mongodb_uri(self, default: str | None = None) -> str | None:
        return self._str("MONGO_URI", default)

    def mongodb_database(self, default: str | None = None) -> str | None:
        return self._str("MONGO_DATABASE", default)

    @property
    def mongo_graph_bfs_mode(self) -> str:
        return self._str("MONGO_GRAPH_BFS_MODE", "bidirectional") or "bidirectional"

    @property
    def eval_llm_binding_api_key(self) -> str | None:
        return self._str("EVAL_LLM_BINDING_API_KEY")

    @property
    def eval_llm_model(self) -> str:
        return self._str("EVAL_LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"

    @property
    def eval_llm_binding_host(self) -> str | None:
        return self._str("EVAL_LLM_BINDING_HOST")

    @property
    def eval_embedding_binding_api_key(self) -> str | None:
        return self._str("EVAL_EMBEDDING_BINDING_API_KEY")

    @property
    def eval_embedding_model(self) -> str:
        return self._str("EVAL_EMBEDDING_MODEL", "text-embedding-3-large") or "text-embedding-3-large"

    @property
    def eval_embedding_binding_host(self) -> str | None:
        return self._str("EVAL_EMBEDDING_BINDING_HOST")

    @property
    def eval_llm_max_retries(self) -> int:
        return self._int("EVAL_LLM_MAX_RETRIES", 5, minimum=0) or 5

    @property
    def eval_llm_timeout(self) -> int:
        return self._int("EVAL_LLM_TIMEOUT", 180, minimum=1) or 180

    def mlx_chat_max_tokens_for(self, default: int) -> int:
        return self._int("MLX_CHAT_MAX_TOKENS", default, minimum=1) or default

    def mlx_chat_timeout_for(self, default: int) -> int:
        return self._int("MLX_CHAT_TIMEOUT", default, minimum=1) or default


    @property
    def eval_query_top_k(self) -> int:
        return self._int("EVAL_QUERY_TOP_K", 10, minimum=1) or 10

    @property
    def eval_max_concurrent(self) -> int:
        return self._int("EVAL_MAX_CONCURRENT", 2, minimum=1) or 2

    @property
    def tiktoken_cache_dir(self) -> str | None:
        return self._str("TIKTOKEN_CACHE_DIR")

    @property
    def postgres_host_configured(self) -> bool:
        return self._has("POSTGRES_HOST")


settings = LightRAGSettings()
