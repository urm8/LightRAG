from ..utils import verbose_debug, VERBOSE_DEBUG
import os
import logging
import json
import time
import warnings

from collections.abc import AsyncIterator

import pipmaster as pm
import tiktoken

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    InternalServerError,
    BadRequestError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    safe_unicode_decode,
    logger,
    clean_chatml_markers,
    contains_chatml_markers,
)

from lightrag.api import __api_version__

import numpy as np
import base64
from typing import Any, Union
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel
from lightrag.config import settings

# Try to import Langfuse for LLM observability (optional)
# Falls back to standard OpenAI client if not available
# Langfuse requires proper configuration to work correctly
LANGFUSE_ENABLED = False
try:
    # Check if required Langfuse environment variables are set
    langfuse_public_key = settings.langfuse_public_key
    langfuse_secret_key = settings.langfuse_secret_key

    # Only enable Langfuse if both keys are configured
    if langfuse_public_key and langfuse_secret_key:
        from langfuse.openai import AsyncOpenAI  # type: ignore[import-untyped]

        LANGFUSE_ENABLED = True
        logger.info("Langfuse observability enabled for OpenAI client")
    else:
        from openai import AsyncOpenAI

        logger.debug(
            "Langfuse environment variables not configured, using standard OpenAI client"
        )
except ImportError:
    from openai import AsyncOpenAI

    logger.debug("Langfuse not available, using standard OpenAI client")


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""

    pass


class TransientBadRequestError(Exception):
    """Wrapper to trigger retry on transient HTTP 400 errors.

    Some 400s are not genuine client errors: the OpenAI API (or a proxy in
    front of it) intermittently returns "We could not parse the JSON body of
    your request" when the request body is corrupted/truncated in transit.
    These succeed on retry, so we re-raise them as this retryable type while
    letting genuine 400s (bad params, content policy, etc.) fail fast.
    """

    pass


def _validate_openai_response_format(response_format: Any | None) -> None:
    """Accept OpenAI wire-format dicts and typed Pydantic schemas."""
    if response_format is None or isinstance(response_format, dict):
        return
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        return

    raise TypeError(
        "openai_complete_if_cache only supports dict response_format payloads "
        "or Pydantic BaseModel schema types."
    )


# Module-level cache for tiktoken encodings
_TIKTOKEN_ENCODING_CACHE: dict[str, Any] = {}


def _should_use_openai_parse(
    *,
    base_url: str | None,
    use_azure: bool,
) -> bool:
    if use_azure:
        return True
    if not base_url:
        return True
    host = (urlparse(base_url).hostname or "").lower()
    return host.endswith("openai.com")


def _response_format_to_schema(
    response_format: Any, *, json_object_only: bool = False
) -> Any:
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        if json_object_only:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            },
        }
    return response_format


def _build_http_client_for_base_url(base_url: str | None) -> httpx.AsyncClient | None:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    if parsed.scheme != "http":
        return None
    return httpx.AsyncClient(verify=False)

# Whether to request base64-encoded embeddings from the API.
# Base64 is more efficient over the wire; set EMBEDDING_USE_BASE64=false for
# providers that don't support it (e.g. Yandex Cloud).
EMBEDDING_USE_BASE64: bool = settings.embedding_use_base64


def _normalize_base_url(base_url: str | None) -> str:
    if not base_url:
        return ""
    return base_url.rstrip("/")


def _is_managed_swift_lm_base_url(base_url: str | None) -> bool:
    return settings.lightrag_manage_swift_lm and _normalize_base_url(
        base_url
    ) == _normalize_base_url(settings.managed_swift_lm_base_url)


def _managed_mlx_openai_server_base_url() -> str:
    return settings.managed_mlx_openai_server_base_url


def _managed_mlx_embeddings_base_url() -> str:
    return settings.managed_mlx_embeddings_base_url


def _get_local_mlx_request_kind(base_url: str | None, model: str | None) -> str | None:
    if settings.lightrag_manage_swift_lm:
        expected_base_url = settings.managed_swift_lm_base_url
    elif settings.lightrag_manage_mlx_openai_server:
        expected_base_url = _managed_mlx_openai_server_base_url()
    else:
        return None
    if _normalize_base_url(base_url) != _normalize_base_url(expected_base_url):
        return None
    extraction_model = settings.mlx_extraction_model or settings.extraction_llm_model or ""
    if model == extraction_model:
        return "extraction"
    return "chat"


def _is_local_mlx_embeddings_base_url(base_url: str | None) -> bool:
    normalized = _normalize_base_url(base_url)
    if not normalized:
        return False
    if settings.lightrag_manage_mlx_openai_server and normalized == _normalize_base_url(
        _managed_mlx_openai_server_base_url()
    ):
        return True
    if settings.lightrag_manage_swift_embeddings and normalized == _normalize_base_url(
        settings.managed_swift_embeddings_base_url
    ):
        return True
    return normalized == _normalize_base_url(_managed_mlx_embeddings_base_url())


def _get_local_mlx_recycle_limit(kind: str) -> int:
    if kind not in {"chat", "extraction", "embedding"}:
        return 0
    return settings.mlx_managed_request_recycle_limit(kind)


# ---------------------------------------------------------------------------
# Hermes / ChatML model support
# ---------------------------------------------------------------------------

# The exact HuggingFace model ID for Hermes-4-abliterated served through
# mlx-openai-server.  When this model is in use, stop sequences and response
# cleanup are activated in the request/response layer to prevent ChatML
# control tokens leaking into downstream consumers.
HERMES_4_CHATML_MODEL_ID = "divinetribe/Hermes-4-14B-abliterated-4bit-mlx"

# Substring check — any model ID containing "Hermes-4" triggers ChatML
# handling.  This covers both the exact ID above and any derivative names.
_CHATML_MODEL_PATTERN = "Hermes-4"


def _is_chatml_model(model: str) -> bool:
    """Return True when *model* is known to produce ChatML control tokens."""
    if not model:
        return False
    return _CHATML_MODEL_PATTERN in model


_CHATML_STOP_SEQUENCES: list[str] = ["<|im_end|>", "<|im_start|>"]


def _get_tiktoken_encoding_for_model(model: str) -> Any:
    """Get tiktoken encoding for the specified model with caching.

    Args:
        model: The model name to get encoding for.

    Returns:
        The tiktoken encoding for the model.
    """
    if model not in _TIKTOKEN_ENCODING_CACHE:
        try:
            _TIKTOKEN_ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.debug(
                f"Encoding for model '{model}' not found, falling back to cl100k_base"
            )
            _TIKTOKEN_ENCODING_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING_CACHE[model]


def _llm_metrics_logging_enabled() -> bool:
    return settings.lightrag_llm_request_metrics_enabled


def _stringify_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(content)


def _estimate_message_tokens(model: str, messages: list[dict[str, Any]]) -> int:
    encoding = _get_tiktoken_encoding_for_model(model)
    token_count = 0
    for message in messages:
        token_count += 4
        token_count += len(encoding.encode(str(message.get("role", ""))))
        if "name" in message:
            token_count += len(encoding.encode(str(message["name"]))) - 1
        token_count += len(
            encoding.encode(_stringify_message_content(message.get("content", "")))
        )
    return token_count + 2


def _estimate_text_tokens(model: str, text: str) -> int:
    encoding = _get_tiktoken_encoding_for_model(model)
    return len(encoding.encode(text or ""))


def _usage_to_token_counts(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens <= 0 and completion_tokens <= 0 and total_tokens <= 0:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _timings_value(timings: Any, key: str) -> float | None:
    if timings is None:
        return None

    value = None
    if isinstance(timings, dict):
        value = timings.get(key)
    else:
        value = getattr(timings, key, None)

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_non_stream_ttft_s(response: Any, total_duration_s: float) -> float | None:
    timings = getattr(response, "timings", None)
    if timings is None:
        return None

    prompt_ms = _timings_value(timings, "prompt_ms")
    if prompt_ms is not None and prompt_ms >= 0:
        return prompt_ms / 1000

    prompt_n = _timings_value(timings, "prompt_n")
    prompt_per_second = _timings_value(timings, "prompt_per_second")
    if (
        prompt_n is not None
        and prompt_per_second is not None
        and prompt_n >= 0
        and prompt_per_second > 0
    ):
        return prompt_n / prompt_per_second

    predicted_ms = _timings_value(timings, "predicted_ms")
    if predicted_ms is None or predicted_ms < 0:
        return None

    decode_duration_s = predicted_ms / 1000
    if decode_duration_s > total_duration_s:
        return None
    return max(total_duration_s - decode_duration_s, 0.0)


def _resolve_llm_token_counts(
    model: str,
    messages: list[dict[str, Any]],
    output_text: str,
    usage: Any,
) -> tuple[dict[str, int], str]:
    usage_counts = _usage_to_token_counts(usage)
    if usage_counts is not None:
        return usage_counts, "api"

    prompt_tokens = _estimate_message_tokens(model, messages)
    completion_tokens = _estimate_text_tokens(model, output_text)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }, "estimated"


def _message_char_count(messages: list[dict[str, Any]]) -> int:
    return sum(
        len(_stringify_message_content(message.get("content", "")))
        for message in messages
    )


def _infer_llm_request_kind(
    explicit_kind: str | None,
    base_url: str | None,
    model: str | None,
    *,
    keyword_extraction: bool,
) -> str:
    if explicit_kind:
        return explicit_kind
    managed_kind = _get_local_mlx_request_kind(base_url, model)
    if managed_kind == "extraction":
        return "extraction"
    if keyword_extraction:
        return "keyword_extraction"
    return managed_kind or "chat"


def _log_llm_request_metrics(
    *,
    request_kind: str,
    model: str,
    base_url: str | None,
    stream: bool,
    token_counts: dict[str, int],
    usage_source: str,
    total_duration_s: float,
    ttft_s: float | None,
    input_chars: int,
    output_chars: int,
) -> None:
    if not _llm_metrics_logging_enabled():
        return

    completion_tokens = token_counts.get("completion_tokens", 0)
    prompt_tokens = token_counts.get("prompt_tokens", 0)
    total_tokens = token_counts.get("total_tokens", 0)
    generation_duration_s = None
    if ttft_s is not None:
        generation_duration_s = max(total_duration_s - ttft_s, 0.0)
    elif not stream:
        generation_duration_s = total_duration_s

    completion_tps = None
    if generation_duration_s and generation_duration_s > 0 and completion_tokens > 0:
        completion_tps = completion_tokens / generation_duration_s

    latency_ms = total_duration_s * 1000
    ttft_ms = ttft_s * 1000 if ttft_s is not None else None
    logger.info(
        "LLM metrics kind=%s stream=%s model=%s base_url=%s latency_ms=%.1f ttft_ms=%s completion_tps=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s usage_source=%s input_chars=%s output_chars=%s",
        request_kind,
        stream,
        model,
        _normalize_base_url(base_url) or "default",
        latency_ms,
        f"{ttft_ms:.1f}" if ttft_ms is not None else "n/a",
        f"{completion_tps:.2f}" if completion_tps is not None else "n/a",
        prompt_tokens,
        completion_tokens,
        total_tokens,
        usage_source,
        input_chars,
        output_chars,
    )


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    use_azure: bool = False,
    azure_deployment: str | None = None,
    api_version: str | None = None,
    timeout: int | None = None,
    client_configs: dict[str, Any] | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI or AsyncAzureOpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        base_url: Base URL for the OpenAI API. If None, uses the default OpenAI API URL.
        use_azure: Whether to create an Azure OpenAI client. Default is False.
        azure_deployment: Azure OpenAI deployment name (only used when use_azure=True).
        api_version: Azure OpenAI API version (only used when use_azure=True).
        timeout: Request timeout in seconds.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        An AsyncOpenAI or AsyncAzureOpenAI client instance.
    """
    if use_azure:
        from openai import AsyncAzureOpenAI

        if not api_key:
            api_key = settings.azure_openai_api_key or settings.llm_binding_api_key

        if client_configs is None:
            client_configs = {}

        # Create a merged config dict with precedence: explicit params > client_configs
        merged_configs = {
            **client_configs,
            "api_key": api_key,
        }

        # Add explicit parameters (override client_configs)
        if base_url is not None:
            merged_configs["azure_endpoint"] = base_url
        if azure_deployment is not None:
            merged_configs["azure_deployment"] = azure_deployment
        if api_version is not None:
            merged_configs["api_version"] = api_version
        if timeout is not None:
            merged_configs["timeout"] = timeout

        return AsyncAzureOpenAI(**merged_configs)
    else:
        if not api_key:
            api_key = settings.openai_api_key
            if api_key is None:
                raise KeyError("OPENAI_API_KEY")

        default_headers = {
            "User-Agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_8) LightRAG/{__api_version__}",
            "Content-Type": "application/json",
        }
        dashscope_workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
        if dashscope_workspace_id:
            default_headers["X-DashScope-Workspace"] = dashscope_workspace_id

        if client_configs is None:
            client_configs = {}

        # Create a merged config dict with precedence: explicit params > client_configs > defaults
        merged_configs = {
            **client_configs,
            "default_headers": default_headers,
            "api_key": api_key,
        }

        if base_url is not None:
            merged_configs["base_url"] = base_url
        else:
            merged_configs["base_url"] = settings.openai_api_base

        if timeout is not None:
            merged_configs["timeout"] = timeout

        if "http_client" not in merged_configs:
            http_client = _build_http_client_for_base_url(base_url)
            if http_client is not None:
                merged_configs["http_client"] = http_client

        return AsyncOpenAI(**merged_configs)


# TODO LengthFinishReasonError should not persist into LLM cache
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
        # Retry transient HTTP 5xx (OpenAI "500 server_error", proxy "upstream
        # connect error"). InternalServerError covers all status >= 500.
        | retry_if_exception_type(InternalServerError)
        # Retry transient "could not parse JSON body" 400s (see handler below).
        | retry_if_exception_type(TransientBadRequestError)
    ),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    timeout: int | None = None,
    keyword_extraction: bool = False,
    use_azure: bool = False,
    azure_deployment: str | None = None,
    api_version: str | None = None,
    image_inputs: list[Any] | None = None,
    **kwargs: Any,
) -> str:
    """Complete a prompt using OpenAI's API with caching support and Chain of Thought (COT) integration.

    This function supports automatic integration of reasoning content from models that provide
    Chain of Thought capabilities. The reasoning content is seamlessly integrated into the response
    using <think>...</think> tags.

    Structured output design note:
    - This adapter supports dict-based OpenAI response_format payloads,
      including ``{"type": "json_object"}`` and dict-form ``json_schema``.
    - Typed/Pydantic ``response_format`` helpers are rejected explicitly.
    - Structured responses are returned as raw text from ``message.content``
      and are not locally schema-validated here.
    - ``keyword_extraction`` is deprecated; prefer
      ``response_format={"type": "json_object"}`` instead.

    Note on truncated structured output: when the OpenAI SDK raises
    `LengthFinishReasonError`, callers may still receive partial raw JSON from
    `completion.choices[0].message.content`. That payload should be treated as
    best-effort recovery only. If the JSON was truncated or repaired after
    truncation, it is safer not to persist it into the LLM cache because later
    runs with a higher token budget could otherwise keep reusing incomplete data.

    Note on `reasoning_content`: This feature relies on a Deepseek Style `reasoning_content`
    in the API response, which may be provided by OpenAI-compatible endpoints that support
    Chain of Thought.

    COT Integration Rules:
    1. COT content is accepted only when regular content is empty and `reasoning_content` has content.
    2. COT processing stops when regular content becomes available.
    3. If both `content` and `reasoning_content` are present simultaneously, reasoning is ignored.
    4. If both fields have content from the start, COT is never activated.
    5. For streaming: COT content is inserted into the content stream with <think> tags.
    6. For non-streaming: COT content is prepended to regular content with <think> tags.

    Args:
        model: The OpenAI model to use. For Azure, this can be the deployment name.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        enable_cot: Whether to enable Chain of Thought (COT) processing. Default is False.
        base_url: Optional base URL for the OpenAI API. For Azure, this should be the
            Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/).
        api_key: Optional API key. For standard OpenAI, uses OPENAI_API_KEY environment
            variable if None. For Azure, uses AZURE_OPENAI_API_KEY if None.
        token_tracker: Optional token usage tracker for monitoring API usage.
        stream: Whether to stream the response. Default is False.
        timeout: Request timeout in seconds. Default is None.
        keyword_extraction: Deprecated compatibility shim. When True and no
            explicit ``response_format`` is supplied, it is mapped to
            ``{"type": "json_object"}``. Prefer passing ``response_format``
            directly. Default is False.
        use_azure: Whether to use Azure OpenAI service instead of standard OpenAI.
            When True, creates an AsyncAzureOpenAI client. Default is False.
        azure_deployment: Azure OpenAI deployment name. Only used when use_azure=True.
            If not specified, falls back to AZURE_OPENAI_DEPLOYMENT environment variable.
        api_version: Azure OpenAI API version (e.g., "2024-02-15-preview"). Only used
            when use_azure=True. If not specified, falls back to AZURE_OPENAI_API_VERSION
            environment variable.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.
            Special kwargs:
            - response_format: Structured output control forwarded to the OpenAI
                chat completions API. This adapter accepts dict payloads such
                as ``{"type": "json_object"}`` and dict-form ``json_schema``,
                but rejects typed/Pydantic response_format values.
            - openai_client_configs: Dict of configuration options for the AsyncOpenAI client.
                These will be passed to the client constructor but will be overridden by
                explicit parameters (api_key, base_url). Supports proxy configuration,
                custom headers, retry policies, etc.

    Returns:
        The completed text (with integrated COT content if available) or an async iterator
        of text chunks if streaming. COT content is wrapped in <think>...</think> tags.

    Raises:
        InvalidResponseError: If the response from OpenAI is invalid or empty.
        APIConnectionError: If there is a connection error with the OpenAI API.
        RateLimitError: If the OpenAI API rate limit is exceeded.
        APITimeoutError: If the OpenAI API request times out.
    """
    if history_messages is None:
        history_messages = []

    # Set openai logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    # Remove special kwargs that shouldn't be passed to OpenAI
    kwargs.pop("hashing_kv", None)
    request_kind = kwargs.pop("_lightrag_request_kind", None)
    kwargs.pop("_lightrag_extraction_request", None)

    # Extract client configuration options
    client_configs = kwargs.pop("openai_client_configs", {})

    # Deprecation shims: map legacy boolean flags to response_format only when
    # an explicit response_format was not supplied by the caller. Prefer passing
    # response_format directly.
    entity_extraction = kwargs.pop("entity_extraction", False)
    if entity_extraction and kwargs.get("response_format") is None:
        warnings.warn(
            "openai_complete_if_cache(entity_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs["response_format"] = {"type": "json_object"}
    if keyword_extraction and kwargs.get("response_format") is None:
        warnings.warn(
            "openai_complete_if_cache(keyword_extraction=True) is deprecated; "
            "pass response_format={'type': 'json_object'} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs["response_format"] = {"type": "json_object"}
    _validate_openai_response_format(kwargs.get("response_format"))
    if kwargs.get("response_format") is not None:
        enable_cot = False

    # Create the OpenAI client (supports both OpenAI and Azure)
    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        use_azure=use_azure,
        azure_deployment=azure_deployment,
        api_version=api_version,
        timeout=timeout,
        client_configs=client_configs,
    )

    # Prepare messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    if image_inputs:
        from lightrag.llm._vision_utils import normalize_image_inputs

        normalized_images = normalize_image_inputs(image_inputs)
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in normalized_images:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.mime_type};base64,{img.base64_str}"
                    },
                }
            )
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": prompt})

    logger.debug("===== Entering func of LLM =====")
    logger.debug(f"Model: {model}   Base URL: {base_url}")
    logger.debug(f"Client Configs: {client_configs}")
    logger.debug(f"Additional kwargs: {kwargs}")
    logger.debug(f"Num of history messages: {len(history_messages)}")
    verbose_debug(f"System prompt: {system_prompt}")
    verbose_debug(f"Query: {prompt}")
    logger.debug("===== Sending Query to LLM =====")

    messages = kwargs.pop("messages", messages)
    request_kind = _infer_llm_request_kind(
        request_kind,
        base_url,
        model,
        keyword_extraction=keyword_extraction,
    )
    request_started = time.perf_counter()
    input_chars = _message_char_count(messages)

    # Add explicit parameters back to kwargs so they're passed to OpenAI API
    if stream is not None:
        kwargs["stream"] = stream
    if timeout is not None:
        kwargs["timeout"] = timeout

    # Determine the correct model identifier to use
    # For Azure OpenAI, we must use the deployment name instead of the model name
    api_model = azure_deployment if use_azure and azure_deployment else model

    # ------------------------------------------------------------------
    # ChatML stop handling — inject stop sequences for Hermes / ChatML
    # models so the server halts generation at message boundaries.
    # ------------------------------------------------------------------
    _chatml_active = _is_chatml_model(model) and not use_azure
    if _chatml_active:
        existing_stop = kwargs.get("stop")
        if existing_stop is None:
            kwargs["stop"] = _CHATML_STOP_SEQUENCES
        elif isinstance(existing_stop, list):
            # Merge without duplicates, preserving caller-provided order
            seen = set(existing_stop)
            merged = list(existing_stop)
            for s in _CHATML_STOP_SEQUENCES:
                if s not in seen:
                    merged.append(s)
                    seen.add(s)
            kwargs["stop"] = merged

    try:
        # Don't use async with context manager, use client directly
        if "response_format" in kwargs:
            if _should_use_openai_parse(base_url=base_url, use_azure=use_azure):
                # beta.chat.completions.parse() provides structured output and is
                # inherently non-streaming; passing `stream=True` would raise a
                # TypeError at runtime.
                parse_kwargs = {k: v for k, v in kwargs.items() if k != "stream"}
                response = await openai_async_client.chat.completions.parse(
                    model=api_model, messages=messages, **parse_kwargs
                )
            else:
                create_kwargs = dict(kwargs)
                create_kwargs.pop("stream", None)
                create_kwargs["response_format"] = _response_format_to_schema(
                    create_kwargs["response_format"],
                    json_object_only=_is_managed_swift_lm_base_url(base_url),
                )
                # Length-truncation is detected via finish_reason below and the
                # raw content is returned unchanged so upstream tolerant JSON parsing
                # can still salvage it.
                response = await openai_async_client.chat.completions.create(
                    model=api_model, messages=messages, **create_kwargs
                )
        else:
            # Single dispatch for non-structured requests
            response = await openai_async_client.chat.completions.create(
                model=api_model, messages=messages, **kwargs
            )
    except APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error: {e}")
        try:
            await openai_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close OpenAI client: {close_error}")
        raise
    except APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error: {e}")
        try:
            await openai_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close OpenAI client: {close_error}")
        raise
    except RateLimitError as e:
        logger.error(f"OpenAI API Rate Limit Error: {e}")
        try:
            await openai_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close OpenAI client: {close_error}")
        raise
    except BadRequestError as e:
        # A "could not parse JSON body" 400 is transient (corrupted/truncated
        # request body in transit) and succeeds on retry; re-raise it as a
        # retryable type. Genuine 400s (bad params, content policy) fail fast.
        # Either way we must close the client before re-raising, matching the
        # other except branches above — otherwise non-transient 400s would
        # leak httpx connections in validation-heavy/misconfigured runs.
        try:
            await openai_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close OpenAI client: {close_error}")
        # Heuristic: match on the provider's error wording. It can drift across
        # providers/proxies or localization, and a genuinely malformed request
        # body (e.g. invalid user-supplied JSON) could also surface this text —
        # in that case we simply retry 3x and still fail fast. We accept that
        # "retry too much" trade-off to recover the common transient case.
        if "could not parse" in str(e).lower():
            logger.warning(f"Transient JSON-parse 400 from OpenAI, will retry: {e}")
            raise TransientBadRequestError(str(e)) from e
        raise
    except Exception as e:
        body = getattr(e, "body", None)
        request_id = getattr(e, "request_id", None)
        req = getattr(e, "request", None)
        extra_parts = []
        if body:
            extra_parts.append(f"Response body: {body}")
        if request_id:
            extra_parts.append(f"Request ID: {request_id}")
        if req is not None:
            extra_parts.append(f"Request URL: {req.url}")
        extra = ("\n" + "\n".join(extra_parts)) if extra_parts else ""
        logger.error(
            f"OpenAI API Call Failed,\nModel: {model},\nParams: {kwargs}, Got: {e}{extra}"
        )
        try:
            await openai_async_client.close()
        except Exception as close_error:
            logger.warning(f"Failed to close OpenAI client: {close_error}")
        raise

    if hasattr(response, "__aiter__"):

        async def inner():
            # Track if we've started iterating
            iteration_started = False
            final_chunk_usage = None
            first_token_at = None
            output_parts: list[str] = []

            # COT (Chain of Thought) state tracking
            cot_active = False
            cot_started = False
            initial_content_seen = False

            # ChatML streaming buffer — accumulates content tokens so we
            # can detect and truncate at leaked control-token boundaries
            # without exposing partial markers to the caller.
            _chatml_accumulated = ""

            def _chatml_yield(content: str) -> str | None:
                """Check content against ChatML markers, yield clean prefix
                or None if the caller should stop iteration."""
                nonlocal _chatml_accumulated
                if not _chatml_active:
                    return content
                _chatml_accumulated += content
                cleaned = clean_chatml_markers(_chatml_accumulated)
                if len(cleaned) < len(_chatml_accumulated):
                    # Marker found somewhere in the accumulated stream.
                    # Emit only the clean prefix portion of this chunk.
                    prefix_len = len(cleaned) - (
                        len(_chatml_accumulated) - len(content)
                    )
                    if prefix_len > 0:
                        return cleaned[-prefix_len:] if prefix_len < len(cleaned) else cleaned
                    return None  # caller should stop
                return content

            try:
                iteration_started = True
                async for chunk in response:
                    # Check if this chunk has usage information (final chunk)
                    if hasattr(chunk, "usage") and chunk.usage:
                        final_chunk_usage = chunk.usage
                        logger.debug(
                            f"Received usage info in streaming chunk: {chunk.usage}"
                        )

                    # Check if choices exists and is not empty
                    if not hasattr(chunk, "choices") or not chunk.choices:
                        # Azure OpenAI sends content filter results in first chunk without choices
                        logger.debug(
                            f"Received chunk without choices (likely Azure content filter): {chunk}"
                        )
                        continue

                    # Check if delta exists
                    if not hasattr(chunk.choices[0], "delta"):
                        # This might be the final chunk, continue to check for usage
                        continue

                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    reasoning_content = getattr(delta, "reasoning_content", "")

                    # Handle COT logic for streaming (only if enabled)
                    if enable_cot:
                        if content:
                            # Regular content is present
                            if not initial_content_seen:
                                initial_content_seen = True
                                # If both content and reasoning_content are present initially, don't start COT
                                if reasoning_content:
                                    cot_active = False
                                    cot_started = False

                            # If COT was active, end it
                            if cot_active:
                                yield "</think>"
                                cot_active = False

                            # Process regular content
                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            # ChatML safety net on content
                            y = _chatml_yield(content)
                            if y is None:
                                return
                            content = y
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            output_parts.append(content)
                            yield content

                        elif reasoning_content:
                            # Only reasoning content is present
                            if not initial_content_seen and not cot_started:
                                # Start COT if we haven't seen initial content yet
                                if not cot_active:
                                    yield "<think>"
                                    cot_active = True
                                    cot_started = True

                            # Process reasoning content if COT is active
                            if cot_active:
                                if r"\u" in reasoning_content:
                                    reasoning_content = safe_unicode_decode(
                                        reasoning_content.encode("utf-8")
                                    )
                                # ChatML safety net on reasoning content
                                y = _chatml_yield(reasoning_content)
                                if y is None:
                                    return
                                reasoning_content = y
                                if first_token_at is None:
                                    first_token_at = time.perf_counter()
                                output_parts.append(reasoning_content)
                                yield reasoning_content
                    else:
                        # COT disabled, only process regular content
                        if content:
                            if r"\u" in content:
                                content = safe_unicode_decode(content.encode("utf-8"))
                            # ChatML safety net on content
                            y = _chatml_yield(content)
                            if y is None:
                                return
                            content = y
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            output_parts.append(content)
                            yield content

                    # If neither content nor reasoning_content, continue to next chunk
                    if content is None and reasoning_content is None:
                        continue

                # Log warning if ChatML markers were detected mid-stream
                if _chatml_active and contains_chatml_markers(_chatml_accumulated):
                    logger.warning(
                        "ChatML control tokens intercepted in streaming response; "
                        "stop sequences should have prevented this. "
                        "Stream was truncated."
                    )

                # Ensure COT is properly closed if still active after stream ends
                if enable_cot and cot_active:
                    yield "</think>"
                    cot_active = False

                # After streaming is complete, track token usage
                output_text = "".join(output_parts)
                token_counts, usage_source = _resolve_llm_token_counts(
                    model,
                    messages,
                    output_text,
                    final_chunk_usage,
                )
                if token_tracker:
                    token_tracker.add_usage(token_counts)
                    logger.debug(
                        "Streaming token usage (%s): %s",
                        usage_source,
                        token_counts,
                    )

                _log_llm_request_metrics(
                    request_kind=request_kind,
                    model=model,
                    base_url=base_url,
                    stream=True,
                    token_counts=token_counts,
                    usage_source=usage_source,
                    total_duration_s=time.perf_counter() - request_started,
                    ttft_s=(first_token_at - request_started)
                    if first_token_at is not None
                    else None,
                    input_chars=input_chars,
                    output_chars=len(output_text),
                )
            except Exception as e:
                # Ensure COT is properly closed before handling exception
                if enable_cot and cot_active:
                    try:
                        yield "</think>"
                        cot_active = False
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close COT tag during exception handling: {close_error}"
                        )

                logger.error(f"Error in stream response: {str(e)}")
                # Try to clean up resources if possible
                if (
                    iteration_started
                    and hasattr(response, "aclose")
                    and callable(getattr(response, "aclose", None))
                ):
                    try:
                        await response.aclose()
                        logger.debug("Successfully closed stream response after error")
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close stream response: {close_error}"
                        )
                # Ensure client is closed in case of exception
                try:
                    await openai_async_client.close()
                except Exception as client_close_error:
                    logger.warning(
                        f"Failed to close OpenAI client after stream error: {client_close_error}"
                    )
                raise
            finally:
                # Final safety check for unclosed COT tags
                if enable_cot and cot_active:
                    try:
                        yield "</think>"
                        cot_active = False
                    except Exception as final_close_error:
                        logger.warning(
                            f"Failed to close COT tag in finally block: {final_close_error}"
                        )

                # Ensure resources are released even if no exception occurs
                # Note: Some wrapped clients (e.g., Langfuse) may not implement aclose() properly
                if iteration_started and hasattr(response, "aclose"):
                    aclose_method = getattr(response, "aclose", None)
                    if callable(aclose_method):
                        try:
                            await response.aclose()
                            logger.debug("Successfully closed stream response")
                        except (AttributeError, TypeError) as close_error:
                            # Some wrapper objects may report hasattr(aclose) but fail when called
                            # This is expected behavior for certain client wrappers
                            logger.debug(
                                f"Stream response cleanup not supported by client wrapper: {close_error}"
                            )
                        except Exception as close_error:
                            logger.warning(
                                f"Unexpected error during stream response cleanup: {close_error}"
                            )

                # This prevents resource leaks since the caller doesn't handle closing
                try:
                    await openai_async_client.close()
                    logger.debug(
                        "Successfully closed OpenAI client for streaming response"
                    )
                except Exception as client_close_error:
                    logger.warning(
                        f"Failed to close OpenAI client in streaming finally block: {client_close_error}"
                    )

        return inner()

    else:
        try:
            if (
                not response
                or not response.choices
                or not hasattr(response.choices[0], "message")
            ):
                logger.error("Invalid response from OpenAI API")
                try:
                    await openai_async_client.close()
                except Exception as close_error:
                    logger.warning(f"Failed to close OpenAI client: {close_error}")
                raise InvalidResponseError("Invalid response from OpenAI API")

            message = response.choices[0].message

            # Handle parsed responses (structured output via response_format)
            # When using beta.chat.completions.parse(), the response is in message.parsed
            if hasattr(message, "parsed") and message.parsed is not None:
                # Serialize the parsed structured response to JSON
                final_content = message.parsed.model_dump_json()
                logger.debug("Using parsed structured response from API")
            else:
                tool_calls = getattr(message, "tool_calls", None) or []
                if tool_calls:
                    tool_call = tool_calls[0]
                    function_payload = getattr(tool_call, "function", None)
                    arguments = getattr(function_payload, "arguments", None)
                    if arguments:
                        final_content = arguments
                    else:
                        logger.error("Received tool call response without arguments")
                        try:
                            await openai_async_client.close()
                        except Exception as close_error:
                            logger.warning(f"Failed to close OpenAI client: {close_error}")
                        raise InvalidResponseError(
                            "Received tool call response without arguments"
                        )
                else:
                # Handle regular content responses
                    content = getattr(message, "content", None)
                    reasoning_content = getattr(message, "reasoning_content", "")

                    # Handle COT logic for non-streaming responses (only if enabled)
                    final_content = ""

                    if enable_cot:
                        # Check if we should include reasoning content
                        should_include_reasoning = False
                        if reasoning_content and reasoning_content.strip():
                            if not content or content.strip() == "":
                                # Case 1: Only reasoning content, should include COT
                                should_include_reasoning = True
                                final_content = (
                                    content or ""
                                )  # Use empty string if content is None
                            else:
                                # Case 3: Both content and reasoning_content present, ignore reasoning
                                should_include_reasoning = False
                                final_content = content
                        else:
                            # No reasoning content, use regular content
                            final_content = content or ""

                        # Apply COT wrapping if needed
                        if should_include_reasoning:
                            if r"\u" in reasoning_content:
                                reasoning_content = safe_unicode_decode(
                                    reasoning_content.encode("utf-8")
                                )
                            final_content = (
                                f"<think>{reasoning_content}</think>{final_content}"
                            )
                    else:
                        # COT disabled, only use regular content
                        final_content = content or ""

                    # Validate final content
                    if not final_content or final_content.strip() == "":
                        logger.error("Received empty content from OpenAI API")
                        try:
                            await openai_async_client.close()
                        except Exception as close_error:
                            logger.warning(f"Failed to close OpenAI client: {close_error}")
                        raise InvalidResponseError("Received empty content from OpenAI API")

            # Apply Unicode decoding to final content if needed
            if r"\u" in final_content:
                final_content = safe_unicode_decode(final_content.encode("utf-8"))

            # ChatML safety net — truncate at any leaked control tokens
            if _chatml_active:
                cleaned = clean_chatml_markers(final_content)
                if len(cleaned) < len(final_content):
                    logger.warning(
                        "ChatML control tokens found in non-streaming response; "
                        "stop sequences should have prevented this. "
                        "Truncated %d chars.", len(final_content) - len(cleaned)
                    )
                    final_content = cleaned

            token_counts, usage_source = _resolve_llm_token_counts(
                model,
                messages,
                final_content,
                getattr(response, "usage", None),
            )
            if token_tracker:
                token_tracker.add_usage(token_counts)

            total_duration_s = time.perf_counter() - request_started
            _log_llm_request_metrics(
                request_kind=request_kind,
                model=model,
                base_url=base_url,
                stream=False,
                token_counts=token_counts,
                usage_source=usage_source,
                total_duration_s=total_duration_s,
                ttft_s=_resolve_non_stream_ttft_s(response, total_duration_s),
                input_chars=input_chars,
                output_chars=len(final_content),
            )

            logger.debug(f"Response content len: {len(final_content)}")
            verbose_debug(f"Response: {response}")

            return final_content
        finally:
            # Ensure client is closed in all cases for non-streaming responses
            try:
                await openai_async_client.close()
            except Exception as close_error:
                logger.warning(
                    f"Failed to close OpenAI client in non-streaming finally block: {close_error}"
                )


async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    # Pop entity_extraction from kwargs if also passed there (avoid duplication)
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=8192,
    model_name="text-embedding-3-small",
    supports_asymmetric=True,
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        # Retry transient HTTP 5xx (OpenAI 500 / proxy upstream errors).
        | retry_if_exception_type(InternalServerError)
    ),
)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    client_configs: dict[str, Any] | None = None,
    token_tracker: Any | None = None,
    use_azure: bool = False,
    azure_deployment: str | None = None,
    api_version: str | None = None,
    context: str = "document",
    query_prefix: str | None = None,
    document_prefix: str | None = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI's API with automatic text truncation.

    This function supports both standard OpenAI and Azure OpenAI services. It automatically
    truncates texts that exceed the model's token limit to prevent API errors.

    Args:
        texts: List of texts to embed.
        model: The embedding model to use. For standard OpenAI (e.g., "text-embedding-3-small").
            For Azure, this can be the deployment name.
        base_url: Optional base URL for the API. For standard OpenAI, uses default OpenAI endpoint.
            For Azure, this should be the Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com/).
        api_key: Optional API key. For standard OpenAI, uses OPENAI_API_KEY environment variable if None.
            For Azure, uses AZURE_EMBEDDING_API_KEY environment variable if None.
        embedding_dim: Optional embedding dimension for dynamic dimension reduction.
            **IMPORTANT**: This parameter is automatically injected by the EmbeddingFunc wrapper.
            Do NOT manually pass this parameter when calling the function directly.
            The dimension is controlled by the @wrap_embedding_func_with_attrs decorator.
            Manually passing a different value will trigger a warning and be ignored.
            When provided (by EmbeddingFunc), it will be passed to the OpenAI API for dimension reduction.
        max_token_size: Maximum tokens per text. Texts exceeding this limit will be truncated.
            **IMPORTANT**: This parameter is automatically injected by the EmbeddingFunc wrapper
            when the underlying function signature supports it (via inspect.signature check).
            The value is controlled by the @wrap_embedding_func_with_attrs decorator.
            Set max_token_size=0 to disable truncation.
        client_configs: Additional configuration options for the AsyncOpenAI/AsyncAzureOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url). Supports proxy configuration,
            custom headers, retry policies, etc.
        token_tracker: Optional token usage tracker for monitoring API usage.
        use_azure: Whether to use Azure OpenAI service instead of standard OpenAI.
            When True, creates an AsyncAzureOpenAI client. Default is False.
        azure_deployment: Azure OpenAI deployment name. Only used when use_azure=True.
            If not specified, falls back to AZURE_EMBEDDING_DEPLOYMENT environment variable.
        api_version: Azure OpenAI API version (e.g., "2024-02-15-preview"). Only used
            when use_azure=True. If not specified, falls back to AZURE_EMBEDDING_API_VERSION
            environment variable.
        context: The embedding context - "query" for search queries, "document" for indexed content.
            **IMPORTANT**: This parameter is automatically injected by the EmbeddingFunc wrapper
            when supports_asymmetric=True. Default is "document".
        query_prefix: Optional prefix to prepend to texts when context="query" (e.g., "search_query: ").
        document_prefix: Optional prefix to prepend to texts when context="document" (e.g., "search_document: ").

    Returns:
        A numpy array of embeddings, one per input text.

    Raises:
        APIConnectionError: If there is a connection error with the OpenAI API.
        RateLimitError: If the OpenAI API rate limit is exceeded.
        APITimeoutError: If the OpenAI API request times out.
    """
    # Apply context-based prefixes if provided
    if context == "query" and query_prefix:
        texts = [query_prefix + text for text in texts]
    elif context == "document" and document_prefix:
        texts = [document_prefix + text for text in texts]
    # Apply text truncation if max_token_size is provided
    if max_token_size is not None and max_token_size > 0:
        encoding = _get_tiktoken_encoding_for_model(model)
        truncated_texts = []
        truncation_count = 0

        for text in texts:
            if not text:
                truncated_texts.append(text)
                continue

            tokens = encoding.encode(text)
            if len(tokens) > max_token_size:
                truncated_tokens = tokens[:max_token_size]
                truncated_texts.append(encoding.decode(truncated_tokens))
                truncation_count += 1
                logger.debug(
                    f"Text truncated from {len(tokens)} to {max_token_size} tokens"
                )
            else:
                truncated_texts.append(text)

        if truncation_count > 0:
            logger.info(
                f"Truncated {truncation_count}/{len(texts)} texts to fit token limit ({max_token_size})"
            )

        texts = truncated_texts

    # Create the OpenAI client (supports both OpenAI and Azure)
    openai_async_client = create_openai_async_client(
        api_key=api_key,
        base_url=base_url,
        use_azure=use_azure,
        azure_deployment=azure_deployment,
        api_version=api_version,
        client_configs=client_configs,
    )

    async with openai_async_client:
        # Determine the correct model identifier to use
        # For Azure OpenAI, we must use the deployment name instead of the model name
        api_model = azure_deployment if use_azure and azure_deployment else model

        # Prepare API call parameters
        api_params = {
            "model": api_model,
            "input": texts,
        }

        # Add encoding_format parameter (some providers like Yandex don't support base64)
        # OpenAI client defaults to base64, so we must explicitly set it to "float" if disabled
        api_params["encoding_format"] = "base64" if EMBEDDING_USE_BASE64 else "float"

        # Add dimensions parameter only if embedding_dim is provided
        if embedding_dim is not None:
            api_params["dimensions"] = embedding_dim

        # Make API call
        response = await openai_async_client.embeddings.create(**api_params)

        if token_tracker and hasattr(response, "usage"):
            token_counts = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            token_tracker.add_usage(token_counts)

        return np.array(
            [
                np.array(dp.embedding, dtype=np.float32)
                if isinstance(dp.embedding, list)
                else np.frombuffer(base64.b64decode(dp.embedding), dtype=np.float32)
                for dp in response.data
            ]
        )


# Azure OpenAI wrapper functions for backward compatibility
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    timeout: int | None = None,
    api_version: str | None = None,
    keyword_extraction: bool = False,
    **kwargs,
):
    """Azure OpenAI completion wrapper function.

    This function provides backward compatibility by wrapping the unified
    openai_complete_if_cache implementation with Azure-specific parameter handling.

    All parameters from the underlying openai_complete_if_cache are exposed to ensure
    full feature parity and API consistency.
    """
    # Handle Azure-specific environment variables and parameters
    deployment = settings.azure_openai_deployment or model or settings.llm_model_configured
    base_url = (
        base_url
        or settings.azure_openai_endpoint
        or settings.llm_binding_host
    )
    api_key = (
        api_key
        or settings.azure_openai_api_key
        or settings.llm_binding_api_key
    )
    api_version = (
        api_version
        or settings.azure_openai_api_version
        or settings.openai_api_version
        or "2024-08-01-preview"
    )

    # Call the unified implementation with Azure-specific parameters
    return await openai_complete_if_cache(
        model=deployment,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        base_url=base_url,
        api_key=api_key,
        token_tracker=token_tracker,
        stream=stream,
        timeout=timeout,
        use_azure=True,
        azure_deployment=deployment,
        api_version=api_version,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


async def azure_openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> str:
    """Azure OpenAI complete wrapper function.

    Provides backward compatibility for azure_openai_complete calls.
    """
    if history_messages is None:
        history_messages = []
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)
    result = await azure_openai_complete_if_cache(
        settings.llm_model_configured or "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        entity_extraction=entity_extraction,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(
    embedding_dim=1536,
    max_token_size=8192,
    model_name="my-text-embedding-3-large-deployment",
    supports_asymmetric=True,
)
async def azure_openai_embed(
    texts: list[str],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    token_tracker: Any | None = None,
    client_configs: dict[str, Any] | None = None,
    api_version: str | None = None,
    context: str = "document",
    query_prefix: str | None = None,
    document_prefix: str | None = None,
) -> np.ndarray:
    """Azure OpenAI embedding wrapper function.

    This function provides backward compatibility by wrapping the unified
    openai_embed implementation with Azure-specific parameter handling.

    All parameters from the underlying openai_embed are exposed to ensure
    full feature parity and API consistency.

    IMPORTANT - Decorator Usage:

    1. This function is decorated with @wrap_embedding_func_with_attrs to provide
       the EmbeddingFunc interface for users who need to access embedding_dim
       and other attributes.

    2. This function does NOT use @retry decorator to avoid double-wrapping,
       since the underlying openai_embed.func already has retry logic.

    3. This function calls openai_embed.func (the unwrapped function) instead of
       openai_embed (the EmbeddingFunc instance) to avoid double decoration issues:

       ✅ Correct: await openai_embed.func(...)  # Calls unwrapped function with retry
       ❌ Wrong:   await openai_embed(...)       # Would cause double EmbeddingFunc wrapping

    Double decoration causes:
    - Double injection of embedding_dim parameter
    - Incorrect parameter passing to the underlying implementation
    - Runtime errors due to parameter conflicts

    The call chain with correct implementation:
    azure_openai_embed(texts)
    → EmbeddingFunc.__call__(texts)              # azure's decorator
      → azure_openai_embed_impl(texts, embedding_dim=1536)
        → openai_embed.func(texts, ...)
          → @retry_wrapper(texts, ...)           # openai's retry (only one layer)
            → openai_embed_impl(texts, ...)
              → actual embedding computation
    """
    # Handle Azure-specific environment variables and parameters
    deployment = (
        settings.azure_embedding_deployment
        or model
        or settings.embedding_model
        or "text-embedding-3-small"
    )
    base_url = (
        base_url
        or settings.azure_embedding_endpoint
        or settings.embedding_binding_host
    )
    api_key = (
        api_key
        or settings.azure_embedding_api_key
        or settings.embedding_binding_api_key
    )
    api_version = (
        api_version
        or settings.azure_embedding_api_version
        or settings.azure_openai_api_version
        or settings.openai_api_version
        or "2024-08-01-preview"
    )

    # CRITICAL: Call openai_embed.func (unwrapped) to avoid double decoration
    # openai_embed is an EmbeddingFunc instance, .func accesses the underlying function
    return await openai_embed.func(
        texts=texts,
        model=deployment,
        base_url=base_url,
        api_key=api_key,
        embedding_dim=embedding_dim,
        token_tracker=token_tracker,
        client_configs=client_configs,
        use_azure=True,
        azure_deployment=deployment,
        api_version=api_version,
        context=context,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )
