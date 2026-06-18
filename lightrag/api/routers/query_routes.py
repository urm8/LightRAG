"""
This module contains all query-related routes for the LightRAG API.
"""

import asyncio
import copy
import json
import re
import time
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException
from lightrag.config import settings
from lightrag.base import QueryParam
from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.prompt import PROMPTS
from lightrag.utils import logger
from pydantic import BaseModel, Field, field_validator

router = APIRouter(tags=["query"])

_REFERENCES_SECTION_RE = re.compile(
    r"(?:\n|^)\s*(?:#{1,6}\s*)?References\b[\s\S]*\Z",
    re.IGNORECASE | re.MULTILINE,
)
def _rerank_enabled_globally() -> bool:
    return settings.lightrag_rerank_enabled


def _get_query_prompt_template(
    prompt_key: str | None, default_key: str = "rag_response"
) -> str:
    prompt_key = str(prompt_key or "").strip()
    if prompt_key:
        if prompt_key in PROMPTS:
            return PROMPTS[prompt_key]
        logger.warning("Ignoring unknown query prompt key %r", prompt_key)
    return PROMPTS[default_key]


def _get_query_prompt_for_request(request: "QueryRequest") -> str:
    """Return the normal-path (deep) query prompt."""
    if request.query_model == "granite":
        return _get_query_prompt_template(
            settings.webui_query_enrichment_prompt,
            "rag_response",
        )
    return PROMPTS["rag_response"]


def _get_fast_path_prompt() -> str:
    """Return the fast-path (shallow, 4K-window) query prompt."""
    return _get_query_prompt_template(
        settings.lightrag_query_fast_prompt, "apfel_rag_response"
    )


def _query_model_name() -> str:
    return str(settings.llm_model_configured or "").strip().lower()


def _query_binding_host() -> str:
    return str(settings.llm_binding_host or "").strip().lower()


def _is_apfel_query_model() -> bool:
    model = _query_model_name()
    host = _query_binding_host()
    return (
        "apple-foundationmodel" in model
        or "apfel" in model
        or "127.0.0.1:11435" in host
        or "localhost:11435" in host
    )


def _fast_query_context_length() -> int | None:
    if _is_apfel_query_model():
        explicit = settings.lightrag_query_fast_context_length
        if explicit:
            return explicit
        return settings.apfel_context_length
    return (
        settings.mlx_openai_server_retrieval_context_length_configured
        or settings.llm_context_length
    )


def _truncate_conversation_history_for_fast_query(param: QueryParam) -> None:
    if not getattr(param, "conversation_history", None):
        return

    history_turns = settings.lightrag_query_fast_history_turns
    if history_turns is None:
        history_turns = 0 if _is_apfel_query_model() else param.history_turns

    if history_turns <= 0:
        removed_count = len(param.conversation_history)
        param.conversation_history = []
        if removed_count:
            logger.info(
                "Dropped %s conversation history messages for fast query context",
                removed_count,
            )
        return

    max_messages = history_turns * 2
    if len(param.conversation_history) > max_messages:
        param.conversation_history = param.conversation_history[-max_messages:]

    max_chars = settings.lightrag_query_fast_history_max_chars
    used_chars = 0
    kept_messages: list[dict[str, Any]] = []
    for message in reversed(param.conversation_history):
        content = str(message.get("content", ""))
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        clipped = content[-remaining:]
        used_chars += len(clipped)
        next_message = dict(message)
        next_message["content"] = clipped
        kept_messages.append(next_message)
    param.conversation_history = list(reversed(kept_messages))


def _apply_fast_query_defaults(param: QueryParam) -> None:
    if not _is_apfel_query_model():
        return

    param.top_k = min(param.top_k, settings.lightrag_query_fast_top_k)
    if param.chunk_top_k:
        param.chunk_top_k = min(
            param.chunk_top_k, settings.lightrag_query_fast_chunk_top_k
        )
    else:
        param.chunk_top_k = settings.lightrag_query_fast_chunk_top_k

    fast_total = settings.lightrag_query_fast_max_total_tokens
    fast_entities = settings.lightrag_query_fast_max_entity_tokens
    fast_relations = settings.lightrag_query_fast_max_relation_tokens

    param.max_total_tokens = min(param.max_total_tokens, fast_total)
    param.max_entity_tokens = min(param.max_entity_tokens, fast_entities)
    param.max_relation_tokens = min(param.max_relation_tokens, fast_relations)
    param.response_type = settings.lightrag_query_fast_response_type

    if not settings.lightrag_query_fast_enable_rerank:
        param.enable_rerank = False

    # Apfel has a 4k window and weak tool-following for this use case. Keep the
    # fast answer path as plain RAG; slower enrichment can use richer prompts.
    param.enable_agent_tools = False
    param.max_completion_tokens = settings.lightrag_query_fast_max_completion_tokens
    _truncate_conversation_history_for_fast_query(param)


def _make_fast_path_param(param: QueryParam) -> QueryParam:
    """Create a fast-path param by applying 4K window settings unconditionally.

    The normal path uses the full model context window (e.g. 16K for Hermes-4).
    This clones the normal param and clamps budgets to the fast / apfel-compatible
    4K profile so both query paths can be launched in parallel.
    """
    fast_param = copy.deepcopy(param)

    fast_param.top_k = min(
        fast_param.top_k, settings.lightrag_query_fast_top_k
    )
    if fast_param.chunk_top_k:
        fast_param.chunk_top_k = min(
            fast_param.chunk_top_k, settings.lightrag_query_fast_chunk_top_k
        )
    else:
        fast_param.chunk_top_k = settings.lightrag_query_fast_chunk_top_k

    fast_param.max_total_tokens = min(
        fast_param.max_total_tokens,
        settings.lightrag_query_fast_max_total_tokens,
    )
    fast_param.max_entity_tokens = min(
        fast_param.max_entity_tokens,
        settings.lightrag_query_fast_max_entity_tokens,
    )
    fast_param.max_relation_tokens = min(
        fast_param.max_relation_tokens,
        settings.lightrag_query_fast_max_relation_tokens,
    )
    fast_param.max_completion_tokens = (
        settings.lightrag_query_fast_max_completion_tokens
    )
    fast_param.response_type = settings.lightrag_query_fast_response_type

    if not settings.lightrag_query_fast_enable_rerank:
        fast_param.enable_rerank = False

    fast_param.enable_agent_tools = False
    _truncate_conversation_history_for_fast_query(fast_param)

    return fast_param


def _build_openai_query_model_func(
    *,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
):
    async def _model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        from lightrag.llm.openai import openai_complete_if_cache

        kwargs.pop("_priority", None)
        kwargs["timeout"] = timeout
        if history_messages is None:
            history_messages = []
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=base_url,
            api_key=api_key,
            keyword_extraction=keyword_extraction,
            **kwargs,
        )

    return _model_func


def _build_webui_enrichment_model_func():
    model = str(settings.webui_query_enrichment_model or "").strip()
    base_url = str(settings.webui_query_enrichment_binding_host or "").strip()
    api_key = settings.webui_query_enrichment_binding_api_key
    timeout = settings.webui_query_enrichment_timeout
    if not model or not base_url:
        return None
    return _build_openai_query_model_func(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


def _build_granite_query_model_func():
    model = (
        str(settings.lightrag_granite_query_model or "").strip()
        or str(settings.webui_query_enrichment_model or "").strip()
    )
    base_url = (
        str(settings.lightrag_granite_query_binding_host or "").strip()
        or str(settings.webui_query_enrichment_binding_host or "").strip()
    )
    api_key = (
        settings.lightrag_granite_query_binding_api_key
        or settings.webui_query_enrichment_binding_api_key
    )
    timeout = (
        settings.lightrag_granite_query_timeout
        or settings.webui_query_enrichment_timeout
    )
    if not model or not base_url:
        return None
    return _build_openai_query_model_func(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )


def _apply_granite_query_defaults(param: QueryParam) -> QueryParam:
    model_func = _build_granite_query_model_func()
    if model_func is None:
        logger.warning(
            "Granite query profile requested but LIGHTRAG_GRANITE_QUERY_MODEL/HOST "
            "or WEBUI_QUERY_ENRICHMENT_MODEL/HOST is not configured"
        )
        return _clamp_query_param_for_local_context(param)

    param.model_func = model_func
    param.top_k = min(param.top_k, settings.webui_query_enrichment_top_k)
    if param.chunk_top_k:
        param.chunk_top_k = min(
            param.chunk_top_k, settings.webui_query_enrichment_chunk_top_k
        )
    else:
        param.chunk_top_k = settings.webui_query_enrichment_chunk_top_k
    param.max_total_tokens = min(
        param.max_total_tokens,
        settings.webui_query_enrichment_max_total_tokens,
    )
    param.max_entity_tokens = min(
        param.max_entity_tokens,
        settings.webui_query_enrichment_max_entity_tokens,
    )
    param.max_relation_tokens = min(
        param.max_relation_tokens,
        settings.webui_query_enrichment_max_relation_tokens,
    )
    param.max_completion_tokens = settings.webui_query_enrichment_max_completion_tokens
    param.enable_agent_tools = settings.webui_query_enrichment_agent_tools
    param.enable_rerank = _rerank_enabled_globally() and settings.webui_query_enrichment_enable_rerank
    return param


def _clamp_query_param_for_local_context(param: QueryParam) -> QueryParam:
    """Keep query prompts inside the managed local retrieval model window.

    Browser settings are persisted client-side and older sessions can still
    submit budgets such as 30k tokens. The local MLX retrieval model may be
    configured with a much smaller context, so clamp API query budgets here
    before retrieval context construction.
    """
    _apply_fast_query_defaults(param)

    context_length = _fast_query_context_length()
    if not context_length:
        return param

    completion_tokens = getattr(param, "max_completion_tokens", None)
    if completion_tokens is None and _is_apfel_query_model():
        completion_tokens = settings.lightrag_query_fast_max_completion_tokens
    completion_tokens = (
        completion_tokens
        or settings.openai_llm_max_completion_tokens
        or settings.mlx_openai_server_retrieval_max_tokens_configured
        or 1024
    )
    # Keep the prompt comfortably below the model window and leave room for
    # completion tokens even when users override server defaults from the UI.
    prompt_cap = max(1024, int(context_length * 0.8))
    if context_length > completion_tokens + 512:
        prompt_cap = min(prompt_cap, context_length - completion_tokens - 512)

    original_total = param.max_total_tokens
    if param.max_total_tokens > prompt_cap:
        param.max_total_tokens = prompt_cap

    kg_cap = max(512, int(param.max_total_tokens * 0.6))
    entity_tokens = max(1, param.max_entity_tokens)
    relation_tokens = max(1, param.max_relation_tokens)
    if entity_tokens + relation_tokens > kg_cap:
        total = entity_tokens + relation_tokens
        param.max_entity_tokens = max(256, int(kg_cap * entity_tokens / total))
        param.max_relation_tokens = max(
            256, kg_cap - param.max_entity_tokens
        )

    if original_total != param.max_total_tokens:
        logger.info(
            "Clamped query token budget for local retrieval model: "
            "requested_total=%s effective_total=%s context_length=%s completion_tokens=%s "
            "top_k=%s chunk_top_k=%s entity_tokens=%s relation_tokens=%s",
            original_total,
            param.max_total_tokens,
            context_length,
            completion_tokens,
            param.top_k,
            param.chunk_top_k,
            param.max_entity_tokens,
            param.max_relation_tokens,
        )
    return param


def _format_reference_source(source: str) -> str:
    source = (source or "").strip()
    if not source:
        return ""
    if source.startswith(("http://", "https://")):
        return f"[{source}]({source})"
    return source


def _iter_reference_sources(data: dict[str, Any]):
    for ref in data.get("references", []) or []:
        yield ref.get("reference_id"), ref.get("file_path")

    for chunk in data.get("chunks", []) or []:
        yield chunk.get("reference_id"), chunk.get("file_path")

    for key in ("entities", "relationships"):
        for item in data.get(key, []) or []:
            file_path = str(item.get("file_path", "")).strip()
            for source in file_path.split("<SEP>"):
                source = source.strip()
                if source and source != "unknown_source":
                    yield None, source


def _dedupe_reference_pairs(
    pairs, max_references: int = 5
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_ids: set[str] = set()
    next_id = 1

    for reference_id, file_path in pairs:
        source = str(file_path or "").strip()
        if not source or source in seen_sources:
            continue
        seen_sources.add(source)

        rendered_id = str(reference_id or "").strip()
        if not rendered_id:
            while str(next_id) in seen_ids:
                next_id += 1
            rendered_id = str(next_id)
            next_id += 1
        seen_ids.add(rendered_id)

        references.append({"reference_id": rendered_id, "file_path": source})
        if len(references) >= max_references:
            break

    return references


def _references_from_query_data(data: dict[str, Any], max_references: int = 5):
    chunk_pairs = [
        (chunk.get("reference_id"), chunk.get("file_path"))
        for chunk in data.get("chunks", []) or []
    ]
    chunk_references = _dedupe_reference_pairs(chunk_pairs, max_references)
    if chunk_references:
        return chunk_references

    return _dedupe_reference_pairs(_iter_reference_sources(data), max_references)


def _build_canonical_references_section(
    references: list[dict[str, Any]], max_references: int = 5
) -> str:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for ref in references:
        reference_id = str(ref.get("reference_id", "")).strip()
        file_path = str(ref.get("file_path", "")).strip()
        key = (reference_id, file_path)
        if not reference_id or not file_path or key in seen:
            continue
        seen.add(key)
        rendered_source = _format_reference_source(file_path)
        if not rendered_source:
            continue
        lines.append(f"- [{reference_id}] {rendered_source}")
        if len(lines) >= max_references:
            break

    if not lines:
        return ""

    return "### References\n\n" + "\n".join(lines)


def _ensure_response_has_canonical_references(
    response_content: str, references: list[dict[str, Any]]
) -> str:
    canonical_section = _build_canonical_references_section(references)
    if not canonical_section:
        return response_content

    base_content = _REFERENCES_SECTION_RE.sub("", response_content or "").rstrip()
    if not base_content:
        return canonical_section
    return f"{base_content}\n\n{canonical_section}"


def _response_content_with_references(
    llm_response: dict[str, Any],
    references: list[dict[str, Any]],
    *,
    include_references: bool,
) -> str:
    response_content = llm_response.get("content", "")
    if not response_content:
        return "No relevant context found for the query."
    if include_references:
        return _ensure_response_has_canonical_references(response_content, references)
    return response_content


def _make_enrichment_param(request_param: QueryParam) -> QueryParam:
    enrich_param = copy.deepcopy(request_param)
    enrich_param.stream = False
    enrich_param.model_func = _build_webui_enrichment_model_func()
    enrich_param.top_k = settings.webui_query_enrichment_top_k
    enrich_param.chunk_top_k = settings.webui_query_enrichment_chunk_top_k
    enrich_param.max_total_tokens = settings.webui_query_enrichment_max_total_tokens
    enrich_param.max_entity_tokens = settings.webui_query_enrichment_max_entity_tokens
    enrich_param.max_relation_tokens = settings.webui_query_enrichment_max_relation_tokens
    enrich_param.max_completion_tokens = (
        settings.webui_query_enrichment_max_completion_tokens
    )
    enrich_param.enable_agent_tools = settings.webui_query_enrichment_agent_tools
    enrich_param.enable_rerank = (
        _rerank_enabled_globally() and settings.webui_query_enrichment_enable_rerank
    )
    marker = "Granite enrichment pass for WebUI. Improve completeness and source grounding."
    enrich_param.user_prompt = (
        f"{request_param.user_prompt}\n\n{marker}"
        if request_param.user_prompt
        else marker
    )
    return enrich_param


async def _run_enrichment_query(
    rag,
    request: "QueryRequest",
    param: QueryParam,
) -> dict[str, Any]:
    started = time.perf_counter()
    if param.model_func is None:
        raise RuntimeError("WEBUI_QUERY_ENRICHMENT_MODEL/HOST is not configured")
    result = await rag.aquery_llm(
        request.query,
        param=param,
        system_prompt=_get_query_prompt_template(
            settings.webui_query_enrichment_prompt,
            "rag_response",
        ),
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    data = result.get("data", {})
    references = _references_from_query_data(data)
    response = _response_content_with_references(
        result.get("llm_response", {}),
        references,
        include_references=bool(request.include_references),
    )
    return {
        "response": response,
        "elapsed_ms": elapsed_ms,
        "model": settings.webui_query_enrichment_model or "",
        "references": references,
    }


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=3,
        description="The query text",
    )

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode",
    )

    only_need_context: Optional[bool] = Field(
        default=None,
        description="If True, only returns the retrieved context without generating a response.",
    )

    only_need_prompt: Optional[bool] = Field(
        default=None,
        description="If True, only returns the generated prompt without producing a response.",
    )

    response_type: Optional[str] = Field(
        min_length=1,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    chunk_top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of text chunks to retrieve initially from vector search and keep after reranking.",
    )

    max_entity_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for entity context in unified token control system.",
        ge=1,
    )

    max_relation_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for relationship context in unified token control system.",
        ge=1,
    )

    max_total_tokens: Optional[int] = Field(
        default=None,
        description="Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt).",
        ge=1,
    )

    hl_keywords: list[str] = Field(
        default_factory=list,
        description="List of high-level keywords to prioritize in retrieval. Leave empty to use the LLM to generate the keywords.",
    )

    ll_keywords: list[str] = Field(
        default_factory=list,
        description="List of low-level keywords to refine retrieval focus. Leave empty to use the LLM to generate the keywords.",
    )

    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="History messages are only sent to LLM for context, not used for retrieval. Format: [{'role': 'user/assistant', 'content': 'message'}].",
    )

    user_prompt: Optional[str] = Field(
        default=None,
        description="User-provided prompt for the query. If provided, this will be used instead of the default value from prompt template.",
    )

    enable_rerank: Optional[bool] = Field(
        default=None,
        description="Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True.",
    )

    include_references: Optional[bool] = Field(
        default=True,
        description="If True, includes reference list in responses. Affects /query and /query/stream endpoints. /query/data always includes references.",
    )

    include_chunk_content: Optional[bool] = Field(
        default=False,
        description="If True, includes actual chunk text content in references. Only applies when include_references=True. Useful for evaluation and debugging.",
    )

    stream: Optional[bool] = Field(
        default=True,
        description="If True, enables streaming output for real-time responses. Only affects /query/stream endpoint.",
    )

    include_debug: Optional[bool] = Field(
        default=False,
        description="If True, includes structured retrieval debug information in /query/stream responses.",
    )

    include_enrichment: Optional[bool] = Field(
        default=False,
        description="If True, WebUI receives a slower Granite enrichment result after the fast primary answer.",
    )

    use_fast_query: Optional[bool] = Field(
        default=False,
        description="If True, also runs the optional Apfel fast-query side path. Disabled by default.",
    )

    query_model: Optional[Literal["default", "granite"]] = Field(
        default=None,
        description="Internal query model profile override. Used by MCP to route query synthesis through Granite.",
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        return query.strip()

    @field_validator("conversation_history", mode="after")
    @classmethod
    def conversation_history_role_check(
        cls, conversation_history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        if conversation_history is None:
            return None
        for msg in conversation_history:
            if "role" not in msg:
                raise ValueError("Each message must have a 'role' key.")
            if not isinstance(msg["role"], str) or not msg["role"].strip():
                raise ValueError("Each message 'role' must be a non-empty string.")
        return conversation_history

    def to_query_params(self, is_stream: bool) -> "QueryParam":
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        # Exclude API-level parameters that don't belong in QueryParam
        request_data = self.model_dump(
            exclude_none=True,
            exclude={
                "query",
                "include_chunk_content",
                "include_debug",
                "include_enrichment",
                "use_fast_query",
                "query_model",
            },
        )

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        if self.query_model == "granite":
            param = _apply_granite_query_defaults(param)
        else:
            param = _clamp_query_param_for_local_context(param)
        if not _rerank_enabled_globally():
            param.enable_rerank = False
        return param


class ReferenceItem(BaseModel):
    """A single reference item in query responses."""

    reference_id: str = Field(description="Unique reference identifier")
    file_path: str = Field(description="Path to the source file")
    content: Optional[List[str]] = Field(
        default=None,
        description="List of chunk contents from this file (only present when include_chunk_content=True)",
    )


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )
    references: Optional[List[ReferenceItem]] = Field(
        default=None,
        description="Reference list (Disabled when include_references=False, /query/data always includes references.)",
    )
    enrichment_response: Optional[str] = Field(
        default=None,
        description="Slower WebUI enrichment answer generated by the configured enrichment model.",
    )
    enrichment_model: Optional[str] = Field(
        default=None,
        description="Model name used for WebUI enrichment.",
    )
    enrichment_elapsed_ms: Optional[int] = Field(
        default=None,
        description="Elapsed time for WebUI enrichment in milliseconds.",
    )
    enrichment_error: Optional[str] = Field(
        default=None,
        description="Enrichment error if the primary fast answer succeeded but enrichment failed.",
    )

    fast_response: Optional[str] = Field(
        default=None,
        description="Fast-path (4K window) answer generated in parallel with the primary response.",
    )
    fast_elapsed_ms: Optional[int] = Field(
        default=None,
        description="Elapsed time for the fast-path query in milliseconds.",
    )
    fast_error: Optional[str] = Field(
        default=None,
        description="Fast-path error if the normal path succeeded but the fast path failed.",
    )


class QueryDataResponse(BaseModel):
    status: str = Field(description="Query execution status")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(
        description="Query result data containing entities, relationships, chunks, and references"
    )
    metadata: Dict[str, Any] = Field(
        description="Query metadata including mode, keywords, and processing information"
    )


class StreamChunkResponse(BaseModel):
    """Response model for streaming chunks in NDJSON format"""

    references: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Reference list (only in first chunk when include_references=True)",
    )
    response: Optional[str] = Field(
        default=None, description="Response content chunk or complete response"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if processing fails"
    )
    debug: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured retrieval debug payload for frontend progress panels",
    )


def _build_stream_debug_payload(
    request: QueryRequest, result: Dict[str, Any]
) -> Dict[str, Any]:
    data = result.get("data", {}) or {}
    metadata = result.get("metadata", {}) or {}
    processing_info = metadata.get("processing_info", {}) or {}
    keywords = metadata.get("keywords", {}) or {}

    entities = data.get("entities", []) or []
    relationships = data.get("relationships", []) or []
    chunks = data.get("chunks", []) or []
    references = data.get("references", []) or []

    def chunk_preview(chunk: Dict[str, Any]) -> Dict[str, Any]:
        content = (chunk.get("content") or "").strip().replace("\n", " ")
        return {
            "reference_id": chunk.get("reference_id"),
            "file_path": chunk.get("file_path"),
            "chunk_id": chunk.get("chunk_id"),
            "preview": content[:220],
        }

    def entity_preview(entity: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_name": entity.get("entity_name"),
            "entity_type": entity.get("entity_type"),
            "file_path": entity.get("file_path"),
            "reference_id": entity.get("reference_id"),
        }

    def relationship_preview(relationship: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "src_id": relationship.get("src_id"),
            "tgt_id": relationship.get("tgt_id"),
            "weight": relationship.get("weight"),
            "file_path": relationship.get("file_path"),
            "reference_id": relationship.get("reference_id"),
        }

    retrieval_steps = [
        {
            "label": "Mode",
            "detail": f"{metadata.get('query_mode', request.mode)} retrieval pipeline selected",
        },
        {
            "label": "Keywords",
            "detail": (
                f"HL={len(keywords.get('high_level', []) or [])}, "
                f"LL={len(keywords.get('low_level', []) or [])}"
            ),
        },
        {
            "label": "Knowledge Graph",
            "detail": (
                f"entities={len(entities)}, relationships={len(relationships)}"
            ),
        },
        {
            "label": "Vector Search",
            "detail": (
                f"chunks={len(chunks)}, references={len(references)}"
            ),
        },
        {
            "label": "Rerank",
            "detail": "enabled"
            if _rerank_enabled_globally() and request.enable_rerank is not False
            else "disabled",
        },
    ]

    payload = {
        "query": request.query,
        "mode": metadata.get("query_mode", request.mode),
        "keywords": {
            "high_level": keywords.get("high_level", []) or [],
            "low_level": keywords.get("low_level", []) or [],
        },
        "processing_info": processing_info,
        "capabilities": {
            "rerank_enabled": _rerank_enabled_globally()
            and request.enable_rerank is not False,
            "tool_calls_visible": False,
            "cosine_scores_visible": False,
        },
        "notes": [
            "This panel reflects real retrieval data returned by LightRAG before answer generation.",
            "Per-item cosine scores and agent tool-call traces are not exposed by this query endpoint yet.",
        ],
        "retrieval_steps": retrieval_steps,
        "samples": {
            "entities": [entity_preview(entity) for entity in entities[:5]],
            "relationships": [
                relationship_preview(relationship) for relationship in relationships[:5]
            ],
            "chunks": [chunk_preview(chunk) for chunk in chunks[:5]],
            "references": references[:5],
        },
    }
    if metadata.get("apfel_iterative"):
        payload["iterative"] = metadata.get("apfel_iterative")
    return payload


def create_query_routes(rag, api_key: Optional[str] = None, top_k: int = 60):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Successful RAG query response",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "response": {
                                    "type": "string",
                                    "description": "The generated response from the RAG system",
                                },
                                "references": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "reference_id": {"type": "string"},
                                            "file_path": {"type": "string"},
                                            "content": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "description": "List of chunk contents from this file (only included when include_chunk_content=True)",
                                            },
                                        },
                                    },
                                    "description": "Reference list (only included when include_references=True)",
                                },
                            },
                            "required": ["response"],
                        },
                        "examples": {
                            "with_references": {
                                "summary": "Response with references",
                                "description": "Example response when include_references=True",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                                    "references": [
                                        {
                                            "reference_id": "1",
                                            "file_path": "/documents/ai_overview.pdf",
                                        },
                                        {
                                            "reference_id": "2",
                                            "file_path": "/documents/machine_learning.txt",
                                        },
                                    ],
                                },
                            },
                            "with_chunk_content": {
                                "summary": "Response with chunk content",
                                "description": "Example response when include_references=True and include_chunk_content=True. Note: content is an array of chunks from the same file.",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                                    "references": [
                                        {
                                            "reference_id": "1",
                                            "file_path": "/documents/ai_overview.pdf",
                                            "content": [
                                                "Artificial Intelligence (AI) represents a transformative field in computer science focused on creating systems that can perform tasks requiring human-like intelligence. These tasks include learning from experience, understanding natural language, recognizing patterns, and making decisions.",
                                                "AI systems can be categorized into narrow AI, which is designed for specific tasks, and general AI, which aims to match human cognitive abilities across a wide range of domains.",
                                            ],
                                        },
                                        {
                                            "reference_id": "2",
                                            "file_path": "/documents/machine_learning.txt",
                                            "content": [
                                                "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of algorithms that can access data and use it to learn for themselves."
                                            ],
                                        },
                                    ],
                                },
                            },
                            "without_references": {
                                "summary": "Response without references",
                                "description": "Example response when include_references=False",
                                "value": {
                                    "response": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
                                },
                            },
                            "different_modes": {
                                "summary": "Different query modes",
                                "description": "Examples of responses from different query modes",
                                "value": {
                                    "local_mode": "Focuses on specific entities and their relationships",
                                    "global_mode": "Provides broader context from relationship patterns",
                                    "hybrid_mode": "Combines local and global approaches",
                                    "naive_mode": "Simple vector similarity search",
                                    "mix_mode": "Integrates knowledge graph and vector retrieval",
                                },
                            },
                        },
                    }
                },
            },
            400: {
                "description": "Bad Request - Invalid input parameters",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Query text must be at least 3 characters long"
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Query processing failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to process query: LLM service unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_text(request: QueryRequest):
        """
        Comprehensive RAG query endpoint with non-streaming response. Parameter "stream" is ignored.

        This endpoint performs Retrieval-Augmented Generation (RAG) queries using various modes
        to provide intelligent responses based on your knowledge base.

        **Query Modes:**
        - **local**: Focuses on specific entities and their direct relationships
        - **global**: Analyzes broader patterns and relationships across the knowledge graph
        - **hybrid**: Combines local and global approaches for comprehensive results
        - **naive**: Simple vector similarity search without knowledge graph
        - **mix**: Integrates knowledge graph retrieval with vector search (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        conversation_history parameteris sent to LLM only, does not affect retrieval results.

        **Usage Examples:**

        Basic query:
        ```json
        {
            "query": "What is machine learning?",
            "mode": "mix"
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Advanced query with references:
        ```json
        {
            "query": "Explain neural networks",
            "mode": "hybrid",
            "include_references": true,
            "response_type": "Multiple Paragraphs",
            "top_k": 10
        }
        ```

        Conversation with history:
        ```json
        {
            "query": "Can you give me more details?",
            "conversation_history": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence..."}
            ]
        }
        ```

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process (min 3 characters)
                - **mode**: Query strategy - "mix" recommended for best results
                - **include_references**: Whether to include source citations
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **conversation_history**: Previous dialogue context
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            QueryResponse: JSON response containing:
                - **response**: The generated answer to your query
                - **references**: Source citations (if include_references=True)

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short)
                - 500: Internal processing error (e.g., LLM service unavailable)
        """
        try:
            normal_param = request.to_query_params(
                False
            )  # Ensure stream=False for non-streaming endpoint
            # Force stream=False for /query endpoint regardless of include_references setting
            normal_param.stream = False

            fast_task = None
            if request.use_fast_query:
                fast_param = _make_fast_path_param(normal_param)
                fast_param.stream = False
                fast_task = asyncio.create_task(
                    rag.aquery_llm(
                        request.query,
                        param=fast_param,
                        system_prompt=_get_fast_path_prompt(),
                    )
                )

            enrichment_task = None
            if request.include_enrichment and settings.webui_query_enrichment_enabled:
                enrichment_task = asyncio.create_task(
                    _run_enrichment_query(
                        rag, request, _make_enrichment_param(normal_param)
                    )
                )

            normal_result = await rag.aquery_llm(
                request.query,
                param=normal_param,
                system_prompt=_get_query_prompt_for_request(request),
            )

            llm_response = normal_result.get("llm_response", {})
            data = normal_result.get("data", {})
            references = _references_from_query_data(data)

            # Get the non-streaming response content
            response_content = llm_response.get("content", "")
            if not response_content:
                response_content = "No relevant context found for the query."
            elif request.include_references:
                response_content = _ensure_response_has_canonical_references(
                    response_content, references
                )

            # Enrich references with chunk content if requested
            if request.include_references and request.include_chunk_content:
                chunks = data.get("chunks", [])
                # Create a mapping from reference_id to chunk content
                ref_id_to_content = {}
                for chunk in chunks:
                    ref_id = chunk.get("reference_id", "")
                    content = chunk.get("content", "")
                    if ref_id and content:
                        # Collect chunk content; join later to avoid quadratic string concatenation
                        ref_id_to_content.setdefault(ref_id, []).append(content)

                # Add content to references
                enriched_references = []
                for ref in references:
                    ref_copy = ref.copy()
                    ref_id = ref.get("reference_id", "")
                    if ref_id in ref_id_to_content:
                        # Keep content as a list of chunks (one file may have multiple chunks)
                        ref_copy["content"] = ref_id_to_content[ref_id]
                    enriched_references.append(ref_copy)
                references = enriched_references

            # Return response with or without references based on request
            if request.include_references:
                response = QueryResponse(response=response_content, references=references)
            else:
                response = QueryResponse(response=response_content, references=None)

            if fast_task is not None:
                fast_result = await asyncio.gather(fast_task, return_exceptions=True)
                fast_outcome = fast_result[0]
                if not isinstance(fast_outcome, Exception):
                    fast_llm = fast_outcome.get("llm_response", {})
                    fast_content = fast_llm.get("content", "")
                    if fast_content:
                        response.fast_response = fast_content
                else:
                    response.fast_error = str(fast_outcome)

            if enrichment_task is not None:
                try:
                    enrichment = await enrichment_task
                    response.enrichment_response = enrichment.get("response")
                    response.enrichment_elapsed_ms = enrichment.get("elapsed_ms")
                    response.enrichment_model = enrichment.get("model")
                except Exception as exc:
                    logger.warning("WebUI query enrichment failed: %s", exc)
                    response.enrichment_error = str(exc)

            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Flexible RAG query response - format depends on stream parameter",
                "content": {
                    "application/x-ndjson": {
                        "schema": {
                            "type": "string",
                            "format": "ndjson",
                            "description": "Newline-delimited JSON (NDJSON) format used for both streaming and non-streaming responses. For streaming: multiple lines with separate JSON objects. For non-streaming: single line with complete JSON object.",
                            "example": '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"response": " a field of computer science"}\n{"response": " that focuses on creating intelligent machines."}',
                        },
                        "examples": {
                            "streaming_with_references": {
                                "summary": "Streaming mode with references (stream=true)",
                                "description": "Multiple NDJSON lines when stream=True and include_references=True. First line contains references, subsequent lines contain response chunks.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai_overview.pdf"}, {"reference_id": "2", "file_path": "/documents/ml_basics.txt"}]}\n{"response": "Artificial Intelligence (AI) is a branch of computer science"}\n{"response": " that aims to create intelligent machines capable of performing"}\n{"response": " tasks that typically require human intelligence, such as learning,"}\n{"response": " reasoning, and problem-solving."}',
                            },
                            "streaming_with_chunk_content": {
                                "summary": "Streaming mode with chunk content (stream=true, include_chunk_content=true)",
                                "description": "Multiple NDJSON lines when stream=True, include_references=True, and include_chunk_content=True. First line contains references with content arrays (one file may have multiple chunks), subsequent lines contain response chunks.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai_overview.pdf", "content": ["Artificial Intelligence (AI) represents a transformative field...", "AI systems can be categorized into narrow AI and general AI..."]}, {"reference_id": "2", "file_path": "/documents/ml_basics.txt", "content": ["Machine learning is a subset of AI that enables computers to learn..."]}]}\n{"response": "Artificial Intelligence (AI) is a branch of computer science"}\n{"response": " that aims to create intelligent machines capable of performing"}\n{"response": " tasks that typically require human intelligence."}',
                            },
                            "streaming_without_references": {
                                "summary": "Streaming mode without references (stream=true)",
                                "description": "Multiple NDJSON lines when stream=True and include_references=False. Only response chunks are sent.",
                                "value": '{"response": "Machine learning is a subset of artificial intelligence"}\n{"response": " that enables computers to learn and improve from experience"}\n{"response": " without being explicitly programmed for every task."}',
                            },
                            "non_streaming_with_references": {
                                "summary": "Non-streaming mode with references (stream=false)",
                                "description": "Single NDJSON line when stream=False and include_references=True. Complete response with references in one message.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/neural_networks.pdf"}], "response": "Neural networks are computational models inspired by biological neural networks that consist of interconnected nodes (neurons) organized in layers. They are fundamental to deep learning and can learn complex patterns from data through training processes."}',
                            },
                            "non_streaming_without_references": {
                                "summary": "Non-streaming mode without references (stream=false)",
                                "description": "Single NDJSON line when stream=False and include_references=False. Complete response only.",
                                "value": '{"response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence deep) to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition."}',
                            },
                            "error_response": {
                                "summary": "Error during streaming",
                                "description": "Error handling in NDJSON format when an error occurs during processing.",
                                "value": '{"references": [{"reference_id": "1", "file_path": "/documents/ai.pdf"}]}\n{"response": "Artificial Intelligence is"}\n{"error": "LLM service temporarily unavailable"}',
                            },
                        },
                    }
                },
            },
            400: {
                "description": "Bad Request - Invalid input parameters",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Query text must be at least 3 characters long"
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Query processing failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to process streaming query: Knowledge graph unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_text_stream(request: QueryRequest):
        """
        Advanced RAG query endpoint with flexible streaming response.

        This endpoint provides the most flexible querying experience, supporting both real-time streaming
        and complete response delivery based on your integration needs.

        **Response Modes:**
        - Real-time response delivery as content is generated
        - NDJSON format: each line is a separate JSON object
        - First line: `{"references": [...]}` (if include_references=True)
        - Subsequent lines: `{"response": "content chunk"}`
        - Error handling: `{"error": "error message"}`

        > If stream parameter is False, or the query hit LLM cache, complete response delivered in a single streaming message.

        **Response Format Details**
        - **Content-Type**: `application/x-ndjson` (Newline-Delimited JSON)
        - **Structure**: Each line is an independent, valid JSON object
        - **Parsing**: Process line-by-line, each line is self-contained
        - **Headers**: Includes cache control and connection management

        **Query Modes (same as /query endpoint)**
        - **local**: Entity-focused retrieval with direct relationships
        - **global**: Pattern analysis across the knowledge graph
        - **hybrid**: Combined local and global strategies
        - **naive**: Vector similarity search only
        - **mix**: Integrated knowledge graph + vector retrieval (recommended)
        - **bypass**: Direct LLM query without knowledge retrieval

        conversation_history parameteris sent to LLM only, does not affect retrieval results.

        **Usage Examples**

        Real-time streaming query:
        ```json
        {
            "query": "Explain machine learning algorithms",
            "mode": "mix",
            "stream": true,
            "include_references": true
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        Complete response query:
        ```json
        {
            "query": "What is deep learning?",
            "mode": "hybrid",
            "stream": false,
            "response_type": "Multiple Paragraphs"
        }
        ```

        Conversation with context:
        ```json
        {
            "query": "Can you elaborate on that?",
            "stream": true,
            "conversation_history": [
                {"role": "user", "content": "What is neural network?"},
                {"role": "assistant", "content": "A neural network is..."}
            ]
        }
        ```

        **Response Processing:**

        ```python
        async for line in response.iter_lines():
            data = json.loads(line)
            if "references" in data:
                # Handle references (first message)
                references = data["references"]
            if "response" in data:
                # Handle content chunk
                content_chunk = data["response"]
            if "error" in data:
                # Handle error
                error_message = data["error"]
        ```

        **Error Handling:**
        - Streaming errors are delivered as `{"error": "message"}` lines
        - Non-streaming errors raise HTTP exceptions
        - Partial responses may be delivered before errors in streaming mode
        - Always check for error objects when processing streaming responses

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The question or prompt to process (min 3 characters)
                - **mode**: Query strategy - "mix" recommended for best results
                - **stream**: Enable streaming (True) or complete response (False)
                - **include_references**: Whether to include source citations
                - **response_type**: Format preference (e.g., "Multiple Paragraphs")
                - **top_k**: Number of top entities/relations to retrieve
                - **conversation_history**: Previous dialogue context for multi-turn conversations
                - **max_total_tokens**: Token budget for the entire response

        Returns:
            StreamingResponse: NDJSON streaming response containing:
                - **Streaming mode**: Multiple JSON objects, one per line
                  - References object (if requested): `{"references": [...]}`
                  - Content chunks: `{"response": "chunk content"}`
                  - Error objects: `{"error": "error message"}`
                - **Non-streaming mode**: Single JSON object
                  - Complete response: `{"references": [...], "response": "complete content"}`

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short, invalid mode)
                - 500: Internal processing error (e.g., LLM service unavailable)

        Note:
            This endpoint is ideal for applications requiring flexible response delivery.
            Use streaming mode for real-time interfaces and non-streaming for batch processing.
        """
        try:
            stream_mode = request.stream if request.stream is not None else True
            normal_param = request.to_query_params(stream_mode)

            from fastapi.responses import StreamingResponse

            enrichment_task = None
            if request.include_enrichment and settings.webui_query_enrichment_enabled:
                enrichment_task = asyncio.create_task(
                    _run_enrichment_query(
                        rag, request, _make_enrichment_param(normal_param)
                    )
                )

            tool_event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            def _stream_event_callback(payload: dict[str, Any]) -> None:
                tool_event_queue.put_nowait(payload)

            normal_param.stream_event_callback = _stream_event_callback
            normal_task = asyncio.create_task(
                rag.aquery_llm(
                    request.query,
                    param=normal_param,
                    system_prompt=_get_query_prompt_for_request(request),
                )
            )

            fast_task = None
            if request.use_fast_query:
                fast_param = _make_fast_path_param(normal_param)
                fast_param.stream_event_callback = None
                fast_task = asyncio.create_task(
                    rag.aquery_llm(
                        request.query,
                        param=fast_param,
                        system_prompt=_get_fast_path_prompt(),
                    )
                )

            async def stream_generator():
                normal_result = None
                while normal_result is None:
                    queue_waiter = asyncio.create_task(tool_event_queue.get())
                    done, pending = await asyncio.wait(
                        {normal_task, queue_waiter},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for pending_task in pending:
                        pending_task.cancel()

                    if queue_waiter in done:
                        yield f"{json.dumps({'tool_event': queue_waiter.result()})}\n"

                    if normal_task in done:
                        normal_result = normal_task.result()

                while not tool_event_queue.empty():
                    yield f"{json.dumps({'tool_event': tool_event_queue.get_nowait()})}\n"

                data = normal_result.get("data", {})
                references = _references_from_query_data(data)
                llm_response = normal_result.get("llm_response", {})

                # Enrich references with chunk content if requested
                if request.include_references and request.include_chunk_content:
                    chunks = data.get("chunks", [])
                    # Create a mapping from reference_id to chunk content
                    ref_id_to_content = {}
                    for chunk in chunks:
                        ref_id = chunk.get("reference_id", "")
                        content = chunk.get("content", "")
                        if ref_id and content:
                            # Collect chunk content
                            ref_id_to_content.setdefault(ref_id, []).append(content)

                    # Add content to references
                    enriched_references = []
                    for ref in references:
                        ref_copy = ref.copy()
                        ref_id = ref.get("reference_id", "")
                        if ref_id in ref_id_to_content:
                            # Keep content as a list of chunks (one file may have multiple chunks)
                            ref_copy["content"] = ref_id_to_content[ref_id]
                        enriched_references.append(ref_copy)
                    references = enriched_references

                if llm_response.get("is_streaming"):
                    # Streaming mode: optionally send debug info, then references, then response chunks
                    if request.include_debug:
                        yield f"{json.dumps({'debug': _build_stream_debug_payload(request, normal_result)})}\n"

                    if request.include_references:
                        yield f"{json.dumps({'references': references})}\n"

                    response_stream = llm_response.get("response_iterator")
                    if response_stream:
                        try:
                            async for chunk in response_stream:
                                if chunk:  # Only send non-empty content
                                    yield f"{json.dumps({'response': chunk})}\n"
                            if request.include_references:
                                canonical_references = (
                                    _build_canonical_references_section(references)
                                )
                                if canonical_references:
                                    trailing_references = (
                                        f"\n\n{canonical_references}"
                                    )
                                    yield f"{json.dumps({'response': trailing_references})}\n"
                        except Exception as e:
                            logger.error(f"Streaming error: {str(e)}")
                            yield f"{json.dumps({'error': str(e)})}\n"
                else:
                    # Non-streaming mode: send complete response in one message
                    response_content = llm_response.get("content", "")
                    if not response_content:
                        response_content = "No relevant context found for the query."
                    elif request.include_references:
                        response_content = _ensure_response_has_canonical_references(
                            response_content, references
                        )

                    # Create complete response object
                    complete_response = {"response": response_content}
                    if request.include_debug:
                        complete_response["debug"] = _build_stream_debug_payload(
                            request, normal_result
                        )
                    if request.include_references:
                        complete_response["references"] = references

                    yield f"{json.dumps(complete_response)}\n"

                if enrichment_task is not None:
                    try:
                        enrichment = await enrichment_task
                        yield f"{json.dumps({'enrichment': enrichment})}\n"
                    except Exception as exc:
                        logger.warning("WebUI query enrichment failed: %s", exc)
                        yield f"{json.dumps({'enrichment_error': str(exc)})}\n"

                if fast_task is not None:
                    fast_result = await asyncio.gather(fast_task, return_exceptions=True)
                    fast_outcome = fast_result[0]
                    if not isinstance(fast_outcome, Exception):
                        fast_llm = fast_outcome.get("llm_response", {})
                        fast_content = fast_llm.get("content", "")
                        if fast_content:
                            yield f"{json.dumps({'fast_response': fast_content})}\n"
                    else:
                        yield f"{json.dumps({'fast_error': str(fast_outcome)})}\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "X-Accel-Buffering": "no",  # Ensure proper handling of streaming response when proxied by Nginx
                },
            )
        except Exception as e:
            logger.error(f"Error processing streaming query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/query/data",
        response_model=QueryDataResponse,
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Successful data retrieval response with structured RAG data",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["success", "failure"],
                                    "description": "Query execution status",
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Status message describing the result",
                                },
                                "data": {
                                    "type": "object",
                                    "properties": {
                                        "entities": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "entity_name": {"type": "string"},
                                                    "entity_type": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "source_id": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                    "reference_id": {"type": "string"},
                                                },
                                            },
                                            "description": "Retrieved entities from knowledge graph",
                                        },
                                        "relationships": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "src_id": {"type": "string"},
                                                    "tgt_id": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "keywords": {"type": "string"},
                                                    "weight": {"type": "number"},
                                                    "source_id": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                    "reference_id": {"type": "string"},
                                                },
                                            },
                                            "description": "Retrieved relationships from knowledge graph",
                                        },
                                        "chunks": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "content": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                    "chunk_id": {"type": "string"},
                                                    "reference_id": {"type": "string"},
                                                },
                                            },
                                            "description": "Retrieved text chunks from vector database",
                                        },
                                        "references": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "reference_id": {"type": "string"},
                                                    "file_path": {"type": "string"},
                                                },
                                            },
                                            "description": "Reference list for citation purposes",
                                        },
                                    },
                                    "description": "Structured retrieval data containing entities, relationships, chunks, and references",
                                },
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "query_mode": {"type": "string"},
                                        "keywords": {
                                            "type": "object",
                                            "properties": {
                                                "high_level": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "low_level": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                            },
                                        },
                                        "processing_info": {
                                            "type": "object",
                                            "properties": {
                                                "total_entities_found": {
                                                    "type": "integer"
                                                },
                                                "total_relations_found": {
                                                    "type": "integer"
                                                },
                                                "entities_after_truncation": {
                                                    "type": "integer"
                                                },
                                                "relations_after_truncation": {
                                                    "type": "integer"
                                                },
                                                "final_chunks_count": {
                                                    "type": "integer"
                                                },
                                            },
                                        },
                                    },
                                    "description": "Query metadata including mode, keywords, and processing information",
                                },
                            },
                            "required": ["status", "message", "data", "metadata"],
                        },
                        "examples": {
                            "successful_local_mode": {
                                "summary": "Local mode data retrieval",
                                "description": "Example of structured data from local mode query focusing on specific entities",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [
                                            {
                                                "entity_name": "Neural Networks",
                                                "entity_type": "CONCEPT",
                                                "description": "Computational models inspired by biological neural networks",
                                                "source_id": "chunk-123",
                                                "file_path": "/documents/ai_basics.pdf",
                                                "reference_id": "1",
                                            }
                                        ],
                                        "relationships": [
                                            {
                                                "src_id": "Neural Networks",
                                                "tgt_id": "Machine Learning",
                                                "description": "Neural networks are a subset of machine learning algorithms",
                                                "keywords": "subset, algorithm, learning",
                                                "weight": 0.85,
                                                "source_id": "chunk-123",
                                                "file_path": "/documents/ai_basics.pdf",
                                                "reference_id": "1",
                                            }
                                        ],
                                        "chunks": [
                                            {
                                                "content": "Neural networks are computational models that mimic the way biological neural networks work...",
                                                "file_path": "/documents/ai_basics.pdf",
                                                "chunk_id": "chunk-123",
                                                "reference_id": "1",
                                            }
                                        ],
                                        "references": [
                                            {
                                                "reference_id": "1",
                                                "file_path": "/documents/ai_basics.pdf",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "local",
                                        "keywords": {
                                            "high_level": ["neural", "networks"],
                                            "low_level": [
                                                "computation",
                                                "model",
                                                "algorithm",
                                            ],
                                        },
                                        "processing_info": {
                                            "total_entities_found": 5,
                                            "total_relations_found": 3,
                                            "entities_after_truncation": 1,
                                            "relations_after_truncation": 1,
                                            "final_chunks_count": 1,
                                        },
                                    },
                                },
                            },
                            "global_mode": {
                                "summary": "Global mode data retrieval",
                                "description": "Example of structured data from global mode query analyzing broader patterns",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [],
                                        "relationships": [
                                            {
                                                "src_id": "Artificial Intelligence",
                                                "tgt_id": "Machine Learning",
                                                "description": "AI encompasses machine learning as a core component",
                                                "keywords": "encompasses, component, field",
                                                "weight": 0.92,
                                                "source_id": "chunk-456",
                                                "file_path": "/documents/ai_overview.pdf",
                                                "reference_id": "2",
                                            }
                                        ],
                                        "chunks": [],
                                        "references": [
                                            {
                                                "reference_id": "2",
                                                "file_path": "/documents/ai_overview.pdf",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "global",
                                        "keywords": {
                                            "high_level": [
                                                "artificial",
                                                "intelligence",
                                                "overview",
                                            ],
                                            "low_level": [],
                                        },
                                    },
                                },
                            },
                            "naive_mode": {
                                "summary": "Naive mode data retrieval",
                                "description": "Example of structured data from naive mode using only vector search",
                                "value": {
                                    "status": "success",
                                    "message": "Query executed successfully",
                                    "data": {
                                        "entities": [],
                                        "relationships": [],
                                        "chunks": [
                                            {
                                                "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers...",
                                                "file_path": "/documents/deep_learning.pdf",
                                                "chunk_id": "chunk-789",
                                                "reference_id": "3",
                                            }
                                        ],
                                        "references": [
                                            {
                                                "reference_id": "3",
                                                "file_path": "/documents/deep_learning.pdf",
                                            }
                                        ],
                                    },
                                    "metadata": {
                                        "query_mode": "naive",
                                        "keywords": {"high_level": [], "low_level": []},
                                    },
                                },
                            },
                        },
                    }
                },
            },
            400: {
                "description": "Bad Request - Invalid input parameters",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Query text must be at least 3 characters long"
                        },
                    }
                },
            },
            500: {
                "description": "Internal Server Error - Data retrieval failed",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"detail": {"type": "string"}},
                        },
                        "example": {
                            "detail": "Failed to retrieve data: Knowledge graph unavailable"
                        },
                    }
                },
            },
        },
    )
    async def query_data(request: QueryRequest):
        """
        Advanced data retrieval endpoint for structured RAG analysis.

        This endpoint provides raw retrieval results without LLM generation, perfect for:
        - **Data Analysis**: Examine what information would be used for RAG
        - **System Integration**: Get structured data for custom processing
        - **Debugging**: Understand retrieval behavior and quality
        - **Research**: Analyze knowledge graph structure and relationships

        **Key Features:**
        - No LLM generation - pure data retrieval
        - Complete structured output with entities, relationships, and chunks
        - Always includes references for citation
        - Detailed metadata about processing and keywords
        - Compatible with all query modes and parameters

        **Query Mode Behaviors:**
        - **local**: Returns entities and their direct relationships + related chunks
        - **global**: Returns relationship patterns across the knowledge graph
        - **hybrid**: Combines local and global retrieval strategies
        - **naive**: Returns only vector-retrieved text chunks (no knowledge graph)
        - **mix**: Integrates knowledge graph data with vector-retrieved chunks
        - **bypass**: Returns empty data arrays (used for direct LLM queries)

        **Data Structure:**
        - **entities**: Knowledge graph entities with descriptions and metadata
        - **relationships**: Connections between entities with weights and descriptions
        - **chunks**: Text segments from documents with source information
        - **references**: Citation information mapping reference IDs to file paths
        - **metadata**: Processing information, keywords, and query statistics

        **Usage Examples:**

        Analyze entity relationships:
        ```json
        {
            "query": "machine learning algorithms",
            "mode": "local",
            "top_k": 10
        }
        ```

        Explore global patterns:
        ```json
        {
            "query": "artificial intelligence trends",
            "mode": "global",
            "max_relation_tokens": 2000
        }
        ```

        Vector similarity search:
        ```json
        {
            "query": "neural network architectures",
            "mode": "naive",
            "chunk_top_k": 5
        }
        ```

        Bypass initial LLM call by providing high-level and low-level keywords:
        ```json
        {
            "query": "What is Retrieval-Augmented-Generation?",
            "hl_keywords": ["machine learning", "information retrieval", "natural language processing"],
            "ll_keywords": ["retrieval augmented generation", "RAG", "knowledge base"],
            "mode": "mix"
        }
        ```

        **Response Analysis:**
        - **Empty arrays**: Normal for certain modes (e.g., naive mode has no entities/relationships)
        - **Processing info**: Shows retrieval statistics and token usage
        - **Keywords**: High-level and low-level keywords extracted from query
        - **Reference mapping**: Links all data back to source documents

        Args:
            request (QueryRequest): The request object containing query parameters:
                - **query**: The search query to analyze (min 3 characters)
                - **mode**: Retrieval strategy affecting data types returned
                - **top_k**: Number of top entities/relationships to retrieve
                - **chunk_top_k**: Number of text chunks to retrieve
                - **max_entity_tokens**: Token limit for entity context
                - **max_relation_tokens**: Token limit for relationship context
                - **max_total_tokens**: Overall token budget for retrieval

        Returns:
            QueryDataResponse: Structured JSON response containing:
                - **status**: "success" or "failure"
                - **message**: Human-readable status description
                - **data**: Complete retrieval results with entities, relationships, chunks, references
                - **metadata**: Query processing information and statistics

        Raises:
            HTTPException:
                - 400: Invalid input parameters (e.g., query too short, invalid mode)
                - 500: Internal processing error (e.g., knowledge graph unavailable)

        Note:
            This endpoint always includes references regardless of the include_references parameter,
            as structured data analysis typically requires source attribution.
        """
        try:
            param = request.to_query_params(False)  # No streaming for data endpoint
            response = await rag.aquery_data(request.query, param=param)

            # aquery_data returns the new format with status, message, data, and metadata
            if isinstance(response, dict):
                return QueryDataResponse(**response)
            else:
                # Handle unexpected response format
                return QueryDataResponse(
                    status="failure",
                    message="Invalid response type",
                    data={},
                    metadata={},
                )
        except Exception as e:
            logger.error(f"Error processing data query: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return router
