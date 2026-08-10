"""FastMCP integration for the LightRAG API service."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

import attrs
from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

AGENTIC_TOOL_DESCRIPTIONS = {
    "query_document": (
        "Agentic development memory search. Use for analytics, planning, QA, "
        "coding, design review, debugging, deployment research, incident context, "
        "and retrieving prior project decisions from LightRAG. Prefer "
        "only_need_context=true when you need raw evidence before changing code."
    ),
    "insert_document": (
        "Persist durable agent memory into LightRAG. Use after meaningful coding, "
        "QA, debugging, design, planning, management, or deployment work to record "
        "decisions, incidents, validation results, architecture notes, and reusable findings."
    ),
    "upload_document": (
        "Add an external project document by file upload for later analysis, QA, "
        "planning, debugging, design review, or deployment research."
    ),
    "insert_file": (
        "Index a local repository file or artifact directly into LightRAG for "
        "agentic coding, QA, analytics, design, debugging, or deployment context."
    ),
    "insert_batch": (
        "Bulk-index a local directory of project artifacts. Use for onboarding "
        "repositories, specs, logs, test reports, design docs, runbooks, and deployment notes."
    ),
    "scan_for_new_documents": (
        "Start ingestion for newly added input documents. Use after adding specs, "
        "logs, QA reports, planning notes, or deployment artifacts to LightRAG inputs."
    ),
    "get_documents": (
        "Inspect indexed document inventory and processing state. Use for knowledge-base "
        "management, QA of ingestion coverage, duplicate checks, and debugging missing context."
    ),
    "get_pipeline_status": (
        "Inspect ingestion pipeline activity and failures. Use for operations, QA, "
        "debugging stuck indexing, and deciding whether newly inserted memory is searchable."
    ),
    "get_graph_labels": (
        "Inspect available graph labels. Use for analytics, ontology management, "
        "QA of extraction quality, planning graph cleanup, and debugging retrieval gaps."
    ),
    "check_lightrag_health": (
        "Check LightRAG service health and runtime configuration before agentic "
        "coding, QA, analytics, management, debugging, deployment, or memory operations."
    ),
    "check_memory_pressure": (
        "Inspect host memory pressure, swap usage, and top resident processes. Use "
        "when debugging Apple Silicon unified-memory pressure, MLX throughput collapse, "
        "or runtime restarts caused by memory contention."
    ),
    "merge_entities": (
        "Merge duplicate graph entities during knowledge-base management. Use for "
        "QA cleanup after extraction, analytics normalization, and improving retrieval quality."
    ),
    "create_entities": (
        "Create validated project graph entities in bulk. Use to curate architecture, "
        "coding, design, planning, QA, debugging, deployment, or management knowledge."
    ),
    "delete_by_entities": (
        "Delete obsolete or incorrect graph entities by name. Use for graph QA, "
        "cleanup, failed extraction repair, and management of stale project knowledge."
    ),
    "delete_by_doc_ids": (
        "Delete graph data associated with specific documents. Use for reindexing, "
        "QA cleanup, removing bad imports, and deployment or debugging data resets."
    ),
    "edit_entities": (
        "Edit existing graph entities in bulk. Use to correct types/descriptions "
        "for coding, design, QA, analytics, planning, debugging, or deployment knowledge."
    ),
    "create_relations": (
        "Create validated graph relations in bulk. Use to capture dependencies, "
        "ownership, failure causes, design links, QA findings, deployment topology, "
        "and coding relationships between project entities."
    ),
    "edit_relations": (
        "Edit graph relations in bulk. Use for QA cleanup, analytics normalization, "
        "debugging retrieval gaps, and maintaining accurate project dependency knowledge."
    ),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_lightrag_mcp_submodule_importable() -> None:
    src_path = _repo_root() / "third_party" / "lightrag-mcp" / "src"
    if not src_path.exists():
        raise RuntimeError(
            "LightRAG MCP submodule is missing. Run "
            "`git submodule update --init --recursive third_party/lightrag-mcp`."
        )
    src = str(src_path)
    if src not in sys.path:
        sys.path.insert(0, src)


def _load_lightrag_client_class() -> type[Any]:
    _ensure_lightrag_mcp_submodule_importable()
    from lightrag_mcp.lightrag_client import LightRAGClient

    return LightRAGClient


def _to_jsonable(value: Any) -> Any:
    value_type = type(value)
    if value_type.__name__ == "Unset" and getattr(value_type, "__module__", "").endswith(
        ".types"
    ):
        return None
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
    if attrs.has(type(value)):
        return _to_jsonable(attrs.asdict(value, recurse=False))
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            rendered = _to_jsonable(item)
            if rendered is not None:
                normalized[str(key)] = rendered
        return normalized
    if isinstance(value, list | tuple | set):
        normalized_items: list[Any] = []
        for item in value:
            rendered = _to_jsonable(item)
            if rendered is not None:
                normalized_items.append(rendered)
        return normalized_items
    if hasattr(value, "dict") and callable(value.dict):
        return _to_jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _to_jsonable(value.__dict__)
    return str(value)


def _format_response(result: Any, *, is_error: bool = False) -> dict[str, Any]:
    if is_error:
        return {"status": "error", "error": str(result)}
    return {"status": "success", "response": _to_jsonable(result)}


def _summarize_value(value: Any, *, limit: int = 240) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        compact = " ".join(value.split())
        return compact[:limit] + ("..." if len(compact) > limit else "")
    if isinstance(value, list | tuple | set):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        preview_keys = list(value.keys())[:8]
        more = "" if len(value) <= 8 else ",..."
        return f"dict(keys={preview_keys}{more})"
    return repr(value)[:limit]


def _summarize_tool_kwargs(kwargs: dict[str, Any]) -> str:
    if not kwargs:
        return "-"
    parts = []
    for key, value in kwargs.items():
        parts.append(f"{key}={_summarize_value(value)}")
    return "; ".join(parts)


def _get_lifespan_context(ctx: Context) -> dict[str, Any]:
    lifespan_context = getattr(ctx, "lifespan_context", None)
    if isinstance(lifespan_context, dict):
        return lifespan_context
    request_context = getattr(ctx, "request_context", None)
    if request_context is not None:
        request_lifespan_context = getattr(request_context, "lifespan_context", None)
        if isinstance(request_lifespan_context, dict):
            return request_lifespan_context
    return {}


async def _execute_lightrag_operation(
    ctx: Context,
    operation_name: str,
    operation_func: Callable[[Any], Awaitable[Any]],
    *,
    tool_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        client = _get_lifespan_context(ctx).get("lightrag_client")
        if client is None:
            logger.error(
                "LightRAG MCP client is not initialized for operation=%s",
                operation_name,
            )
            return _format_response(
                f"LightRAG MCP client is not initialized for {operation_name}",
                is_error=True,
            )

        logger.info(
            "Executing LightRAG MCP operation: %s args=%s",
            operation_name,
            _summarize_tool_kwargs(tool_kwargs or {}),
        )
        result = await operation_func(client)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "Completed LightRAG MCP operation: %s duration_ms=%.2f response_type=%s result=%s",
            operation_name,
            elapsed_ms,
            type(result).__name__,
            _summarize_value(result),
        )
        return _format_response(result)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "LightRAG MCP operation failed: %s duration_ms=%.2f error_type=%s args=%s",
            operation_name,
            elapsed_ms,
            type(exc).__name__,
            _summarize_tool_kwargs(tool_kwargs or {}),
        )
        return _format_response(exc, is_error=True)


def _default_api_base_url(args: Any) -> str:
    explicit = os.getenv("LIGHTRAG_MCP_API_BASE_URL", "").strip()
    if explicit:
        return explicit
    port = int(getattr(args, "port", 9621))
    return f"http://127.0.0.1:{port}"


def _default_api_key(api_key: str | None) -> str:
    return os.getenv("LIGHTRAG_MCP_API_KEY", "").strip() or api_key or ""


def _run_command_output(command: list[str], timeout_s: float = 5.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except FileNotFoundError:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": "command not found",
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"timed out after {timeout_s}s",
        }


def _memory_pressure_level(available_ratio: float, swap_used_mb: float) -> str:
    if available_ratio < 0.10 or swap_used_mb >= 4096:
        return "critical"
    if available_ratio < 0.20 or swap_used_mb >= 1024:
        return "high"
    if available_ratio < 0.35 or swap_used_mb > 0:
        return "moderate"
    return "normal"


def _memory_pressure_snapshot(top_process_limit: int = 8) -> dict[str, Any]:
    import psutil

    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    available_ratio = (
        virtual_memory.available / virtual_memory.total if virtual_memory.total else 0.0
    )
    swap_used_mb = round(swap_memory.used / 1024 / 1024, 2)
    process_rows: list[dict[str, Any]] = []

    for process in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            info = process.info
            rss_bytes = getattr(info.get("memory_info"), "rss", 0)
            process_rows.append(
                {
                    "pid": info.get("pid"),
                    "name": info.get("name") or "unknown",
                    "rss_mb": round(rss_bytes / 1024 / 1024, 2),
                    "cmdline": " ".join(info.get("cmdline") or []),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    top_processes = sorted(
        process_rows,
        key=lambda item: item.get("rss_mb", 0.0),
        reverse=True,
    )[: max(1, top_process_limit)]

    snapshot = {
        "platform": platform.platform(),
        "memory_pressure": {
            "level": _memory_pressure_level(available_ratio, swap_used_mb),
            "available_ratio": round(available_ratio, 4),
            "total_mb": round(virtual_memory.total / 1024 / 1024, 2),
            "available_mb": round(virtual_memory.available / 1024 / 1024, 2),
            "used_mb": round(virtual_memory.used / 1024 / 1024, 2),
            "percent": virtual_memory.percent,
        },
        "swap": {
            "total_mb": round(swap_memory.total / 1024 / 1024, 2),
            "used_mb": swap_used_mb,
            "free_mb": round(swap_memory.free / 1024 / 1024, 2),
            "percent": swap_memory.percent,
        },
        "current_process": {
            "pid": psutil.Process().pid,
            "rss_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
        },
        "top_processes": top_processes,
    }

    if sys.platform == "darwin":
        snapshot["macos"] = {
            "memory_pressure_q": _run_command_output(["memory_pressure", "-Q"]),
            "vm_stat": _run_command_output(["vm_stat"]),
            "swapusage": _run_command_output(["sysctl", "vm.swapusage"]),
        }

    return snapshot


def _configure_lightrag_client_auth(client: Any, api_key: str) -> None:
    """Ensure generated LightRAG API clients send the server's X-API-Key."""

    if not api_key:
        return

    generated_client = getattr(client, "client", None)
    if generated_client is None:
        return

    if hasattr(generated_client, "auth_header_name"):
        generated_client.auth_header_name = "X-API-Key"
    if hasattr(generated_client, "prefix"):
        generated_client.prefix = ""

    with_headers = getattr(generated_client, "with_headers", None)
    if callable(with_headers):
        client.client = with_headers({"X-API-Key": api_key})

    for attr_name in ("_client", "_async_client"):
        http_client = getattr(client.client, attr_name, None)
        if http_client is not None:
            http_client.headers.pop("Authorization", None)
            http_client.headers["X-API-Key"] = api_key


def create_lightrag_mcp(args: Any, api_key: str | None) -> FastMCP:
    """Create the curated LightRAG MCP server using current FastMCP APIs."""

    base_url = _default_api_base_url(args)
    resolved_api_key = _default_api_key(api_key)

    @asynccontextmanager
    async def lifespan(_server: FastMCP):
        client_class = _load_lightrag_client_class()
        client = client_class(base_url=base_url, api_key=resolved_api_key)
        _configure_lightrag_client_auth(client, resolved_api_key)
        logger.info(
            "Integrated LightRAG MCP server started base_url=%s api_key_configured=%s transport=%s",
            base_url,
            bool(resolved_api_key),
            os.getenv("LIGHTRAG_MCP_TRANSPORT", "streamable-http"),
        )
        try:
            yield {"lightrag_client": client}
        finally:
            await client.close()
            logger.info("Integrated LightRAG MCP server stopped")

    mcp = FastMCP(
        name=os.getenv("LIGHTRAG_MCP_NAME", "LightRAG"),
        lifespan=lifespan,
    )

    @mcp.tool(name="query_document", description=AGENTIC_TOOL_DESCRIPTIONS["query_document"])
    async def query_document(
        ctx: Context,
        query: str = Field(description="Query text"),
        mode: str = Field(
            default="mix",
            description="Search mode: mix, semantic, keyword, global, hybrid, local, or naive",
        ),
        top_k: int = Field(default=60, description="Number of candidate results"),
        only_need_context: bool = Field(
            default=False,
            description="Return retrieved context without generating an answer",
        ),
        only_need_prompt: bool = Field(
            default=False,
            description="Return the generated prompt without generating an answer",
        ),
        response_type: str = Field(
            default="Multiple Paragraphs",
            description="Answer format requested from LightRAG",
        ),
        max_token_for_text_unit: int = Field(
            default=4096,
            description="Maximum tokens for retrieved text chunks",
        ),
        max_token_for_global_context: int = Field(
            default=4096,
            description="Maximum tokens for global graph context",
        ),
        max_token_for_local_context: int = Field(
            default=4096,
            description="Maximum tokens for local graph context",
        ),
        hl_keywords: list[str] = Field(
            default_factory=list,
            description="High-level keywords for prioritization",
        ),
        ll_keywords: list[str] = Field(
            default_factory=list,
            description="Low-level keywords for search refinement",
        ),
        history_turns: int = Field(
            default=10,
            description="Conversation turns included in response context",
        ),
    ) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.query(
                query_text=query,
                mode=mode,
                top_k=top_k,
                only_need_context=only_need_context,
                only_need_prompt=only_need_prompt,
                response_type=response_type,
                max_token_for_text_unit=max_token_for_text_unit,
                max_token_for_global_context=max_token_for_global_context,
                max_token_for_local_context=max_token_for_local_context,
                hl_keywords=hl_keywords,
                ll_keywords=ll_keywords,
                history_turns=history_turns,
            )

        return await _execute_lightrag_operation(
            ctx,
            "query_document",
            _operation,
            tool_kwargs={
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "only_need_context": only_need_context,
                "only_need_prompt": only_need_prompt,
                "response_type": response_type,
                "hl_keywords": hl_keywords,
                "ll_keywords": ll_keywords,
                "history_turns": history_turns,
            },
        )

    @mcp.tool(
        name="insert_document",
        description=AGENTIC_TOOL_DESCRIPTIONS["insert_document"],
    )
    async def insert_document(
        ctx: Context,
        text: str | list[str] = Field(description="Text or list of texts to insert"),
    ) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.insert_text(text=text)

        return await _execute_lightrag_operation(
            ctx,
            "insert_document",
            _operation,
            tool_kwargs={"text": text},
        )

    @mcp.tool(
        name="upload_document",
        description=AGENTIC_TOOL_DESCRIPTIONS["upload_document"],
    )
    async def upload_document(
        ctx: Context,
        file_path: str = Field(description="Local path to the file to upload"),
    ) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.upload_document(file_path=file_path)

        return await _execute_lightrag_operation(
            ctx,
            "upload_document",
            _operation,
            tool_kwargs={"file_path": file_path},
        )

    @mcp.tool(name="insert_file", description=AGENTIC_TOOL_DESCRIPTIONS["insert_file"])
    async def insert_file(
        ctx: Context,
        file_path: str = Field(description="Local path to the file to insert"),
    ) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.insert_file(file_path=file_path)

        return await _execute_lightrag_operation(
            ctx,
            "insert_file",
            _operation,
            tool_kwargs={"file_path": file_path},
        )

    @mcp.tool(name="insert_batch", description=AGENTIC_TOOL_DESCRIPTIONS["insert_batch"])
    async def insert_batch(
        ctx: Context,
        directory_path: str = Field(description="Directory containing files to insert"),
        recursive: bool = Field(default=False, description="Walk subdirectories"),
        depth: int = Field(default=1, description="Maximum recursion depth"),
        include_only: list[str] = Field(
            default_factory=list,
            description="Regular expressions for file names to include",
        ),
        ignore_files: list[str] = Field(
            default_factory=list,
            description="Regular expressions for file names to skip",
        ),
        ignore_directories: list[str] = Field(
            default_factory=list,
            description="Regular expressions for directory names to skip",
        ),
    ) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.insert_batch(
                directory_path=directory_path,
                recursive=recursive,
                depth=depth,
                include_only=include_only,
                ignore_directories=ignore_directories,
                ignore_files=ignore_files,
            )

        return await _execute_lightrag_operation(
            ctx,
            "insert_batch",
            _operation,
            tool_kwargs={
                "directory_path": directory_path,
                "recursive": recursive,
                "depth": depth,
                "include_only": include_only,
                "ignore_files": ignore_files,
                "ignore_directories": ignore_directories,
            },
        )

    @mcp.tool(
        name="scan_for_new_documents",
        description=AGENTIC_TOOL_DESCRIPTIONS["scan_for_new_documents"],
    )
    async def scan_for_new_documents(ctx: Context) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.scan_for_new_documents()

        return await _execute_lightrag_operation(
            ctx, "scan_for_new_documents", _operation
        )

    @mcp.tool(name="get_documents", description=AGENTIC_TOOL_DESCRIPTIONS["get_documents"])
    async def get_documents(ctx: Context) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.get_documents()

        return await _execute_lightrag_operation(ctx, "get_documents", _operation)

    @mcp.tool(
        name="get_pipeline_status",
        description=AGENTIC_TOOL_DESCRIPTIONS["get_pipeline_status"],
    )
    async def get_pipeline_status(ctx: Context) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.get_pipeline_status()

        return await _execute_lightrag_operation(
            ctx, "get_pipeline_status", _operation
        )

    @mcp.tool(
        name="get_graph_labels",
        description=AGENTIC_TOOL_DESCRIPTIONS["get_graph_labels"],
    )
    async def get_graph_labels(ctx: Context) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.get_graph_labels()

        return await _execute_lightrag_operation(ctx, "get_graph_labels", _operation)

    @mcp.tool(
        name="check_lightrag_health",
        description=AGENTIC_TOOL_DESCRIPTIONS["check_lightrag_health"],
    )
    async def check_lightrag_health(ctx: Context) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.get_health()

        return await _execute_lightrag_operation(
            ctx, "check_lightrag_health", _operation
        )

    @mcp.tool(
        name="check_memory_pressure",
        description=AGENTIC_TOOL_DESCRIPTIONS["check_memory_pressure"],
    )
    async def check_memory_pressure(
        top_process_limit: int = Field(
            default=8,
            description="Number of top resident processes to include",
        ),
    ) -> dict[str, Any]:
        return _format_response(_memory_pressure_snapshot(top_process_limit))

    @mcp.tool(name="merge_entities", description=AGENTIC_TOOL_DESCRIPTIONS["merge_entities"])
    async def merge_entities(
        ctx: Context,
        source_entities: list[str] = Field(description="Entity names to merge"),
        target_entity: str = Field(description="Target entity name"),
        merge_strategy: dict[str, str] = Field(
            default_factory=dict,
            description="Property merge strategy by field name",
        ),
    ) -> dict[str, Any]:
        async def _operation(client: Any) -> Any:
            return await client.merge_entities(
                source_entities=source_entities,
                target_entity=target_entity,
                merge_strategy=merge_strategy,
            )

        return await _execute_lightrag_operation(
            ctx,
            "merge_entities",
            _operation,
            tool_kwargs={
                "source_entities": source_entities,
                "target_entity": target_entity,
                "merge_strategy": merge_strategy,
            },
        )

    @mcp.tool(
        name="create_entities",
        description=AGENTIC_TOOL_DESCRIPTIONS["create_entities"],
    )
    async def create_entities(
        ctx: Context,
        entities: list[dict[str, Any]] = Field(
            description="Entities with entity_name, entity_type, description, source_id"
        ),
    ) -> dict[str, Any]:
        async def _create_entity(client: Any, data: dict[str, Any]) -> dict[str, Any]:
            entity_name = data.get("entity_name")
            entity_type = data.get("entity_type")
            description = data.get("description")
            source_id = data.get("source_id")
            if not all([entity_name, entity_type, description, source_id]):
                return {
                    "entity_name": str(entity_name or "unknown"),
                    "status": "error",
                    "error": "Missing required fields",
                }
            try:
                result = await client.create_entity(
                    entity_name=str(entity_name),
                    entity_type=str(entity_type),
                    description=str(description),
                    source_id=str(source_id),
                )
                return {
                    "entity_name": str(entity_name),
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {
                    "entity_name": str(entity_name),
                    "status": "error",
                    "error": str(exc),
                }

        async def _operation(client: Any) -> Any:
            results = await asyncio.gather(
                *(_create_entity(client, entity) for entity in entities)
            )
            return {
                "total": len(entities),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            ctx,
            "create_entities",
            _operation,
            tool_kwargs={"entities": entities},
        )

    @mcp.tool(
        name="delete_by_entities",
        description=AGENTIC_TOOL_DESCRIPTIONS["delete_by_entities"],
    )
    async def delete_by_entities(
        ctx: Context,
        entity_names: list[str] = Field(description="Entity names to delete"),
    ) -> dict[str, Any]:
        async def _delete_entity(client: Any, entity_name: str) -> dict[str, Any]:
            try:
                result = await client.delete_by_entity(entity_name=entity_name)
                return {
                    "entity_name": entity_name,
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {"entity_name": entity_name, "status": "error", "error": str(exc)}

        async def _operation(client: Any) -> Any:
            results = await asyncio.gather(
                *(_delete_entity(client, entity_name) for entity_name in entity_names)
            )
            return {
                "total": len(entity_names),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            ctx,
            "delete_by_entities",
            _operation,
            tool_kwargs={"entity_names": entity_names},
        )

    @mcp.tool(
        name="delete_by_doc_ids",
        description=AGENTIC_TOOL_DESCRIPTIONS["delete_by_doc_ids"],
    )
    async def delete_by_doc_ids(
        ctx: Context,
        doc_ids: list[str] = Field(description="Document IDs to delete"),
    ) -> dict[str, Any]:
        async def _delete_by_doc_id(client: Any, doc_id: str) -> dict[str, Any]:
            try:
                result = await client.delete_by_doc_id(doc_id=doc_id)
                return {"doc_id": doc_id, "status": "success", "result": _to_jsonable(result)}
            except Exception as exc:
                return {"doc_id": doc_id, "status": "error", "error": str(exc)}

        async def _operation(client: Any) -> Any:
            results = await asyncio.gather(
                *(_delete_by_doc_id(client, doc_id) for doc_id in doc_ids)
            )
            return {
                "total": len(doc_ids),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            ctx,
            "delete_by_doc_ids",
            _operation,
            tool_kwargs={"doc_ids": doc_ids},
        )

    @mcp.tool(name="edit_entities", description=AGENTIC_TOOL_DESCRIPTIONS["edit_entities"])
    async def edit_entities(
        ctx: Context,
        entities: list[dict[str, Any]] = Field(
            description="Entities with entity_name, entity_type, description, source_id"
        ),
    ) -> dict[str, Any]:
        async def _edit_entity(client: Any, data: dict[str, Any]) -> dict[str, Any]:
            entity_name = data.get("entity_name")
            entity_type = data.get("entity_type")
            description = data.get("description")
            source_id = data.get("source_id")
            if not all([entity_name, entity_type, description, source_id]):
                return {
                    "entity_name": str(entity_name or "unknown"),
                    "status": "error",
                    "error": "Missing required fields",
                }
            try:
                result = await client.edit_entity(
                    entity_name=str(entity_name),
                    entity_type=str(entity_type),
                    description=str(description),
                    source_id=str(source_id),
                )
                return {
                    "entity_name": str(entity_name),
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {
                    "entity_name": str(entity_name),
                    "status": "error",
                    "error": str(exc),
                }

        async def _operation(client: Any) -> Any:
            results = await asyncio.gather(
                *(_edit_entity(client, entity) for entity in entities)
            )
            return {
                "total": len(entities),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            ctx,
            "edit_entities",
            _operation,
            tool_kwargs={"entities": entities},
        )

    @mcp.tool(
        name="create_relations",
        description=AGENTIC_TOOL_DESCRIPTIONS["create_relations"],
    )
    async def create_relations(
        ctx: Context,
        relations: list[dict[str, Any]] = Field(
            description="Relations with source, target, description, keywords, optional source_id and weight"
        ),
    ) -> dict[str, Any]:
        async def _create_relation(client: Any, data: dict[str, Any]) -> dict[str, Any]:
            source = data.get("source")
            target = data.get("target")
            description = data.get("description")
            keywords = data.get("keywords")
            source_id = data.get("source_id")
            weight = data.get("weight")
            label = f"{source or 'unknown'} -> {target or 'unknown'}"
            if not all([source, target, description, keywords]):
                return {
                    "relation": label,
                    "status": "error",
                    "error": "Missing required fields",
                }
            try:
                result = await client.create_relation(
                    source=str(source),
                    target=str(target),
                    description=str(description),
                    keywords=str(keywords),
                    source_id=str(source_id) if source_id else None,
                    weight=float(weight) if weight is not None else None,
                )
                return {
                    "relation": label,
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {"relation": label, "status": "error", "error": str(exc)}

        async def _operation(client: Any) -> Any:
            results = await asyncio.gather(
                *(_create_relation(client, relation) for relation in relations)
            )
            return {
                "total": len(relations),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            ctx,
            "create_relations",
            _operation,
            tool_kwargs={"relations": relations},
        )

    @mcp.tool(name="edit_relations", description=AGENTIC_TOOL_DESCRIPTIONS["edit_relations"])
    async def edit_relations(
        ctx: Context,
        relations: list[dict[str, Any]] = Field(
            description="Relations with source, target, description, keywords, relation_type, optional source_id and weight"
        ),
    ) -> dict[str, Any]:
        async def _edit_relation(client: Any, data: dict[str, Any]) -> dict[str, Any]:
            source = data.get("source")
            target = data.get("target")
            description = data.get("description")
            keywords = data.get("keywords")
            relation_type = data.get("relation_type")
            source_id = data.get("source_id")
            weight = data.get("weight")
            label = f"{source or 'unknown'} -> {target or 'unknown'}"
            if not all([source, target, description, keywords, relation_type]):
                return {
                    "relation": label,
                    "status": "error",
                    "error": "Missing required fields",
                }
            try:
                result = await client.edit_relation(
                    source=str(source),
                    target=str(target),
                    description=str(description),
                    keywords=str(keywords),
                    relation_type=str(relation_type),
                    source_id=str(source_id) if source_id else None,
                    weight=float(weight) if weight is not None else None,
                )
                return {
                    "relation": label,
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {"relation": label, "status": "error", "error": str(exc)}

        async def _operation(client: Any) -> Any:
            results = await asyncio.gather(
                *(_edit_relation(client, relation) for relation in relations)
            )
            return {
                "total": len(relations),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            ctx,
            "edit_relations",
            _operation,
            tool_kwargs={"relations": relations},
        )

    return mcp


def create_lightrag_mcp_http_app(args: Any, api_key: str | None) -> Any:
    mcp = create_lightrag_mcp(args=args, api_key=api_key)
    return mcp.http_app(
        path="/",
        transport=cast(
            Literal["http", "streamable-http", "sse"],
            os.getenv("LIGHTRAG_MCP_TRANSPORT", "streamable-http"),
        ),
        json_response=_env_bool("LIGHTRAG_MCP_JSON_RESPONSE", False),
        stateless_http=_env_bool("LIGHTRAG_MCP_STATELESS_HTTP", False),
    )


class _MCPMountRootEndpoint:
    """Forward exact /mcp requests to a mounted FastMCP app without redirecting."""

    def __init__(self, app: ASGIApp, mount_path: str):
        self.app = app
        self.mount_path = mount_path.rstrip("/") or "/"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        child_scope = dict(scope)
        root_path = scope.get("root_path", "")
        child_scope["root_path"] = f"{root_path}{self.mount_path}"
        child_scope["path"] = "/"
        child_scope["raw_path"] = b"/"
        await self.app(child_scope, receive, send)


def mount_lightrag_mcp_http_app(
    app: Any, mcp_http_app: ASGIApp, mount_path: str
) -> None:
    """Mount FastMCP at /mcp and /mcp/ without POST slash redirects.

    Starlette redirects exact POST /mcp to /mcp/ for mounted apps. Some MCP
    bridges do not preserve streamable-http session state across that redirect,
    leaving tools stuck even though the operation completes server-side.
    """

    normalized_path = mount_path.rstrip("/") or "/mcp"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    app.router.routes.insert(
        0,
        Route(
            normalized_path,
            endpoint=_MCPMountRootEndpoint(mcp_http_app, normalized_path),
            methods=["GET", "POST", "DELETE"],
            include_in_schema=False,
        ),
    )
    app.mount(normalized_path, mcp_http_app)
