"""FastMCP integration for the LightRAG API service."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import secrets
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_access_token
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

_MCP_MATCH_LIMIT = 6
_MCP_EXCERPT_CHARS = 700
_QUERY_STOP_WORDS = {
    "about",
    "after",
    "before",
    "find",
    "from",
    "have",
    "into",
    "light",
    "lightrag",
    "path",
    "project",
    "repository",
    "that",
    "their",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

AGENTIC_TOOL_DESCRIPTIONS = {
    "query_document": (
        "Agentic development memory search. Use for analytics, planning, QA, "
        "coding, design review, debugging, deployment research, incident context, "
        "and retrieving prior project decisions from LightRAG. Returns matching "
        "source excerpts with provenance in one call. Prefer only_need_context=true "
        "when you need raw evidence before changing code."
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
    "get_document_content": (
        "Fetch one complete indexed document by the document_id returned by query_document. "
        "Use only when the bounded matching excerpts are insufficient and the full source is required."
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
        "Check LightRAG service health after an MCP timeout, connection failure, "
        "or invalid response. Do not use as a preflight for normal memory operations."
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


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "value") and isinstance(value.value, str | int | float | bool):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_jsonable(value.to_dict())
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


def _query_terms(query: str, keywords: list[str]) -> list[str]:
    query_body = "\n".join(
        line
        for line in query.splitlines()
        if not line.casefold().startswith(("project:", "project path:", "repository:"))
    )
    words = re.findall(r"[\w.-]{4,}", " ".join([*keywords, query_body]).casefold())
    return list(dict.fromkeys(word for word in words if word not in _QUERY_STOP_WORDS))


def _matching_excerpt(content: str, terms: list[str]) -> tuple[str, list[str]]:
    """Return one small source window centered on the strongest lexical match."""

    compact = content.strip()
    if len(compact) <= _MCP_EXCERPT_CHARS:
        matched = [term for term in terms if term in compact.casefold()]
        return compact, matched

    folded = compact.casefold()
    positions = [(folded.find(term), term) for term in terms if term in folded]
    if positions:
        best_position, _ = max(
            positions,
            key=lambda item: sum(
                term in folded[
                    max(0, item[0] - _MCP_EXCERPT_CHARS // 2) : item[0]
                    + _MCP_EXCERPT_CHARS // 2
                ]
                for term in terms
            ),
        )
    else:
        best_position = 0

    start = max(0, best_position - _MCP_EXCERPT_CHARS // 3)
    end = min(len(compact), start + _MCP_EXCERPT_CHARS)
    start = max(0, end - _MCP_EXCERPT_CHARS)
    excerpt = compact[start:end].strip()
    matched = [term for term in terms if term in excerpt.casefold()]
    if start:
        excerpt = f"...{excerpt}"
    if end < len(compact):
        excerpt = f"{excerpt}..."
    return excerpt, matched


def _normalize_source_path(file_path: str | None) -> str:
    normalized = (file_path or "").strip()
    return normalized if normalized and normalized != "no-file-path" else "unknown_source"


async def _execute_lightrag_operation(
    operation_name: str,
    operation_func: Callable[[], Awaitable[Any]],
    *,
    tool_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        logger.info(
            "Executing LightRAG MCP operation: %s args=%s",
            operation_name,
            _summarize_tool_kwargs(tool_kwargs or {}),
        )
        result = await operation_func()
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

    if platform.system() == "Darwin":
        snapshot["macos"] = {
            "memory_pressure_q": _run_command_output(["memory_pressure", "-Q"]),
            "vm_stat": _run_command_output(["vm_stat"]),
            "swapusage": _run_command_output(["sysctl", "vm.swapusage"]),
        }

    return snapshot


@dataclass(slots=True)
class LightRAGMCPRuntime:
    """Direct in-process adapter between MCP tools and a live LightRAG instance."""

    rag_provider: Callable[[], Any]
    doc_manager: Any
    args: Any
    background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    @property
    def rag(self) -> Any:
        rag = self.rag_provider()
        if rag is None:
            raise RuntimeError("LightRAG runtime is not initialized")
        return rag

    def schedule(self, coroutine: Awaitable[Any], *, name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self.background_tasks.add(task)

        def _finished(done: asyncio.Task[Any]) -> None:
            self.background_tasks.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                logger.error(
                    "LightRAG MCP background task failed name=%s error=%s",
                    done.get_name(),
                    error,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(_finished)

    async def close(self) -> None:
        tasks = tuple(self.background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def query(
        self,
        *,
        query: str,
        mode: str,
        top_k: int,
        only_need_context: bool,
        only_need_prompt: bool,
        response_type: str,
        max_token_for_text_unit: int,
        max_token_for_global_context: int,
        max_token_for_local_context: int,
        hl_keywords: list[str],
        ll_keywords: list[str],
        history_turns: int,
    ) -> dict[str, Any]:
        from lightrag.base import QueryParam

        normalized_mode = {"semantic": "naive", "keyword": "local"}.get(mode, mode)
        param = QueryParam(
            mode=normalized_mode,
            stream=False,
            top_k=top_k,
            chunk_top_k=top_k,
            only_need_context=only_need_context,
            only_need_prompt=only_need_prompt,
            response_type=response_type,
            max_entity_tokens=max_token_for_local_context,
            max_relation_tokens=max_token_for_global_context,
            max_total_tokens=(
                max_token_for_text_unit
                + max_token_for_global_context
                + max_token_for_local_context
            ),
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            conversation_history=[],
            include_references=True,
        )
        result = await self.rag.aquery_llm(query, param=param)
        llm_response = result.get("llm_response", {})
        data = result.get("data", {})
        chunks = [
            chunk for chunk in data.get("chunks", []) if chunk.get("content")
        ][:_MCP_MATCH_LIMIT]
        chunk_rows: list[dict[str, Any] | None] = [None] * len(chunks)
        text_chunks = getattr(self.rag, "text_chunks", None)
        if chunks and text_chunks is not None:
            chunk_rows = await text_chunks.get_by_ids(
                [chunk.get("chunk_id", "") for chunk in chunks]
            )

        terms = _query_terms(query, [*hl_keywords, *ll_keywords])
        matches = []
        for chunk, stored_chunk in zip(chunks, chunk_rows, strict=True):
            excerpt, matched_terms = _matching_excerpt(chunk["content"], terms)
            matches.append(
                {
                    "content": excerpt,
                    "file_path": chunk.get("file_path", "unknown_source"),
                    "reference_id": chunk.get("reference_id", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "document_id": chunk.get("full_doc_id")
                    or (stored_chunk or {}).get("full_doc_id"),
                    "matched_terms": matched_terms,
                    "is_excerpt": len(chunk["content"].strip()) > _MCP_EXCERPT_CHARS,
                }
            )
        selected_reference_ids = {match["reference_id"] for match in matches}
        references = [
            reference
            for reference in data.get("references", [])
            if reference.get("reference_id") in selected_reference_ids
        ]
        response = llm_response.get("content")
        if only_need_context:
            response = f"Retrieved {len(matches)} bounded source excerpt(s)."
        return {
            "matches": matches,
            "response": response or "No relevant context found for the query.",
            "references": references,
            "history_turns": history_turns,
        }

    async def document_content(self, document_id: str) -> dict[str, Any]:
        from lightrag.utils_pipeline import doc_status_field

        content_data = await self.rag.full_docs.get_by_id(document_id)
        status_doc = await self.rag.doc_status.get_by_id(document_id)
        if content_data is None and status_doc is None:
            raise ValueError(f"Document not found: {document_id}")

        content = str((content_data or {}).get("content", "")).strip()
        if not content and status_doc is not None:
            chunk_ids = doc_status_field(status_doc, "chunks_list", []) or []
            chunk_rows = await self.rag.text_chunks.get_by_ids(list(chunk_ids))
            content = "\n\n".join(
                str(chunk.get("content", "")).strip()
                for chunk in sorted(
                    (chunk for chunk in chunk_rows if chunk),
                    key=lambda chunk: int(chunk.get("chunk_order_index", 0)),
                )
                if str(chunk.get("content", "")).strip()
            )
        return {
            "document_id": document_id,
            "file_path": _normalize_source_path(
                doc_status_field(status_doc, "file_path", "")
            ),
            "content": content,
            "metadata": doc_status_field(status_doc, "metadata", {}) or {},
        }

    async def insert_text(
        self, text: str | list[str], tags: list[str] | None = None
    ) -> dict[str, Any]:
        from lightrag.utils_pipeline import normalize_document_tags

        texts = [text] if isinstance(text, str) else list(text)
        if not texts or any(not item.strip() for item in texts):
            raise ValueError("text must contain at least one non-empty document")
        track_id = f"mcp-{uuid4().hex}"
        normalized_tags = normalize_document_tags(tags)
        file_paths = [f"mcp-memory/{track_id}-{index}.txt" for index in range(len(texts))]
        self.schedule(
            self.rag.ainsert(
                texts,
                file_paths=file_paths,
                track_id=track_id,
                tags=normalized_tags,
            ),
            name=f"mcp-insert-{track_id}",
        )
        return {
            "status": "success",
            "message": "Text accepted for background processing",
            "track_id": track_id,
            "tags": normalized_tags,
        }

    async def _enqueue_file(self, file_path: Path, track_id: str) -> bool:
        from lightrag.api.routers.document_routes import pipeline_enqueue_file

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not self.doc_manager.is_supported_file(file_path.name):
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        success, _ = await pipeline_enqueue_file(self.rag, file_path, track_id)
        return success

    async def insert_file(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path).expanduser().resolve()
        track_id = f"mcp-file-{uuid4().hex}"
        if not await self._enqueue_file(path, track_id):
            raise RuntimeError(f"File was not enqueued: {path.name}")
        self.schedule(
            self.rag.apipeline_process_enqueue_documents(),
            name=f"mcp-process-{track_id}",
        )
        return {"status": "success", "track_id": track_id, "file_path": str(path)}

    async def upload_file(self, file_path: str) -> dict[str, Any]:
        source = Path(file_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"File not found: {source}")
        target = self.doc_manager.input_dir / source.name
        if target.exists():
            raise FileExistsError(f"Input file already exists: {target.name}")
        await asyncio.to_thread(shutil.copy2, source, target)
        try:
            return await self.insert_file(str(target))
        except Exception:
            target.unlink(missing_ok=True)
            raise

    async def insert_batch(
        self,
        *,
        directory_path: str,
        recursive: bool,
        depth: int,
        include_only: list[str],
        ignore_files: list[str],
        ignore_directories: list[str],
    ) -> dict[str, Any]:
        if include_only and ignore_files:
            raise ValueError("include_only and ignore_files cannot both be set")
        root = Path(directory_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")
        include_patterns = [re.compile(pattern) for pattern in include_only]
        ignore_file_patterns = [re.compile(pattern) for pattern in ignore_files]
        ignore_dir_patterns = [re.compile(pattern) for pattern in ignore_directories]
        candidates: list[Path] = []
        paths = root.rglob("*") if recursive else root.iterdir()
        for path in paths:
            if not path.is_file():
                continue
            relative_parts = path.relative_to(root).parts
            if len(relative_parts) > depth + 1:
                continue
            if any(
                pattern.search(part)
                for part in relative_parts[:-1]
                for pattern in ignore_dir_patterns
            ):
                continue
            if include_patterns and not any(
                pattern.search(path.name) for pattern in include_patterns
            ):
                continue
            if any(pattern.search(path.name) for pattern in ignore_file_patterns):
                continue
            candidates.append(path)

        track_id = f"mcp-batch-{uuid4().hex}"
        failures: list[dict[str, str]] = []
        accepted = 0
        for path in candidates:
            try:
                accepted += int(await self._enqueue_file(path, track_id))
            except Exception as exc:
                failures.append({"file_path": str(path), "error": str(exc)})
        if accepted:
            self.schedule(
                self.rag.apipeline_process_enqueue_documents(),
                name=f"mcp-process-{track_id}",
            )
        return {
            "status": "success" if not failures else "partial_success",
            "track_id": track_id,
            "accepted": accepted,
            "failed": len(failures),
            "failures": failures,
        }

    async def scan(self) -> dict[str, Any]:
        track_id = f"mcp-scan-{uuid4().hex}"
        accepted = 0
        for path in self.doc_manager.iter_new_files():
            accepted += int(await self._enqueue_file(path, track_id))
        if accepted:
            self.schedule(
                self.rag.apipeline_process_enqueue_documents(),
                name=f"mcp-process-{track_id}",
            )
        return {"status": "scanning_started", "track_id": track_id, "accepted": accepted}

    async def documents(self, tags: list[str] | None = None) -> dict[str, Any]:
        from lightrag.base import DocStatus
        from lightrag.utils_pipeline import normalize_document_tags

        normalized_tags = set(normalize_document_tags(tags))
        statuses = tuple(DocStatus)
        rows = await asyncio.gather(
            *(self.rag.get_docs_by_status(status) for status in statuses)
        )
        return {
            "statuses": {
                status.value: [
                    {"id": doc_id, **(_to_jsonable(doc) or {})}
                    for doc_id, doc in result.items()
                    if not normalized_tags
                    or normalized_tags.issubset(
                        set((getattr(doc, "metadata", {}) or {}).get("tags", []))
                    )
                ]
                for status, result in zip(statuses, rows, strict=True)
            }
        }

    async def pipeline_status(self) -> dict[str, Any]:
        from lightrag.kg.shared_storage import (
            _INTERNAL_PIPELINE_STATUS_FIELDS,
            get_namespace_data,
        )

        status = (
            await get_namespace_data("pipeline_status", workspace=self.rag.workspace)
        ).copy()
        for field_name in _INTERNAL_PIPELINE_STATUS_FIELDS:
            status.pop(field_name, None)
        history = status.get("history_messages")
        if history is not None and not isinstance(history, list):
            status["history_messages"] = history[-1000:]
        return _to_jsonable(status)

    async def health(self) -> dict[str, Any]:
        status = await self.pipeline_status()
        return {
            "status": "healthy",
            "workspace": self.rag.workspace,
            "pipeline_busy": bool(status.get("busy", False)),
            "configuration": {
                "kv_storage": self.args.kv_storage,
                "doc_status_storage": self.args.doc_status_storage,
                "graph_storage": self.args.graph_storage,
                "vector_storage": self.args.vector_storage,
                "llm_model": self.args.llm_model,
                "embedding_model": self.args.embedding_model,
            },
        }


def create_lightrag_mcp(
    rag_provider: Callable[[], Any], doc_manager: Any, args: Any
) -> FastMCP:
    """Create the curated LightRAG MCP server using current FastMCP APIs."""

    runtime = LightRAGMCPRuntime(rag_provider, doc_manager, args)

    @asynccontextmanager
    async def lifespan(_server: FastMCP):
        logger.info(
            "Integrated LightRAG MCP server started mode=direct transport=%s",
            os.getenv("LIGHTRAG_MCP_TRANSPORT", "streamable-http"),
        )
        try:
            yield {}
        finally:
            await runtime.close()
            logger.info("Integrated LightRAG MCP server stopped")

    mcp = FastMCP(
        name=os.getenv("LIGHTRAG_MCP_NAME", "LightRAG"),
        lifespan=lifespan,
    )

    @mcp.tool(name="query_document", description=AGENTIC_TOOL_DESCRIPTIONS["query_document"])
    async def query_document(
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
        async def _operation() -> Any:
            return await runtime.query(
                query=query,
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
        text: str | list[str] = Field(description="Text or list of texts to insert"),
        tags: list[str] = Field(
            default_factory=list,
            description="Tags such as skill, workflow, or agentic-development",
        ),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.insert_text(text, tags)

        return await _execute_lightrag_operation(
            "insert_document",
            _operation,
            tool_kwargs={"text": text, "tags": tags},
        )

    @mcp.tool(
        name="upload_document",
        description=AGENTIC_TOOL_DESCRIPTIONS["upload_document"],
    )
    async def upload_document(
        file_path: str = Field(description="Local path to the file to upload"),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.upload_file(file_path)

        return await _execute_lightrag_operation(
            "upload_document",
            _operation,
            tool_kwargs={"file_path": file_path},
        )

    @mcp.tool(name="insert_file", description=AGENTIC_TOOL_DESCRIPTIONS["insert_file"])
    async def insert_file(
        file_path: str = Field(description="Local path to the file to insert"),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.insert_file(file_path)

        return await _execute_lightrag_operation(
            "insert_file",
            _operation,
            tool_kwargs={"file_path": file_path},
        )

    @mcp.tool(name="insert_batch", description=AGENTIC_TOOL_DESCRIPTIONS["insert_batch"])
    async def insert_batch(
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
        async def _operation() -> Any:
            return await runtime.insert_batch(
                directory_path=directory_path,
                recursive=recursive,
                depth=depth,
                include_only=include_only,
                ignore_directories=ignore_directories,
                ignore_files=ignore_files,
            )

        return await _execute_lightrag_operation(
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
    async def scan_for_new_documents() -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.scan()

        return await _execute_lightrag_operation(
            "scan_for_new_documents", _operation
        )

    @mcp.tool(name="get_documents", description=AGENTIC_TOOL_DESCRIPTIONS["get_documents"])
    async def get_documents(
        tags: list[str] = Field(
            default_factory=list,
            description="Return only documents containing every requested tag",
        ),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.documents(tags)

        return await _execute_lightrag_operation(
            "get_documents", _operation, tool_kwargs={"tags": tags}
        )

    @mcp.tool(
        name="get_document_content",
        description=AGENTIC_TOOL_DESCRIPTIONS["get_document_content"],
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
        },
    )
    async def get_document_content(
        document_id: str = Field(
            description="Document ID returned by query_document"
        ),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.document_content(document_id)

        return await _execute_lightrag_operation(
            "get_document_content",
            _operation,
            tool_kwargs={"document_id": document_id},
        )

    @mcp.tool(
        name="get_pipeline_status",
        description=AGENTIC_TOOL_DESCRIPTIONS["get_pipeline_status"],
    )
    async def get_pipeline_status() -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.pipeline_status()

        return await _execute_lightrag_operation(
            "get_pipeline_status", _operation
        )

    @mcp.tool(
        name="get_graph_labels",
        description=AGENTIC_TOOL_DESCRIPTIONS["get_graph_labels"],
    )
    async def get_graph_labels() -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.rag.get_graph_labels()

        return await _execute_lightrag_operation("get_graph_labels", _operation)

    @mcp.tool(
        name="check_lightrag_health",
        description=AGENTIC_TOOL_DESCRIPTIONS["check_lightrag_health"],
    )
    async def check_lightrag_health() -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.health()

        return await _execute_lightrag_operation(
            "check_lightrag_health", _operation
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
        source_entities: list[str] = Field(description="Entity names to merge"),
        target_entity: str = Field(description="Target entity name"),
        merge_strategy: dict[str, str] = Field(
            default_factory=dict,
            description="Property merge strategy by field name",
        ),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            from lightrag.api.routers.document_routes import check_pipeline_busy_or_raise

            await check_pipeline_busy_or_raise(runtime.rag)
            return await runtime.rag.amerge_entities(
                source_entities, target_entity, merge_strategy
            )

        return await _execute_lightrag_operation(
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
        entities: list[dict[str, Any]] = Field(
            description="Entities with entity_name, entity_type, description, source_id"
        ),
    ) -> dict[str, Any]:
        async def _create_entity(data: dict[str, Any]) -> dict[str, Any]:
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
                result = await runtime.rag.acreate_entity(
                    str(entity_name),
                    {
                        "entity_type": str(entity_type),
                        "description": str(description),
                        "source_id": str(source_id),
                    },
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

        async def _operation() -> Any:
            from lightrag.api.routers.document_routes import check_pipeline_busy_or_raise

            await check_pipeline_busy_or_raise(runtime.rag)
            results = await asyncio.gather(
                *(_create_entity(entity) for entity in entities)
            )
            return {
                "total": len(entities),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            "create_entities",
            _operation,
            tool_kwargs={"entities": entities},
        )

    @mcp.tool(
        name="delete_by_entities",
        description=AGENTIC_TOOL_DESCRIPTIONS["delete_by_entities"],
    )
    async def delete_by_entities(
        entity_names: list[str] = Field(description="Entity names to delete"),
    ) -> dict[str, Any]:
        async def _delete_entity(entity_name: str) -> dict[str, Any]:
            try:
                result = await runtime.rag.adelete_by_entity(entity_name)
                return {
                    "entity_name": entity_name,
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {"entity_name": entity_name, "status": "error", "error": str(exc)}

        async def _operation() -> Any:
            from lightrag.api.routers.document_routes import check_pipeline_busy_or_raise

            await check_pipeline_busy_or_raise(runtime.rag)
            results = await asyncio.gather(
                *(_delete_entity(entity_name) for entity_name in entity_names)
            )
            return {
                "total": len(entity_names),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            "delete_by_entities",
            _operation,
            tool_kwargs={"entity_names": entity_names},
        )

    @mcp.tool(
        name="delete_by_doc_ids",
        description=AGENTIC_TOOL_DESCRIPTIONS["delete_by_doc_ids"],
    )
    async def delete_by_doc_ids(
        doc_ids: list[str] = Field(description="Document IDs to delete"),
    ) -> dict[str, Any]:
        async def _delete_by_doc_id(doc_id: str) -> dict[str, Any]:
            try:
                result = await runtime.rag.adelete_by_doc_id(doc_id)
                return {"doc_id": doc_id, "status": "success", "result": _to_jsonable(result)}
            except Exception as exc:
                return {"doc_id": doc_id, "status": "error", "error": str(exc)}

        async def _operation() -> Any:
            results = await asyncio.gather(
                *(_delete_by_doc_id(doc_id) for doc_id in doc_ids)
            )
            return {
                "total": len(doc_ids),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            "delete_by_doc_ids",
            _operation,
            tool_kwargs={"doc_ids": doc_ids},
        )

    @mcp.tool(name="edit_entities", description=AGENTIC_TOOL_DESCRIPTIONS["edit_entities"])
    async def edit_entities(
        entities: list[dict[str, Any]] = Field(
            description="Entities with entity_name, entity_type, description, source_id"
        ),
    ) -> dict[str, Any]:
        async def _edit_entity(data: dict[str, Any]) -> dict[str, Any]:
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
                result = await runtime.rag.aedit_entity(
                    str(entity_name),
                    {
                        "entity_type": str(entity_type),
                        "description": str(description),
                        "source_id": str(source_id),
                    },
                    allow_rename=False,
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

        async def _operation() -> Any:
            from lightrag.api.routers.document_routes import check_pipeline_busy_or_raise

            await check_pipeline_busy_or_raise(runtime.rag)
            results = await asyncio.gather(
                *(_edit_entity(entity) for entity in entities)
            )
            return {
                "total": len(entities),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            "edit_entities",
            _operation,
            tool_kwargs={"entities": entities},
        )

    @mcp.tool(
        name="create_relations",
        description=AGENTIC_TOOL_DESCRIPTIONS["create_relations"],
    )
    async def create_relations(
        relations: list[dict[str, Any]] = Field(
            description="Relations with source, target, description, keywords, optional source_id and weight"
        ),
    ) -> dict[str, Any]:
        async def _create_relation(data: dict[str, Any]) -> dict[str, Any]:
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
                relation_data: dict[str, Any] = {
                    "description": str(description),
                    "keywords": str(keywords),
                }
                if source_id:
                    relation_data["source_id"] = str(source_id)
                if weight is not None:
                    relation_data["weight"] = float(weight)
                result = await runtime.rag.acreate_relation(
                    str(source), str(target), relation_data
                )
                return {
                    "relation": label,
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {"relation": label, "status": "error", "error": str(exc)}

        async def _operation() -> Any:
            from lightrag.api.routers.document_routes import check_pipeline_busy_or_raise

            await check_pipeline_busy_or_raise(runtime.rag)
            results = await asyncio.gather(
                *(_create_relation(relation) for relation in relations)
            )
            return {
                "total": len(relations),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            "create_relations",
            _operation,
            tool_kwargs={"relations": relations},
        )

    @mcp.tool(name="edit_relations", description=AGENTIC_TOOL_DESCRIPTIONS["edit_relations"])
    async def edit_relations(
        relations: list[dict[str, Any]] = Field(
            description="Relations with source, target, description, keywords, relation_type, optional source_id and weight"
        ),
    ) -> dict[str, Any]:
        async def _edit_relation(data: dict[str, Any]) -> dict[str, Any]:
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
                updated_data: dict[str, Any] = {
                    "description": str(description),
                    "keywords": str(keywords),
                    "relation_type": str(relation_type),
                }
                if source_id:
                    updated_data["source_id"] = str(source_id)
                if weight is not None:
                    updated_data["weight"] = float(weight)
                result = await runtime.rag.aedit_relation(
                    str(source), str(target), updated_data
                )
                return {
                    "relation": label,
                    "status": "success",
                    "result": _to_jsonable(result),
                }
            except Exception as exc:
                return {"relation": label, "status": "error", "error": str(exc)}

        async def _operation() -> Any:
            from lightrag.api.routers.document_routes import check_pipeline_busy_or_raise

            await check_pipeline_busy_or_raise(runtime.rag)
            results = await asyncio.gather(
                *(_edit_relation(relation) for relation in relations)
            )
            return {
                "total": len(relations),
                "successful": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "error"),
                "results": results,
            }

        return await _execute_lightrag_operation(
            "edit_relations",
            _operation,
            tool_kwargs={"relations": relations},
        )

    return mcp


def create_lightrag_mcp_http_app(
    rag_provider: Callable[[], Any], doc_manager: Any, args: Any
) -> Any:
    mcp = create_lightrag_mcp(rag_provider, doc_manager, args)
    return mcp.http_app(
        path="/",
        transport=cast(
            Literal["http", "streamable-http", "sse"],
            os.getenv("LIGHTRAG_MCP_TRANSPORT", "streamable-http"),
        ),
        json_response=_env_bool("LIGHTRAG_MCP_JSON_RESPONSE", False),
        stateless_http=_env_bool("LIGHTRAG_MCP_STATELESS_HTTP", False),
    )


def create_chatgpt_mcp_http_app(
    rag_provider: Callable[[], Any], doc_manager: Any, args: Any, base_url: str
) -> Any:
    """Create the OAuth-protected, least-privilege ChatGPT memory endpoint."""

    from lightrag.api.auth import auth_handler
    from lightrag.api.chatgpt_oauth import (
        READ_SCOPE,
        WRITE_SCOPE,
        LightRAGChatGPTOAuthProvider,
    )

    provider = LightRAGChatGPTOAuthProvider(
        base_url=base_url,
        secret=auth_handler.secret,
        password_verifier=auth_handler.verify_password,
        algorithm=auth_handler.algorithm,
    )
    runtime = LightRAGMCPRuntime(rag_provider, doc_manager, args)

    @asynccontextmanager
    async def lifespan(_server: FastMCP):
        try:
            yield {}
        finally:
            await runtime.close()

    mcp = FastMCP(
        name="LightRAG Memory",
        instructions=(
            "Search memory when prior context may help. Cite each factual claim with "
            "the source returned by search_memory. Save memory only when the user asks "
            "you to remember something or when a durable preference/decision is explicit. "
            "Expand a full source only after its excerpt is relevant."
        ),
        auth=provider,
        lifespan=lifespan,
    )

    @mcp.tool(
        name="search_memory",
        title="Search LightRAG memory",
        description=(
            "Search prior knowledge and chat memory. Returns compact relevant excerpts, "
            "document IDs, and source references suitable for citation."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
        },
        meta={"securitySchemes": [{"type": "oauth2", "scopes": [READ_SCOPE]}]},
    )
    async def search_memory(
        query: str = Field(description="What to recall from LightRAG memory"),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.query(
                query=query,
                mode="mix",
                top_k=12,
                only_need_context=True,
                only_need_prompt=False,
                response_type="Multiple Paragraphs",
                max_token_for_text_unit=2048,
                max_token_for_global_context=2048,
                max_token_for_local_context=2048,
                hl_keywords=[],
                ll_keywords=[],
                history_turns=0,
            )

        return await _execute_lightrag_operation(
            "chatgpt_search_memory", _operation, tool_kwargs={"query": query}
        )

    @mcp.tool(
        name="get_memory_source",
        title="Read a LightRAG memory source",
        description=(
            "Read one full source selected from search_memory by document_id. "
            "Use only when the compact excerpt is relevant but insufficient."
        ),
        annotations={
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
        },
        meta={"securitySchemes": [{"type": "oauth2", "scopes": [READ_SCOPE]}]},
    )
    async def get_memory_source(
        document_id: str = Field(description="Document ID from search_memory"),
    ) -> dict[str, Any]:
        async def _operation() -> Any:
            return await runtime.document_content(document_id)

        return await _execute_lightrag_operation(
            "chatgpt_get_memory_source",
            _operation,
            tool_kwargs={"document_id": document_id},
        )

    @mcp.tool(
        name="save_memory",
        title="Save knowledge to LightRAG",
        description=(
            "Save durable knowledge, a user preference, or a decision for future chats. "
            "Use only with clear user intent; this changes persistent memory."
        ),
        annotations={
            "readOnlyHint": False,
            "destructiveHint": False,
            "openWorldHint": False,
        },
        meta={
            "securitySchemes": [
                {"type": "oauth2", "scopes": [READ_SCOPE, WRITE_SCOPE]}
            ]
        },
    )
    async def save_memory(
        text: str = Field(description="Concise knowledge to remember"),
        tags: list[str] = Field(
            default_factory=list, description="Optional retrieval tags"
        ),
    ) -> dict[str, Any]:
        access_token = get_access_token()
        if access_token is None or WRITE_SCOPE not in access_token.scopes:
            raise PermissionError("memory:write scope is required")
        username = str(access_token.claims.get("sub", "unknown"))

        async def _operation() -> Any:
            return await runtime.insert_text(
                f"Memory owner: {username}\n{text}", ["chatgpt-memory", *tags]
            )

        return await _execute_lightrag_operation(
            "chatgpt_save_memory",
            _operation,
            tool_kwargs={"text": text, "tags": tags},
        )

    return mcp.http_app(
        path="/mcp",
        transport="streamable-http",
        json_response=True,
        stateless_http=True,
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


class _MCPAPIKeyMiddleware:
    """Require the REST API key before exposing MCP management tools."""

    def __init__(self, app: ASGIApp, api_key: str | None):
        self.app = app
        self.api_key = api_key or ""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.api_key and scope["type"] == "http":
            supplied_key = Headers(scope=scope).get("x-api-key", "")
            if not secrets.compare_digest(supplied_key, self.api_key):
                response = JSONResponse(
                    {"detail": "Invalid or missing API key"}, status_code=403
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def mount_lightrag_mcp_http_app(
    app: Any,
    mcp_http_app: ASGIApp,
    mount_path: str,
    api_key: str | None = None,
) -> None:
    """Mount FastMCP at /mcp and /mcp/ without POST slash redirects.

    Starlette redirects exact POST /mcp to /mcp/ for mounted apps. Some MCP
    bridges do not preserve streamable-http session state across that redirect,
    leaving tools stuck even though the operation completes server-side.
    """

    normalized_path = mount_path.rstrip("/") or "/mcp"
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    protected_mcp_app = _MCPAPIKeyMiddleware(mcp_http_app, api_key)
    app.router.routes.insert(
        0,
        Route(
            normalized_path,
            endpoint=_MCPMountRootEndpoint(protected_mcp_app, normalized_path),
            methods=["GET", "POST", "DELETE"],
            include_in_schema=False,
        ),
    )
    app.mount(normalized_path, protected_mcp_app)
