from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.constants import DEFAULT_LOG_FILENAME


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runtime_log_dir() -> Path:
    override = os.getenv("LIGHTRAG_RUNTIME_DATA_DIR", "").strip()
    data_dir = Path(override).expanduser() if override else (_repo_root() / "data")
    return data_dir / "logs"


def _desktop_log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "lightrag"


def _main_log_file() -> Path:
    log_dir = os.getenv("LOG_DIR", "").strip()
    if log_dir:
        return Path(log_dir).expanduser().resolve() / DEFAULT_LOG_FILENAME
    return _runtime_log_dir() / DEFAULT_LOG_FILENAME


class LogFileInfo(BaseModel):
    id: str = Field(description="Stable log file identifier used by the UI")
    label: str = Field(description="User-facing log file label")
    path: str = Field(description="Absolute path to the log file")
    size_bytes: int = Field(description="Current log file size in bytes")
    modified_at: Optional[float] = Field(
        default=None, description="Last modification time as UNIX timestamp"
    )
    is_default: bool = Field(
        default=False, description="Whether this file is the default UI selection"
    )


class LogFilesResponse(BaseModel):
    default_file_id: Optional[str] = Field(
        default=None, description="Default log file id for initial selection"
    )
    files: list[LogFileInfo] = Field(description="Discovered log files")


def _format_combined_line(label: str, line: str) -> str:
    return f"[{label}] {line}"


def _discover_log_files() -> list[tuple[str, str, Path]]:
    discovered: list[tuple[str, str, Path]] = []

    main_log = _main_log_file()
    discovered.append(("server", "LightRAG Server", main_log))

    runtime_log_dir = _runtime_log_dir()
    if runtime_log_dir.exists():
        for path in sorted(runtime_log_dir.glob("*.log")):
            discovered.append((f"runtime:{path.name}", path.name, path.resolve()))

    desktop_log_dir = _desktop_log_dir()
    if desktop_log_dir.exists():
        for path in sorted(desktop_log_dir.glob("*.log")):
            discovered.append((f"desktop:{path.name}", f"desktop/{path.name}", path.resolve()))

    return discovered


def _available_log_map() -> dict[str, tuple[str, Path]]:
    log_map: dict[str, tuple[str, Path]] = {}
    for file_id, label, path in _discover_log_files():
        if path.exists() and path.is_file():
            log_map[file_id] = (label, path)
    return log_map


def _combined_snapshot(
    sources: list[tuple[str, str, Path]], max_lines: int
) -> tuple[list[str], bool]:
    combined_lines: list[str] = []
    truncated = False
    for _, label, path in sources:
        lines, source_truncated = _read_tail_lines(path, max_lines)
        if lines:
            combined_lines.extend(_format_combined_line(label, line) for line in lines)
        truncated = truncated or source_truncated
    return combined_lines, truncated


def _read_tail_lines(path: Path, max_lines: int) -> tuple[list[str], bool]:
    if max_lines <= 0:
        return [], False

    tail_buffer: deque[str] = deque(maxlen=max_lines)
    total_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            total_lines += 1
            tail_buffer.append(line.rstrip("\n"))
    return list(tail_buffer), total_lines > max_lines


def _serialize_event(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def create_logs_routes(api_key: Optional[str] = None):
    router = APIRouter(tags=["logs"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/logs/files",
        response_model=LogFilesResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def list_log_files():
        log_map = _available_log_map()
        files: list[LogFileInfo] = []

        if len(log_map) > 1:
            latest_mtime = max(
                path.stat().st_mtime for _, path in log_map.values()
            )
            combined_size = sum(path.stat().st_size for _, path in log_map.values())
            files.append(
                LogFileInfo(
                    id="combined",
                    label="Combined",
                    path="multiple sources",
                    size_bytes=combined_size,
                    modified_at=latest_mtime,
                    is_default=True,
                )
            )

        for index, (file_id, (label, path)) in enumerate(log_map.items()):
            stat = path.stat()
            files.append(
                LogFileInfo(
                    id=file_id,
                    label=label,
                    path=str(path),
                    size_bytes=stat.st_size,
                    modified_at=stat.st_mtime,
                    is_default=not files and index == 0,
                )
            )

        default_file_id = files[0].id if files else None
        return LogFilesResponse(default_file_id=default_file_id, files=files)

    @router.get(
        "/logs/stream",
        dependencies=[Depends(combined_auth)],
        responses={
            200: {
                "description": "Realtime NDJSON stream of backend log events",
                "content": {
                    "application/x-ndjson": {
                        "schema": {
                            "type": "string",
                            "format": "ndjson",
                        }
                    }
                },
            }
        },
    )
    async def stream_logs(
        file_id: str = Query(..., description="Log file id from /logs/files"),
        tail_lines: int = Query(
            200, ge=0, le=2000, description="How many trailing lines to send first"
        ),
        poll_interval: float = Query(
            0.75, ge=0.2, le=5.0, description="Filesystem polling interval in seconds"
        ),
    ):
        log_map = _available_log_map()
        if file_id == "combined":
            combined_sources = [
                (source_id, label, path)
                for source_id, (label, path) in log_map.items()
            ]
            if not combined_sources:
                raise HTTPException(status_code=404, detail="Log file not found")
        elif file_id not in log_map:
            raise HTTPException(status_code=404, detail="Log file not found")
        else:
            label, path = log_map[file_id]
            if not path.exists():
                raise HTTPException(status_code=404, detail="Log file does not exist")

        async def event_stream():
            try:
                if file_id == "combined":
                    snapshot_lines, truncated = _combined_snapshot(
                        combined_sources, tail_lines
                    )
                    yield _serialize_event(
                        {
                            "type": "snapshot",
                            "file_id": file_id,
                            "label": "Combined",
                            "path": "multiple sources",
                            "lines": snapshot_lines,
                            "truncated": truncated,
                        }
                    )

                    state: dict[str, dict[str, object]] = {}
                    for source_id, source_label, source_path in combined_sources:
                        stat = source_path.stat()
                        state[source_id] = {
                            "label": source_label,
                            "path": source_path,
                            "last_stat": stat,
                            "position": stat.st_size,
                            "partial_line": "",
                        }
                else:
                    snapshot_lines, truncated = _read_tail_lines(path, tail_lines)
                    yield _serialize_event(
                        {
                            "type": "snapshot",
                            "file_id": file_id,
                            "label": label,
                            "path": str(path),
                            "lines": snapshot_lines,
                            "truncated": truncated,
                        }
                    )

                    last_stat = path.stat()
                    position = last_stat.st_size
                    partial_line = ""

                while True:
                    await asyncio.sleep(poll_interval)

                    if file_id == "combined":
                        for source_id, source_state in state.items():
                            source_label = str(source_state["label"])
                            source_path = source_state["path"]
                            previous_stat = source_state["last_stat"]
                            position = int(source_state["position"])
                            partial_line = str(source_state["partial_line"])

                            if not source_path.exists():
                                continue

                            current_stat = source_path.stat()
                            rotated = (
                                getattr(current_stat, "st_ino", None)
                                != getattr(previous_stat, "st_ino", None)
                            )
                            truncated_file = current_stat.st_size < position

                            if rotated or truncated_file:
                                snapshot_lines, truncated = _read_tail_lines(
                                    source_path, tail_lines
                                )
                                yield _serialize_event(
                                    {
                                        "type": "reset",
                                        "file_id": file_id,
                                        "label": "Combined",
                                        "path": "multiple sources",
                                        "lines": [
                                            _format_combined_line(source_label, line)
                                            for line in snapshot_lines
                                        ],
                                        "truncated": truncated,
                                        "reason": "rotated"
                                        if rotated
                                        else "truncated",
                                        "source_file_id": source_id,
                                    }
                                )
                                source_state["last_stat"] = current_stat
                                source_state["position"] = current_stat.st_size
                                source_state["partial_line"] = ""
                                continue

                            if current_stat.st_size == position:
                                continue

                            with source_path.open(
                                "r", encoding="utf-8", errors="replace"
                            ) as handle:
                                handle.seek(position)
                                chunk = handle.read()
                                position = handle.tell()

                            if not chunk:
                                source_state["last_stat"] = current_stat
                                source_state["position"] = position
                                continue

                            combined = partial_line + chunk
                            raw_lines = combined.split("\n")
                            partial_line = raw_lines.pop()
                            appended_lines = [line.rstrip("\r") for line in raw_lines]

                            if appended_lines:
                                yield _serialize_event(
                                    {
                                        "type": "append",
                                        "file_id": file_id,
                                        "lines": [
                                            _format_combined_line(source_label, line)
                                            for line in appended_lines
                                        ],
                                        "source_file_id": source_id,
                                    }
                                )

                            source_state["last_stat"] = current_stat
                            source_state["position"] = position
                            source_state["partial_line"] = partial_line
                    else:
                        if not path.exists():
                            yield _serialize_event(
                                {
                                    "type": "error",
                                    "file_id": file_id,
                                    "message": "Log file disappeared while streaming.",
                                }
                            )
                            return

                        current_stat = path.stat()
                        rotated = (
                            getattr(current_stat, "st_ino", None)
                            != getattr(last_stat, "st_ino", None)
                        )
                        truncated_file = current_stat.st_size < position

                        if rotated or truncated_file:
                            snapshot_lines, truncated = _read_tail_lines(
                                path, tail_lines
                            )
                            yield _serialize_event(
                                {
                                    "type": "reset",
                                    "file_id": file_id,
                                    "label": label,
                                    "path": str(path),
                                    "lines": snapshot_lines,
                                    "truncated": truncated,
                                    "reason": "rotated" if rotated else "truncated",
                                }
                            )
                            last_stat = current_stat
                            position = current_stat.st_size
                            partial_line = ""
                            continue

                        if current_stat.st_size == position:
                            continue

                        with path.open(
                            "r", encoding="utf-8", errors="replace"
                        ) as handle:
                            handle.seek(position)
                            chunk = handle.read()
                            position = handle.tell()

                        if not chunk:
                            last_stat = current_stat
                            continue

                        combined = partial_line + chunk
                        raw_lines = combined.split("\n")
                        partial_line = raw_lines.pop()
                        appended_lines = [line.rstrip("\r") for line in raw_lines]

                        if appended_lines:
                            yield _serialize_event(
                                {
                                    "type": "append",
                                    "file_id": file_id,
                                    "lines": appended_lines,
                                }
                            )

                        last_stat = current_stat
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                yield _serialize_event(
                    {
                        "type": "error",
                        "file_id": file_id,
                        "message": str(exc),
                    }
                )

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/x-ndjson",
                "X-Accel-Buffering": "no",
            },
        )

    return router
