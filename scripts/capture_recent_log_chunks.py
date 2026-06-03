from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from lightrag.config import settings


DEFAULT_LOG_PATH = Path("/Users/max/Library/Logs/lightrag/lightrag.err.log")
DEFAULT_OUTPUT_PATH = Path("evals/captured/recent_log_chunks.jsonl")

USER_MESSAGE_START = re.compile(
    r"^INFO: MLX_FULL_MESSAGE .* idx=1 role=user chars=\d+\s*$"
)
CHUNK_RESULT = re.compile(
    r"^INFO: Chunk \d+ of \d+ extracted (?P<entities>\d+) Ent \+ (?P<relations>\d+) Rel (?P<chunk_key>chunk-[0-9a-f]+)\s*$"
)
FILE_START = re.compile(r"^INFO: Extracting stage \d+/\d+: (?P<file_path>.+)\s*$")


def _extract_document_payload(block: str) -> str:
    if "<Document Payload>" not in block or "<Output>" not in block:
        return ""
    payload = block.split("<Document Payload>", 1)[1].split("<Output>", 1)[0]
    payload = payload.strip()
    if payload.startswith("<Current Document Context>"):
        payload = payload.split("</Current Document Context>", 1)[-1].strip()
    if payload.startswith("<Current Chunk>"):
        payload = payload[len("<Current Chunk>") :].strip()
    if payload.endswith("</Current Chunk>"):
        payload = payload[: -len("</Current Chunk>")].strip()
    return payload.strip()


def _load_existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = item.get("capture_key")
        if key:
            keys.add(str(key))
    return keys


def _make_capture_key(focus: str, payload: str) -> str:
    digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{focus}:{digest}"


def _iter_recent_user_blocks(lines: list[str], max_blocks: int) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if USER_MESSAGE_START.match(lines[i]):
            start_index = i
            collected: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("INFO: MLX_FULL_MESSAGES_END"):
                collected.append(lines[i])
                i += 1
            block = "\n".join(collected).strip()
            if block:
                blocks.append((start_index, block))
        else:
            i += 1
    return blocks[-max_blocks:]


def _nearest_context(lines: list[str], start_index: int) -> tuple[str | None, str | None]:
    chunk_key = None
    file_path = None
    for j in range(start_index, min(len(lines), start_index + 120)):
        chunk_match = CHUNK_RESULT.match(lines[j])
        if chunk_match:
            chunk_key = chunk_match.group("chunk_key")
            break
    for j in range(start_index, max(-1, start_index - 120), -1):
        file_match = FILE_START.match(lines[j])
        if file_match:
            file_path = file_match.group("file_path")
            break
    return chunk_key, file_path


def main() -> int:
    log_path = Path(settings.lightrag_log_path or str(DEFAULT_LOG_PATH))
    output_path = Path(
        settings.lightrag_promptfoo_log_capture_file or str(DEFAULT_OUTPUT_PATH)
    )
    max_blocks = settings.lightrag_promptfoo_log_capture_max_blocks

    if not log_path.exists():
        print(f"log file not found: {log_path}")
        return 1

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    existing_keys = _load_existing_keys(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    captured = 0
    blocks = _iter_recent_user_blocks(lines, max_blocks)
    with output_path.open("a", encoding="utf-8") as handle:
        for start_index, block in blocks:
            payload = _extract_document_payload(block)
            if not payload:
                continue
            chunk_key, file_path = _nearest_context(lines, start_index)
            for focus in ("entity", "relation"):
                capture_key = _make_capture_key(focus, payload)
                if capture_key in existing_keys:
                    continue
                record = {
                    "capture_key": capture_key,
                    "description": f"recent log chunk {focus} {chunk_key or capture_key[-12:]}",
                    "chunk_key": chunk_key,
                    "file_path": file_path,
                    "warning_classes": [],
                    "input_text": payload,
                    "metadata": {
                        "source": "recent-log-chunk",
                        "extraction_focus": focus,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_keys.add(capture_key)
                captured += 1

    print(f"captured {captured} recent log chunk cases into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
