from __future__ import annotations

import asyncio
import time
from typing import Any

from lightrag.base import DocStatus


async def persist_processing_stage(
    rag: Any,
    *,
    doc_id: str,
    status_doc: Any,
    file_path: str,
    chunks: dict[str, Any],
    extraction_meta: dict[str, Any],
    pipeline_status: dict[str, Any],
    pipeline_status_lock: Any,
) -> int:
    """Persist stage-1 document state before KG extraction begins."""
    process_start_time = int(time.time())

    await rag._raise_if_cancelled(pipeline_status, pipeline_status_lock)

    await asyncio.gather(
        rag._upsert_doc_status_transition(
            doc_id=doc_id,
            status=DocStatus.PROCESSING,
            status_doc=status_doc,
            file_path=file_path,
            extra_fields={
                "chunks_count": len(chunks),
                "chunks_list": list(chunks.keys()),
            },
            metadata_extra={
                "process_start_time": process_start_time,
                **extraction_meta,
            },
        ),
        rag.chunks_vdb.upsert(chunks),
        rag.text_chunks.upsert(chunks),
    )

    return process_start_time
