import re

from lightrag.base import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.config import settings
from lightrag.prompt import PROMPTS
from lightrag.utils import Tokenizer, logger, remove_think_tags


from typing import Any, AsyncIterator, Callable

_APFEL_REFERENCES_RE = re.compile(
    r"(?:\n|^)\s*(?:#{1,6}\s*)?References\b[\s\S]*\Z",
    re.IGNORECASE | re.MULTILINE,
)

_APFEL_ANSWER_DELTA_RE = re.compile(
    r"###\s*Answer\s+Delta\s*(.*?)(?=###\s*Carry\s+Summary|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_APFEL_CARRY_SUMMARY_RE = re.compile(
    r"###\s*Carry\s+Summary\s*(.*)\Z",
    re.IGNORECASE | re.DOTALL,
)



def _format_apfel_item(kind: str, item: dict[str, Any]) -> tuple[str, set[str]]:
    reference_ids: set[str] = set()

    if kind == "chunk":
        ref_id = _format_apfel_reference_id(item.get("reference_id"))
        if ref_id:
            reference_ids.add(ref_id)
        source = str(item.get("file_path") or "unknown_source").strip()
        chunk_id = str(item.get("chunk_id") or "").strip()
        content = str(item.get("content") or "").strip()
        header = f"CHUNK ref=[{ref_id or '?'}] source={source}"
        if chunk_id:
            header = f"{header} chunk_id={chunk_id}"
        return f"{header}\n{content}", reference_ids

    if kind == "relationship":
        source_ids = str(item.get("source_id") or "").strip()
        for source_id in source_ids.split(GRAPH_FIELD_SEP):
            source_id = source_id.strip()
            if source_id:
                reference_ids.add(source_id)
        src = str(item.get("src_id") or "").strip()
        tgt = str(item.get("tgt_id") or "").strip()
        keywords = str(item.get("keywords") or "").strip()
        description = str(item.get("description") or "").strip()
        file_path = str(item.get("file_path") or "").strip()
        return (
            f"RELATION {src} -> {tgt}"
            f"\nkeywords={keywords}"
            f"\nsource_ids={source_ids}"
            f"\nfile_path={file_path}"
            f"\ndescription={description}"
        ), reference_ids

    source_ids = str(item.get("source_id") or "").strip()
    for source_id in source_ids.split(GRAPH_FIELD_SEP):
        source_id = source_id.strip()
        if source_id:
            reference_ids.add(source_id)
    name = str(item.get("entity_name") or "").strip()
    entity_type = str(item.get("entity_type") or "").strip()
    description = str(item.get("description") or "").strip()
    file_path = str(item.get("file_path") or "").strip()
    return (
        f"ENTITY {name}"
        f"\ntype={entity_type}"
        f"\nsource_ids={source_ids}"
        f"\nfile_path={file_path}"
        f"\ndescription={description}"
    ), reference_ids


def _new_apfel_portion() -> dict[str, Any]:
    return {
        "items": [],
        "token_estimate": 0,
        "reference_ids": set(),
        "counts": {"chunks": 0, "relationships": 0, "entities": 0},
    }


def _finalize_apfel_portion(
    portion: dict[str, Any], references: list[dict[str, Any]]
) -> dict[str, Any]:
    reference_ids = portion["reference_ids"]
    sections: list[str] = []
    grouped: dict[str, list[str]] = {
        "chunks": [],
        "relationships": [],
        "entities": [],
    }

    for item in portion["items"]:
        grouped[item["group"]].append(item["text"])

    for group, title in (
        ("chunks", "Document Chunks"),
        ("relationships", "Knowledge Graph Relationships"),
        ("entities", "Knowledge Graph Entities"),
    ):
        if grouped[group]:
            sections.append(f"{title}:\n\n" + "\n\n---\n\n".join(grouped[group]))

    reference_lines = _build_apfel_reference_lines(references, reference_ids)
    if reference_lines:
        sections.append("Reference Document List:\n" + "\n".join(reference_lines))

    return {
        "context": "\n\n".join(sections).strip(),
        "token_estimate": portion["token_estimate"],
        "counts": portion["counts"],
        "reference_ids": sorted(reference_ids),
    }


def _clip_text_by_tokens(tokenizer: Tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""

    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    clipped = tokenizer.decode(tokens[:max_tokens]).rstrip()
    return f"{clipped}\n[truncated]"


def _count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text or ""))


def _build_apfel_iterative_portions(
    raw_data: dict[str, Any],
    tokenizer: Tokenizer,
    max_total_tokens: int,
) -> list[dict[str, Any]]:
    """Pack retrieved query data into Apfel-sized source-bearing portions."""

    data = raw_data.get("data", {}) or {}
    references = data.get("references", []) or []
    source_items: list[tuple[str, dict[str, Any]]] = []

    for chunk in data.get("chunks", []) or []:
        source_items.append(("chunk", chunk))
    for relation in data.get("relationships", []) or []:
        source_items.append(("relationship", relation))
    for entity in data.get("entities", []) or []:
        source_items.append(("entity", entity))

    if not source_items:
        return []

    item_budget = max(192, int(max_total_tokens * 0.65))
    portion_budget = max(512, int(max_total_tokens * 0.72))
    portions: list[dict[str, Any]] = []
    current = _new_apfel_portion()

    for kind, source_item in source_items:
        text, reference_ids = _format_apfel_item(kind, source_item)
        if _count_tokens(tokenizer, text) > item_budget:
            text = _clip_text_by_tokens(tokenizer, text, item_budget)
        item_tokens = max(1, _count_tokens(tokenizer, text))

        if current["items"] and current["token_estimate"] + item_tokens > portion_budget:
            portions.append(_finalize_apfel_portion(current, references))
            current = _new_apfel_portion()

        group = {
            "chunk": "chunks",
            "relationship": "relationships",
        }.get(kind, "entities")
        current["items"].append({"group": group, "text": text})
        current["token_estimate"] += item_tokens
        current["reference_ids"].update(reference_ids)
        current["counts"][group] += 1

    if current["items"]:
        portions.append(_finalize_apfel_portion(current, references))

    return portions




def _strip_apfel_section_noise(text: str) -> str:
    text = remove_think_tags(text or "").strip()
    text = _APFEL_REFERENCES_RE.sub("", text).strip()
    return text


def _parse_apfel_iterative_response(text: str) -> tuple[str, str]:
    text = _strip_apfel_section_noise(text)
    answer_match = _APFEL_ANSWER_DELTA_RE.search(text)
    carry_match = _APFEL_CARRY_SUMMARY_RE.search(text)

    answer = answer_match.group(1).strip() if answer_match else text.strip()
    carry = carry_match.group(1).strip() if carry_match else ""

    answer = _strip_apfel_section_noise(answer)
    carry = _strip_apfel_section_noise(carry)

    normalized_answer = re.sub(r"[\W_]+", " ", answer.lower()).strip()
    if normalized_answer in {"none", "n a", "empty"} or (
        len(normalized_answer.split()) <= 8
        and normalized_answer.startswith(("nothing new", "no new", "no useful"))
    ):
        answer = ""
    return answer, carry


def _apfel_iterative_metadata(portions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "iterative": True,
        "portion_count": len(portions),
        "current_portion": 0,
        "token_estimates": [
            portion.get("token_estimate", 0) for portion in portions
        ],
        "counts": [portion.get("counts", {}) for portion in portions],
    }


def _make_apfel_iterative_user_prompt(
    *,
    query: str,
    response_type: str,
    carry_summary: str,
    portion: dict[str, Any],
    portion_index: int,
    portion_count: int,
) -> str:
    return PROMPTS["apfel_iterative_rag_user"].format(
        query=query,
        response_type=response_type,
        carry_summary=carry_summary or "None yet.",
        portion_index=portion_index,
        portion_count=portion_count,
        context_data=portion.get("context", ""),
    )


def _make_apfel_iterative_final_prompt(
    *, query: str, response_type: str, accumulated_answer: str, carry_summary: str
) -> str:
    return PROMPTS["apfel_iterative_rag_final_user"].format(
        query=query,
        response_type=response_type,
        accumulated_answer=accumulated_answer or "No answer deltas.",
        carry_summary=carry_summary or "No carry summary.",
    )


def _fallback_apfel_answer_from_portions(
    portions: list[dict[str, Any]], tokenizer: Tokenizer, max_tokens: int
) -> str:
    for portion in portions:
        context = portion.get("context", "")
        chunk_match = re.search(
            r"CHUNK\s+ref=\[[^\]]+\][^\n]*\n"
            r"(.*?)(?=\n\n---|\n\nKnowledge Graph|\n\nRELATION|\Z)",
            context,
            flags=re.DOTALL,
        )
        if chunk_match:
            content = chunk_match.group(1).strip()
            if content:
                return _clip_text_by_tokens(tokenizer, content, max_tokens)

    for portion in portions:
        context = re.sub(r"\s+", " ", portion.get("context", "")).strip()
        if context:
            return _clip_text_by_tokens(tokenizer, context, max_tokens)

    return ""


async def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "__aiter__"):
        chunks: list[str] = []
        async for chunk in response:
            if chunk:
                chunks.append(str(chunk))
        return "".join(chunks)
    return str(response or "")


async def _run_apfel_iterative_answer_generator(
    *,
    query: str,
    response_type: str,
    query_param: QueryParam,
    use_model_func: Callable[..., Any],
    tokenizer: Tokenizer,
    portions: list[dict[str, Any]],
    raw_data: dict[str, Any],
) -> AsyncIterator[str]:
    carry_summary = ""
    answer_deltas: list[str] = []
    system_prompt = PROMPTS["apfel_iterative_rag_system"]
    max_completion_tokens = getattr(query_param, "max_completion_tokens", None)
    llm_call_kwargs = {"max_completion_tokens": max_completion_tokens}
    if not max_completion_tokens:
        llm_call_kwargs = {}

    metadata = raw_data.setdefault("metadata", {}).setdefault(
        "apfel_iterative", _apfel_iterative_metadata(portions)
    )

    for index, portion in enumerate(portions, start=1):
        metadata["current_portion"] = index
        logger.info(
            "APFEL_ITERATIVE_PORTION_START index=%s total=%s tokens=%s counts=%s",
            index,
            len(portions),
            portion.get("token_estimate"),
            portion.get("counts"),
        )
        prompt = _make_apfel_iterative_user_prompt(
            query=query,
            response_type=response_type,
            carry_summary=carry_summary,
            portion=portion,
            portion_index=index,
            portion_count=len(portions),
        )
        response = await use_model_func(
            prompt,
            system_prompt=system_prompt,
            history_messages=[],
            enable_cot=False,
            stream=False,
            _lightrag_request_kind="query",
            **llm_call_kwargs,
        )
        response_text = await _response_to_text(response)
        answer_delta, next_carry = _parse_apfel_iterative_response(response_text)

        if answer_delta:
            answer_deltas.append(answer_delta)
            yield answer_delta.rstrip() + "\n\n"

        if next_carry:
            carry_summary = _clip_text_by_tokens(
                tokenizer,
                next_carry,
                max(96, int((max_completion_tokens or 384) * 0.75)),
            )
        elif answer_delta:
            carry_summary = _clip_text_by_tokens(
                tokenizer,
                "\n".join(answer_deltas[-3:]),
                max(96, int((max_completion_tokens or 384) * 0.75)),
            )

    if not answer_deltas:
        fallback_answer = _fallback_apfel_answer_from_portions(
            portions,
            tokenizer,
            max(64, int((max_completion_tokens or 384) * 0.5)),
        )
        if fallback_answer:
            logger.warning(
                "APFEL_ITERATIVE_FALLBACK no usable model delta; using retrieved context excerpt"
            )
            answer_deltas.append(fallback_answer)
            yield fallback_answer.rstrip()

    if len(answer_deltas) > 1:
        final_prompt = _make_apfel_iterative_final_prompt(
            query=query,
            response_type=response_type,
            accumulated_answer="\n\n".join(answer_deltas),
            carry_summary=carry_summary,
        )
        response = await use_model_func(
            final_prompt,
            system_prompt=system_prompt,
            history_messages=[],
            enable_cot=False,
            stream=False,
            _lightrag_request_kind="query",
            **llm_call_kwargs,
        )
        response_text = await _response_to_text(response)
        final_answer, _ = _parse_apfel_iterative_response(response_text)
        if final_answer:
            yield "### Final\n\n" + final_answer.rstrip()

    metadata["current_portion"] = len(portions)
    logger.info("APFEL_ITERATIVE_DONE portions=%s", len(portions))


async def _run_apfel_iterative_answer(
    **kwargs: Any,
) -> str:
    chunks: list[str] = []
    async for chunk in _run_apfel_iterative_answer_generator(**kwargs):
        if chunk:
            chunks.append(chunk)
    return "".join(chunks).strip()


def _is_apfel_fast_query(
    global_config: dict[str, Any], query_param: QueryParam
) -> bool:
    """Detect the local Apfel fast query path without adding config flags."""

    if query_param.model_func is not None:
        return False

    model_name = str(
        global_config.get("llm_model_name") or settings.llm_model_configured or ""
    ).lower()
    binding_host = str(settings.llm_binding_host or "").lower()
    return (
        "apple-foundationmodel" in model_name
        or "apfel" in model_name
        or "127.0.0.1:11435" in binding_host
        or "localhost:11435" in binding_host
    )


def _format_apfel_reference_id(value: Any) -> str:
    return str(value or "").strip()


def _build_apfel_reference_lines(
    references: list[dict[str, Any]], reference_ids: set[str]
) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()

    for ref in references:
        ref_id = _format_apfel_reference_id(ref.get("reference_id"))
        file_path = str(ref.get("file_path") or "").strip()
        if not ref_id or not file_path:
            continue
        if reference_ids and ref_id not in reference_ids:
            continue
        key = (ref_id, file_path)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"[{ref_id}] {file_path}")

    return lines