#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from scripts.local_env import env_bool, env_int, env_str

MODEL_PATH = env_str("MLX_EMBEDDINGS_MODEL_PATH")
MODEL_REPO = env_str("MLX_EMBEDDINGS_MODEL", "mlx-community/bge-m3-mlx-4bit")
MODEL_NAME = env_str("EMBEDDING_MODEL") or MODEL_REPO
HOST = env_str("MLX_EMBEDDINGS_HOST", "127.0.0.1")
PORT = env_int("MLX_EMBEDDINGS_PORT", 11439)
MAX_SEQ_LEN = env_int("EMBEDDING_TOKEN_LIMIT", 8192)
API_KEY = env_str("EMBEDDING_BINDING_API_KEY", "dummy")
RERANK_MODEL_PATH = env_str("MLX_RERANK_MODEL_PATH")
RERANK_MODEL_REPO = env_str("MLX_RERANK_MODEL")
RERANK_MODEL_NAME = env_str("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_MAX_LENGTH = env_int("RERANK_MAX_TOKENS_PER_DOC", 1024)
RERANK_API_KEY = env_str("RERANK_BINDING_API_KEY") or API_KEY
RERANK_BATCH_SIZE = env_int("MLX_RERANK_BATCH_SIZE", 8)
RERANK_ENABLED = env_bool("LIGHTRAG_RERANK_ENABLED")
RERANK_MEMORY_LIMIT_MB = env_int("MLX_RERANK_SERVER_MAX_RSS_MB", 0)
RERANK_CACHE_LIMIT_MB = env_int("MLX_RERANK_CACHE_MAX_MB", 0) or min(
    1024,
    max(128, RERANK_MEMORY_LIMIT_MB // 8),
)

if RERANK_MEMORY_LIMIT_MB:
    mx.metal.set_memory_limit(RERANK_MEMORY_LIMIT_MB * 1024 * 1024)
if RERANK_CACHE_LIMIT_MB:
    mx.metal.set_cache_limit(RERANK_CACHE_LIMIT_MB * 1024 * 1024)


class EmbeddingsRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    encoding_format: str | None = "float"
    dimensions: int | None = None


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    model: str | None = None
    top_n: int | None = None
    return_documents: bool | None = None


app = FastAPI(title="Managed MLX Embeddings", version="1.0")
_MODEL = None
_TOKENIZER = None
_RERANK_MODEL = None
_RERANK_TOKENIZER = None


def _resolve_model_source() -> str:
    if MODEL_PATH:
        model_path = Path(MODEL_PATH)
        if model_path.exists():
            return str(model_path)
    return MODEL_REPO


def _resolve_rerank_model_source() -> str:
    if RERANK_MODEL_PATH:
        model_path = Path(RERANK_MODEL_PATH)
        if model_path.exists():
            return str(model_path)
    return RERANK_MODEL_REPO


def _ensure_model_loaded():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    from mlx_embeddings.utils import load

    _MODEL, _TOKENIZER = load(_resolve_model_source())
    return _MODEL, _TOKENIZER


class _SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)

    def __call__(self, sequence_output: mx.array) -> mx.array:
        hidden = sequence_output[:, 0, :]
        hidden = self.dropout(hidden)
        hidden = self.dense(hidden)
        hidden = nn.tanh(hidden)
        hidden = self.dropout(hidden)
        return self.out_proj(hidden)


def _build_rerank_model(model_path: str):
    from mlx_embeddings.models.xlm_roberta import (
        ModelArgs,
        XLMRobertaEmbeddings,
        XLMRobertaEncoder,
    )
    from mlx_embeddings.tokenizer_utils import load_tokenizer
    from mlx_embeddings.utils import get_model_path, load_config

    class _XLMRobertaReranker(nn.Module):
        def __init__(self, config: ModelArgs):
            super().__init__()
            self.config = config
            self.embeddings = XLMRobertaEmbeddings(config)
            self.encoder = XLMRobertaEncoder(config)
            self.classifier = _SequenceClassificationHead(
                config.hidden_size,
                getattr(config, "classifier_dropout", config.hidden_dropout_prob),
            )

        def get_extended_attention_mask(
            self, attention_mask: mx.array, input_shape: tuple[int, ...]
        ) -> mx.array:
            if attention_mask.ndim == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            else:
                raise ValueError(
                    f"Wrong shape for attention_mask (shape {attention_mask.shape})"
                )
            return (1.0 - extended_attention_mask) * -10000.0

        def __call__(
            self,
            input_ids: mx.array,
            attention_mask: mx.array | None = None,
            token_type_ids: mx.array | None = None,
            position_ids: mx.array | None = None,
        ) -> mx.array:
            input_shape = input_ids.shape
            if attention_mask is None:
                attention_mask = mx.ones(input_shape)
            if token_type_ids is None:
                token_type_ids = mx.zeros(input_shape, dtype=mx.int64)
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape
            )
            embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
            sequence_output = self.encoder(
                embedding_output,
                extended_attention_mask,
            )[0]
            return self.classifier(sequence_output)

        def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
            return {
                key: value
                for key, value in weights.items()
                if "position_ids" not in key
            }

    resolved_model_path = get_model_path(model_path)
    config = load_config(resolved_model_path)
    config["classifier_dropout"] = config.get(
        "classifier_dropout", config.get("hidden_dropout_prob", 0.1)
    )
    model_args = ModelArgs.from_dict(config)
    model = _XLMRobertaReranker(model_args)

    weights: dict[str, mx.array] = {}
    for weight_file in resolved_model_path.rglob("model*.safetensors"):
        weights.update(mx.load(str(weight_file)))

    quantization = config.get("quantization")
    if quantization is not None:
        def _class_predicate(path: str, module: nn.Module) -> bool:
            if path in quantization:
                return bool(quantization[path])
            if not hasattr(module, "to_quantized"):
                return False
            if hasattr(module, "weight") and module.weight.size % 64 != 0:
                return False
            return f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=_class_predicate,
        )

    model.load_weights(list(model.sanitize(weights).items()))
    mx.eval(model.parameters())
    model.eval()
    tokenizer = load_tokenizer(resolved_model_path, {})
    return model, tokenizer


def _ensure_rerank_model_loaded():
    if not RERANK_ENABLED:
        raise HTTPException(status_code=503, detail="Rerank is disabled")

    global _RERANK_MODEL, _RERANK_TOKENIZER
    if _RERANK_MODEL is not None and _RERANK_TOKENIZER is not None:
        return _RERANK_MODEL, _RERANK_TOKENIZER
    _RERANK_MODEL, _RERANK_TOKENIZER = _build_rerank_model(
        _resolve_rerank_model_source()
    )
    return _RERANK_MODEL, _RERANK_TOKENIZER


def _normalize_inputs(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _rerank_encode_pairs(
    tokenizer: Any, queries: list[str], documents: list[str]
) -> dict[str, mx.array]:
    """Encode text pairs for rerank using the underlying HF tokenizer."""
    batch = tokenizer._tokenizer(
        queries,
        documents,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=RERANK_MAX_LENGTH,
    )
    return {
        "input_ids": mx.array(batch["input_ids"]),
        "attention_mask": mx.array(batch["attention_mask"]),
    }


def _count_tokens(inputs: dict[str, Any]) -> int:
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        return int(np.array(attention_mask).sum())
    input_ids = inputs.get("input_ids")
    if input_ids is None:
        return 0
    return int(np.array(input_ids).shape[-1])


def _batched(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield start, values[start : start + size]


@app.get("/health")
async def health() -> dict[str, str]:
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_source": _resolve_model_source(),
        "rerank_model": RERANK_MODEL_NAME,
        "rerank_model_source": _resolve_rerank_model_source(),
        "rerank_enabled": str(RERANK_ENABLED).lower(),
        "rerank_batch_size": str(RERANK_BATCH_SIZE),
        "rerank_memory_limit_mb": str(RERANK_MEMORY_LIMIT_MB),
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            },
            *(
                [
                    {
                        "id": RERANK_MODEL_NAME,
                        "object": "model",
                        "owned_by": "local",
                    }
                ]
                if RERANK_ENABLED
                else []
            ),
        ],
    }


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingsRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    if API_KEY and API_KEY != "dummy":
        expected = f"Bearer {API_KEY}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    if request.dimensions is not None and request.dimensions != 1024:
        raise HTTPException(
            status_code=400,
            detail="This managed MLX embeddings server only supports 1024 dimensions.",
        )

    if request.encoding_format not in (None, "float"):
        raise HTTPException(
            status_code=400,
            detail="Only encoding_format=float is supported.",
        )

    model, tokenizer = _ensure_model_loaded()
    texts = _normalize_inputs(request.input)
    inputs = tokenizer.batch_encode_plus(
        texts,
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )
    text_embeds = outputs.text_embeds
    mx.eval(text_embeds)
    embeddings_array = np.asarray(text_embeds.tolist(), dtype=np.float32)

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": embeddings_array[index].tolist(),
            }
            for index in range(len(texts))
        ],
        "model": request.model or MODEL_NAME,
        "usage": {
            "prompt_tokens": _count_tokens(inputs),
            "total_tokens": _count_tokens(inputs),
        },
    }


@app.post("/v1/rerank")
async def rerank(
    request: RerankRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    if RERANK_API_KEY and RERANK_API_KEY != "dummy":
        expected = f"Bearer {RERANK_API_KEY}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    if not request.documents:
        return {"results": []}

    model, tokenizer = _ensure_rerank_model_loaded()
    documents = [str(document) for document in request.documents]
    score_values: list[float] = []
    prompt_tokens = 0
    started = time.perf_counter()
    for _, batch_documents in _batched(documents, RERANK_BATCH_SIZE):
        query_batch = [request.query] * len(batch_documents)
        inputs = _rerank_encode_pairs(tokenizer, query_batch, batch_documents)
        logits = model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        mx.eval(logits)
        prompt_tokens += _count_tokens(inputs)
        scores = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
        score_values.extend(float(score) for score in scores.tolist())
        mx.clear_cache()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    indexed_scores = [
        {
            "index": index,
            "relevance_score": float(score),
            "document": request.documents[index]
            if request.return_documents
            else None,
        }
        for index, score in enumerate(score_values)
    ]
    indexed_scores.sort(key=lambda item: item["relevance_score"], reverse=True)
    if request.top_n is not None:
        indexed_scores = indexed_scores[: request.top_n]
    for item in indexed_scores:
        if item["document"] is None:
            item.pop("document")
    return {
        "results": indexed_scores,
        "model": request.model or RERANK_MODEL_NAME,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
            "batch_size": RERANK_BATCH_SIZE,
            "elapsed_ms": elapsed_ms,
        },
    }


def main() -> None:
    app_dir = str(Path(__file__).resolve().parent.parent)
    uvicorn.run(
        "scripts.managed_mlx_embeddings_server:app",
        host=HOST,
        port=PORT,
        log_level="info",
        app_dir=app_dir,
    )


if __name__ == "__main__":
    main()
