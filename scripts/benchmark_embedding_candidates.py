#!/usr/bin/env python3
"""Benchmark isolated embedding-model candidates against LightRAG retrieval inputs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 11438


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    model: str
    dimensions: int
    query_prefix: str
    document_prefix: str
    max_tokens: int
    source: str = "hub"


PROFILES = {
    "modernbert": EmbeddingProfile(
        name="modernbert",
        model="nomic-ai/modernbert-embed-base",
        dimensions=768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        max_tokens=8192,
    ),
    "e5": EmbeddingProfile(
        name="e5",
        model="intfloat/multilingual-e5-large",
        dimensions=1024,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_tokens=512,
    ),
    "harrier": EmbeddingProfile(
        name="harrier",
        model="microsoft/harrier-oss-v1-0.6b",
        dimensions=1024,
        query_prefix=(
            "Instruct: Given a query, retrieve relevant passages that answer "
            "the query\nQuery: "
        ),
        document_prefix="",
        max_tokens=32768,
    ),
    # This is the locally installed, MLX-converted retrieval specialization of
    # jinaai/jina-embeddings-v5-text-small. The raw all-task checkpoint needs
    # Jina's custom model implementation and is not served by mlx-embeddings.
    "jina-retrieval": EmbeddingProfile(
        name="jina-retrieval",
        model=str(
            ROOT / "models" / "jina-embeddings-v5-text-small-retrieval-mlx"
        ),
        dimensions=1024,
        query_prefix="Query: ",
        document_prefix="Document: ",
        max_tokens=32768,
        source="local MLX retrieval conversion",
    ),
}

BASELINE = EmbeddingProfile(
    name="bge-m3",
    model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
    dimensions=1024,
    query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", "search_query: ").strip('"'),
    document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", "search_document: ").strip(
        '"'
    ),
    max_tokens=8192,
    source="active service",
)

DOCUMENTS = [
    "Redis stores completed LLM responses so repeated prompts can be served from cache.",
    "Redis coordinates shared pipeline worker state and cross-process locks.",
    "The graph storage persists entities, relations, and their metadata.",
    "Graph retrieval expands an entity into its connected relationships and communities.",
    "Vector storage indexes embeddings for document chunks, entities, and relationships.",
    "Keyword extraction produces high-level and low-level search terms before retrieval.",
    "The agent tool loop dispatches retrieval tools before the final answer is generated.",
    "A reranker reorders candidate chunks after initial vector and graph retrieval.",
    "ModernBERT is an efficient encoder model for text representation learning.",
    "BGE-M3 is a multilingual embedding model that can create dense vectors.",
    "The document pipeline queues uploads before parsing and extracting entities.",
    "The managed MLX embeddings server exposes OpenAI-compatible embedding requests.",
]
QUERIES = [
    ("What component caches previous LLM answers?", 0),
    ("How does LightRAG coordinate pipeline workers?", 1),
    ("How are entity relationships persisted?", 2),
    ("Where are chunk embeddings searched?", 4),
    ("What stage yields high-level and low-level keywords?", 5),
    ("What component reorders retrieved chunks?", 7),
    ("Где хранятся кэшированные ответы LLM?", 0),
    ("How can an application call local embeddings?", 11),
]


def _prefixed(values: list[str], prefix: str) -> list[str]:
    return [prefix + value for value in values]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def _embed(
    client: httpx.Client, base_url: str, model: str, texts: list[str]
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url}/v1/embeddings",
        headers={"Authorization": "Bearer dummy"},
        json={"input": texts, "model": model, "encoding_format": "float"},
    )
    response.raise_for_status()
    payload = response.json()
    vectors = np.asarray([item["embedding"] for item in payload["data"]], dtype=np.float32)
    return vectors, (time.perf_counter() - started) * 1000


def _wait_for_health(
    client: httpx.Client, base_url: str, server: subprocess.Popen[str], timeout_seconds: int
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"embedding server exited with code {server.returncode}")
        try:
            response = client.get(f"{base_url}/health")
            if response.is_success:
                return
            last_error = f"HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(1)
    raise RuntimeError(f"embedding server did not become healthy: {last_error}")


def _retrieval_metrics(
    document_vectors: np.ndarray, query_vectors: np.ndarray
) -> dict[str, object]:
    document_vectors = _normalize(document_vectors)
    query_vectors = _normalize(query_vectors)
    ranks: list[int] = []
    top_document_indexes: list[int] = []
    for vector, (_, expected_index) in zip(query_vectors, QUERIES, strict=True):
        order = np.argsort(-(document_vectors @ vector))
        ranks.append(int(np.where(order == expected_index)[0][0]) + 1)
        top_document_indexes.append(int(order[0]))
    return {
        "expected_document_ranks": ranks,
        "top_document_indexes": top_document_indexes,
        "mrr": round(float(np.mean([1 / rank for rank in ranks])), 4),
        "recall_at_1": round(float(np.mean([rank == 1 for rank in ranks])), 4),
        "recall_at_3": round(float(np.mean([rank <= 3 for rank in ranks])), 4),
    }


def _benchmark_provider(
    client: httpx.Client, base_url: str, profile: EmbeddingProfile
) -> dict[str, object]:
    document_inputs = _prefixed(DOCUMENTS, profile.document_prefix)
    query_inputs = _prefixed([query for query, _ in QUERIES], profile.query_prefix)
    batch_inputs = (document_inputs * 3)[:16]

    vectors, cold_batch_ms = _embed(client, base_url, profile.model, batch_inputs)
    warm_batch_ms = [_embed(client, base_url, profile.model, batch_inputs)[1] for _ in range(3)]
    warm_query_ms = [
        _embed(client, base_url, profile.model, [query_inputs[0]])[1] for _ in range(3)
    ]
    document_vectors, document_ms = _embed(client, base_url, profile.model, document_inputs)
    query_vectors, query_ms = _embed(client, base_url, profile.model, query_inputs)

    return {
        "profile": asdict(profile),
        "dimensions_returned": int(vectors.shape[1]),
        "cold_document_batch_16_ms": round(cold_batch_ms, 1),
        "warm_document_batch_16_ms": [round(value, 1) for value in warm_batch_ms],
        "warm_document_batch_16_median_ms": round(float(np.median(warm_batch_ms)), 1),
        "warm_single_query_ms": [round(value, 1) for value in warm_query_ms],
        "warm_single_query_median_ms": round(float(np.median(warm_query_ms)), 1),
        "retrieval_document_encode_ms": round(document_ms, 1),
        "retrieval_query_encode_ms": round(query_ms, 1),
        "retrieval": _retrieval_metrics(document_vectors, query_vectors),
    }


async def _verify_lightrag_adapter(base_url: str, profile: EmbeddingProfile) -> list[int]:
    from lightrag.llm.openai import openai_embed

    vectors = await openai_embed.func(
        ["What does LightRAG use Redis for?"],
        model=profile.model,
        base_url=f"{base_url}/v1",
        api_key="dummy",
        context="query",
        query_prefix=profile.query_prefix,
        document_prefix=profile.document_prefix,
    )
    return list(vectors.shape)


def _run_isolated_profile(
    profile: EmbeddingProfile, port: int, startup_timeout: int
) -> dict[str, object]:
    base_url = f"http://127.0.0.1:{port}"
    environment = os.environ.copy()
    environment.update(
        {
            "EMBEDDING_MODEL": profile.model,
            "EMBEDDING_DIM": str(profile.dimensions),
            "EMBEDDING_TOKEN_LIMIT": str(profile.max_tokens),
            "EMBEDDING_BINDING_API_KEY": "dummy",
            "MLX_EMBEDDINGS_MODEL": profile.model,
            "MLX_EMBEDDINGS_MODEL_PATH": "",
            "MLX_EMBEDDINGS_HOST": "127.0.0.1",
            "MLX_EMBEDDINGS_PORT": str(port),
        }
    )
    with tempfile.NamedTemporaryFile(mode="w+", prefix="lightrag-embed-", delete=False) as log:
        log_path = Path(log.name)
    server = subprocess.Popen(
        [sys.executable, "scripts/managed_mlx_embeddings_server.py"],
        cwd=ROOT,
        env=environment,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with httpx.Client(timeout=300) as client:
            _wait_for_health(client, base_url, server, startup_timeout)
            report = _benchmark_provider(client, base_url, profile)
            report["lightrag_openai_adapter_shape"] = asyncio.run(
                _verify_lightrag_adapter(base_url, profile)
            )
            report["status"] = "passed"
            return report
    except Exception as exc:
        details = log_path.read_text(errors="replace")[-4000:]
        return {
            "profile": asdict(profile),
            "status": "unsupported_or_failed",
            "error": str(exc),
            "server_log_tail": details,
        }
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILES),
        default=sorted(PROFILES),
        help="Candidates to run sequentially.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--baseline-url", default="http://127.0.0.1:11439")
    args = parser.parse_args()

    report: dict[str, object] = {
        "benchmark": "isolated_embedding_candidates",
        "baseline": None,
        "candidates": {},
    }
    try:
        with httpx.Client(timeout=300) as client:
            report["baseline"] = _benchmark_provider(client, args.baseline_url, BASELINE)
    except Exception as exc:
        report["baseline"] = {"status": "failed", "error": str(exc)}

    candidates = report["candidates"]
    assert isinstance(candidates, dict)
    for name in args.profiles:
        candidates[name] = _run_isolated_profile(
            PROFILES[name], args.port, args.startup_timeout
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if any(item["status"] == "passed" for item in candidates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
