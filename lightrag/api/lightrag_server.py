"""
LightRAG FastAPI Server
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
import os
import re
import logging
import logging.config
import sys
import uvicorn
import pipmaster as pm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from typing import Any, Literal
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import json
import subprocess
import threading
import time
from urllib import request as urllib_request
from urllib.parse import urlparse
from lightrag.config import settings
from lightrag.api.utils_api import (
    get_combined_auth_dependency,
    display_splash_screen,
    check_env_file,
)
from .config import (
    global_args,
    update_uvicorn_mode_config,
    get_default_host,
    resolve_asymmetric_embedding_opt_in,
    PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS,
)
from lightrag import LightRAG, __version__ as core_version
from lightrag.api import __api_version__
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc, priority_limit_async_func_call
from lightrag.constants import (
    DEFAULT_LOG_FILENAME,
)
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes,
)
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
from lightrag.api.routers.ollama_api import OllamaAPI

from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_default_workspace,
    # set_default_workspace,
    cleanup_keyed_lock,
    finalize_share_data,
)
from fastapi.security import OAuth2PasswordRequestForm
from lightrag.api.auth import auth_handler


webui_title = settings.webui_title
webui_description = settings.webui_description

# Global authentication configuration
auth_configured = bool(auth_handler.accounts)
def _is_structured_extraction_response_format(response_format: Any) -> bool:
    return getattr(response_format, "__name__", "") == "ExtractionStructuredOutput"


def _is_extraction_request(
    response_format: Any, forced_extraction_request: bool = False
) -> bool:
    return forced_extraction_request or _is_structured_extraction_response_format(
        response_format
    )


def _get_extraction_openai_override(args) -> tuple[str, str, str | None]:
    model = (
        settings.extraction_llm_model
        or settings.mlx_extraction_model
        or args.llm_model
    )
    base_url = settings.extraction_llm_binding_host or args.llm_binding_host
    api_key = settings.extraction_llm_binding_api_key or args.llm_binding_api_key
    return model, base_url, api_key


def _get_extraction_max_completion_tokens() -> int | None:
    return (
        settings.extraction_openai_llm_max_completion_tokens
        or settings.mlx_openai_server_extraction_max_tokens_configured
    )


def _get_extraction_max_async(default: int) -> int:
    configured = settings.extraction_max_async
    if configured is None:
        return max(1, default)
    return max(1, configured)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_data_dir() -> Path:
    override = str(settings.lightrag_runtime_data_dir or "").strip()
    data_dir = Path(override).expanduser() if override else (_repo_root() / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _runtime_log_dir() -> Path:
    log_dir = _runtime_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _url_is_healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib_request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _wait_for_healthy_url(url: str, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _url_is_healthy(url):
            return True
        time.sleep(1.0)
    return False


def _send_warmup_request(
    url: str, payload: dict[str, Any], description: str, timeout_s: float = 30.0
) -> None:
    """Send a warmup request to preload models on an MLX server.

    This is best-effort; failures are logged as warnings and do not
    block server startup.
    """
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            if 200 <= resp.status < 300:
                logger.info("Warmup OK (%s): %s", description, url)
            else:
                logger.warning(
                    "Warmup returned status %d (%s): %s", resp.status, description, url
                )
    except Exception as exc:
        logger.warning("Warmup failed (%s): %s - %s", description, url, exc)


def _build_mlx_openai_server_config(args, extraction_args=None) -> dict[str, Any]:
    retrieval_model = settings.mlx_agentcpm_model or args.llm_model
    retrieval_model_path = str(settings.mlx_agentcpm_model_path or "").strip()
    extraction_model = (
        settings.mlx_extraction_model
        or getattr(extraction_args, "llm_model", None)
        or settings.extraction_llm_model
        or args.llm_model
    )
    extraction_model_path = str(settings.mlx_extraction_model_path or "").strip()
    embedding_model = settings.mlx_embeddings_model or args.embedding_model
    embedding_model_path = str(settings.mlx_embeddings_model_path or "").strip()

    def _lm_config(
        *,
        served_model_name: str,
        model_path: str,
        role: Literal["retrieval", "extraction"],
    ) -> dict[str, Any]:
        role_config = settings.mlx_openai_server_lm_role_config(role)
        config: dict[str, Any] = {
            "model_path": model_path,
            "model_type": "lm",
            "served_model_name": served_model_name,
            "queue_timeout": role_config.queue_timeout_s,
            "trust_remote_code": True,
            "context_length": role_config.context_length,
            "batch_completion_size": role_config.decode_concurrency,
            "batch_prefill_size": role_config.prompt_concurrency,
            "batch_prefill_step_size": role_config.prefill_step_size,
            "default_max_tokens": role_config.max_tokens,
            "default_temperature": role_config.temperature,
            "prompt_cache_size": role_config.prompt_cache_size,
            "prompt_cache_max_bytes": role_config.prompt_cache_max_bytes,
            "disable_batching": role_config.disable_batching,
            "on_demand": role_config.on_demand,
            "on_demand_idle_timeout": role_config.idle_timeout_s,
        }
        if role_config.prompt_cache_dir:
            config["prompt_cache_dir"] = role_config.prompt_cache_dir
        if role == "retrieval" and role_config.draft_model_path:
            config["draft_model_path"] = role_config.draft_model_path
            config["num_draft_tokens"] = role_config.num_draft_tokens
        if role_config.kv_bits > 0:
            config["kv_bits"] = role_config.kv_bits
            config["kv_group_size"] = role_config.kv_group_size
            config["quantized_kv_start"] = role_config.quantized_kv_start
        return config

    retrieval_config = _lm_config(
        served_model_name=retrieval_model,
        model_path=retrieval_model_path,
        role="retrieval",
    )
    extraction_config = _lm_config(
        served_model_name=extraction_model,
        model_path=extraction_model_path,
        role="extraction",
    )

    models: list[dict[str, Any]] = [retrieval_config]
    if extraction_config["served_model_name"] != retrieval_config["served_model_name"]:
        models.append(extraction_config)
    else:
        retrieval_config["queue_timeout"] = max(
            retrieval_config["queue_timeout"],
            extraction_config["queue_timeout"],
        )
        logger.info(
            "Unified MLX config deduplicated retrieval/extraction model entry for '%s' queue_timeout=%s",
            retrieval_config["served_model_name"],
            retrieval_config["queue_timeout"],
        )

    if not settings.lightrag_manage_swift_embeddings:
        models.append(
            {
                "model_path": embedding_model_path,
                "model_type": "embeddings",
                "served_model_name": embedding_model,
                "queue_timeout": settings.mlx_openai_server_embeddings_config().queue_timeout_s,
                "on_demand": settings.mlx_openai_server_embeddings_config().on_demand,
                "on_demand_idle_timeout": settings.mlx_openai_server_embeddings_config().idle_timeout_s,
            }
        )

    return {
        "server": {
            "host": settings.mlx_openai_server_host,
            "port": settings.mlx_openai_server_port,
            "log_level": settings.mlx_openai_server_log_level,
        },
        "models": models,
    }

def _managed_mlx_rerank_enabled(args) -> bool:
    settings.refresh()
    if not settings.lightrag_rerank_enabled:
        return False
    rerank_binding = getattr(args, "rerank_binding", "null")
    if not rerank_binding or rerank_binding == "null":
        return False
    if rerank_binding != "jina":
        return False
    rerank_host = getattr(args, "rerank_binding_host", None)
    if not rerank_host:
        return False
    parsed = urlparse(rerank_host)
    host = parsed.hostname or ""
    port = parsed.port or 80
    expected_host = settings.mlx_rerank_host
    expected_port = settings.mlx_rerank_port
    return host in {"127.0.0.1", "localhost", expected_host} and port == expected_port


def _managed_mlx_rerank_tree_rss_limit_kb() -> int:
    return max(0, settings.mlx_rerank_server_max_rss_mb) * 1024


def _managed_mlx_rerank_watchdog_interval_s() -> int:
    return settings.mlx_rerank_watchdog_interval_s


def _terminate_managed_process(process: subprocess.Popen[Any] | None, name: str) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=10)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=5)
        except Exception as exc:
            logger.warning(f"Failed to stop managed {name}: {exc}")


def _close_managed_log_handles(app: FastAPI, prefix: str) -> None:
    for stream_name in ("stdout", "stderr"):
        handle_name = f"{prefix}_{stream_name}"
        handle = getattr(app.state, handle_name, None)
        if handle is None:
            continue
        try:
            handle.close()
        except Exception:
            pass
        setattr(app.state, handle_name, None)


def _collect_process_snapshot() -> dict[int, dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,command="],
        capture_output=True,
        check=True,
        text=True,
    )
    snapshot: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kb = int(parts[2])
        except ValueError:
            continue
        snapshot[pid] = {
            "pid": pid,
            "ppid": ppid,
            "rss_kb": rss_kb,
            "command": parts[3],
        }
    return snapshot


def _collect_descendants(snapshot: dict[int, dict[str, Any]], root_pid: int) -> list[dict[str, Any]]:
    descendants: list[dict[str, Any]] = []
    queue = [root_pid]
    while queue:
        current = queue.pop(0)
        for proc in snapshot.values():
            if proc["ppid"] != current:
                continue
            descendants.append(proc)
            queue.append(proc["pid"])
    return descendants


def _find_managed_mlx_openai_processes(config_path: Path) -> list[dict[str, Any]]:
    snapshot = _collect_process_snapshot()
    matches: list[dict[str, Any]] = []
    config_marker = str(config_path)
    for proc in snapshot.values():
        command = proc["command"]
        if "lightrag.api.mlx_openai_server_wrapper" not in command:
            continue
        if config_marker not in command:
            continue
        matches.append(proc)
    return matches


def _terminate_pid(pid: int, name: str) -> None:
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return
    except Exception as exc:
        logger.warning("Failed SIGTERM for %s pid=%s: %s", name, pid, exc)
        return

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except Exception:
            return
        time.sleep(0.25)

    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        return
    except Exception as exc:
        logger.warning("Failed SIGKILL for %s pid=%s: %s", name, pid, exc)


def _terminate_orphaned_managed_mlx_openai_servers(config_path: Path, protected_pids: set[int] | None = None) -> None:
    protected = protected_pids or set()
    matching = _find_managed_mlx_openai_processes(config_path)
    if not matching:
        return
    for proc in matching:
        pid = proc["pid"]
        if pid in protected:
            continue
        logger.warning(
            "Stopping orphan managed MLX OpenAI proc pid=%s cmd=%s",
            pid,
            proc["command"],
        )
        _terminate_pid(pid, "managed MLX OpenAI server")


def _find_managed_mlx_rerank_processes() -> list[dict[str, Any]]:
    snapshot = _collect_process_snapshot()
    matches: list[dict[str, Any]] = []
    for proc in snapshot.values():
        command = proc["command"]
        if "managed_mlx_embeddings_server.py" not in command:
            continue
        matches.append(proc)
    return matches


def _terminate_orphaned_managed_mlx_rerank_servers(
    protected_pids: set[int] | None = None,
) -> None:
    protected = protected_pids or set()
    matching = _find_managed_mlx_rerank_processes()
    if not matching:
        return
    for proc in matching:
        pid = proc["pid"]
        if pid in protected:
            continue
        logger.warning(
            "Stopping orphan managed MLX rerank proc pid=%s cmd=%s",
            pid,
            proc["command"],
        )
        _terminate_pid(pid, "managed MLX rerank sidecar")


def _find_managed_native_processes(binary: str) -> list[dict[str, Any]]:
    snapshot = _collect_process_snapshot()
    marker = str(Path(binary).expanduser())
    matches: list[dict[str, Any]] = []
    for proc in snapshot.values():
        command = proc["command"]
        if marker not in command:
            continue
        matches.append(proc)
    return matches


def _terminate_orphaned_managed_native_servers(
    binary: str,
    name: str,
    protected_pids: set[int] | None = None,
) -> None:
    protected = protected_pids or set()
    for proc in _find_managed_native_processes(binary):
        pid = proc["pid"]
        if pid in protected:
            continue
        logger.warning(
            "Stopping orphan managed %s proc pid=%s cmd=%s",
            name,
            pid,
            proc["command"],
        )
        _terminate_pid(pid, name)


def _managed_mlx_tree_rss_limit_kb() -> int:
    return settings.mlx_openai_server_max_rss_mb * 1024


def _managed_mlx_watchdog_interval_s() -> int:
    return settings.mlx_openai_server_watchdog_interval_s


def _restart_managed_mlx_openai_server(app: FastAPI, args, reason: str) -> subprocess.Popen[Any] | None:
    logger.warning("Recycling managed MLX OpenAI server: %s", reason)
    process = getattr(app.state, "managed_mlx_openai_process", None)
    _terminate_managed_process(process, "unified MLX OpenAI server")
    app.state.managed_mlx_openai_process = None
    _close_managed_log_handles(app, "managed_mlx_openai")
    return _start_managed_mlx_openai_server(app, args)


def _start_managed_mlx_openai_watchdog(app: FastAPI, args) -> threading.Thread | None:
    if not settings.lightrag_manage_mlx_openai_server:
        return None
    stop_event = threading.Event()
    app.state.managed_mlx_openai_watchdog_stop = stop_event
    rss_limit_kb = _managed_mlx_tree_rss_limit_kb()
    interval_s = _managed_mlx_watchdog_interval_s()
    restart_lock = threading.Lock()
    logger.warning(
        "Managed MLX OpenAI watchdog enabled rss_cap_mb=%.1f interval_s=%s",
        rss_limit_kb / 1024,
        interval_s,
    )

    def _watchdog() -> None:
        while not stop_event.wait(interval_s):
            process = getattr(app.state, "managed_mlx_openai_process", None)
            if process is None or process.poll() is not None:
                continue
            try:
                snapshot = _collect_process_snapshot()
                descendants = _collect_descendants(snapshot, process.pid)
                if not descendants:
                    continue
                worst = max(descendants, key=lambda proc: proc["rss_kb"])
                total_rss_kb = sum(proc["rss_kb"] for proc in descendants)
                if total_rss_kb <= rss_limit_kb and worst["rss_kb"] <= rss_limit_kb:
                    continue
                reason = (
                    f"rss cap exceeded total_mb={total_rss_kb / 1024:.1f} "
                    f"worst_pid={worst['pid']} worst_mb={worst['rss_kb'] / 1024:.1f}"
                )
                logger.warning(
                    "Managed MLX RSS cap hit: %s cmd=%s",
                    reason,
                    worst["command"],
                )
                with restart_lock:
                    if stop_event.is_set():
                        return
                    app.state.managed_mlx_openai_process = _restart_managed_mlx_openai_server(
                        app, args, reason
                    )
            except Exception:
                logger.exception("Managed MLX OpenAI watchdog failed")

    thread = threading.Thread(
        target=_watchdog,
        name="managed-mlx-openai-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _start_managed_mlx_openai_server(app: FastAPI, args) -> subprocess.Popen[Any] | None:
    if not settings.lightrag_manage_mlx_openai_server:
        return None

    host = settings.mlx_openai_server_host
    port = settings.mlx_openai_server_port
    models_url = f"http://{host}:{port}/v1/models"
    config = _build_mlx_openai_server_config(args)
    config_path = _runtime_data_dir() / "mlx-openai-server-config.yaml"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    _terminate_orphaned_managed_mlx_openai_servers(config_path)

    if _url_is_healthy(models_url):
        logger.warning(
            f"Managed unified MLX OpenAI server already healthy at http://{host}:{port}"
        )
        return None

    log_dir = _runtime_log_dir()
    _close_managed_log_handles(app, "managed_mlx_openai")
    stdout_handle = open(log_dir / "managed-mlx-openai.out.log", "ab")
    stderr_handle = open(log_dir / "managed-mlx-openai.err.log", "ab")
    app.state.managed_mlx_openai_stdout = stdout_handle
    app.state.managed_mlx_openai_stderr = stderr_handle

    command = _managed_mlx_openai_launch_command(config_path)
    startup_dir = _repo_root() / "lightrag" / "api" / "mlx_openai_server_startup"
    existing_pythonpath = settings.pythonpath
    pythonpath_parts = [str(startup_dir)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        },
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
    )
    if not _wait_for_healthy_url(models_url, timeout_s=180):
        _terminate_managed_process(process, "unified MLX OpenAI server")
        raise RuntimeError(
            f"Managed unified MLX OpenAI server failed to become healthy at {models_url}"
        )
    logger.warning(
        f"Managed unified MLX OpenAI server started under LightRAG pid={process.pid} on {host}:{port} config={config_path}"
    )

    # Warmup: fire-and-forget to preload models without blocking startup
    embed_model = next(
        (
            model.get("served_model_name", "")
            for model in config.get("models", [])
            if model.get("model_type") == "embeddings"
        ),
        "",
    )
    if embed_model:
        threading.Thread(
            target=_send_warmup_request,
            args=(
                f"http://{host}:{port}/v1/embeddings",
                {"input": "warmup", "model": embed_model},
                "unified MLX OpenAI embeddings",
            ),
            daemon=True,
        ).start()
    retrieval_model = config.get("models", [{}])[0].get("served_model_name", "")
    if retrieval_model:
        threading.Thread(
            target=_send_warmup_request,
            args=(
                f"http://{host}:{port}/v1/chat/completions",
                {
                    "model": retrieval_model,
                    "messages": [{"role": "user", "content": "warmup"}],
                    "max_tokens": 1,
                },
                "unified MLX OpenAI retrieval LM",
            ),
            daemon=True,
        ).start()

    return process


def _require_native_runtime_path(value: str | None, env_name: str) -> str:
    path = str(value or "").strip()
    if not path:
        raise RuntimeError(f"{env_name} is required")
    if not Path(path).expanduser().exists():
        raise RuntimeError(f"{env_name} not found: {path}")
    return path


def _append_optional_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _append_optional_value(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    rendered = str(value).strip()
    if not rendered:
        return
    command.extend([flag, rendered])


def _managed_swift_lm_launch_command() -> list[str]:
    model_path = _require_native_runtime_path(
        settings.swift_lm_model_path, "SWIFT_LM_MODEL_PATH"
    )
    binary = _require_native_runtime_path(settings.swift_lm_binary, "SWIFT_LM_BINARY")
    command = [
        binary,
        "--model",
        model_path,
        "--host",
        settings.swift_lm_host,
        "--port",
        str(settings.swift_lm_port),
        "--ctx-size",
        str(settings.swift_lm_context_size),
        "--max-tokens",
        str(settings.swift_lm_max_tokens),
        "--temp",
        "0",
        "--parallel",
        str(settings.swift_lm_parallel),
        "--mem-limit",
        str(settings.swift_lm_mem_limit_mb),
        "--prefill-size",
        str(settings.swift_lm_prefill_size),
    ]
    _append_optional_value(command, "--gpu-layers", settings.swift_lm_gpu_layers)
    _append_optional_flag(command, "--stream-experts", settings.swift_lm_stream_experts)
    _append_optional_flag(command, "--ssd-prefetch", settings.swift_lm_ssd_prefetch)
    _append_optional_flag(command, "--turbo-kv", settings.swift_lm_turbo_kv)
    _append_optional_value(command, "--draft-model", settings.swift_lm_draft_model_path)
    _append_optional_value(
        command, "--num-draft-tokens", settings.swift_lm_num_draft_tokens
    )
    _append_optional_flag(command, "--mtp", settings.swift_lm_mtp)
    _append_optional_value(command, "--num-mtp-tokens", settings.swift_lm_num_mtp_tokens)
    return command


def _start_managed_swift_lm_server(app: FastAPI) -> subprocess.Popen[Any] | None:
    if not settings.lightrag_manage_swift_lm:
        return None

    host = settings.swift_lm_host
    port = settings.swift_lm_port
    health_url = f"http://{host}:{port}/health"
    _terminate_orphaned_managed_native_servers(settings.swift_lm_binary, "SwiftLM")
    if _url_is_healthy(health_url):
        logger.warning("Managed SwiftLM already healthy at http://%s:%s", host, port)
        return None

    log_dir = _runtime_log_dir()
    _close_managed_log_handles(app, "managed_swift_lm")
    stdout_handle = open(log_dir / "managed-swift-lm.out.log", "ab")
    stderr_handle = open(log_dir / "managed-swift-lm.err.log", "ab")
    app.state.managed_swift_lm_stdout = stdout_handle
    app.state.managed_swift_lm_stderr = stderr_handle
    process = subprocess.Popen(
        _managed_swift_lm_launch_command(),
        cwd=str(_repo_root()),
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
    )
    if not _wait_for_healthy_url(health_url, timeout_s=180):
        _terminate_managed_process(process, "SwiftLM")
        raise RuntimeError(f"Managed SwiftLM failed to become healthy at {health_url}")
    logger.warning("Managed SwiftLM started under LightRAG pid=%s on %s:%s", process.pid, host, port)
    return process


def _managed_swift_embeddings_launch_command() -> list[str]:
    model_path = _require_native_runtime_path(
        settings.swift_embeddings_model_path, "SWIFT_EMBEDDINGS_MODEL_PATH"
    )
    binary = _require_native_runtime_path(
        settings.swift_embeddings_server_binary, "SWIFT_EMBEDDINGS_SERVER_BINARY"
    )
    command = [
        binary,
        "--model",
        model_path,
        "--host",
        settings.swift_embeddings_host,
        "--port",
        str(settings.swift_embeddings_port),
        "--max-tokens",
        str(settings.swift_embeddings_max_tokens),
    ]
    _append_optional_value(
        command,
        "--served-model-name",
        settings.embedding_model or Path(model_path).name,
    )
    _append_optional_value(
        command,
        "--idle-timeout-s",
        settings.swift_embeddings_idle_timeout_s,
    )
    return command


def _start_managed_swift_embeddings_server(app: FastAPI) -> subprocess.Popen[Any] | None:
    if not settings.lightrag_manage_swift_embeddings:
        return None

    host = settings.swift_embeddings_host
    port = settings.swift_embeddings_port
    health_url = f"http://{host}:{port}/health"
    _terminate_orphaned_managed_native_servers(
        settings.swift_embeddings_server_binary, "Swift embeddings"
    )
    if _url_is_healthy(health_url):
        logger.warning("Managed Swift embeddings already healthy at http://%s:%s", host, port)
        return None

    log_dir = _runtime_log_dir()
    _close_managed_log_handles(app, "managed_swift_embeddings")
    stdout_handle = open(log_dir / "managed-swift-embeddings.out.log", "ab")
    stderr_handle = open(log_dir / "managed-swift-embeddings.err.log", "ab")
    app.state.managed_swift_embeddings_stdout = stdout_handle
    app.state.managed_swift_embeddings_stderr = stderr_handle
    process = subprocess.Popen(
        _managed_swift_embeddings_launch_command(),
        cwd=str(_repo_root()),
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
    )
    if not _wait_for_healthy_url(health_url, timeout_s=180):
        _terminate_managed_process(process, "Swift embeddings")
        raise RuntimeError(
            f"Managed Swift embeddings failed to become healthy at {health_url}"
        )
    logger.warning(
        "Managed Swift embeddings started under LightRAG pid=%s on %s:%s",
        process.pid,
        host,
        port,
    )
    threading.Thread(
        target=_send_warmup_request,
        args=(
            f"http://{host}:{port}/v1/embeddings",
            {"input": "search_document: warmup", "model": settings.embedding_model},
            "managed Swift embeddings",
        ),
        daemon=True,
    ).start()
    return process


def _restart_managed_swift_embeddings_server(
    app: FastAPI, reason: str
) -> subprocess.Popen[Any] | None:
    logger.warning("Recycling managed Swift embeddings server: %s", reason)
    process = getattr(app.state, "managed_swift_embeddings_process", None)
    _terminate_managed_process(process, "Swift embeddings server")
    app.state.managed_swift_embeddings_process = None
    _close_managed_log_handles(app, "managed_swift_embeddings")
    return _start_managed_swift_embeddings_server(app)


def _start_managed_swift_embeddings_watchdog(app: FastAPI) -> threading.Thread | None:
    if not settings.lightrag_manage_swift_embeddings:
        return None

    rss_limit_kb = settings.swift_embeddings_max_rss_mb * 1024
    if rss_limit_kb <= 0:
        logger.info("Managed Swift embeddings watchdog disabled by rss cap <= 0")
        return None

    stop_event = threading.Event()
    app.state.managed_swift_embeddings_watchdog_stop = stop_event
    interval_s = settings.swift_embeddings_watchdog_interval_s
    restart_lock = threading.Lock()
    logger.warning(
        "Managed Swift embeddings watchdog enabled rss_cap_mb=%.1f interval_s=%s",
        rss_limit_kb / 1024,
        interval_s,
    )

    def _watchdog() -> None:
        while not stop_event.wait(interval_s):
            process = getattr(app.state, "managed_swift_embeddings_process", None)
            if process is None or process.poll() is not None:
                continue
            try:
                snapshot = _collect_process_snapshot()
                descendants = _collect_descendants(snapshot, process.pid)
                root_proc = snapshot.get(process.pid)
                processes = descendants + ([root_proc] if root_proc else [])
                if not processes:
                    continue
                worst = max(processes, key=lambda proc: proc["rss_kb"])
                total_rss_kb = sum(proc["rss_kb"] for proc in processes)
                if total_rss_kb <= rss_limit_kb and worst["rss_kb"] <= rss_limit_kb:
                    continue
                reason = (
                    f"rss cap exceeded total_mb={total_rss_kb / 1024:.1f} "
                    f"worst_pid={worst['pid']} worst_mb={worst['rss_kb'] / 1024:.1f}"
                )
                logger.warning(
                    "Managed Swift embeddings RSS cap hit: %s cmd=%s",
                    reason,
                    worst["command"],
                )
                with restart_lock:
                    if stop_event.is_set():
                        return
                    app.state.managed_swift_embeddings_process = (
                        _restart_managed_swift_embeddings_server(app, reason)
                    )
            except Exception:
                logger.exception("Managed Swift embeddings watchdog failed")

    thread = threading.Thread(
        target=_watchdog,
        name="managed-swift-embeddings-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _managed_mlx_openai_launch_command(config_path: Path) -> list[str]:
    package_version = settings.mlx_openai_server_package_version.strip()
    mlx_lm_version = settings.mlx_lm_package_version.strip()
    if not package_version:
        package_version = "1.8.1"
    if not mlx_lm_version:
        mlx_lm_version = "0.31.3"
    return [
        "uv",
        "run",
        "--with",
        f"mlx-openai-server=={package_version}",
        "--with",
        f"mlx-lm=={mlx_lm_version}",
        "python",
        "-m",
        "lightrag.api.mlx_openai_server_wrapper",
        "launch",
        "--config",
        str(config_path),
    ]


def _restart_managed_mlx_rerank_sidecar(
    app: FastAPI, args, reason: str
) -> subprocess.Popen[Any] | None:
    logger.warning("Recycling managed MLX rerank sidecar: %s", reason)
    process = getattr(app.state, "managed_mlx_rerank_process", None)
    _terminate_managed_process(process, "MLX rerank sidecar")
    app.state.managed_mlx_rerank_process = None
    _close_managed_log_handles(app, "managed_mlx_rerank")
    return _start_managed_mlx_rerank_sidecar(app, args)


def _start_managed_mlx_rerank_watchdog(
    app: FastAPI, args
) -> threading.Thread | None:
    if not _managed_mlx_rerank_enabled(args):
        return None

    rss_limit_kb = _managed_mlx_rerank_tree_rss_limit_kb()
    if rss_limit_kb <= 0:
        logger.info("Managed MLX rerank watchdog disabled by rss cap <= 0")
        return None

    stop_event = threading.Event()
    app.state.managed_mlx_rerank_watchdog_stop = stop_event
    interval_s = _managed_mlx_rerank_watchdog_interval_s()
    restart_lock = threading.Lock()
    logger.warning(
        "Managed MLX rerank watchdog enabled rss_cap_mb=%.1f interval_s=%s",
        rss_limit_kb / 1024,
        interval_s,
    )

    def _watchdog() -> None:
        while not stop_event.wait(interval_s):
            process = getattr(app.state, "managed_mlx_rerank_process", None)
            if process is None or process.poll() is not None:
                continue
            try:
                snapshot = _collect_process_snapshot()
                descendants = _collect_descendants(snapshot, process.pid)
                root_proc = snapshot.get(process.pid)
                processes = descendants + ([root_proc] if root_proc else [])
                if not processes:
                    continue
                worst = max(processes, key=lambda proc: proc["rss_kb"])
                total_rss_kb = sum(proc["rss_kb"] for proc in processes)
                if total_rss_kb <= rss_limit_kb and worst["rss_kb"] <= rss_limit_kb:
                    continue
                reason = (
                    f"rss cap exceeded total_mb={total_rss_kb / 1024:.1f} "
                    f"worst_pid={worst['pid']} worst_mb={worst['rss_kb'] / 1024:.1f}"
                )
                logger.warning(
                    "Managed MLX rerank RSS cap hit: %s cmd=%s",
                    reason,
                    worst["command"],
                )
                with restart_lock:
                    if stop_event.is_set():
                        return
                    app.state.managed_mlx_rerank_process = (
                        _restart_managed_mlx_rerank_sidecar(app, args, reason)
                    )
            except Exception:
                logger.exception("Managed MLX rerank watchdog failed")

    thread = threading.Thread(
        target=_watchdog,
        name="managed-mlx-rerank-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def _start_managed_mlx_rerank_sidecar(
    app: FastAPI, args
) -> subprocess.Popen[Any] | None:
    if not settings.lightrag_rerank_enabled:
        _terminate_orphaned_managed_mlx_rerank_servers()
        logger.info("Managed MLX rerank sidecar disabled by LIGHTRAG_RERANK_ENABLED=0")
        return None

    if not _managed_mlx_rerank_enabled(args):
        return None

    host = settings.mlx_rerank_host
    port = settings.mlx_rerank_port
    health_url = f"http://{host}:{port}/health"
    _terminate_orphaned_managed_mlx_rerank_servers()
    if _url_is_healthy(health_url):
        logger.warning(f"Managed MLX rerank sidecar already healthy at http://{host}:{port}")
        return None

    log_dir = _runtime_log_dir()
    stdout_handle = open(log_dir / "managed-mlx-rerank.out.log", "ab")
    stderr_handle = open(log_dir / "managed-mlx-rerank.err.log", "ab")
    app.state.managed_mlx_rerank_stdout = stdout_handle
    app.state.managed_mlx_rerank_stderr = stderr_handle

    command = [sys.executable, "scripts/managed_mlx_embeddings_server.py"]
    process = subprocess.Popen(
        command,
        cwd=str(_repo_root()),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
    )
    if not _wait_for_healthy_url(health_url, timeout_s=120):
        _terminate_managed_process(process, "MLX rerank sidecar")
        raise RuntimeError(
            f"Managed MLX rerank sidecar failed to become healthy at {health_url}"
        )
    logger.warning(
        f"Managed MLX rerank sidecar started under LightRAG pid={process.pid} on {host}:{port}"
    )

    # Warm only rerank here. Embeddings are served by the unified MLX OpenAI
    # server; loading them in this sidecar doubles memory pressure.
    threading.Thread(
        target=_send_warmup_request,
        args=(
            f"http://{host}:{port}/v1/rerank",
            {"query": "warmup", "documents": ["warmup document"], "top_n": 1},
            "MLX rerank sidecar rerank",
        ),
        daemon=True,
    ).start()

    return process


class LLMConfigCache:
    """Smart LLM and Embedding configuration cache class"""

    def __init__(self, args):
        self.args = args

        # Initialize configurations based on binding conditions
        self.openai_llm_options = None
        self.gemini_llm_options = None
        self.gemini_embedding_options = None
        self.ollama_llm_options = None
        self.ollama_embedding_options = None

        # Only initialize and log OpenAI options when using OpenAI-related bindings
        if args.llm_binding in ["openai", "azure_openai"]:
            from lightrag.llm.binding_options import OpenAILLMOptions

            self.openai_llm_options = OpenAILLMOptions.options_dict(args)
            logger.info(f"OpenAI LLM Options: {self.openai_llm_options}")

        if args.llm_binding == "gemini":
            from lightrag.llm.binding_options import GeminiLLMOptions

            self.gemini_llm_options = GeminiLLMOptions.options_dict(args)
            logger.info(f"Gemini LLM Options: {self.gemini_llm_options}")

        # Only initialize and log Ollama LLM options when using Ollama LLM binding
        if args.llm_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                self.ollama_llm_options = OllamaLLMOptions.options_dict(args)
                logger.info(f"Ollama LLM Options: {self.ollama_llm_options}")
            except ImportError:
                logger.warning(
                    "OllamaLLMOptions not available, using default configuration"
                )
                self.ollama_llm_options = {}

        # Only initialize and log Ollama Embedding options when using Ollama Embedding binding
        if args.embedding_binding == "ollama":
            try:
                from lightrag.llm.binding_options import OllamaEmbeddingOptions

                self.ollama_embedding_options = OllamaEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Ollama Embedding Options: {self.ollama_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "OllamaEmbeddingOptions not available, using default configuration"
                )
                self.ollama_embedding_options = {}

        # Only initialize and log Gemini Embedding options when using Gemini Embedding binding
        if args.embedding_binding == "gemini":
            try:
                from lightrag.llm.binding_options import GeminiEmbeddingOptions

                self.gemini_embedding_options = GeminiEmbeddingOptions.options_dict(
                    args
                )
                logger.info(
                    f"Gemini Embedding Options: {self.gemini_embedding_options}"
                )
            except ImportError:
                logger.warning(
                    "GeminiEmbeddingOptions not available, using default configuration"
                )
                self.gemini_embedding_options = {}


def check_frontend_build():
    """Check if frontend is built and optionally check if source is up-to-date

    Returns:
        tuple: (assets_exist: bool, is_outdated: bool)
            - assets_exist: True if WebUI build files exist
            - is_outdated: True if source is newer than build (only in dev environment)
    """
    webui_dir = Path(__file__).parent / "webui"
    index_html = webui_dir / "index.html"

    # 1. Check if build files exist
    if not index_html.exists():
        ASCIIColors.yellow("\n" + "=" * 80)
        ASCIIColors.yellow("WARNING: Frontend Not Built")
        ASCIIColors.yellow("=" * 80)
        ASCIIColors.yellow("The WebUI frontend has not been built yet.")
        ASCIIColors.yellow("The API server will start without the WebUI interface.")
        ASCIIColors.yellow(
            "\nTo enable WebUI, build the frontend using these commands:\n"
        )
        ASCIIColors.cyan("    cd lightrag_webui")
        ASCIIColors.cyan("    bun install --frozen-lockfile")
        ASCIIColors.cyan("    bun run build")
        ASCIIColors.cyan("    cd ..")
        ASCIIColors.yellow("\nThen restart the service.\n")
        ASCIIColors.cyan(
            "Note: Make sure you have Bun installed. Visit https://bun.sh for installation."
        )
        ASCIIColors.yellow("=" * 80 + "\n")
        return (False, False)  # Assets don't exist, not outdated

    # 2. Check if this is a development environment (source directory exists)
    try:
        source_dir = Path(__file__).parent.parent.parent / "lightrag_webui"
        src_dir = source_dir / "src"

        # Determine if this is a development environment: source directory exists and contains src directory
        if not source_dir.exists() or not src_dir.exists():
            # Production environment, skip source code check
            logger.debug(
                "Production environment detected, skipping source freshness check"
            )
            return (True, False)  # Assets exist, not outdated (prod environment)

        # Development environment, perform source code timestamp check
        logger.debug("Development environment detected, checking source freshness")

        # Source code file extensions (files to check)
        source_extensions = {
            ".ts",
            ".tsx",
            ".js",
            ".jsx",
            ".mjs",
            ".cjs",  # TypeScript/JavaScript
            ".css",
            ".scss",
            ".sass",
            ".less",  # Style files
            ".json",
            ".jsonc",  # Configuration/data files
            ".html",
            ".htm",  # Template files
            ".md",
            ".mdx",  # Markdown
        }

        # Key configuration files (in lightrag_webui root directory)
        key_files = [
            source_dir / "package.json",
            source_dir / "bun.lock",
            source_dir / "vite.config.ts",
            source_dir / "tsconfig.json",
            source_dir / "tailraid.config.js",
            source_dir / "index.html",
        ]

        # Get the latest modification time of source code
        latest_source_time = 0

        # Check source code files in src directory
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                # Only check source code files, ignore temporary files and logs
                if file_path.suffix.lower() in source_extensions:
                    mtime = file_path.stat().st_mtime
                    latest_source_time = max(latest_source_time, mtime)

        # Check key configuration files
        for key_file in key_files:
            if key_file.exists():
                mtime = key_file.stat().st_mtime
                latest_source_time = max(latest_source_time, mtime)

        # Get build time
        build_time = index_html.stat().st_mtime

        # Compare timestamps (5 second tolerance to avoid file system time precision issues)
        if latest_source_time > build_time + 5:
            ASCIIColors.yellow("\n" + "=" * 80)
            ASCIIColors.yellow("WARNING: Frontend Source Code Has Been Updated")
            ASCIIColors.yellow("=" * 80)
            ASCIIColors.yellow(
                "The frontend source code is newer than the current build."
            )
            ASCIIColors.yellow(
                "This might happen after 'git pull' or manual code changes.\n"
            )
            ASCIIColors.cyan(
                "Recommended: Rebuild the frontend to use the latest changes:"
            )
            ASCIIColors.cyan("    cd lightrag_webui")
            ASCIIColors.cyan("    bun install --frozen-lockfile")
            ASCIIColors.cyan("    bun run build")
            ASCIIColors.cyan("    cd ..")
            ASCIIColors.yellow("\nThe server will continue with the current build.")
            ASCIIColors.yellow("=" * 80 + "\n")
            return (True, True)  # Assets exist, outdated
        else:
            logger.info("Frontend build is up-to-date")
            return (True, False)  # Assets exist, up-to-date

    except Exception as e:
        # If check fails, log warning but don't affect startup
        logger.warning(f"Failed to check frontend source freshness: {e}")
        return (True, False)  # Assume assets exist and up-to-date on error


def create_app(args):
    # Check frontend build first and get status
    webui_assets_exist, is_frontend_outdated = check_frontend_build()

    # Create unified API version display with warning symbol if frontend is outdated
    api_version_display = (
        f"{__api_version__}⚠️" if is_frontend_outdated else __api_version__
    )

    # Setup logging
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)

    # Create configuration cache (this will output configuration logs)
    config_cache = LLMConfigCache(args)

    # Verify that bindings are correctly setup
    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
        "gemini",
    ]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
        "jina",
        "gemini",
        "voyageai",
    ]:
        raise Exception(f"embedding binding '{args.embedding_binding}' not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)

    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Check if API key is provided either through env var or args
    api_key = settings.lightrag_api_key or args.key

    # Initialize document manager with workspace support for data isolation
    doc_manager = DocumentManager(args.input_dir, workspace=args.workspace)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()
        app.state.managed_mlx_openai_process = None
        app.state.managed_swift_lm_process = None
        app.state.managed_swift_embeddings_process = None
        app.state.managed_swift_embeddings_watchdog = None
        app.state.managed_swift_embeddings_watchdog_stop = None
        app.state.managed_mlx_rerank_process = None
        app.state.managed_mlx_openai_watchdog = None
        app.state.managed_mlx_openai_watchdog_stop = None
        app.state.managed_mlx_rerank_watchdog = None
        app.state.managed_mlx_rerank_watchdog_stop = None

        try:
            app.state.managed_mlx_openai_process = _start_managed_mlx_openai_server(
                app, args
            )
            app.state.managed_swift_lm_process = _start_managed_swift_lm_server(
                app
            )
            app.state.managed_mlx_openai_watchdog = _start_managed_mlx_openai_watchdog(
                app, args
            )
            app.state.managed_swift_embeddings_process = (
                _start_managed_swift_embeddings_server(app)
            )
            app.state.managed_swift_embeddings_watchdog = (
                _start_managed_swift_embeddings_watchdog(app)
            )
            app.state.managed_mlx_rerank_process = _start_managed_mlx_rerank_sidecar(
                app, args
            )
            app.state.managed_mlx_rerank_watchdog = _start_managed_mlx_rerank_watchdog(
                app, args
            )

            # Initialize database connections
            # Note: initialize_storages() now auto-initializes pipeline_status for rag.workspace
            await rag.initialize_storages()

            # Data migration regardless of storage implementation
            await rag.check_and_migrate_data()

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up database connections
            await rag.finalize_storages()

            watchdog_stop = getattr(app.state, "managed_mlx_openai_watchdog_stop", None)
            if watchdog_stop is not None:
                watchdog_stop.set()
            swift_embeddings_watchdog_stop = getattr(
                app.state, "managed_swift_embeddings_watchdog_stop", None
            )
            if swift_embeddings_watchdog_stop is not None:
                swift_embeddings_watchdog_stop.set()
            rerank_watchdog_stop = getattr(
                app.state, "managed_mlx_rerank_watchdog_stop", None
            )
            if rerank_watchdog_stop is not None:
                rerank_watchdog_stop.set()
            _terminate_managed_process(
                getattr(app.state, "managed_mlx_rerank_process", None),
                "MLX rerank sidecar",
            )
            _terminate_managed_process(
                getattr(app.state, "managed_swift_embeddings_process", None),
                "Swift embeddings server",
            )
            _terminate_managed_process(
                getattr(app.state, "managed_swift_lm_process", None),
                "SwiftLM",
            )
            _terminate_managed_process(
                getattr(app.state, "managed_mlx_openai_process", None),
                "unified MLX OpenAI server",
            )
            for handle_name in (
                "managed_mlx_rerank_stdout",
                "managed_mlx_rerank_stderr",
                "managed_swift_embeddings_stdout",
                "managed_swift_embeddings_stderr",
                "managed_swift_lm_stdout",
                "managed_swift_lm_stderr",
                "managed_mlx_openai_stdout",
                "managed_mlx_openai_stderr",
            ):
                handle = getattr(app.state, handle_name, None)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass

            if not settings.lightrag_gunicorn_mode_configured:
                # Only perform cleanup in Uvicorn single-process mode
                logger.debug("Unvicorn Mode: finalizing shared storage...")
                finalize_share_data()
            else:
                # In Gunicorn mode with preload_app=True, cleanup is handled by on_exit hooks
                logger.debug(
                    "Gunicorn Mode: postpone shared storage finalization to master process"
                )

    mcp_http_app = None
    mcp_mount_path = str(settings.lightrag_mcp_path).strip() or "/mcp"
    if not mcp_mount_path.startswith("/"):
        mcp_mount_path = f"/{mcp_mount_path}"

    app_lifespan = lifespan
    if settings.lightrag_mcp_enabled:
        from lightrag.api.mcp_server import (
            create_lightrag_mcp_http_app,
            mount_lightrag_mcp_http_app,
        )

        mcp_http_app = create_lightrag_mcp_http_app(args=args, api_key=api_key)

        @asynccontextmanager
        async def app_lifespan(app: FastAPI):
            async with lifespan(app):
                async with mcp_http_app.router.lifespan_context(mcp_http_app):
                    yield

    # Initialize FastAPI
    base_description = (
        "Providing API for LightRAG core, Web UI and Ollama Model Emulation"
    )
    swagger_description = (
        base_description
        + (" (API-Key Enabled)" if api_key else "")
        + "\n\n[View ReDoc documentation](/redoc)"
    )
    app_kwargs = {
        "title": "LightRAG Server API",
        "description": swagger_description,
        "version": __api_version__,
        "openapi_url": "/openapi.json",  # Explicitly set OpenAPI schema URL
        "docs_url": None,  # Disable default docs, we'll create custom endpoint
        "redoc_url": "/redoc",  # Explicitly set redoc URL
        "lifespan": app_lifespan,
    }

    # Configure Swagger UI parameters
    # Enable persistAuthorization and tryItOutEnabled for better user experience
    app_kwargs["swagger_ui_parameters"] = {
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    }

    app = FastAPI(**app_kwargs)

    def is_mcp_request(path: str) -> bool:
        return path == mcp_mount_path or path.startswith(f"{mcp_mount_path}/")

    @app.middleware("http")
    async def mcp_http_logging_middleware(request: Request, call_next):
        if mcp_http_app is None or not is_mcp_request(request.url.path):
            return await call_next(request)

        request_start = time.perf_counter()
        client_host = request.client.host if request.client else "-"
        query_string = request.url.query or "-"
        content_length = request.headers.get("content-length", "-")
        user_agent = request.headers.get("user-agent", "-")

        logger.info(
            "[mcp][http] request method=%s path=%s query=%s client=%s content_length=%s user_agent=%s",
            request.method,
            request.url.path,
            query_string,
            client_host,
            content_length,
            user_agent,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - request_start) * 1000
            logger.exception(
                "[mcp][http] unhandled_exception method=%s path=%s query=%s client=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                query_string,
                client_host,
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - request_start) * 1000
        log_message = (
            "[mcp][http] response method=%s path=%s query=%s client=%s status=%s duration_ms=%.2f content_type=%s"
        )
        log_args = (
            request.method,
            request.url.path,
            query_string,
            client_host,
            response.status_code,
            duration_ms,
            response.headers.get("content-type", "-"),
        )

        if response.status_code >= 400:
            logger.error(log_message, *log_args)
        else:
            logger.info(log_message, *log_args)

        return response

    if mcp_http_app is not None:
        mount_lightrag_mcp_http_app(app, mcp_http_app, mcp_mount_path)
        logger.info("Integrated LightRAG MCP mounted at %s", mcp_mount_path)

    # Add custom validation error handler for /query/data endpoint
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        # Check if this is a request to /query/data endpoint
        if request.url.path.endswith("/query/data"):
            # Extract error details
            error_details = []
            for error in exc.errors():
                field_path = " -> ".join(str(loc) for loc in error["loc"])
                error_details.append(f"{field_path}: {error['msg']}")

            error_message = "; ".join(error_details)

            # Return in the expected format for /query/data
            return JSONResponse(
                status_code=400,
                content={
                    "status": "failure",
                    "message": f"Validation error: {error_message}",
                    "data": {},
                    "metadata": {},
                },
            )
        else:
            # For other endpoints, return the default FastAPI validation error
            return JSONResponse(status_code=422, content={"detail": exc.errors()})

    def get_cors_origins():
        """Get allowed origins from global_args
        Returns a list of allowed origins, defaults to ["*"] if not set
        """
        origins_str = global_args.cors_origins
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-New-Token"
        ],  # Expose token renewal header for cross-origin requests
    )

    # Create combined auth dependency for all endpoints
    combined_auth = get_combined_auth_dependency(api_key)

    def get_workspace_from_request(request: Request) -> str | None:
        """
        Extract workspace from HTTP request header or use default.

        This enables multi-workspace API support by checking the custom
        'LIGHTRAG-WORKSPACE' header. If not present, falls back to the
        server's default workspace configuration.

        Args:
            request: FastAPI Request object

        Returns:
            Workspace identifier (may be empty string for global namespace)
        """
        # Check custom header first
        workspace = request.headers.get("LIGHTRAG-WORKSPACE", "").strip()

        if not workspace:
            workspace = None
        else:
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", workspace)
            if sanitized != workspace:
                logger.warning(
                    f"Workspace header '{workspace}' contains invalid characters. "
                    f"Sanitized to '{sanitized}'."
                )
                workspace = sanitized

        return workspace

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    def create_optimized_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized OpenAI LLM function with pre-processed configuration"""
        from lightrag.llm.openai import openai_complete_if_cache

        async def dispatch_openai_completion(
            target_model,
            prompt,
            system_prompt,
            history_messages,
            target_base_url,
            target_api_key,
            **kwargs,
        ) -> str:
            return await openai_complete_if_cache(
                target_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=target_base_url,
                api_key=target_api_key,
                **kwargs,
            )

        extraction_openai_completion = priority_limit_async_func_call(
            _get_extraction_max_async(getattr(args, "max_async", 1)),
            llm_timeout=llm_timeout,
            queue_name="Extraction LLM func",
        )(dispatch_openai_completion)

        async def optimized_openai_alike_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            keyword_extraction = kwargs.pop("keyword_extraction", None)
            request_kind = kwargs.pop("_lightrag_request_kind", None)
            if keyword_extraction:
                kwargs["response_format"] = GPTKeywordExtractionFormat
            if history_messages is None:
                history_messages = []

            forced_extraction_request = bool(
                kwargs.pop("_lightrag_extraction_request", False)
            )
            response_format = kwargs.get("response_format")
            is_extraction_request = _is_extraction_request(
                response_format, forced_extraction_request
            )
            target_model = args.llm_model
            target_base_url = args.llm_binding_host
            target_api_key = args.llm_binding_api_key

            # Use pre-processed configuration to avoid repeated parsing
            kwargs["timeout"] = llm_timeout
            explicit_completion_limit = (
                "max_completion_tokens" in kwargs or "max_tokens" in kwargs
            )
            if config_cache.openai_llm_options:
                kwargs = {**config_cache.openai_llm_options, **kwargs}

            if is_extraction_request:
                target_model, target_base_url, target_api_key = (
                    _get_extraction_openai_override(args)
                )
                request_kind = "extraction"
                if not explicit_completion_limit:
                    extraction_max_completion_tokens = (
                        _get_extraction_max_completion_tokens()
                    )
                    if extraction_max_completion_tokens is not None:
                        kwargs["max_completion_tokens"] = extraction_max_completion_tokens
                logger.info(
                    "Routing structured extraction LLM request model=%s base_url=%s",
                    target_model,
                    target_base_url,
                )

            if request_kind:
                kwargs["_lightrag_request_kind"] = request_kind

            completion_func = (
                extraction_openai_completion
                if is_extraction_request
                else dispatch_openai_completion
            )
            return await completion_func(
                target_model,
                prompt,
                system_prompt,
                history_messages,
                target_base_url,
                target_api_key,
                **kwargs,
            )

        return optimized_openai_alike_model_complete

    def create_optimized_azure_openai_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Azure OpenAI LLM function with pre-processed configuration"""

        async def optimized_azure_openai_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            from lightrag.llm.azure_openai import azure_openai_complete_if_cache

            keyword_extraction = kwargs.pop("keyword_extraction", None)
            if keyword_extraction:
                kwargs["response_format"] = GPTKeywordExtractionFormat
            if history_messages is None:
                history_messages = []

            # Use pre-processed configuration to avoid repeated parsing
            kwargs["timeout"] = llm_timeout
            if config_cache.openai_llm_options:
                kwargs = {**config_cache.openai_llm_options, **kwargs}

            return await azure_openai_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=args.llm_binding_host,
                api_key=settings.azure_openai_api_key or args.llm_binding_api_key,
                api_version=settings.azure_openai_api_version,
                **kwargs,
            )

        return optimized_azure_openai_model_complete

    def create_optimized_gemini_llm_func(
        config_cache: LLMConfigCache, args, llm_timeout: int
    ):
        """Create optimized Gemini LLM function with cached configuration"""

        async def optimized_gemini_model_complete(
            prompt,
            system_prompt=None,
            history_messages=None,
            keyword_extraction=False,
            **kwargs,
        ) -> str:
            from lightrag.llm.gemini import gemini_complete_if_cache

            if history_messages is None:
                history_messages = []

            # Use pre-processed configuration to avoid repeated parsing
            kwargs["timeout"] = llm_timeout
            if (
                config_cache.gemini_llm_options is not None
                and "generation_config" not in kwargs
            ):
                kwargs["generation_config"] = dict(config_cache.gemini_llm_options)

            return await gemini_complete_if_cache(
                args.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=args.llm_binding_api_key,
                base_url=args.llm_binding_host,
                keyword_extraction=keyword_extraction,
                **kwargs,
            )

        return optimized_gemini_model_complete

    def create_llm_model_func(binding: str):
        """
        Create LLM model function based on binding type.
        Uses optimized functions for OpenAI bindings and lazy import for others.
        """
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_model_complete

                return lollms_model_complete
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_model_complete

                return ollama_model_complete
            elif binding == "aws_bedrock":
                return bedrock_model_complete  # Already defined locally
            elif binding == "azure_openai":
                # Use optimized function with pre-processed configuration
                return create_optimized_azure_openai_llm_func(
                    config_cache, args, llm_timeout
                )
            elif binding == "gemini":
                return create_optimized_gemini_llm_func(config_cache, args, llm_timeout)
            else:  # openai and compatible
                # Use optimized function with pre-processed configuration
                return create_optimized_openai_llm_func(config_cache, args, llm_timeout)
        except ImportError as e:
            raise Exception(f"Failed to import {binding} LLM binding: {e}")

    def create_llm_model_kwargs(binding: str, args, llm_timeout: int) -> dict:
        """
        Create LLM model kwargs based on binding type.
        Uses lazy import for binding-specific options.
        """
        if binding in ["lollms", "ollama"]:
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                return {
                    "host": args.llm_binding_host,
                    "timeout": llm_timeout,
                    "options": OllamaLLMOptions.options_dict(args),
                    "api_key": args.llm_binding_api_key,
                }
            except ImportError as e:
                raise Exception(f"Failed to import {binding} options: {e}")
        return {}

    def create_optimized_embedding_function(
        config_cache: LLMConfigCache,
        binding,
        model,
        host,
        api_key,
        args,
        document_prefix=None,
        query_prefix=None,
    ) -> EmbeddingFunc:
        """
        Create optimized embedding function and return an EmbeddingFunc instance
        with proper max_token_size inheritance from provider defaults.

        This function:
        1. Imports the provider embedding function
        2. Extracts max_token_size and embedding_dim from provider if it's an EmbeddingFunc
        3. Creates an optimized wrapper that calls the underlying function directly (avoiding double-wrapping)
        4. Returns a properly configured EmbeddingFunc instance

        Configuration Rules:
        - When EMBEDDING_MODEL is not set: Uses provider's default model and dimension
          (e.g., jina-embeddings-v4 with 2048 dims, text-embedding-3-small with 1536 dims)
        - When EMBEDDING_MODEL is set to a custom model: User MUST also set EMBEDDING_DIM
          to match the custom model's dimension (e.g., for jina-embeddings-v3, set EMBEDDING_DIM=1024)

        Note: The embedding_dim parameter is automatically injected by EmbeddingFunc wrapper
        when send_dimensions=True (enabled for Jina and Gemini bindings). This wrapper calls
        the underlying provider function directly (.func) to avoid double-wrapping, so we must
        explicitly pass embedding_dim to the provider's underlying function.
        """

        # Step 1: Import provider function and extract default attributes
        provider_func = None
        provider_max_token_size = None
        provider_embedding_dim = None
        provider_supports_asymmetric = False

        try:
            if binding == "openai":
                from lightrag.llm.openai import openai_embed

                provider_func = openai_embed
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_embed

                provider_func = ollama_embed
            elif binding == "gemini":
                from lightrag.llm.gemini import gemini_embed

                provider_func = gemini_embed
            elif binding == "jina":
                from lightrag.llm.jina import jina_embed

                provider_func = jina_embed
            elif binding == "azure_openai":
                from lightrag.llm.azure_openai import azure_openai_embed

                provider_func = azure_openai_embed
            elif binding == "aws_bedrock":
                from lightrag.llm.bedrock import bedrock_embed

                provider_func = bedrock_embed
            elif binding == "lollms":
                from lightrag.llm.lollms import lollms_embed

                provider_func = lollms_embed
            elif binding == "voyageai":
                from lightrag.llm.voyageai import voyageai_embed

                provider_func = voyageai_embed
            # Extract attributes if provider is an EmbeddingFunc
            if provider_func and isinstance(provider_func, EmbeddingFunc):
                provider_max_token_size = provider_func.max_token_size
                provider_embedding_dim = provider_func.embedding_dim
                provider_supports_asymmetric = provider_func.supports_asymmetric
                logger.debug(
                    f"Extracted from {binding} provider: "
                    f"max_token_size={provider_max_token_size}, "
                    f"embedding_dim={provider_embedding_dim}, "
                    f"supports_asymmetric={provider_supports_asymmetric}"
                )
        except ImportError as e:
            logger.warning(f"Could not import provider function for {binding}: {e}")

        # Step 2: Apply priority (user config > provider default)
        # For max_token_size: explicit env var > provider default > None
        final_max_token_size = args.embedding_token_limit or provider_max_token_size
        # For embedding_dim: user config (always has value) takes priority
        # Only use provider default if user config is explicitly None (which shouldn't happen)
        final_embedding_dim = (
            args.embedding_dim if args.embedding_dim else provider_embedding_dim
        )
        # Asymmetric embedding is explicit opt-in only. Provider-specific
        # validation decides whether task parameters or prefixes are required.
        asymmetric_opt_in = resolve_asymmetric_embedding_opt_in(
            binding=binding,
            embedding_asymmetric=args.embedding_asymmetric,
            embedding_asymmetric_configured=args.embedding_asymmetric_configured,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            query_prefix_configured=args.embedding_query_prefix_configured,
            document_prefix_configured=args.embedding_document_prefix_configured,
        )

        # Step 3: Create optimized embedding function (calls underlying function directly)
        # Note: When model is None, each binding will use its own default model
        async def optimized_embedding_function(
            texts, embedding_dim=None, context="document"
        ):
            try:
                if binding == "lollms":
                    from lightrag.llm.lollms import lollms_embed

                    # Get real function, skip EmbeddingFunc wrapper if present
                    actual_func = (
                        lollms_embed.func
                        if isinstance(lollms_embed, EmbeddingFunc)
                        else lollms_embed
                    )
                    # lollms embed_model is not used (server uses configured vectorizer)
                    # Only pass base_url and api_key
                    return await actual_func(texts, base_url=host, api_key=api_key)
                elif binding == "ollama":
                    from lightrag.llm.ollama import ollama_embed

                    # Get real function, skip EmbeddingFunc wrapper if present
                    actual_func = (
                        ollama_embed.func
                        if isinstance(ollama_embed, EmbeddingFunc)
                        else ollama_embed
                    )

                    # Use pre-processed configuration if available
                    if config_cache.ollama_embedding_options is not None:
                        ollama_options = config_cache.ollama_embedding_options
                    else:
                        from lightrag.llm.binding_options import OllamaEmbeddingOptions

                        ollama_options = OllamaEmbeddingOptions.options_dict(args)

                    # Pass embed_model only if provided, let function use its default (bge-m3:latest)
                    kwargs = {
                        "texts": texts,
                        "host": host,
                        "api_key": api_key,
                        "options": ollama_options,
                    }
                    if provider_supports_asymmetric and asymmetric_opt_in:
                        kwargs["context"] = context
                        if query_prefix:
                            kwargs["query_prefix"] = query_prefix
                        if document_prefix:
                            kwargs["document_prefix"] = document_prefix
                    if model:
                        kwargs["embed_model"] = model
                    return await actual_func(**kwargs)
                elif binding == "azure_openai":
                    from lightrag.llm.azure_openai import azure_openai_embed

                    actual_func = (
                        azure_openai_embed.func
                        if isinstance(azure_openai_embed, EmbeddingFunc)
                        else azure_openai_embed
                    )
                    # Pass model only if provided, let function use its default otherwise
                    kwargs = {
                        "texts": texts,
                        "api_key": api_key,
                        "embedding_dim": embedding_dim,
                    }
                    if model:
                        kwargs["model"] = model
                    if provider_supports_asymmetric and asymmetric_opt_in:
                        kwargs["context"] = context
                        if query_prefix:
                            kwargs["query_prefix"] = query_prefix
                        if document_prefix:
                            kwargs["document_prefix"] = document_prefix
                    return await actual_func(**kwargs)
                elif binding == "aws_bedrock":
                    from lightrag.llm.bedrock import bedrock_embed

                    actual_func = (
                        bedrock_embed.func
                        if isinstance(bedrock_embed, EmbeddingFunc)
                        else bedrock_embed
                    )
                    # Pass model only if provided, let function use its default otherwise
                    kwargs = {"texts": texts}
                    if model:
                        kwargs["model"] = model
                    return await actual_func(**kwargs)
                elif binding == "jina":
                    from lightrag.llm.jina import jina_embed

                    actual_func = (
                        jina_embed.func
                        if isinstance(jina_embed, EmbeddingFunc)
                        else jina_embed
                    )
                    # Pass model only if provided, let function use its default (jina-embeddings-v4)
                    kwargs = {
                        "texts": texts,
                        "embedding_dim": embedding_dim,
                        "base_url": host,
                        "api_key": api_key,
                    }
                    if model:
                        kwargs["model"] = model
                    if provider_supports_asymmetric and asymmetric_opt_in:
                        kwargs["context"] = context
                        kwargs["task"] = None
                    return await actual_func(**kwargs)
                elif binding == "gemini":
                    from lightrag.llm.gemini import gemini_embed

                    actual_func = (
                        gemini_embed.func
                        if isinstance(gemini_embed, EmbeddingFunc)
                        else gemini_embed
                    )

                    # Use pre-processed configuration if available
                    if config_cache.gemini_embedding_options is not None:
                        gemini_options = config_cache.gemini_embedding_options
                    else:
                        from lightrag.llm.binding_options import GeminiEmbeddingOptions

                        gemini_options = GeminiEmbeddingOptions.options_dict(args)
                    # Pass model only if provided, let function use its default (gemini-embedding-001)
                    kwargs = {
                        "texts": texts,
                        "base_url": host,
                        "api_key": api_key,
                        "embedding_dim": embedding_dim,
                    }
                    if model:
                        kwargs["model"] = model
                    task_type = gemini_options.get("task_type")
                    if task_type is not None:
                        kwargs["task_type"] = task_type
                    if provider_supports_asymmetric and asymmetric_opt_in:
                        kwargs["context"] = context
                    return await actual_func(**kwargs)
                elif binding == "voyageai":
                    from lightrag.llm.voyageai import voyageai_embed

                    actual_func = (
                        voyageai_embed.func
                        if isinstance(voyageai_embed, EmbeddingFunc)
                        else voyageai_embed
                    )
                    kwargs = {
                        "texts": texts,
                        "api_key": api_key,
                        "embedding_dim": embedding_dim,
                    }
                    if model:
                        kwargs["model"] = model
                    if provider_supports_asymmetric and asymmetric_opt_in:
                        kwargs["context"] = context
                    return await actual_func(**kwargs)
                else:  # openai and compatible
                    from lightrag.llm.openai import openai_embed

                    actual_func = (
                        openai_embed.func
                        if isinstance(openai_embed, EmbeddingFunc)
                        else openai_embed
                    )
                    # Pass model only if provided, let function use its default (text-embedding-3-small)
                    kwargs = {
                        "texts": texts,
                        "base_url": host,
                        "api_key": api_key,
                        "embedding_dim": embedding_dim,
                    }
                    if model:
                        kwargs["model"] = model
                    if provider_supports_asymmetric and asymmetric_opt_in:
                        kwargs["context"] = context
                        if query_prefix:
                            kwargs["query_prefix"] = query_prefix
                        if document_prefix:
                            kwargs["document_prefix"] = document_prefix
                    return await actual_func(**kwargs)
            except ImportError as e:
                raise Exception(f"Failed to import {binding} embedding: {e}")

        # Step 4: Wrap in EmbeddingFunc and return
        embedding_func_instance = EmbeddingFunc(
            embedding_dim=final_embedding_dim,
            func=optimized_embedding_function,
            max_token_size=final_max_token_size,
            send_dimensions=False,  # Will be set later based on binding requirements
            model_name=model,
            supports_asymmetric=provider_supports_asymmetric and asymmetric_opt_in,
        )

        # Log final embedding configuration. Only include prefix info when
        # prefixes will actually be applied (prefix-based asymmetric mode).
        prefix_info = ""
        if (
            asymmetric_opt_in
            and binding in PREFIX_ASYMMETRIC_EMBEDDING_BINDINGS
            and (document_prefix or query_prefix)
        ):
            prefix_info = f" document_prefix={repr(document_prefix)} query_prefix={repr(query_prefix)}"
        logger.info(
            f"Embedding config: binding={binding} model={model} "
            f"embedding_dim={final_embedding_dim} max_token_size={final_max_token_size}{prefix_info}"
        )

        return embedding_func_instance

    llm_timeout = settings.llm_timeout
    embedding_timeout = settings.embedding_timeout

    async def bedrock_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        # Lazy import
        from lightrag.llm.bedrock import bedrock_complete_if_cache

        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []

        # Use global temperature for Bedrock
        kwargs["temperature"] = settings.bedrock_llm_temperature

        return await bedrock_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    # Create embedding function with optimized configuration and max_token_size inheritance
    import inspect

    # Create the EmbeddingFunc instance (now returns complete EmbeddingFunc with max_token_size)
    embedding_func = create_optimized_embedding_function(
        config_cache=config_cache,
        binding=args.embedding_binding,
        model=args.embedding_model,
        host=args.embedding_binding_host,
        api_key=args.embedding_binding_api_key,
        args=args,
        document_prefix=args.embedding_document_prefix,
        query_prefix=args.embedding_query_prefix,
    )

    # Get embedding_send_dim from centralized configuration
    embedding_send_dim = args.embedding_send_dim

    # Check if the underlying function signature has embedding_dim parameter
    sig = inspect.signature(embedding_func.func)
    has_embedding_dim_param = "embedding_dim" in sig.parameters

    # Determine send_dimensions value based on binding type
    # Jina and Gemini REQUIRE dimension parameter (forced to True)
    # OpenAI and others: controlled by EMBEDDING_SEND_DIM environment variable
    if args.embedding_binding in ["jina", "gemini"]:
        # Jina and Gemini APIs require dimension parameter - always send it
        send_dimensions = has_embedding_dim_param
        dimension_control = f"forced by {args.embedding_binding.title()} API"
    else:
        # For OpenAI and other bindings, respect EMBEDDING_SEND_DIM setting
        send_dimensions = embedding_send_dim and has_embedding_dim_param
        if send_dimensions or not embedding_send_dim:
            dimension_control = "by env var"
        else:
            dimension_control = "by not hasparam"

    # Set send_dimensions on the EmbeddingFunc instance
    embedding_func.send_dimensions = send_dimensions

    logger.info(
        f"Send embedding dimension: {send_dimensions} {dimension_control} "
        f"(dimensions={embedding_func.embedding_dim}, has_param={has_embedding_dim_param}, "
        f"binding={args.embedding_binding})"
    )

    # Log max_token_size source
    if embedding_func.max_token_size:
        source = (
            "env variable"
            if args.embedding_token_limit
            else f"{args.embedding_binding} provider default"
        )
        logger.info(
            f"Embedding max_token_size: {embedding_func.max_token_size} (from {source})"
        )
    else:
        logger.info(
            "Embedding max_token_size: None (Embedding token limit is disabled)."
        )

    # Configure rerank function based on args.rerank_bindingparameter
    rerank_model_func = None
    rerank_enabled = settings.lightrag_rerank_enabled
    if not rerank_enabled:
        logger.info("Reranking is disabled by LIGHTRAG_RERANK_ENABLED=0")
    elif args.rerank_binding != "null":
        from lightrag.rerank import cohere_rerank, jina_rerank, ali_rerank

        # Map rerank binding to corresponding function
        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
        }

        # Select the appropriate rerank function based on binding
        selected_rerank_func = rerank_functions.get(args.rerank_binding)
        if not selected_rerank_func:
            logger.error(f"Unsupported rerank binding: {args.rerank_binding}")
            raise ValueError(f"Unsupported rerank binding: {args.rerank_binding}")

        # Get default values from selected_rerank_func if args values are None
        if args.rerank_model is None or args.rerank_binding_host is None:
            sig = inspect.signature(selected_rerank_func)

            # Set default model if args.rerank_model is None
            if args.rerank_model is None and "model" in sig.parameters:
                default_model = sig.parameters["model"].default
                if default_model != inspect.Parameter.empty:
                    args.rerank_model = default_model

            # Set default base_url if args.rerank_binding_host is None
            if args.rerank_binding_host is None and "base_url" in sig.parameters:
                default_base_url = sig.parameters["base_url"].default
                if default_base_url != inspect.Parameter.empty:
                    args.rerank_binding_host = default_base_url

        async def server_rerank_func(
            query: str, documents: list, top_n: int = None, extra_body: dict = None
        ):
            """Server rerank function with configuration from environment variables"""
            # Prepare kwargs for rerank function
            kwargs = {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "api_key": args.rerank_binding_api_key,
                "model": args.rerank_model,
                "base_url": args.rerank_binding_host,
            }

            # Add Cohere-specific parameters if using cohere binding
            if args.rerank_binding == "cohere":
                # Enable chunking if configured (useful for models with token limits like ColBERT)
                kwargs["enable_chunking"] = settings.rerank_enable_chunking
                kwargs["max_tokens_per_doc"] = settings.rerank_max_tokens_per_doc

            return await selected_rerank_func(**kwargs, extra_body=extra_body)

        rerank_model_func = server_rerank_func
        logger.info(
            f"Reranking is enabled: {args.rerank_model or 'default model'} using {args.rerank_binding} provider"
        )
    else:
        logger.info("Reranking is disabled")

    # Create ollama_server_infos from command line arguments
    from lightrag.api.config import OllamaServerInfos

    ollama_server_infos = OllamaServerInfos(
        name=args.simulated_model_name, tag=args.simulated_model_tag
    )

    # Initialize RAG with unified configuration
    try:
        rag = LightRAG(
            working_dir=args.working_dir,
            workspace=args.workspace,
            llm_model_func=create_llm_model_func(args.llm_binding),
            llm_model_name=args.llm_model,
            llm_model_max_async=args.max_async,
            summary_max_tokens=args.summary_max_tokens,
            summary_context_size=args.summary_context_size,
            chunk_token_size=int(args.chunk_size),
            chunk_overlap_token_size=int(args.chunk_overlap_size),
            llm_model_kwargs=create_llm_model_kwargs(
                args.llm_binding, args, llm_timeout
            ),
            embedding_func=embedding_func,
            default_llm_timeout=llm_timeout,
            default_embedding_timeout=embedding_timeout,
            kv_storage=args.kv_storage,
            graph_storage=args.graph_storage,
            vector_storage=args.vector_storage,
            doc_status_storage=args.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": args.cosine_threshold
            },
            enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
            enable_llm_cache=args.enable_llm_cache,
            rerank_model_func=rerank_model_func,
            max_parallel_insert=args.max_parallel_insert,
            max_graph_nodes=args.max_graph_nodes,
            addon_params={
                "language": args.summary_language,
                "entity_types": args.entity_types,
                "relation_labels": args.relation_labels,
            },
            ollama_server_infos=ollama_server_infos,
        )
    except Exception as e:
        logger.error(f"Failed to initialize LightRAG: {e}")
        raise

    # Add routes
    app.include_router(
        create_document_routes(
            rag,
            doc_manager,
            api_key,
        )
    )
    app.include_router(create_query_routes(rag, api_key, args.top_k))
    app.include_router(create_graph_routes(rag, api_key))

    # Add Ollama API routes
    ollama_api = OllamaAPI(rag, top_k=args.top_k, api_key=api_key)
    app.include_router(ollama_api.router, prefix="/api")

    # Custom Swagger UI endpoint for offline support
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Custom Swagger UI HTML with local static files"""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - Swagger UI",
            oauth2_redirect_url="/docs/oauth2-redirect",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
            swagger_ui_parameters=app.swagger_ui_parameters,
        )

    @app.get("/docs/oauth2-redirect", include_in_schema=False)
    async def swagger_ui_redirect():
        """OAuth2 redirect for Swagger UI"""
        return get_swagger_ui_oauth2_redirect_html()

    @app.get("/")
    async def redirect_to_webui():
        """Redirect root path based on WebUI availability"""
        if webui_assets_exist:
            return RedirectResponse(url="/webui")
        else:
            return RedirectResponse(url="/docs")

    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status and guest token if auth is not configured"""

        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }

        return {
            "auth_configured": True,
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        username = form_data.username
        if not auth_handler.verify_password(username, form_data.password):
            raise HTTPException(status_code=401, detail="Incorrect credentials")

        # Regular user login
        user_token = auth_handler.create_token(
            username=username, role="user", metadata={"auth_mode": "enabled"}
        )
        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": api_version_display,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.get(
        "/health",
        dependencies=[Depends(combined_auth)],
        summary="Get system health and configuration status",
        description="Returns comprehensive system status including WebUI availability, configuration, and operational metrics",
        response_description="System health status with configuration details",
        responses={
            200: {
                "description": "Successful response with system status",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "healthy",
                            "webui_available": True,
                            "working_directory": "/path/to/working/dir",
                            "input_directory": "/path/to/input/dir",
                            "configuration": {
                                "llm_binding": "openai",
                                "llm_model": "gpt-4",
                                "embedding_binding": "openai",
                                "embedding_model": "text-embedding-ada-002",
                                "workspace": "default",
                            },
                            "auth_mode": "enabled",
                            "pipeline_busy": False,
                            "core_version": "0.0.1",
                            "api_version": "0.0.1",
                        }
                    }
                },
            }
        },
    )
    async def get_status(request: Request):
        """Get current system status including WebUI availability"""
        try:
            workspace = get_workspace_from_request(request)
            default_workspace = get_default_workspace()
            if workspace is None:
                workspace = default_workspace
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=workspace
            )

            if not auth_configured:
                auth_mode = "disabled"
            else:
                auth_mode = "enabled"

            # Cleanup expired keyed locks and get status
            keyed_lock_info = cleanup_keyed_lock()

            return {
                "status": "healthy",
                "webui_available": webui_assets_exist,
                "working_directory": str(args.working_dir),
                "input_directory": str(args.input_dir),
                "configuration": {
                    # LLM configuration binding/host address (if applicable)/model (if applicable)
                    "llm_binding": args.llm_binding,
                    "llm_binding_host": args.llm_binding_host,
                    "llm_model": args.llm_model,
                    # embedding model configuration binding/host address (if applicable)/model (if applicable)
                    "embedding_binding": args.embedding_binding,
                    "embedding_binding_host": args.embedding_binding_host,
                    "embedding_model": args.embedding_model,
                    "summary_max_tokens": args.summary_max_tokens,
                    "summary_context_size": args.summary_context_size,
                    "kv_storage": args.kv_storage,
                    "doc_status_storage": args.doc_status_storage,
                    "graph_storage": args.graph_storage,
                    "vector_storage": args.vector_storage,
                    "enable_llm_cache_for_extract": args.enable_llm_cache_for_extract,
                    "enable_llm_cache": args.enable_llm_cache,
                    "workspace": default_workspace,
                    "max_graph_nodes": args.max_graph_nodes,
                    # Rerank configuration
                    "enable_rerank": rerank_model_func is not None,
                    "rerank_binding": args.rerank_binding,
                    "rerank_model": args.rerank_model if rerank_model_func else None,
                    "rerank_binding_host": args.rerank_binding_host
                    if rerank_model_func
                    else None,
                    # Environment variable status (requested configuration)
                    "summary_language": args.summary_language,
                    "force_llm_summary_on_merge": args.force_llm_summary_on_merge,
                    "max_parallel_insert": args.max_parallel_insert,
                    "cosine_threshold": args.cosine_threshold,
                    "min_rerank_score": args.min_rerank_score,
                    "related_chunk_number": args.related_chunk_number,
                    "max_async": args.max_async,
                    "embedding_func_max_async": args.embedding_func_max_async,
                    "embedding_batch_num": args.embedding_batch_num,
                },
                "auth_mode": auth_mode,
                "pipeline_busy": pipeline_status.get("busy", False),
                "keyed_locks": keyed_lock_info,
                "core_version": core_version,
                "api_version": api_version_display,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # Custom StaticFiles class for smart caching
    class SmartStaticFiles(StaticFiles):  # Renamed from NoCacheStaticFiles
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)

            is_html = path.endswith(".html") or response.media_type == "text/html"

            if is_html:
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            elif (
                "/assets/" in path
            ):  # Assets (JS, CSS, images, fonts) generated by Vite with hash in filename
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            # Add other rules here if needed for non-HTML, non-asset files

            # Ensure correct Content-Type
            if path.endswith(".js"):
                response.headers["Content-Type"] = "application/javascript"
            elif path.endswith(".css"):
                response.headers["Content-Type"] = "text/css"

            return response

    # Mount Swagger UI static files for offline support
    swagger_static_dir = Path(__file__).parent / "static" / "swagger-ui"
    if swagger_static_dir.exists():
        app.mount(
            "/static/swagger-ui",
            StaticFiles(directory=swagger_static_dir),
            name="swagger-ui-static",
        )

    # Conditionally mount WebUI only if assets exist
    if webui_assets_exist:
        static_dir = Path(__file__).parent / "webui"
        static_dir.mkdir(exist_ok=True)
        app.mount(
            "/webui",
            SmartStaticFiles(
                directory=static_dir, html=True, check_dir=True
            ),  # Use SmartStaticFiles
            name="webui",
        )
        logger.info("WebUI assets mounted at /webui")
    else:
        logger.info("WebUI assets not available, /webui route not mounted")

        # Add redirect for /webui when assets are not available
        @app.get("/webui")
        @app.get("/webui/")
        async def webui_redirect_to_docs():
            """Redirect /webui to /docs when WebUI is not available"""
            return RedirectResponse(url="/docs")

    return app


def get_application(args=None):
    """Factory function for creating the FastAPI application"""
    if args is None:
        args = global_args
    return create_app(args)


def configure_logging():
    """Configure logging for uvicorn startup"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    # Get log directory path from environment variable
    log_dir = settings.log_dir
    log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))

    print(f"\nLightRAG log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = settings.log_max_bytes
    log_backup_count = settings.log_backup_count

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                # Configure all uvicorn related loggers
                "uvicorn": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
                "uvicorn.error": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                    "filters": ["path_filter"],
                },
            },
            "filters": {
                "path_filter": {
                    "()": "lightrag.utils.LightragPathFilter",
                },
            },
        }
    )


def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        "uvicorn",
        "tiktoken",
        "fastapi",
        # Add other required packages here
    ]

    for package in required_packages:
        if not pm.is_installed(package):
            print(f"Installing {package}...")
            pm.install(package)
            print(f"{package} installed successfully")


def main():
    # On Windows, ProactorEventLoop (default since Python 3.8) has known
    # race conditions with uvicorn's socket binding that can cause the server
    # to report it's running while the port is never actually bound.
    # Using SelectorEventLoop resolves this issue.
    # See: https://github.com/HKUDS/LightRAG/issues/2438
    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Explicitly initialize configuration for clarity
    # (The proxy will auto-initialize anyway, but this makes intent clear)
    from .config import initialize_config

    initialize_config()

    # Check if running under Gunicorn
    if settings.gunicorn_cmd_args_configured:
        # If started with Gunicorn, return directly as Gunicorn will call get_application
        print("Running under Gunicorn - worker management handled by Gunicorn")
        return

    # Check .env file
    if not check_env_file():
        sys.exit(1)

    # Check and install dependencies
    check_and_install_dependencies()

    from multiprocessing import freeze_support

    freeze_support()

    # Configure logging before parsing args
    configure_logging()
    update_uvicorn_mode_config()
    display_splash_screen(global_args)

    # Note: Signal handlers are NOT registered here because:
    # - Uvicorn has built-in signal handling that properly calls lifespan shutdown
    # - Custom signal handlers can interfere with uvicorn's graceful shutdown
    # - Cleanup is handled by the lifespan context manager's finally block

    # Create application instance directly instead of using factory function
    app = create_app(global_args)

    # Start Uvicorn in single process mode
    uvicorn_config = {
        "app": app,  # Pass application instance directly instead of string path
        "host": global_args.host,
        "port": global_args.port,
        "log_config": None,  # Disable default config
    }

    if global_args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": global_args.ssl_certfile,
                "ssl_keyfile": global_args.ssl_keyfile,
            }
        )

    print(
        f"Starting Uvicorn server in single-process mode on {global_args.host}:{global_args.port}"
    )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
