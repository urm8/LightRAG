import json
import sys
import importlib
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

sys.argv = ["pytest"]
_log_routes = importlib.import_module("lightrag.api.routers.log_routes")
create_logs_routes = _log_routes.create_logs_routes


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_logs_files_lists_server_and_runtime_logs(monkeypatch, tmp_path):
    log_dir = tmp_path / "server-logs"
    runtime_dir = tmp_path / "runtime-data"
    home_dir = tmp_path / "fake-home"

    _write_log(log_dir / "lightrag.log", ["server-1"])
    _write_log(runtime_dir / "logs" / "managed-swift-lm.err.log", ["runtime-1"])
    _write_log(home_dir / "Library" / "Logs" / "lightrag" / "lightrag.err.log", ["desktop-1"])

    monkeypatch.setenv("LOG_DIR", str(log_dir))
    monkeypatch.setenv("LIGHTRAG_RUNTIME_DATA_DIR", str(runtime_dir))
    monkeypatch.setenv("HOME", str(home_dir))

    app = FastAPI()
    app.include_router(create_logs_routes())
    client = TestClient(app)

    response = client.get("/logs/files")
    assert response.status_code == 200

    payload = response.json()
    file_ids = {item["id"] for item in payload["files"]}
    assert payload["default_file_id"] == "combined"
    assert "combined" in file_ids
    assert "server" in file_ids
    assert "runtime:managed-swift-lm.err.log" in file_ids
    assert "desktop:lightrag.err.log" in file_ids


@pytest.mark.asyncio
async def test_logs_stream_emits_initial_snapshot(monkeypatch, tmp_path):
    log_dir = tmp_path / "server-logs"
    runtime_dir = tmp_path / "runtime-data"

    _write_log(log_dir / "lightrag.log", ["one", "two", "three"])
    monkeypatch.setenv("LOG_DIR", str(log_dir))
    monkeypatch.setenv("LIGHTRAG_RUNTIME_DATA_DIR", str(runtime_dir))

    router = create_logs_routes()
    stream_endpoint = next(
      route.endpoint for route in router.routes if route.path == "/logs/stream"
    )
    response = await stream_endpoint(file_id="server", tail_lines=2, poll_interval=0.2)
    body_iterator = response.body_iterator
    chunk = await body_iterator.__anext__()
    await body_iterator.aclose()

    payload = json.loads(chunk.decode("utf-8"))
    assert payload["type"] == "snapshot"
    assert payload["file_id"] == "server"
    assert payload["lines"] == ["two", "three"]
    assert payload["truncated"] is True


@pytest.mark.asyncio
async def test_combined_logs_stream_emits_prefixed_snapshot(monkeypatch, tmp_path):
    log_dir = tmp_path / "server-logs"
    runtime_dir = tmp_path / "runtime-data"
    home_dir = tmp_path / "fake-home"

    _write_log(log_dir / "lightrag.log", ["server-a"])
    _write_log(runtime_dir / "logs" / "managed-swift-lm.err.log", ["runtime-a"])
    _write_log(home_dir / "Library" / "Logs" / "lightrag" / "lightrag.err.log", ["desktop-a"])

    monkeypatch.setenv("LOG_DIR", str(log_dir))
    monkeypatch.setenv("LIGHTRAG_RUNTIME_DATA_DIR", str(runtime_dir))
    monkeypatch.setenv("HOME", str(home_dir))

    router = create_logs_routes()
    stream_endpoint = next(
        route.endpoint for route in router.routes if route.path == "/logs/stream"
    )
    response = await stream_endpoint(file_id="combined", tail_lines=5, poll_interval=0.2)
    body_iterator = response.body_iterator
    chunk = await body_iterator.__anext__()
    await body_iterator.aclose()

    payload = json.loads(chunk.decode("utf-8"))
    assert payload["type"] == "snapshot"
    assert payload["file_id"] == "combined"
    assert payload["label"] == "Combined"
    assert "[LightRAG Server] server-a" in payload["lines"]
    assert "[managed-swift-lm.err.log] runtime-a" in payload["lines"]
    assert "[desktop/lightrag.err.log] desktop-a" in payload["lines"]
