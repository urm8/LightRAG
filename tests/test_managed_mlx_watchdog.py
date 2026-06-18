import importlib
import sys
from types import SimpleNamespace


def _load_lightrag_server(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    return importlib.import_module("lightrag.api.lightrag_server")


def test_collect_descendants_walks_process_tree(monkeypatch):
    server = _load_lightrag_server(monkeypatch)
    snapshot = {
        10: {"pid": 10, "ppid": 1, "rss_kb": 100, "command": "root"},
        11: {"pid": 11, "ppid": 10, "rss_kb": 200, "command": "child-a"},
        12: {"pid": 12, "ppid": 10, "rss_kb": 300, "command": "child-b"},
        13: {"pid": 13, "ppid": 12, "rss_kb": 400, "command": "grandchild"},
    }

    descendants = server._collect_descendants(snapshot, 10)

    assert [proc["pid"] for proc in descendants] == [11, 12, 13]


def test_managed_mlx_tree_rss_limit_uses_default_and_env(monkeypatch):
    server = _load_lightrag_server(monkeypatch)
    monkeypatch.setenv("MLX_OPENAI_SERVER_MAX_RSS_MB", "20480")
    server.settings.refresh()
    assert server._managed_mlx_tree_rss_limit_kb() == 20480 * 1024

    monkeypatch.setenv("MLX_OPENAI_SERVER_MAX_RSS_MB", "16384")
    server.settings.refresh()
    assert server._managed_mlx_tree_rss_limit_kb() == 16384 * 1024


def test_managed_swift_lm_watchdog_restarts_on_healthcheck_failure(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_MANAGE_SWIFT_LM", "true")
    monkeypatch.setenv("SWIFT_LM_HOST", "127.0.0.1")
    monkeypatch.setenv("SWIFT_LM_PORT", "11436")
    monkeypatch.setenv("SWIFT_LM_WATCHDOG_INTERVAL_S", "5")
    server = _load_lightrag_server(monkeypatch)

    class _FakeEvent:
        def __init__(self):
            self.wait_calls = 0
            self._is_set = False

        def wait(self, _timeout):
            self.wait_calls += 1
            return self.wait_calls > 1

        def set(self):
            self._is_set = True

        def is_set(self):
            return self._is_set

    class _FakeThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            self.target()

    health_checks = []
    restarted = []

    def _fake_url_is_healthy(url, timeout=2.0):
        health_checks.append((url, timeout))
        return False

    def _fake_restart(app, reason):
        restarted.append(reason)
        return "restarted-process"

    fake_process = SimpleNamespace(poll=lambda: None)
    app = SimpleNamespace(
        state=SimpleNamespace(
            managed_swift_lm_process=fake_process,
            managed_swift_lm_watchdog_stop=None,
        )
    )

    monkeypatch.setattr(server.threading, "Event", _FakeEvent)
    monkeypatch.setattr(server.threading, "Thread", _FakeThread)
    monkeypatch.setattr(server, "_url_is_healthy", _fake_url_is_healthy)
    monkeypatch.setattr(server, "_restart_managed_swift_lm_server", _fake_restart)

    thread = server._start_managed_swift_lm_watchdog(app)

    assert thread is not None
    assert restarted == ["healthcheck failed url=http://127.0.0.1:11436/health"]
    assert app.state.managed_swift_lm_process == "restarted-process"
    assert health_checks == [
        ("http://127.0.0.1:11436/health", 2.0),
        ("http://127.0.0.1:11436/health", 2.0),
    ]
