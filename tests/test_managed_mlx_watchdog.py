import importlib
import sys


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
    monkeypatch.delenv("MLX_OPENAI_SERVER_MAX_RSS_MB", raising=False)
    assert server._managed_mlx_tree_rss_limit_kb() == 20480 * 1024

    monkeypatch.setenv("MLX_OPENAI_SERVER_MAX_RSS_MB", "16384")
    assert server._managed_mlx_tree_rss_limit_kb() == 16384 * 1024
