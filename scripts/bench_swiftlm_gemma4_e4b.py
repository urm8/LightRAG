import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


CONFIGS = [
    {
        "name": "compact-4k-p256",
        "ctx_size": 4096,
        "prefill_size": 256,
        "parallel": 1,
        "turbo_kv": False,
        "prompt_sizes": [2048],
    },
    {
        "name": "balanced-8k-p512",
        "ctx_size": 8192,
        "prefill_size": 512,
        "parallel": 1,
        "turbo_kv": False,
        "prompt_sizes": [2048, 6144],
    },
    {
        "name": "fast-8k-p1024",
        "ctx_size": 8192,
        "prefill_size": 1024,
        "parallel": 1,
        "turbo_kv": False,
        "prompt_sizes": [2048, 6144],
    },
    {
        "name": "long-16k-p512",
        "ctx_size": 16384,
        "prefill_size": 512,
        "parallel": 1,
        "turbo_kv": False,
        "prompt_sizes": [2048, 6144, 12000],
    },
    {
        "name": "long-16k-p512-turbo",
        "ctx_size": 16384,
        "prefill_size": 512,
        "parallel": 1,
        "turbo_kv": True,
        "prompt_sizes": [2048, 6144, 12000],
    },
]


def get_gpu_alloc_gb() -> tuple[float, float]:
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        alloc_match = re.search(r'"Alloc system memory"=(\d+)', result.stdout)
        in_use_match = re.search(r'"In use system memory"=(\d+)', result.stdout)
        alloc_gb = int(alloc_match.group(1)) / (1024**3) if alloc_match else 0.0
        in_use_gb = int(in_use_match.group(1)) / (1024**3) if in_use_match else 0.0
        return alloc_gb, in_use_gb
    except Exception:
        return 0.0, 0.0


def wait_for_health(port: int, timeout_s: float = 180.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except Exception:
            pass


def build_prompt(token_count: int) -> str:
    return "apple silicon throughput test " * max(1, token_count // 4)


def parse_prefill_seconds(log_text: str) -> float | None:
    matches = re.findall(r"prefill done \| n_tokens=\d+, t=([0-9.]+)s", log_text)
    if not matches:
        return None
    return float(matches[-1])


def run_request(port: int, model_name: str, prompt_tokens: int, max_tokens: int, log_path: Path) -> dict:
    prompt = build_prompt(prompt_tokens)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    peak_alloc_gb = 0.0
    peak_in_use_gb = 0.0
    stop_event = threading.Event()

    def poll_memory() -> None:
        nonlocal peak_alloc_gb, peak_in_use_gb
        while not stop_event.is_set():
            alloc_gb, in_use_gb = get_gpu_alloc_gb()
            peak_alloc_gb = max(peak_alloc_gb, alloc_gb)
            peak_in_use_gb = max(peak_in_use_gb, in_use_gb)
            stop_event.wait(0.25)

    poller = threading.Thread(target=poll_memory, daemon=True)
    poller.start()

    log_before = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            body = response.read().decode("utf-8")
    finally:
        elapsed_s = time.perf_counter() - started
        stop_event.set()
        poller.join(timeout=2)

    time.sleep(0.5)
    log_after = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    appended_log = log_after[len(log_before):]
    response_json = json.loads(body)
    usage = response_json.get("usage", {})
    timings = response_json.get("timings", {})
    prefill_s = parse_prefill_seconds(appended_log)
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens_used = usage.get("prompt_tokens", 0)
    predicted_tps = timings.get("predicted_per_second")
    if predicted_tps is None and elapsed_s > 0 and completion_tokens > 0:
        decode_window = max(elapsed_s - (prefill_s or 0.0), 1e-6)
        predicted_tps = completion_tokens / decode_window

    return {
        "prompt_tokens": prompt_tokens_used,
        "requested_prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_latency_s": round(elapsed_s, 3),
        "prefill_s": round(prefill_s, 3) if prefill_s is not None else None,
        "decode_tps": round(float(predicted_tps), 2) if predicted_tps is not None else None,
        "peak_gpu_alloc_gb": round(peak_alloc_gb, 2),
        "peak_gpu_in_use_gb": round(peak_in_use_gb, 2),
        "memory_efficiency": round(float(predicted_tps) / max(peak_in_use_gb, 0.1), 2)
        if predicted_tps is not None
        else None,
    }


def benchmark_config(swiftlm_path: Path, model_path: Path, config: dict, port: int, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{config['name']}.log"
    command = [
        str(swiftlm_path),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(config["ctx_size"]),
        "--max-tokens",
        "128",
        "--temp",
        "0",
        "--parallel",
        str(config["parallel"]),
        "--mem-limit",
        "16384",
        "--prefill-size",
        str(config["prefill_size"]),
        "--gpu-layers",
        "auto",
    ]
    if config["turbo_kv"]:
        command.append("--turbo-kv")

    with open(log_path, "w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(model_path.parents[1]),
            start_new_session=True,
            text=True,
        )
    try:
        if not wait_for_health(port):
            raise RuntimeError(f"SwiftLM failed to become healthy for config {config['name']}")

        warmup = run_request(port, str(model_path), 128, 16, log_path)
        results: list[dict] = []
        for prompt_tokens in config["prompt_sizes"]:
            result = run_request(port, str(model_path), prompt_tokens, 96, log_path)
            result.update(
                {
                    "config": config["name"],
                    "ctx_size": config["ctx_size"],
                    "prefill_size": config["prefill_size"],
                    "parallel": config["parallel"],
                    "turbo_kv": config["turbo_kv"],
                    "warmup_decode_tps": warmup.get("decode_tps"),
                }
            )
            results.append(result)
        return results
    finally:
        terminate_process(process)
        time.sleep(4)


def pick_winners(results: list[dict]) -> dict[str, dict]:
    short_runs = [r for r in results if r["requested_prompt_tokens"] == 2048 and r["decode_tps"] is not None]
    medium_runs = [r for r in results if r["requested_prompt_tokens"] == 6144 and r["decode_tps"] is not None]
    long_runs = [r for r in results if r["requested_prompt_tokens"] == 12000 and r["decode_tps"] is not None]
    efficiency_runs = [r for r in results if r["memory_efficiency"] is not None]
    best_short = max(short_runs, key=lambda r: r["decode_tps"]) if short_runs else {}
    best_medium = max(medium_runs, key=lambda r: r["decode_tps"]) if medium_runs else {}
    best_long = max(long_runs, key=lambda r: r["decode_tps"]) if long_runs else {}
    best_efficiency = (
        max(efficiency_runs, key=lambda r: r["memory_efficiency"])
        if efficiency_runs
        else {}
    )
    return {
        "best_short_tps": best_short,
        "best_medium_tps": best_medium,
        "best_long_tps": best_long,
        "best_efficiency": best_efficiency,
        "recommended_default": best_medium or best_short,
    }


def write_reports(results: list[dict], winners: dict[str, dict], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "swiftlm_gemma4_e4b_tuning_results.json"
    md_path = output_dir / "swiftlm_gemma4_e4b_tuning_results.md"
    json_path.write_text(json.dumps({"results": results, "winners": winners}, indent=2), encoding="utf-8")

    lines = []
    lines.append("# SwiftLM Gemma 4 E4B Knob Benchmark")
    lines.append("")
    lines.append("## Matrix")
    lines.append("")
    lines.append("| Config | Ctx | Prefill | Turbo KV | Prompt | Prefill s | Decode tok/s | Peak GPU alloc GB | Peak GPU in-use GB | tok/s per in-use GB |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in results:
        lines.append(
            f"| {row['config']} | {row['ctx_size']} | {row['prefill_size']} | {'on' if row['turbo_kv'] else 'off'} | "
            f"{row['requested_prompt_tokens']} | {row['prefill_s']} | {row['decode_tps']} | {row['peak_gpu_alloc_gb']} | "
            f"{row['peak_gpu_in_use_gb']} | {row['memory_efficiency']} |"
        )
    lines.append("")
    lines.append("## Winners")
    lines.append("")
    for label, row in winners.items():
        if not row:
            continue
        lines.append(
            f"- {label}: {row['config']} at prompt {row['requested_prompt_tokens']} with "
            f"{row['decode_tps']} tok/s and {row['peak_gpu_in_use_gb']} GB peak in-use GPU memory"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    recommended_default = winners.get("recommended_default") or {}
    best_eff = winners.get("best_efficiency") or {}
    best_long = winners.get("best_long_tps") or {}
    if recommended_default:
        lines.append(
            f"- Default interactive profile: `{recommended_default['config']}` because it was the strongest non-long-context profile once prompts reached ~6k tokens."
        )
    if best_eff:
        lines.append(
            f"- Best performance-to-memory ratio: `{best_eff['config']}` with {best_eff['memory_efficiency']} tok/s per GB of active GPU memory."
        )
    if best_long:
        lines.append(
            f"- Long-context profile: `{best_long['config']}` for prompts around 12k tokens."
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 E4B SwiftLM tuning knobs")
    parser.add_argument("--swiftlm", default="data/SwiftLM/.build/release/SwiftLM")
    parser.add_argument("--model", default="models/gemma-4-e4b-it-4bit")
    parser.add_argument("--output-dir", default="temp/benchmarks/swiftlm_gemma4_e4b")
    parser.add_argument("--base-port", type=int, default=11540)
    args = parser.parse_args()

    swiftlm_path = Path(args.swiftlm).resolve()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not swiftlm_path.exists():
        raise SystemExit(f"SwiftLM binary not found: {swiftlm_path}")
    if not model_path.exists():
        raise SystemExit(f"Model path not found: {model_path}")

    all_results: list[dict] = []
    for index, config in enumerate(CONFIGS):
        port = args.base_port + index
        print(f"Benchmarking {config['name']} on port {port}...")
        config_results = benchmark_config(swiftlm_path, model_path, config, port, output_dir)
        for row in config_results:
            print(
                f"  prompt={row['requested_prompt_tokens']} prefill={row['prefill_s']}s "
                f"decode={row['decode_tps']} tok/s peak={row['peak_gpu_in_use_gb']} GB"
            )
        all_results.extend(config_results)

    winners = pick_winners(all_results)
    json_path, md_path = write_reports(all_results, winners, output_dir)
    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()