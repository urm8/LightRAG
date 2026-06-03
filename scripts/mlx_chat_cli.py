import json
import sys
import urllib.error
import urllib.request

from lightrag.config import settings


def build_url() -> str:
    base_url = (
        settings.mlx_chat_url
        or settings.llm_binding_host
        or "http://127.0.0.1:11436/v1"
    )
    if base_url.endswith("/v1/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url.rstrip("/") + "/chat/completions"
    return base_url


def chat_completion(url: str, model: str, messages: list[dict[str, str]], max_tokens: int, timeout: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.replace("<end_of_turn>", "").strip()


def main() -> int:
    url = build_url()
    model = (
        settings.mlx_chat_model
        or settings.llm_model
        or "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit"
    )
    max_tokens = settings.mlx_chat_max_tokens_for(512)
    timeout = settings.mlx_chat_timeout_for(300)

    history: list[dict[str, str]] = []

    print(f"mlx-chat model={model}")
    print("commands: /exit /quit /reset")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/reset":
            history.clear()
            print("history cleared")
            continue

        history.append({"role": "user", "content": user_input})
        try:
            reply = chat_completion(url, model, history, max_tokens, timeout)
        except urllib.error.HTTPError as exc:
            print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
            history.pop()
            continue
        except Exception as exc:
            print(f"mlx-chat failed: {exc}", file=sys.stderr)
            history.pop()
            continue

        print(f"model> {reply}")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
