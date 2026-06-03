import json
import sys
import urllib.error
import urllib.request

from lightrag.config import settings


def main() -> int:
    base_url = (
        settings.mlx_chat_url
        or settings.llm_binding_host
        or "http://127.0.0.1:11436/v1"
    )
    if base_url.endswith("/v1"):
        url = base_url.rstrip("/") + "/chat/completions"
    else:
        url = base_url

    model = (
        settings.mlx_chat_model
        or settings.llm_model
        or "TheCluster/amoral-gemma-3-12B-v2-mlx-4bit"
    )
    prompt = settings.mlx_chat_prompt
    max_tokens = settings.mlx_chat_max_tokens_for(16)
    timeout = settings.mlx_chat_timeout_for(300)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
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

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"))
        raise
    except Exception as exc:
        print(f"mlx-chat-test failed: {exc}", file=sys.stderr)
        raise

    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(content.replace("<end_of_turn>", "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
