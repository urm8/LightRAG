#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="${SWIFTLM_RUNTIME_DIR:-${ROOT_DIR}/data/SwiftLM}"
EMBEDDING_MODEL_DIR="${SWIFT_EMBEDDINGS_MODEL_PATH:-${ROOT_DIR}/models/bge-m3-mlx-4bit}"

apply_patch_once() {
    local repository="$1"
    local patch_file="$2"
    if git -C "$repository" apply --check "$patch_file" >/dev/null 2>&1; then
        git -C "$repository" apply "$patch_file"
    elif git -C "$repository" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
        printf 'Already applied: %s\n' "$patch_file"
    else
        printf 'Patch does not apply cleanly: %s\n' "$patch_file" >&2
        exit 1
    fi
}

if [ ! -d "$RUNTIME_DIR/.git" ]; then
    git clone --recursive https://github.com/SharpAI/SwiftLM "$RUNTIME_DIR"
else
    git -C "$RUNTIME_DIR" submodule update --init --recursive
fi

apply_patch_once "$RUNTIME_DIR" "$ROOT_DIR/patches/swiftlm-nomic-embeddings-target.patch"
apply_patch_once \
    "$RUNTIME_DIR/mlx-swift-lm" \
    "$ROOT_DIR/patches/mlx-swift-lm-nomic-v2-moe.patch"

mkdir -p "$RUNTIME_DIR/Sources/NomicEmbeddingsServer"
cp \
    "$ROOT_DIR/swift_runtime/NomicEmbeddingsServer/main.swift" \
    "$RUNTIME_DIR/Sources/NomicEmbeddingsServer/main.swift"

(
    cd "$RUNTIME_DIR"
    ./build.sh
    cp .build/arm64-apple-macosx/release/default.metallib \
        .build/arm64-apple-macosx/release/mlx.metallib
    swift build -c release --product NomicEmbeddingsServer
)

uv run --with huggingface_hub python - "$EMBEDDING_MODEL_DIR" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

target = Path(sys.argv[1])
snapshot_download(
    repo_id="mlx-community/bge-m3-mlx-4bit",
    local_dir=target,
)
print(target)
PY

printf 'SwiftLM runtime ready: %s\n' "$RUNTIME_DIR"
printf 'BGE-M3 MLX embedding model ready: %s\n' "$EMBEDDING_MODEL_DIR"
