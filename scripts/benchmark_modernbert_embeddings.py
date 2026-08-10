#!/usr/bin/env python3
"""Backward-compatible ModernBERT-only entry point."""

import sys

from benchmark_embedding_candidates import main


if __name__ == "__main__":
    if "--profiles" not in sys.argv:
        sys.argv[1:1] = ["--profiles", "modernbert"]
    raise SystemExit(main())
