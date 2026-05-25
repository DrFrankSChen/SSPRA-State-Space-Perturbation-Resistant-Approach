#!/usr/bin/env python3
"""Evaluate the SSPRA Stage 2 demo outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sspra.metrics import evaluate_from_metadata  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate prepared SSPRA demo results.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=REPO_ROOT / "data" / "demo" / "stage2" / "metadata.json",
    )
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "outputs" / "demo")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "demo" / "metrics.json")
    args = parser.parse_args()

    if not args.metadata.exists():
        parser.error(f"missing {args.metadata}; run `python scripts/prepare_demo_data.py` first")
    payload = evaluate_from_metadata(args.metadata, args.results, args.out)
    print(f"Wrote metrics for {len(payload['tests'])} tests to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
