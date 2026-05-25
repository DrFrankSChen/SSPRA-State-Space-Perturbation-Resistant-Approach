#!/usr/bin/env python3
"""Prepare a short BB-MAS Stage 2 demo stream for SSPRA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sspra.data import prepare_bbmas_demo  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local BB-MAS Stage 2 demo CSVs.")
    parser.add_argument(
        "--bbmas-root",
        type=Path,
        default=REPO_ROOT / "data" / "BB-MAS_Dataset",
        help="Downloaded BB-MAS folder. Default: data/BB-MAS_Dataset",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "demo" / "stage2")
    parser.add_argument("--user", type=int, default=1, help="Genuine user. Packaged assets support user 1.")
    parser.add_argument("--imposter-user", type=int, default=2, help="Raw stream used to inject attack samples.")
    parser.add_argument("--seconds", type=int, default=45, help="Seconds of Stage 2 walking data to extract.")
    args = parser.parse_args()

    if args.user != 1:
        parser.error("the packaged demo assets contain templates and PMFs for user 1 only")
    if not args.bbmas_root.exists():
        parser.error(
            f"BB-MAS folder not found: {args.bbmas_root}\n"
            "Download BB-MAS, then unzip it under data/ so that data/BB-MAS_Dataset exists."
        )

    metadata = prepare_bbmas_demo(
        bbmas_root=args.bbmas_root,
        out_dir=args.out,
        user_id=args.user,
        imposter_user_id=args.imposter_user,
        stage=2,
        seconds=args.seconds,
    )

    print(f"Wrote demo data to {args.out}")
    for test in metadata["tests"]:
        print(f"- {test['name']}: {args.out / test['input']}")
    print(f"- metadata: {args.out / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
