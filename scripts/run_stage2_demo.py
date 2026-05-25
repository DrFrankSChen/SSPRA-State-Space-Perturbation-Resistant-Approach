#!/usr/bin/env python3
"""Run SSPRA on the prepared BB-MAS Stage 2 demo streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sspra.config import MonitorConfig  # noqa: E402
from sspra.monitoring import SSPRAMonitor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the SSPRA Stage 2 gait demo.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "experiments" / "stage2_demo.yaml")
    parser.add_argument("--demo-dir", type=Path, default=REPO_ROOT / "data" / "demo" / "stage2")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "demo")
    args = parser.parse_args()

    metadata_path = args.demo_dir / "metadata.json"
    if not metadata_path.exists():
        parser.error(f"missing {metadata_path}; run `python scripts/prepare_demo_data.py` first")

    config = _load_config(args.config)
    monitor = SSPRAMonitor(config)
    metadata = json.loads(metadata_path.read_text())

    for test in metadata.get("tests", []):
        input_path = args.demo_dir / str(test["input"])
        output_dir = args.out / str(test["name"])
        result = monitor.simulate_gait_monitoring(input_path, user_id=config.user_id, out_dir=output_dir)
        print(f"{test['name']}: wrote {len(result.states)} state records to {output_dir}")
    return 0


def _load_config(path: Path) -> MonitorConfig:
    values = _read_flat_yaml(path)
    if "asset_dir" in values:
        asset_dir = Path(str(values["asset_dir"]))
        if not asset_dir.is_absolute():
            values["asset_dir"] = str((REPO_ROOT / asset_dir).resolve())
    return MonitorConfig.from_mapping(values)


def _read_flat_yaml(path: Path) -> Dict[str, object]:
    values: Dict[str, object] = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = _parse_scalar(value.strip().strip("\"'"))
    return values


def _parse_scalar(value: str) -> object:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


if __name__ == "__main__":
    raise SystemExit(main())
