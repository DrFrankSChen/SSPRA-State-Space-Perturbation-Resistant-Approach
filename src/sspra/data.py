"""CSV I/O, validation, and BB-MAS demo extraction helpers."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .config import STAGE2_MODALITIES

STAGE2_COLUMNS: Tuple[str, ...] = tuple(
    column for modality in STAGE2_MODALITIES for column in modality.columns
)


def write_csv_dicts(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_stage2_timeseries(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in STAGE2_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Stage 2 CSV is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Stage 2 CSV has no data rows: {path}")
    return fieldnames, rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare_bbmas_demo(
    bbmas_root: Path,
    out_dir: Path,
    user_id: int,
    imposter_user_id: int,
    stage: int = 2,
    seconds: int = 45,
    seed: int = 13,
) -> Dict[str, object]:
    """Extract a short Stage 2 demo from a locally downloaded BB-MAS dataset."""

    if stage != 2:
        raise ValueError("The packaged demo assets support Stage 2 only.")

    random.seed(seed)
    raw_root = _resolve_raw_root(Path(bbmas_root))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    genuine = _extract_stage2_walk2(raw_root, user_id, seconds)
    imposter = _extract_stage2_walk2(raw_root, imposter_user_id, seconds)

    test1_path = out_dir / f"test1_user{user_id}_genuine.csv"
    test2_path = out_dir / f"test2_user{user_id}_attack_user{imposter_user_id}.csv"
    test3_path = out_dir / f"test3_user{user_id}_missing_pp_gyr.csv"

    write_csv_dicts(test1_path, genuine, STAGE2_COLUMNS)

    missing_rows = [row.copy() for row in genuine]
    missing_start = min(1000, max(len(missing_rows) // 3, 0))
    missing_end = min(missing_start + 500, len(missing_rows))
    for row in missing_rows[missing_start:missing_end]:
        for column in ("PP_Gyr_Xvalue", "PP_Gyr_Yvalue", "PP_Gyr_Zvalue"):
            row[column] = "0"
    write_csv_dicts(test3_path, missing_rows, STAGE2_COLUMNS)

    attack_rows = [row.copy() for row in genuine]
    attack_start = min(1000, max(len(attack_rows) // 3, 0))
    for index in range(attack_start, len(attack_rows)):
        if index - attack_start >= len(imposter):
            break
        for column in STAGE2_COLUMNS:
            attack_rows[index][column] = imposter[index - attack_start][column]
    write_csv_dicts(test2_path, attack_rows, STAGE2_COLUMNS)

    metadata: Dict[str, object] = {
        "source": "SU-AIS BB-MAS local download",
        "stage": stage,
        "sampling_rate_hz": 100,
        "test_duration": len(genuine),
        "user_id": user_id,
        "imposter_user_id": imposter_user_id,
        "tests": [
            {
                "name": "test1",
                "input": test1_path.name,
                "kind": "genuine",
                "description": f"Test 1 uses user {user_id}'s Stage 2 Walk2 stream with all four modalities unchanged.",
            },
            {
                "name": "test2",
                "input": test2_path.name,
                "kind": "attack",
                "description": (
                    f"Test 2 simulates a non-zero-effort attack from user {imposter_user_id} "
                    f"on all four modalities starting at row {attack_start}."
                ),
                "attack_start": attack_start,
                "attacked_modalities": [modality.label for modality in STAGE2_MODALITIES],
                "attack_source_user_id": imposter_user_id,
            },
            {
                "name": "test3",
                "input": test3_path.name,
                "kind": "missing",
                "description": (
                    "Test 3 simulates temporary disconnection of the PocketPhone gyroscope "
                    f"from row {missing_start} to row {missing_end - 1}."
                ),
                "missing_start": missing_start,
                "missing_end": missing_end,
                "missing_modalities": ["PP_Gyr"],
            },
        ],
    }
    write_json(out_dir / "metadata.json", metadata)
    return metadata


def _optional_float(value: object, default: float) -> float:
    if value in ("", None):
        return default
    return float(value)


def _resolve_raw_root(bbmas_root: Path) -> Path:
    candidates = [
        bbmas_root,
        bbmas_root / "BB-MAS_Dataset",
        bbmas_root / "Raw-BB-MAS_Dataset",
        bbmas_root / "BB-MAS_Dataset" / "Raw-BB-MAS_Dataset",
        bbmas_root / "Data" / "BB-MAS_Dataset" / "Raw-BB-MAS_Dataset",
    ]
    for candidate in candidates:
        if _looks_like_bbmas_user_root(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find BB-MAS user folders. Expected data/BB-MAS_Dataset to "
        "contain numeric user folders such as 1, 2, 3, ... ."
    )


def _looks_like_bbmas_user_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and child.name.isdigit():
            return True
    return False


def _extract_stage2_walk2(raw_root: Path, user_id: int, seconds: int) -> List[Dict[str, str]]:
    user_dir = raw_root / str(user_id)
    if not user_dir.exists():
        raise FileNotFoundError(f"BB-MAS user folder not found: {user_dir}")

    checkpoint_path = _first_existing(
        user_dir,
        [
            f"{user_id}_HandPhone_Checkpoints_(Samsung_S6).csv",
            f"{user_id}_HandPhone_Checkpoints_(HTC_One).csv",
        ],
    )
    checkpoints = _read_raw_csv(checkpoint_path)

    streams = {
        "HP_Acc": _read_walk2_stream(user_dir, user_id, "HandPhone", "Accelerometer", checkpoints),
        "HP_Gyr": _read_walk2_stream(user_dir, user_id, "HandPhone", "Gyroscope", checkpoints),
        "PP_Acc": _read_walk2_stream(user_dir, user_id, "PocketPhone", "Accelerometer", checkpoints),
        "PP_Gyr": _read_walk2_stream(user_dir, user_id, "PocketPhone", "Gyroscope", checkpoints),
    }
    target_rows = min(seconds * 100, *(len(stream) for stream in streams.values()))
    if target_rows <= 0:
        raise ValueError(f"No usable Stage 2 Walk2 rows found for user {user_id}.")

    combined: List[Dict[str, str]] = []
    for index in range(target_rows):
        row: Dict[str, str] = {}
        for prefix, stream in streams.items():
            source = stream[index]
            row[f"{prefix}_Xvalue"] = _value(source, "Xvalue", 2)
            row[f"{prefix}_Yvalue"] = _value(source, "Yvalue", 3)
            row[f"{prefix}_Zvalue"] = _value(source, "Zvalue", 4)
        combined.append(row)
    return combined


def _read_walk2_stream(
    user_dir: Path,
    user_id: int,
    device: str,
    sensor: str,
    checkpoints: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    filename = _sensor_filename(user_dir, user_id, device, sensor)
    rows = _read_raw_csv(filename)
    matched_lines = _match_lines(rows, checkpoints)
    start = matched_lines[7]
    end = matched_lines[8] + 1
    return rows[start:end]


def _sensor_filename(user_dir: Path, user_id: int, device: str, sensor: str) -> Path:
    device_suffixes = ["Samsung_S6", "HTC_One"] if device == "HandPhone" else ["Samsung_S6", "Nexus9"]
    names = [f"{user_id}_{device}_{sensor}_({suffix}).csv" for suffix in device_suffixes]
    return _first_existing(user_dir, names)


def _first_existing(parent: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = parent / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these files exist under {parent}: {list(names)}")


def _read_raw_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _match_lines(data: Sequence[Dict[str, str]], checkpoints: Sequence[Dict[str, str]]) -> List[int]:
    lines: List[int] = []
    start = 0
    for checkpoint in checkpoints[:16]:
        target = _split_time(checkpoint["time"])
        line_number = _match_time(data, target, start)
        lines.append(line_number)
        start = line_number + 1
    if len(lines) < 16:
        raise ValueError("BB-MAS checkpoint file has fewer than 16 checkpoints.")
    return lines


def _match_time(data: Sequence[Dict[str, str]], target_time: Sequence[str], start_line: int) -> int:
    for index in range(start_line, len(data)):
        current = _split_time(data[index]["time"])
        if current[0] == target_time[0] and current[1] == target_time[1]:
            current_second = float(current[2] + current[3])
            target_second = float(target_time[2] + target_time[3])
            if abs(current_second - target_second) <= 10:
                return index
            if current_second > target_second:
                previous = _split_time(data[index - 1]["time"]) if index > 0 else current
                previous_second = float(previous[2] + previous[3])
                if abs(current_second - target_second) < abs(previous_second - target_second):
                    return index
                return max(index - 1, 0)
    return len(data) - 1


def _split_time(value: str) -> List[str]:
    return re.split(r"[: .]", value.split(" ")[1])


def _value(row: Dict[str, str], preferred_name: str, fallback_index: int) -> str:
    if preferred_name in row:
        return row[preferred_name]
    values = list(row.values())
    return values[fallback_index]
