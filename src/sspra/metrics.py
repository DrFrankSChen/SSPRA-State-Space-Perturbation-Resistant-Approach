"""Temporal evaluation metrics from the SSPRA paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def evaluate_trace(
    states_path: Path,
    kind: str = "genuine",
    sampling_rate_hz: int = 100,
    test_duration: Optional[int] = None,
    attack_start: Optional[int] = None,
    missing_start: Optional[int] = None,
    missing_end: Optional[int] = None,
) -> Dict[str, object]:
    """Evaluate one SSPRA state trace with the paper's temporal metrics."""

    states = _read_states(states_path)
    if not states:
        return {"metrics": {}, "events": {}}

    attacked_times = _times_for_state(states, "Attacked")
    suspense_times = _times_for_state(states, "Suspense")
    trace_end = _required_int(test_duration, "test_duration") if test_duration is not None else int(float(states[-1]["Time"]))

    if kind == "attack":
        return _evaluate_test2_attack(
            attacked_times=attacked_times,
            suspense_times=suspense_times,
            attack_start=_required_int(attack_start, "attack_start"),
            trace_end=trace_end,
            sampling_rate_hz=sampling_rate_hz,
        )

    if kind == "missing":
        return _evaluate_test3_disconnection(
            attacked_times=attacked_times,
            suspense_times=suspense_times,
            missing_start=_required_int(missing_start, "missing_start"),
            missing_end=_required_int(missing_end, "missing_end"),
            trace_end=trace_end,
            sampling_rate_hz=sampling_rate_hz,
        )

    return _evaluate_test1_genuine(
        attacked_times=attacked_times,
        suspense_times=suspense_times,
        trace_end=trace_end,
        sampling_rate_hz=sampling_rate_hz,
    )


def evaluate_from_metadata(metadata_path: Path, results_dir: Path, out_path: Path) -> Dict[str, object]:
    metadata = json.loads(Path(metadata_path).read_text())
    sampling_rate_hz = int(metadata.get("sampling_rate_hz", 100))
    results_dir = Path(results_dir)
    tests: Dict[str, object] = {}

    for test in metadata.get("tests", []):
        states_path = _states_path_for_test(results_dir, test)
        if not states_path.exists():
            continue
        result = evaluate_trace(
            states_path,
            kind=str(test.get("kind", "genuine")),
            sampling_rate_hz=sampling_rate_hz,
            test_duration=_optional_int(test.get("test_duration", metadata.get("test_duration"))),
            attack_start=_optional_int(test.get("attack_start")),
            missing_start=_optional_int(test.get("missing_start")),
            missing_end=_optional_int(test.get("missing_end")),
        )
        result["input"] = test.get("input")
        result["description"] = test.get("description", "")
        tests[str(test.get("name", states_path.parent.name))] = result

    payload: Dict[str, object] = {
        "sampling_rate_hz": sampling_rate_hz,
        "time_unit": "seconds",
        "tests": tests,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _evaluate_test1_genuine(
    attacked_times: Sequence[int],
    suspense_times: Sequence[int],
    trace_end: int,
    sampling_rate_hz: int,
) -> Dict[str, object]:
    first_attacked = attacked_times[0] if attacked_times else None
    return {
        "metrics": {
            "FAR": int(first_attacked is not None),
            "RRT": _seconds(first_attacked if first_attacked is not None else trace_end, sampling_rate_hz),
        },
        "events": {
            "first_suspense": _maybe_seconds(suspense_times[0] if suspense_times else None, sampling_rate_hz),
            "first_attacked": _maybe_seconds(first_attacked, sampling_rate_hz),
            "trace_end": _seconds(trace_end, sampling_rate_hz),
        },
    }


def _evaluate_test2_attack(
    attacked_times: Sequence[int],
    suspense_times: Sequence[int],
    attack_start: int,
    trace_end: int,
    sampling_rate_hz: int,
) -> Dict[str, object]:
    first_false_alarm = next((time for time in attacked_times if time < attack_start), None)
    first_attack_alarm = next((time for time in attacked_times if time >= attack_start), None)
    first_suspense = next((time for time in suspense_times if time >= attack_start), None)
    false_alarm = first_false_alarm is not None
    true_alarm = (not false_alarm) and first_attack_alarm is not None
    false_pass = (not false_alarm) and first_attack_alarm is None

    return {
        "metrics": {
            "FAR": int(false_alarm),
            "TAR": int(true_alarm),
            "TCA": (
                _seconds(first_attack_alarm - attack_start, sampling_rate_hz)
                if true_alarm and first_attack_alarm is not None
                else None
            ),
            "FPaR": int(false_pass),
        },
        "events": {
            "attack_start": _seconds(attack_start, sampling_rate_hz),
            "first_suspense": _maybe_seconds(first_suspense, sampling_rate_hz),
            "first_attacked": _maybe_seconds(
                first_false_alarm if first_false_alarm is not None else first_attack_alarm,
                sampling_rate_hz,
            ),
            "trace_end": _seconds(trace_end, sampling_rate_hz),
        },
    }


def _evaluate_test3_disconnection(
    attacked_times: Sequence[int],
    suspense_times: Sequence[int],
    missing_start: int,
    missing_end: int,
    trace_end: int,
    sampling_rate_hz: int,
) -> Dict[str, object]:
    alarm_during_disconnection = next(
        (time for time in attacked_times if missing_start <= time < missing_end),
        None,
    )
    return {
        "metrics": {
            "FARDD": int(alarm_during_disconnection is not None),
            "TPaR": int(not attacked_times),
        },
        "events": {
            "disconnection_start": _seconds(missing_start, sampling_rate_hz),
            "disconnection_end": _seconds(missing_end, sampling_rate_hz),
            "first_suspense": _maybe_seconds(suspense_times[0] if suspense_times else None, sampling_rate_hz),
            "first_attacked": _maybe_seconds(attacked_times[0] if attacked_times else None, sampling_rate_hz),
            "trace_end": _seconds(trace_end, sampling_rate_hz),
        },
    }


def _states_path_for_test(results_dir: Path, test: Dict[str, object]) -> Path:
    if (results_dir / "states.csv").exists():
        return results_dir / "states.csv"
    return results_dir / str(test.get("name")) / "states.csv"


def _read_states(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _times_for_state(states: Sequence[Dict[str, str]], state: str) -> List[int]:
    return [int(float(row["Time"])) for row in states if row["State"] == state]


def _seconds(time_steps: int, sampling_rate_hz: int) -> float:
    return round(float(time_steps) / float(sampling_rate_hz), 6)


def _maybe_seconds(value: Optional[int], sampling_rate_hz: int) -> Optional[float]:
    if value is None:
        return None
    return _seconds(value, sampling_rate_hz)


def _optional_int(value: object) -> Optional[int]:
    if value in ("", None):
        return None
    return int(float(value))


def _required_int(value: Optional[int], name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required metadata field: {name}")
    return int(value)
