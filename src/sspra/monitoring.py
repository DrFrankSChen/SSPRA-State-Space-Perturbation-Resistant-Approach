"""High-level SSPRA monitoring APIs."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .config import MonitorConfig, STAGE2_MODALITIES
from .data import validate_stage2_timeseries, write_csv_dicts
from .fusion import ObservationBatch, fuse_observations, normalize
from .gait_cycles import extract_gait_cycle
from .stm import ATTACKED, SAFE, SUSPENSE, StateTransitionMachine


@dataclass(frozen=True)
class ProbabilityRecord:
    time: int
    safe: float
    suspense: float
    attacked: float

@dataclass(frozen=True)
class StateRecord:
    time: int
    state: str

@dataclass
class MonitorResult:
    probabilities: List[ProbabilityRecord]
    states: List[StateRecord]

    def write(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv_dicts(
            out_dir / "probabilities.csv",
            (
                {
                    "Time": row.time,
                    "Safe": row.safe,
                    "Suspense": row.suspense,
                    "Attacked": row.attacked,
                }
                for row in self.probabilities
            ),
            ("Time", "Safe", "Suspense", "Attacked"),
        )
        write_csv_dicts(
            out_dir / "states.csv",
            ({"Time": row.time, "State": row.state} for row in self.states),
            ("Time", "State"),
        )


class SSPRAMonitor:
    """Run SSPRA inference and the BB-MAS Stage 2 monitoring simulation."""

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self.config = config or MonitorConfig()
        self.stm = StateTransitionMachine(self.config)

    def run_sspra_inference(
        self,
        batches: Iterable[ObservationBatch],
        initial_probabilities: Optional[ProbabilityRecord] = None,
        initial_state: str = SAFE,
        include_initial: bool = True,
    ) -> MonitorResult:
        """Apply SSPRA state inference to prepared inspection observations."""

        initial = initial_probabilities or ProbabilityRecord(time=0, safe=1.0, suspense=0.0, attacked=0.0)
        probabilities = [initial]
        states = [StateRecord(time=initial.time, state=initial_state)]

        for batch in sorted(batches, key=lambda item: item.time):
            previous_probability = probabilities[-1]
            previous = {
                SAFE: previous_probability.safe,
                SUSPENSE: previous_probability.suspense,
                ATTACKED: previous_probability.attacked,
            }
            delta = batch.delta
            if delta is None:
                delta = max(batch.time - previous_probability.time, 0)
            previous_state = states[-1].state
            past = self.stm.past_probabilities(previous, delta, previous_state=previous_state)
            evidence = fuse_observations(
                batch.observations,
                rule=self.config.fusion_rule,
                epsilon=self.config.epsilon,
            )
            if evidence is None:
                current = normalize(past, epsilon=self.config.epsilon)
            else:
                if previous_state == SUSPENSE:
                    current = normalize(
                        {
                            SAFE: past[SAFE] * evidence[SAFE],
                            SUSPENSE: 0.0,
                            ATTACKED: past[ATTACKED] * evidence[ATTACKED],
                        },
                        epsilon=self.config.epsilon,
                    )
                else:
                    current = normalize(
                        {
                            SAFE: past[SAFE] * evidence[SAFE],
                            SUSPENSE: past[SUSPENSE] * evidence[SUSPENSE],
                            ATTACKED: past[ATTACKED] * evidence[ATTACKED],
                        },
                        epsilon=self.config.epsilon,
                    )
            next_state = self.stm.decide(current, previous_state)
            if previous_state == SUSPENSE and next_state == SAFE:
                current = {
                    SAFE: current[SAFE],
                    SUSPENSE: current[ATTACKED],
                    ATTACKED: 0.0,
                }

            probabilities.append(
                ProbabilityRecord(
                    time=batch.time,
                    safe=current[SAFE],
                    suspense=current[SUSPENSE],
                    attacked=current[ATTACKED],
                )
            )
            states.append(
                StateRecord(
                    time=batch.time,
                    state=next_state,
                )
            )
        if include_initial:
            return MonitorResult(probabilities=probabilities, states=states)
        return MonitorResult(probabilities=probabilities[1:], states=states[1:])

    def simulate_gait_monitoring(
        self,
        input_path: Path,
        user_id: Optional[int] = None,
        out_dir: Optional[Path] = None,
    ) -> MonitorResult:
        """Simulate continuous Stage 2 gait monitoring over a time series."""

        from .gait_model import GaitAuthenticator

        _validate_monitoring_intervals(self.config)
        user = user_id if user_id is not None else self.config.user_id
        _, rows = validate_stage2_timeseries(Path(input_path))
        authenticator = GaitAuthenticator(self.config.asset_dir, bins=self.config.bins)
        missing = authenticator.check_assets()
        if missing:
            formatted = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing gait assets:\n{formatted}")
        probabilities = [ProbabilityRecord(time=0, safe=1.0, suspense=0.0, attacked=0.0)]
        states = [StateRecord(time=0, state=SAFE)]
        previous_time = 0
        rng = random.Random(self.config.inspection_random_seed)
        current_time = rng.randint(
            self.config.inspection_interval_min_steps,
            self.config.inspection_interval_max_steps,
        )

        while current_time <= len(rows):
            observations: List[Dict[str, float]] = []
            for modality in STAGE2_MODALITIES:
                window = _extract_window(
                    rows,
                    modality.columns,
                    current_time - self.config.observation_window_steps,
                    self.config.observation_window_steps,
                )
                if _zero_count(window) > self.config.missing_zero_threshold:
                    continue
                cycle = extract_gait_cycle(window)
                if cycle is None:
                    continue
                intra, inter, raw_score = authenticator.score_cycle(modality.key, user, cycle)
                observations.append(
                    {
                        "modality": modality.key,
                        "intra": intra,
                        "inter": inter,
                        "raw_score": raw_score,
                    }
                )

            result = self.run_sspra_inference(
                [
                    ObservationBatch(
                        time=current_time,
                        delta=current_time - previous_time,
                        observations=observations,
                    )
                ],
                initial_probabilities=probabilities[-1],
                initial_state=states[-1].state,
                include_initial=False,
            )
            probabilities.extend(result.probabilities)
            states.extend(result.states)
            previous_time = current_time

            if states[-1].state == ATTACKED:
                break
            if states[-1].state == SUSPENSE:
                current_time += self.config.observation_window_steps
            else:
                current_time += rng.randint(
                    self.config.inspection_interval_min_steps,
                    self.config.inspection_interval_max_steps,
                )

        result = MonitorResult(probabilities=probabilities, states=states)
        if out_dir is not None:
            result.write(Path(out_dir))
        return result


def _extract_window(
    rows: Sequence[Dict[str, str]],
    columns: Sequence[str],
    start: int,
    length: int,
) -> List[tuple]:
    output = []
    for row in rows[start : start + length]:
        output.append(tuple(float(row[column]) for column in columns))
    return output


def _inspection_times(length: int, config: MonitorConfig) -> List[int]:
    """Return paper-style inspection times for a real-time stream."""

    _validate_monitoring_intervals(config)
    if length < config.observation_window_steps:
        return []

    rng = random.Random(config.inspection_random_seed)
    current_time = 0
    times: List[int] = []
    while True:
        current_time += rng.randint(
            config.inspection_interval_min_steps,
            config.inspection_interval_max_steps,
        )
        if current_time > length:
            break
        times.append(current_time)
    return times


def _validate_monitoring_intervals(config: MonitorConfig) -> None:
    if config.observation_window_steps <= 0:
        raise ValueError("observation_window_steps must be positive")
    if config.inspection_interval_min_steps <= 0:
        raise ValueError("inspection_interval_min_steps must be positive")
    if config.inspection_interval_max_steps < config.inspection_interval_min_steps:
        raise ValueError("inspection_interval_max_steps must be >= inspection_interval_min_steps")
    if config.inspection_interval_min_steps < config.observation_window_steps:
        raise ValueError("inspection_interval_min_steps must be >= observation_window_steps")


def _zero_count(window: Sequence[Sequence[float]]) -> int:
    return sum(1 for row in window for value in row if float(value) == 0.0)
