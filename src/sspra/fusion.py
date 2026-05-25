"""Vertical fusion of per-modality observation likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .stm import ATTACKED, SAFE, SUSPENSE


@dataclass(frozen=True)
class ObservationBatch:
    """All modality likelihoods obtained at one inspection time."""

    time: int
    observations: List[Dict[str, float]]
    delta: Optional[int] = None


def normalize(values: Dict[str, float], epsilon: float = 1e-12) -> Dict[str, float]:
    """Normalize a state-probability dictionary."""

    total = sum(max(value, 0.0) for value in values.values())
    if total <= epsilon:
        states = list(values.keys())
        return {state: 1.0 / len(states) for state in states}
    return {state: max(value, 0.0) / total for state, value in values.items()}


def fuse_observations(
    observations: Iterable[Dict[str, float]],
    rule: str = "product",
    epsilon: float = 1e-12,
) -> Optional[Dict[str, float]]:
    """Fuse available modality likelihoods using Eq. (1) from the paper.

    Missing modalities should be omitted by the caller. PMF-bin likelihoods are
    floored by epsilon, matching the paper's non-zero PMF-bin guard.
    """

    available: List[Dict[str, float]] = []
    for observation in observations:
        intra = max(float(observation.get("intra", 0.0)), epsilon)
        inter = max(float(observation.get("inter", 0.0)), epsilon)
        available.append({"intra": intra, "inter": inter})

    if not available:
        return None

    if rule == "product":
        evidence = {SAFE: 1.0, SUSPENSE: 1.0, ATTACKED: 1.0}
        for item in available:
            evidence[SAFE] *= item["intra"]
            evidence[SUSPENSE] *= item["inter"]
            evidence[ATTACKED] *= item["inter"]
        return evidence

    raise ValueError(f"Unsupported fusion rule: {rule}")
