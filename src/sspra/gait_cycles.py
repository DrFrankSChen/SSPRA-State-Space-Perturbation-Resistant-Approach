"""Small gait-cycle extraction helpers for the packaged BB-MAS demo."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def local_minima(values: Sequence[float], order: int = 30) -> List[int]:
    """Return indices that are less than or equal to neighbors within `order`."""

    if len(values) < (2 * order + 1):
        return []
    minima: List[int] = []
    for index in range(order, len(values) - order):
        value = values[index]
        window = values[index - order : index + order + 1]
        if all(value <= other for other in window):
            if not minima or index - minima[-1] >= order:
                minima.append(index)
    return minima


def linear_resample(values: Sequence[float], target_size: int = 100) -> List[float]:
    """Resample a 1D signal to `target_size` with linear interpolation."""

    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if not values:
        return [0.0] * target_size
    if len(values) == 1:
        return [float(values[0])] * target_size
    if target_size == 1:
        return [float(values[0])]

    source_last = len(values) - 1
    output: List[float] = []
    for target_index in range(target_size):
        position = target_index * source_last / (target_size - 1)
        left = int(position)
        right = min(left + 1, source_last)
        fraction = position - left
        output.append(float(values[left]) * (1.0 - fraction) + float(values[right]) * fraction)
    return output


def extract_gait_cycle(
    window_xyz: Sequence[Tuple[float, float, float]],
    target_size: int = 100,
    min_cycle_steps: int = 70,
    max_cycle_steps: int = 180,
) -> Optional[List[List[float]]]:
    """Extract and resample one 3-axis gait cycle from a time window.

    This helper is intentionally compact for the public demo. It uses z-axis
    minima to estimate cycle boundaries, then resamples the same interval from
    all three axes.
    """

    if len(window_xyz) < min_cycle_steps:
        return None

    x_values = [float(row[0]) for row in window_xyz]
    y_values = [float(row[1]) for row in window_xyz]
    z_values = [float(row[2]) for row in window_xyz]

    minima = local_minima(z_values, order=30)
    start = 0
    end = min(len(window_xyz), max_cycle_steps)
    for first, second in zip(minima, minima[1:]):
        length = second - first
        if min_cycle_steps <= length <= max_cycle_steps:
            start = first
            end = second
            break
    else:
        if minima:
            start = minima[0]
            end = min(len(window_xyz), start + max_cycle_steps)

    if end - start < min_cycle_steps:
        return None

    return [
        linear_resample(x_values[start:end], target_size),
        linear_resample(y_values[start:end], target_size),
        linear_resample(z_values[start:end], target_size),
    ]
