"""State transition machines used by SSPRA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .config import MonitorConfig

SAFE = "Safe"
SUSPENSE = "Suspense"
ATTACKED = "Attacked"


@dataclass
class StateTransitionMachine:
    """Dual STM logic from the SSPRA paper.

    P1 is used under normal operation and maps Normal/Suspense to
    Normal/Suspense. P2 is used after a Suspense decision and maps
    Normal/Suspense to Normal/Alert.
    """

    config: MonitorConfig

    @property
    def k1(self) -> float:
        return -math.log(2.0) / self.config.normal_half_life_seconds

    @property
    def k2(self) -> float:
        return -math.log(2.0) / self.config.suspense_half_life_seconds

    def _seconds(self, delta_steps: int) -> float:
        return max(float(delta_steps), 0.0) / float(self.config.sampling_rate_hz)

    def p(self, delta_steps: int) -> float:
        return math.exp(self.k1 * self._seconds(delta_steps))

    def u_or_h(self, delta_steps: int) -> float:
        return math.exp(self.k2 * self._seconds(delta_steps))

    def safe_to_safe(self, delta_steps: int) -> float:
        return self.p(delta_steps)

    def safe_to_suspense(self, delta_steps: int) -> float:
        return 1.0 - self.p(delta_steps)

    def p1_suspense_to_safe(self, delta_steps: int) -> float:
        return self.u_or_h(delta_steps)

    def p1_suspense_to_suspense(self, delta_steps: int) -> float:
        return 1.0 - self.u_or_h(delta_steps)

    def p2_suspense_to_safe(self, delta_steps: int) -> float:
        return self.u_or_h(delta_steps)

    def p2_suspense_to_attacked(self, delta_steps: int) -> float:
        return 1.0 - self.u_or_h(delta_steps)

    def suspense_to_attacked(self, delta_steps: int) -> float:
        return self.p2_suspense_to_attacked(delta_steps)

    def attacked_to_attacked(self, delta_steps: int) -> float:
        return 1.0

    def past_probabilities(
        self,
        previous: Dict[str, float],
        delta_steps: int,
        previous_state: str = SAFE,
    ) -> Dict[str, float]:
        """Apply P1 or P2 to previous state probabilities."""

        safe_prev = previous.get(SAFE, 0.0)
        suspense_prev = previous.get(SUSPENSE, 0.0)
        attacked_prev = previous.get(ATTACKED, 0.0)

        if previous_state == SUSPENSE:
            return {
                SAFE: self.p(delta_steps) * safe_prev + self.u_or_h(delta_steps) * suspense_prev,
                SUSPENSE: 0.0,
                ATTACKED: (
                    (1.0 - self.p(delta_steps)) * safe_prev
                    + (1.0 - self.u_or_h(delta_steps)) * suspense_prev
                    + attacked_prev
                ),
            }

        if previous_state == ATTACKED:
            return {SAFE: 0.0, SUSPENSE: 0.0, ATTACKED: 1.0}

        return {
            SAFE: self.p(delta_steps) * safe_prev + self.u_or_h(delta_steps) * suspense_prev,
            SUSPENSE: (
                (1.0 - self.p(delta_steps)) * safe_prev
                + (1.0 - self.u_or_h(delta_steps)) * suspense_prev
            ),
            ATTACKED: attacked_prev,
        }

    def decide(self, probabilities: Dict[str, float], previous_state: str) -> str:
        """Convert state probabilities to the SSPRA discrete state."""

        p_safe = probabilities.get(SAFE, 0.0)

        if previous_state == SAFE:
            if p_safe >= self.config.stay_safe_threshold:
                return SAFE
            return SUSPENSE

        if previous_state == SUSPENSE:
            if p_safe >= self.config.back_to_safe_threshold:
                return SAFE
            return ATTACKED

        return ATTACKED
