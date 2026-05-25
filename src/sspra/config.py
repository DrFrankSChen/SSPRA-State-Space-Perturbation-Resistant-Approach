"""Configuration objects and BB-MAS Stage 2 modality metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ModalitySpec:
    """Description of one sensor modality used by the BB-MAS gait demo."""

    key: str
    label: str
    device: str
    sensor: str
    columns: Tuple[str, str, str]


STAGE2_MODALITIES: Tuple[ModalitySpec, ...] = (
    ModalitySpec(
        key="handphone_acc",
        label="HP_Acc",
        device="HandPhone",
        sensor="Accelerometer",
        columns=("HP_Acc_Xvalue", "HP_Acc_Yvalue", "HP_Acc_Zvalue"),
    ),
    ModalitySpec(
        key="handphone_gyro",
        label="HP_Gyr",
        device="HandPhone",
        sensor="Gyroscope",
        columns=("HP_Gyr_Xvalue", "HP_Gyr_Yvalue", "HP_Gyr_Zvalue"),
    ),
    ModalitySpec(
        key="pocketphone_acc",
        label="PP_Acc",
        device="PocketPhone",
        sensor="Accelerometer",
        columns=("PP_Acc_Xvalue", "PP_Acc_Yvalue", "PP_Acc_Zvalue"),
    ),
    ModalitySpec(
        key="pocketphone_gyro",
        label="PP_Gyr",
        device="PocketPhone",
        sensor="Gyroscope",
        columns=("PP_Gyr_Xvalue", "PP_Gyr_Yvalue", "PP_Gyr_Zvalue"),
    ),
)

STAGE2_MODALITY_BY_KEY: Dict[str, ModalitySpec] = {
    modality.key: modality for modality in STAGE2_MODALITIES
}


@dataclass
class MonitorConfig:
    """Runtime settings for SSPRA monitoring."""

    sampling_rate_hz: int = 100
    normal_half_life_seconds: float = 10.0
    suspense_half_life_seconds: float = 4.0
    stay_safe_threshold: float = 0.5
    back_to_safe_threshold: float = 0.6
    observation_window_steps: int = 200
    inspection_interval_min_steps: int = 200
    inspection_interval_max_steps: int = 400
    inspection_random_seed: Optional[int] = 13
    missing_zero_threshold: int = 100
    fusion_rule: str = "product"
    epsilon: float = 1e-12
    asset_dir: Path = Path("assets/stage2")
    bins: int = 25
    user_id: int = 1

    @classmethod
    def from_mapping(cls, values: Dict[str, object]) -> "MonitorConfig":
        """Build a config from a flat dict, ignoring unknown keys."""

        known = cls.__dataclass_fields__
        clean = {key: value for key, value in values.items() if key in known}
        if "asset_dir" in clean:
            clean["asset_dir"] = Path(str(clean["asset_dir"]))
        return cls(**clean)
