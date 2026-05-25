"""Siamese gait authenticator used by the BB-MAS Stage 2 demo."""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .config import STAGE2_MODALITY_BY_KEY, STAGE2_MODALITIES


class GaitAuthenticator:
    """Load packaged gait assets and convert a gait cycle to SSPRA likelihoods."""

    def __init__(self, asset_dir: Path, bins: int = 25, device: str = "cpu") -> None:
        self.asset_dir = Path(asset_dir)
        self.bins = bins
        self.device = device
        self._template_cache: Dict[str, object] = {}
        self._model_cache: Dict[str, object] = {}

    def check_assets(self) -> List[Path]:
        """Return missing required Stage 2 asset paths."""

        missing: List[Path] = []
        for modality in STAGE2_MODALITIES:
            for path in (
                self.model_path(modality.key),
                self.template_path(modality.key),
            ):
                if not path.exists():
                    missing.append(path)
        rates = self.asset_dir / "pmfs" / "Bins-25-Rates.csv"
        if not rates.exists():
            missing.append(rates)
            return missing

        with rates.open(newline="") as handle:
            user_ids = [row["UserID"] for row in csv.DictReader(handle)]
        for user_id in user_ids:
            for modality in STAGE2_MODALITIES:
                base = (
                    self.asset_dir
                    / "pmfs"
                    / "PdfsAndBins"
                    / f"Bins-{self.bins}"
                    / modality.sensor
                    / str(int(float(user_id)))
                )
                prefix = f"{int(float(user_id))}_{modality.device}_{modality.sensor}"
                for suffix in ("intra_pdf", "intra_bin_edges", "inter_pdf", "inter_bin_edges"):
                    path = base / f"{prefix}_{suffix}.csv"
                    if not path.exists():
                        missing.append(path)
        return missing

    def score_cycle(
        self, modality_key: str, user_id: int, cycle: Sequence[Sequence[float]]
    ) -> Tuple[float, float, float]:
        """Return intra, inter, and raw Siamese score for one 3-axis cycle."""

        torch = _import_torch()
        model = self._load_model(modality_key, torch)
        template = self._load_template(modality_key)[str(user_id)]

        template_tensor = torch.as_tensor(template, dtype=torch.float64)
        cycle_tensor = torch.as_tensor(cycle, dtype=torch.float64)
        template_tensor = template_tensor[None, None, :, :].to(self.device)
        cycle_tensor = cycle_tensor[None, None, :, :].to(self.device)

        model.eval()
        with torch.no_grad():
            score = float(model(template_tensor, cycle_tensor).item())

        intra, inter = self.lookup_pmfs(modality_key, user_id, score)
        return intra, inter, score

    def lookup_pmfs(self, modality_key: str, user_id: int, score: float) -> Tuple[float, float]:
        modality = STAGE2_MODALITY_BY_KEY[modality_key]
        base = (
            self.asset_dir
            / "pmfs"
            / "PdfsAndBins"
            / f"Bins-{self.bins}"
            / modality.sensor
            / str(user_id)
        )
        prefix = f"{user_id}_{modality.device}_{modality.sensor}"
        intra_pdf = _read_single_column_csv(base / f"{prefix}_intra_pdf.csv")
        intra_edges = _read_single_column_csv(base / f"{prefix}_intra_bin_edges.csv")
        inter_pdf = _read_single_column_csv(base / f"{prefix}_inter_pdf.csv")
        inter_edges = _read_single_column_csv(base / f"{prefix}_inter_bin_edges.csv")

        intra = _lookup_histogram(score, intra_pdf, intra_edges)
        inter = _lookup_histogram(score, inter_pdf, inter_edges)
        return intra, inter

    def model_path(self, modality_key: str) -> Path:
        return self.asset_dir / "models" / modality_key / "BestModel.pth"

    def template_path(self, modality_key: str) -> Path:
        return self.asset_dir / "templates" / f"{modality_key}_template_3axis.pickle"

    def _load_template(self, modality_key: str) -> object:
        if modality_key not in self._template_cache:
            with self.template_path(modality_key).open("rb") as handle:
                self._template_cache[modality_key] = pickle.load(handle)
        return self._template_cache[modality_key]

    def _load_model(self, modality_key: str, torch):
        if modality_key not in self._model_cache:
            model = _make_siamese_model(torch).double().to(self.device)
            checkpoint = self.model_path(modality_key)
            try:
                state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(state)
            self._model_cache[modality_key] = model
        return self._model_cache[modality_key]


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Gait inference requires PyTorch. Install the repository dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return torch


def _make_siamese_model(torch):
    nn = torch.nn
    functional = torch.nn.functional

    class SiameseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn1 = nn.Sequential(
                nn.Conv2d(1, 128, (3, 5), stride=(1, 3)),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
            )
            self.cnn2 = nn.Sequential(
                nn.Conv1d(128, 128, 4, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2, stride=2),
            )
            self.embedding = nn.Sequential(
                nn.Linear(896, 512),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
                nn.Dropout(0.3),
                nn.ReLU(inplace=True),
            )
            self.score_layer = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

        def forward_once(self, x):
            output = self.cnn1(x)
            output = output.squeeze(2)
            output = self.cnn2(output)
            output = output.view(output.size(0), -1)
            return self.embedding(output)

        def forward(self, x1, x2):
            y1 = self.forward_once(x1)
            y2 = self.forward_once(x2)
            dist = functional.pairwise_distance(y1, y2, 1, keepdim=True)
            return self.score_layer(dist)

    return SiameseNet()


def _read_single_column_csv(path: Path) -> List[float]:
    values: List[float] = []
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[0]))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"No numeric values found in {path}")
    return values


def _lookup_histogram(score: float, pdf: Sequence[float], edges: Sequence[float]) -> float:
    if len(edges) < 2:
        raise ValueError("Histogram edges must contain at least two values.")
    for index, probability in enumerate(pdf):
        lower = edges[min(index, len(edges) - 1)]
        upper = edges[min(index + 1, len(edges) - 1)]
        if lower < score <= upper:
            return float(probability)
    if score <= edges[0]:
        return float(pdf[0])
    return float(pdf[-1])
