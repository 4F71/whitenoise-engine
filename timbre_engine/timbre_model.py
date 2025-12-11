from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timbre_engine.timbre_params import DEFAULT_RANGE_MAP, NOISE_COLORS, TimbreParams, clamp_params


def _params_to_tensors(params: Sequence[TimbreParams], numeric_keys: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of TimbreParams objects into numeric targets and noise_color indices."""
    numeric: List[List[float]] = []
    color_idx: List[int] = []
    color_to_idx: Mapping[str, int] = {c: i for i, c in enumerate(NOISE_COLORS)}

    for p in params:
        data = asdict(p)
        numeric.append([float(data[k]) for k in numeric_keys])
        color_idx.append(color_to_idx.get(str(p.noise_color).lower(), 0))

    return torch.tensor(numeric, dtype=torch.float32), torch.tensor(color_idx, dtype=torch.long)


class TimbreModel(nn.Module):
    """Simple MLP that regresses timbre parameters and classifies noise_color."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, num_hidden: int = 2, dropout: float = 0.1):
        super().__init__()
        self.numeric_keys: Tuple[str, ...] = tuple(DEFAULT_RANGE_MAP.keys())
        self.num_numeric = len(self.numeric_keys)
        self.num_colors = len(NOISE_COLORS)

        layers: List[nn.Module] = []
        in_dim = feature_dim
        for _ in range(num_hidden):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.num_numeric + self.num_colors))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (numeric_pred, noise_color_logits)."""
        out = self.net(x)
        numeric_pred = out[:, : self.num_numeric]
        color_logits = out[:, self.num_numeric :]
        return numeric_pred, color_logits

    def predict_params(self, features: np.ndarray | torch.Tensor, clamp: bool = True, device: str | torch.device | None = None) -> TimbreParams:
        """Run a forward pass and convert outputs to TimbreParams."""
        self.eval()
        device = device or next(self.parameters()).device
        feats = torch.as_tensor(features, dtype=torch.float32, device=device)
        if feats.ndim == 1:
            feats = feats.unsqueeze(0)
        with torch.no_grad():
            numeric_pred, color_logits = self.forward(feats)
        numeric_vec = numeric_pred[0].cpu().numpy()
        color_idx = int(color_logits[0].argmax(dim=-1))
        param_dict: Dict[str, float | str] = {k: v for k, v in zip(self.numeric_keys, numeric_vec)}
        param_dict["noise_color"] = NOISE_COLORS[color_idx % self.num_colors]
        params = TimbreParams(**param_dict)
        return clamp_params(params) if clamp else params

    def train_step(
        self,
        batch_features: torch.Tensor,
        batch_params: Sequence[TimbreParams],
        optimizer: torch.optim.Optimizer,
        mse_weight: float = 1.0,
        ce_weight: float = 0.2,
    ) -> Dict[str, float]:
        """One training step with combined MSE (numeric) and CE (noise_color)."""
        self.train()
        target_numeric, target_color = _params_to_tensors(batch_params, self.numeric_keys)
        target_numeric = target_numeric.to(batch_features.device)
        target_color = target_color.to(batch_features.device)

        pred_numeric, color_logits = self.forward(batch_features)
        mse = F.mse_loss(pred_numeric, target_numeric)
        ce = F.cross_entropy(color_logits, target_color)
        loss = mse_weight * mse + ce_weight * ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": float(loss.item()), "mse": float(mse.item()), "ce": float(ce.item())}

    def save(self, path: str | Path) -> None:
        """Save model weights and minimal config."""
        payload = {
            "state_dict": self.state_dict(),
            "feature_dim": self.net[0].in_features,  # type: ignore[index]
            "hidden_dim": self.net[0].out_features,  # type: ignore[index]
            "num_hidden": sum(isinstance(m, nn.Linear) for m in self.net) - 1,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device | None = None) -> "TimbreModel":
        """Load model with stored config."""
        payload = torch.load(Path(path), map_location=map_location)
        model = cls(
            feature_dim=payload["feature_dim"],
            hidden_dim=payload["hidden_dim"],
            num_hidden=payload["num_hidden"],
        )
        model.load_state_dict(payload["state_dict"])
        return model
