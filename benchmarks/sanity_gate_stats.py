# xxxDSP V2 — C2 Benchmark
# Sanity Gate Çarpma Oranı
# Amaç: ML çıktılarının sanity gate tarafından ne sıklıkla clamp edildiğini ölçmek
# Not: Model ve sanity gate siyah kutudur; burada tanımlanmaz veya değiştirilmez.

import argparse
import importlib
import json
from typing import Callable, Dict

import numpy as np


def load_callable(path: str) -> Callable:
    """
    Dışarıdan verilen callable'ı yükler.
    Format: module_name:function_name
    """
    try:
        module_name, fn_name = path.split(":")
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
    except Exception as exc:
        raise RuntimeError(f"Callable yüklenemedi: {path}") from exc

    if not callable(fn):
        raise TypeError("Verilen nesne callable olmalıdır")

    return fn


def sanity_gate_stats(
    features: np.ndarray,
    model_fn: Callable[[np.ndarray], np.ndarray],
    sanity_gate_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, object]:
    """
    Sanity gate çarpma oranlarını hesaplar.
    """
    if features.ndim != 2:
        raise ValueError("Feature matrisi (N, F) boyutunda olmalıdır")

    n_samples = features.shape[0]

    raw_params = model_fn(features)
    gated_params = np.zeros_like(raw_params)

    clamp_counts = np.zeros(raw_params.shape[1], dtype=int)

    for i in range(n_samples):
        gated = sanity_gate_fn(raw_params[i])
        gated_params[i] = gated

        clamp_mask = gated != raw_params[i]
        clamp_counts += clamp_mask.astype(int)

    total_params = n_samples * raw_params.shape[1]

    return {
        "per_param_clamp_count": clamp_counts.tolist(),
        "per_param_clamp_ratio": (clamp_counts / n_samples).tolist(),
        "total_clamp_ratio": float(np.sum(clamp_counts) / total_params),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="xxxDSP V2 C2 — Sanity Gate Çarpma Oranı"
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Önceden hesaplanmış feature matrisi (.npy), şekil (N, F)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model callable yolu (module:function)",
    )
    parser.add_argument(
        "--sanity-gate",
        required=True,
        help="Sanity gate callable yolu (module:function)",
    )
    parser.add_argument(
        "--output",
        default="sanity_gate_stats.json",
        help="Çıktı JSON dosyası",
    )

    args = parser.parse_args()

    features = np.load(args.features)
    model_fn = load_callable(args.model)
    sanity_gate_fn = load_callable(args.sanity_gate)

    results = sanity_gate_stats(
        features=features,
        model_fn=model_fn,
        sanity_gate_fn=sanity_gate_fn,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Sanity gate çarpma oranı testi tamamlandı.")
    print("Toplam clamp oranı:", results["total_clamp_ratio"])


if __name__ == "__main__":
    main()
