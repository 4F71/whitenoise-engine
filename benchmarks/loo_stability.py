# UltraGen V2 — C2 Benchmark
# Leave-One-Out (LOO) Stabilite Testi
# Amaç: Feature → parametre projeksiyonunun davranışsal stabilitesini ölçmek
# Not: Model siyah kutudur; burada tanımlanmaz, eğitilmez veya seçilmez.

import argparse
import importlib
import json
from typing import Callable, Dict, List

import numpy as np


def load_model_callable(path: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Dışarıdan verilen siyah-kutu model callable'ını yükler.
    Format: module_name:function_name
    """
    try:
        module_name, fn_name = path.split(":")
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
    except Exception as exc:
        raise RuntimeError(f"Model callable yüklenemedi: {path}") from exc

    if not callable(fn):
        raise TypeError("Model callable olmalıdır")

    return fn


def loo_stability(
    features: np.ndarray,
    model_fn: Callable[[np.ndarray], np.ndarray],
    spread: np.ndarray,
) -> Dict[str, object]:
    """
    Leave-One-Out stabilite testini uygular.

    Ölçümler:
    - Parametre bazlı mutlak sapma (L1)
    - Spread dışına çıkan parametre sayısı
    """
    if features.ndim != 2:
        raise ValueError("Feature matrisi (N, F) boyutunda olmalıdır")

    n_samples = features.shape[0]

    full_pred = model_fn(features)
    if full_pred.shape[0] != n_samples:
        raise ValueError("Model çıktısı boyutu (N, P) olmalıdır")

    if spread.ndim != 1 or spread.shape[0] != full_pred.shape[1]:
        raise ValueError("Spread vektörü (P,) boyutunda olmalıdır")

    l1_per_sample: List[float] = []
    out_of_spread_per_sample: List[int] = []

    for i in range(n_samples):
        ref_pred = full_pred[i]
        loo_pred = model_fn(features[i : i + 1])[0]

        diff = np.abs(ref_pred - loo_pred)
        l1_per_sample.append(float(np.mean(diff)))

        out_count = int(np.sum(diff > spread))
        out_of_spread_per_sample.append(out_count)

    return {
        "l1_per_sample": l1_per_sample,
        "l1_mean": float(np.mean(l1_per_sample)),
        "l1_std": float(np.std(l1_per_sample)),
        "l1_max": float(np.max(l1_per_sample)),
        "out_of_spread_per_sample": out_of_spread_per_sample,
        "out_of_spread_mean": float(np.mean(out_of_spread_per_sample)),
        "out_of_spread_max": int(np.max(out_of_spread_per_sample)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UltraGen V2 C2 — Leave-One-Out Stabilite Testi"
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
        "--spread",
        required=True,
        help="Parametre spread vektörü (.npy), normalize uzayda (P,)",
    )
    parser.add_argument(
        "--output",
        default="loo_stability_results.json",
        help="Sonuçların yazılacağı JSON dosyası",
    )

    args = parser.parse_args()

    features = np.load(args.features)
    spread = np.load(args.spread)
    model_fn = load_model_callable(args.model)

    results = loo_stability(
        features=features,
        model_fn=model_fn,
        spread=spread,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("LOO stabilite testi tamamlandı.")
    print("L1 ortalama:", results["l1_mean"])
    print("L1 maksimum:", results["l1_max"])
    print("Ortalama spread dışı parametre sayısı:", results["out_of_spread_mean"])


if __name__ == "__main__":
    np.random.seed(0)
    main()
