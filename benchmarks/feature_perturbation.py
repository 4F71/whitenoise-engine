# xxxDSP V2 — C2 Benchmark
# Feature Perturbation Testi
# Amaç: Feature uzayındaki küçük değişimlerin parametre uzayında
#       monoton, sınırlı ve sezgisel tepkiler üretip üretmediğini gözlemlemek.
# Not: Model siyah kutudur; burada tanımlanmaz, eğitilmez veya seçilmez.

import argparse
import importlib
import json
from typing import Callable, Dict

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


def apply_perturbations(
    feature_vector: np.ndarray,
    feature_indices: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """
    Zorunlu perturbation'ları uygular.
    Feature vektörü normalize uzayda varsayılır.
    """
    perturbed = {}

    base = feature_vector.copy()

    rms_idx = feature_indices["RMS_mean"]
    rms_vec = base.copy()
    rms_vec[rms_idx] *= 1.10
    rms_vec[rms_idx] = np.clip(rms_vec[rms_idx], -1.0, 1.0)
    perturbed["RMS_mean_plus_10pct"] = rms_vec

    centroid_idx = feature_indices["Spectral_centroid_mean"]
    centroid_vec = base.copy()
    centroid_vec[centroid_idx] += 0.05
    centroid_vec[centroid_idx] = np.clip(
        centroid_vec[centroid_idx], -1.0, 1.0
    )
    perturbed["Spectral_centroid_mean_small_plus"] = centroid_vec

    tilt_idx = feature_indices["Spectral_tilt_estimate"]
    tilt_vec = base.copy()
    tilt_vec[tilt_idx] -= 0.10
    tilt_vec[tilt_idx] = np.clip(tilt_vec[tilt_idx], -1.0, 1.0)
    perturbed["Spectral_tilt_minus"] = tilt_vec

    am_idx = feature_indices["Amplitude_modulation_index_slow"]
    am_vec = base.copy()
    am_vec[am_idx] += 0.05
    am_vec[am_idx] = np.clip(am_vec[am_idx], -1.0, 1.0)
    perturbed["Amplitude_modulation_small_plus"] = am_vec

    return perturbed


def feature_perturbation_test(
    features: np.ndarray,
    model_fn: Callable[[np.ndarray], np.ndarray],
    feature_indices: Dict[str, int],
) -> Dict[str, Dict[str, list]]:
    """
    Tek referans örnek üzerinden feature perturbation testini çalıştırır.
    """
    if features.ndim != 2:
        raise ValueError("Feature matrisi (N, F) boyutunda olmalıdır")

    ref_feature = features[0]
    ref_param = model_fn(ref_feature[None, :])[0]

    perturbed_features = apply_perturbations(
        ref_feature, feature_indices
    )

    results: Dict[str, Dict[str, list]] = {}

    for name, vec in perturbed_features.items():
        pert_param = model_fn(vec[None, :])[0]
        delta = pert_param - ref_param

        results[name] = {
            "original_params": ref_param.tolist(),
            "perturbed_params": pert_param.tolist(),
            "delta": delta.tolist(),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="xxxDSP V2 C2 — Feature Perturbation Testi"
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
        "--feature-index-map",
        required=True,
        help="Feature adı → index eşlemesi (JSON)",
    )
    parser.add_argument(
        "--output",
        default="feature_perturbation_results.json",
        help="Çıktı JSON dosyası",
    )

    args = parser.parse_args()

    features = np.load(args.features)
    model_fn = load_model_callable(args.model)

    with open(args.feature_index_map, "r", encoding="utf-8") as f:
        feature_indices = json.load(f)

    results = feature_perturbation_test(
        features=features,
        model_fn=model_fn,
        feature_indices=feature_indices,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Feature perturbation testi tamamlandı.")
    for k in results:
        print("Perturbation:", k)


if __name__ == "__main__":
    np.random.seed(0)
    main()
