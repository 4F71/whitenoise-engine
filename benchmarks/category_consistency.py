# xxxDSP V2 — C2 Benchmark
# Kategori İçi Tutarlılık Testi
# Amaç: Aynı kategori örneklerinin parametre uzayında yakın,
#       farklı kategorilerin ise ayrışmış olup olmadığını ölçmek.
# Not: Model siyah kutudur; burada tanımlanmaz, eğitilmez veya seçilmez.

import argparse
import importlib
import json
from typing import Callable, Dict, List

import numpy as np


def load_callable(path: str) -> Callable[[np.ndarray], np.ndarray]:
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


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    İki vektör arasındaki L2 (Euclidean) mesafeyi hesaplar.
    """
    return float(np.linalg.norm(a - b))


def category_consistency(
    features: np.ndarray,
    categories: List[str],
    model_fn: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, object]:
    """
    Kategori içi ve kategoriler arası parametre mesafelerini hesaplar.
    """
    if features.ndim != 2:
        raise ValueError("Feature matrisi (N, F) boyutunda olmalıdır")

    if len(categories) != features.shape[0]:
        raise ValueError("Kategori sayısı örnek sayısı ile eşleşmelidir")

    params = model_fn(features)

    unique_categories = sorted(set(categories))

    within_distances: Dict[str, List[float]] = {c: [] for c in unique_categories}
    between_distances: List[float] = []

    n = features.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            d = pairwise_l2(params[i], params[j])
            if categories[i] == categories[j]:
                within_distances[categories[i]].append(d)
            else:
                between_distances.append(d)

    within_means = {
        c: float(np.mean(dists)) if dists else None
        for c, dists in within_distances.items()
    }

    return {
        "within_category_mean_distance": within_means,
        "between_category_mean_distance": float(np.mean(between_distances))
        if between_distances
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="xxxDSP V2 C2 — Kategori İçi Tutarlılık Testi"
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Önceden hesaplanmış feature matrisi (.npy), şekil (N, F)",
    )
    parser.add_argument(
        "--categories",
        required=True,
        help="Kategori etiketleri (.json list), uzunluk N",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model callable yolu (module:function)",
    )
    parser.add_argument(
        "--output",
        default="category_consistency_results.json",
        help="Çıktı JSON dosyası",
    )

    args = parser.parse_args()

    features = np.load(args.features)
    with open(args.categories, "r", encoding="utf-8") as f:
        categories = json.load(f)

    model_fn = load_callable(args.model)

    results = category_consistency(
        features=features,
        categories=categories,
        model_fn=model_fn,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Kategori içi tutarlılık testi tamamlandı.")
    print("Kategoriler arası ortalama mesafe:", results["between_category_mean_distance"])


if __name__ == "__main__":
    main()
