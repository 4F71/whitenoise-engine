import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import euclidean_distances

with open("v2_ml/models/ridge_baseline.pkl", "rb") as f:
    model = pickle.load(f)

with open("v2_ml/models/encoding_maps.pkl", "rb") as f:
    encoding_maps = pickle.load(f)

features_df = pd.read_csv("v2_ml/data/features_normalized.csv")
X = features_df.drop(columns=["filename"]).values
filenames = features_df["filename"].tolist()

feature_names = [
    "RMS_mean", "RMS_std_slow", "Crest_factor",
    "Spectral_centroid_mean", "Spectral_rolloff_85_mean",
    "Spectral_tilt_estimate", "Spectral_flatness_mean",
    "Low_band_ratio", "Mid_band_ratio",
    "Zero_crossing_rate_mean", "Amplitude_modulation_index_slow"
]

print("Model ve veri yuklendi")

with open("v2_ml/models/model_metrics.json", "r") as f:
    model_metrics = json.load(f)

loo_r2 = model_metrics["loo_r2"]
loo_mae = model_metrics["loo_mae"]
param_r2 = model_metrics.get("param_r2", {})

print("\n" + "="*70)
print("C2.1 - Leave-One-Out Stability Test")
print("="*70)
print(f"LOO R2:  {loo_r2:.3f}")
print(f"LOO MAE: {loo_mae:.3f}")

low_r2_params = {k: v for k, v in param_r2.items() if v < 0.70}
if low_r2_params:
    print(f"\nDusuk R2 parametreler:")
    for param, r2 in low_r2_params.items():
        print(f"   {param}: R2={r2:.3f}")
else:
    print(f"Tum parametreler R2 > 0.70")

c2_1_pass = loo_r2 > 0.70
print(f"\nSonuc: {'PASS' if c2_1_pass else 'FAIL'} (threshold: R2 > 0.70)")

print("\n" + "="*70)
print("C2.2 - Feature Perturbation Test")
print("="*70)

test_idx = 0
X_base = X[test_idx:test_idx+1]
y_base = model.predict(X_base)[0]

print(f"Test sesi: {filenames[test_idx]}\n")

perturbations = {
    "RMS +10%": (0, 0.1),
    "Centroid +5%": (3, 0.05),
    "AM_index +10%": (10, 0.1),
}

perturbation_results = {}

for pert_name, (feature_idx, delta) in perturbations.items():
    X_pert = X_base.copy()
    X_pert[0, feature_idx] += delta
    X_pert = np.clip(X_pert, -1, 1)
    
    y_pert = model.predict(X_pert)[0]
    param_diff = y_pert - y_base
    
    perturbation_results[pert_name] = {
        "feature_idx": feature_idx,
        "delta": delta,
        "param_diff": param_diff,
    }
    
    top_changes = np.argsort(np.abs(param_diff))[-3:][::-1]
    
    print(f"{pert_name}:")
    print(f"  En cok degişenler:")
    param_names_list = [
        "noise_type_enc", "gain_db", "lp_cutoff_hz", "hp_cutoff_hz",
        "spectral_tilt_db", "soft_sat", "normalize_target",
        "lfo_rate_hz", "lfo_depth", "lfo_target_enc", "stereo_width", "reverb_send"
    ]
    for idx in top_changes:
        print(f"    {param_names_list[idx]}: {param_diff[idx]:+.3f}")

expected_responses = {
    "RMS +10%": (1, 0.1, "gain_db"),
    "Centroid +5%": (2, 100, "lp_cutoff_hz"),
    "AM_index +10%": (8, 0.01, "lfo_depth"),
}

print(f"\nMonoton Davranis Kontrolu:")
monotonic_count = 0

for pert_name, (param_idx, min_change, param_name) in expected_responses.items():
    param_change = perturbation_results[pert_name]["param_diff"][param_idx]
    
    if param_change > min_change:
        print(f"  {pert_name} -> {param_name} artti (+{param_change:.3f})")
        monotonic_count += 1
    else:
        print(f"  {pert_name} -> {param_name} zayif tepki ({param_change:+.3f})")

c2_2_pass = monotonic_count >= 2
print(f"\nSonuc: {'PASS' if c2_2_pass else 'WARNING'} ({monotonic_count}/3 monoton)")

print("\n" + "="*70)
print("C2.3 - Sanity Gate Clamp Ratio")
print("="*70)

y_pred_all = model.predict(X)

limits = {
    "lp_cutoff_hz": (200, 16000),
    "hp_cutoff_hz": (20, 200),
    "gain_db": (-24, 0),
    "lfo_rate_hz": (0.001, 0.5),
    "lfo_depth": (0.0, 0.3),
    "stereo_width": (0.8, 1.5),
    "soft_saturation": (0.0, 0.1),
    "spectral_tilt_db": (-6, 0),
}

param_indices = {
    "lp_cutoff_hz": 2,
    "hp_cutoff_hz": 3,
    "gain_db": 1,
    "lfo_rate_hz": 7,
    "lfo_depth": 8,
    "stereo_width": 10,
    "soft_saturation": 5,
    "spectral_tilt_db": 4,
}

total_values = 0
clamped_values = 0
clamp_details = defaultdict(int)

for param_name, (min_val, max_val) in limits.items():
    param_idx = param_indices[param_name]
    param_values = y_pred_all[:, param_idx]
    
    below_min = (param_values < min_val).sum()
    above_max = (param_values > max_val).sum()
    
    total_values += len(param_values)
    clamped_values += below_min + above_max
    
    if below_min + above_max > 0:
        clamp_details[param_name] = below_min + above_max

clamp_ratio = clamped_values / total_values

print(f"Toplam deger:      {total_values}")
print(f"Clamped deger:     {clamped_values}")
print(f"Clamp ratio:       {clamp_ratio:.3f} ({clamp_ratio*100:.1f}%)")

if clamp_details:
    print(f"\nClamped parametreler:")
    for param, count in clamp_details.items():
        print(f"  {param}: {count} / 17")
else:
    print(f"Hicbir parametre clamp edilmedi")

c2_3_pass = clamp_ratio < 0.10
print(f"\nSonuc: {'PASS' if c2_3_pass else 'WARNING'} (threshold: <10%)")

print("\n" + "="*70)
print("C2.4 - Category Consistency (Kategori Ici Tutarlilik)")
print("="*70)

p0_dir = "v2_ml/presets/p0"
categories = defaultdict(list)
category_params = defaultdict(list)

for i, json_file in enumerate(sorted(os.listdir(p0_dir))):
    if json_file.endswith(".json"):
        category = json_file.split("_")[0]
        categories[category].append(i)
        
        y_pred_i = y_pred_all[i]
        category_params[category].append(y_pred_i)

print(f"\n{'Kategori':<20s} {'Ornek Sayisi':>12s} {'Avg L2 Dist':>15s}")
print("-" * 50)

category_distances = {}

for category, indices in categories.items():
    if len(indices) < 2:
        print(f"{category:<20s} {len(indices):>12d} {'-':>15s} (tek ornek)")
        continue
    
    params = np.array(category_params[category])
    distances = euclidean_distances(params, params)
    upper_triangle = distances[np.triu_indices_from(distances, k=1)]
    avg_dist = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
    
    category_distances[category] = avg_dist
    
    print(f"{category:<20s} {len(indices):>12d} {avg_dist:>15.2f}")

if len(category_distances) >= 2:
    top_cats = sorted(categories.items(), key=lambda x: len(x[1]), reverse=True)[:2]
    
    cat1_name = top_cats[0][0]
    cat2_name = top_cats[1][0]
    
    cat1_params = np.array(category_params[cat1_name])
    cat2_params = np.array(category_params[cat2_name])
    
    inter_distances = euclidean_distances(cat1_params, cat2_params)
    inter_avg = np.mean(inter_distances)
    
    intra_avg = np.mean(list(category_distances.values()))
    
    print(f"\nKategoriler Arasi Mesafe:")
    print(f"  {cat1_name} <-> {cat2_name}: {inter_avg:.2f}")
    print(f"  Kategori ici ortalama: {intra_avg:.2f}")
    
    c2_4_pass = intra_avg < inter_avg
    
    print(f"\nSonuc: {'PASS' if c2_4_pass else 'FAIL'} (kategori ici < arasi)")
else:
    print(f"\nYeterli kategori yok (>2 gerekli)")
    c2_4_pass = True

print("\n" + "="*70)
print("C2 BENCHMARK OZET")
print("="*70)

all_pass = c2_1_pass and c2_2_pass and c2_3_pass and c2_4_pass

results = {
    "C2.1 LOO Stability": "PASS" if c2_1_pass else "FAIL",
    "C2.2 Feature Perturbation": "PASS" if c2_2_pass else "WARNING",
    "C2.3 Sanity Gate Clamp": "PASS" if c2_3_pass else "WARNING",
    "C2.4 Category Consistency": "PASS" if c2_4_pass else "FAIL",
}

for test, result in results.items():
    print(f"{test:<30s} {result}")

print(f"\nGenel Sonuc: {'C2 BASARILI' if all_pass else 'C2 KISMI BASARILI'}")

os.makedirs("v2_ml/benchmarks", exist_ok=True)

report = {
    "c2_benchmarks": {
        "c2_1_loo_stability": {
            "loo_r2": float(loo_r2),
            "loo_mae": float(loo_mae),
            "low_r2_params": {k: float(v) for k, v in low_r2_params.items()},
            "pass": bool(c2_1_pass)
        },
        "c2_2_feature_perturbation": {
            "monotonic_responses": int(monotonic_count),
            "total_tests": int(len(expected_responses)),
            "perturbations": {k: {
                "feature": feature_names[v["feature_idx"]],
                "delta": float(v["delta"]),
                "max_param_change": float(np.max(np.abs(v["param_diff"])))
            } for k, v in perturbation_results.items()},
            "pass": bool(c2_2_pass)
        },
        "c2_3_sanity_gate": {
            "total_values": int(total_values),
            "clamped_values": int(clamped_values),
            "clamp_ratio": float(clamp_ratio),
            "clamped_params": {k: int(v) for k, v in clamp_details.items()},
            "pass": bool(c2_3_pass)
        },
        "c2_4_category_consistency": {
            "category_distances": {k: float(v) for k, v in category_distances.items()},
            "intra_category_avg": float(intra_avg) if len(category_distances) >= 2 else 0.0,
            "inter_category_avg": float(inter_avg) if len(category_distances) >= 2 else 0.0,
            "pass": bool(c2_4_pass)
        },
        "overall_pass": bool(all_pass)
    }
}

with open("v2_ml/benchmarks/c2_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nRapor kaydedildi: v2_ml/benchmarks/c2_report.json")

