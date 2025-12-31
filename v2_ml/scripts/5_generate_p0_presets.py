import pickle
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

with open("v2_ml/models/ridge_baseline.pkl", "rb") as f:
    model = pickle.load(f)

with open("v2_ml/models/encoding_maps.pkl", "rb") as f:
    encoding_maps = pickle.load(f)

features_df = pd.read_csv("v2_ml/data/features_normalized.csv")
X = features_df.drop(columns=["filename"]).values
filenames = features_df["filename"].tolist()

y_pred = model.predict(X)


def decode_and_sanitize(y_pred_row, encoding_maps):
    """Model çıktısını decode et ve sanity gate uygula"""
    
    noise_idx = int(np.clip(round(y_pred_row[0]), 0, 4))
    noise_type = encoding_maps["noise_type_inverse"][noise_idx]
    
    lfo_target_idx = int(np.clip(round(y_pred_row[9]), 0, 2))
    lfo_target = encoding_maps["lfo_target_inverse"][lfo_target_idx]
    
    params = {
        "noise_type": noise_type,
        "gain_db": float(np.clip(y_pred_row[1], -24, 0)),
        "lp_cutoff_hz": float(np.clip(y_pred_row[2], 200, 16000)),
        "hp_cutoff_hz": float(np.clip(y_pred_row[3], 20, 200)),
        "spectral_tilt_db": float(np.clip(y_pred_row[4], -6, 0)),
        "soft_saturation_amount": float(np.clip(y_pred_row[5], 0.0, 0.1)),
        "normalize_target_db": float(y_pred_row[6]),
        "lfo_rate_hz": float(np.clip(y_pred_row[7], 0.001, 0.5)),
        "lfo_depth": float(np.clip(y_pred_row[8], 0.0, 0.3)),
        "lfo_target": lfo_target,
        "stereo_width": float(np.clip(y_pred_row[10], 0.8, 1.5)),
        "reverb_send": float(np.clip(y_pred_row[11], 0.0, 0.5)),
    }
    
    if params["hp_cutoff_hz"] >= params["lp_cutoff_hz"]:
        params["hp_cutoff_hz"] = params["lp_cutoff_hz"] - 50
    
    return params


def create_preset_json(params, filename):
    """preset_schema.py formatında JSON oluştur"""
    
    linear_gain = 10 ** (params["gain_db"] / 20.0)
    linear_gain = float(np.clip(linear_gain, 0.0, 1.0))
    
    preset_name = filename.replace(".wav", "_P0").replace("__", " ").replace("_", " ").title()
    
    preset = {
        "name": preset_name,
        "description": f"P0 baseline preset generated from ML model (Ridge)",
        "author": "xxxDSP V2 ML",
        "version": "2.0",
        "tags": ["v2", "ml-generated", "p0", "baseline"],
        "layers": [
            {
                "name": "ML Predicted Layer",
                "enabled": True,
                "noise_type": params["noise_type"],
                "gain": linear_gain,
                "pan": 0.0,
                "filter_config": {
                    "enabled": True,
                    "filter_type": "lowpass",
                    "cutoff_hz": params["lp_cutoff_hz"],
                    "resonance_q": 0.707,
                    "gain_db": 0.0
                },
                "lfo_config": {
                    "enabled": params["lfo_depth"] > 0.01,
                    "waveform": "sine",
                    "target": params["lfo_target"],
                    "rate_hz": params["lfo_rate_hz"],
                    "depth": params["lfo_depth"],
                    "phase_offset": 0.0
                }
            }
        ],
        "master_gain": 0.8,
        "fx_config": {
            "saturation_amount": params["soft_saturation_amount"],
            "stereo_width": params["stereo_width"],
            "reverb_mix": params["reverb_send"],
            "reverb_decay": 1.0
        },
        "duration_sec": 60.0,
        "sample_rate": 48000,
        "seed": None
    }
    
    return preset


os.makedirs("v2_ml/presets/p0", exist_ok=True)

print("P0 Preset Generation basliyor...")
print(f"Toplam: {len(filenames)} ses\n")

for i, filename in enumerate(filenames):
    y_pred_row = y_pred[i]
    
    params = decode_and_sanitize(y_pred_row, encoding_maps)
    
    preset_json = create_preset_json(params, filename)
    
    json_filename = filename.replace(".wav", "_p0.json")
    json_path = f"v2_ml/presets/p0/{json_filename}"
    
    with open(json_path, "w") as f:
        json.dump(preset_json, f, indent=2)
    
    print(f"[{i+1:2d}/17] {json_filename}")
    print(f"        noise: {params['noise_type']}, "
          f"gain: {params['gain_db']:.1f} dB, "
          f"lp: {params['lp_cutoff_hz']:.0f} Hz")

print(f"\n17 P0 preset olusturuldu: v2_ml/presets/p0/")

noise_types = [decode_and_sanitize(y_pred[i], encoding_maps)["noise_type"] 
               for i in range(len(filenames))]

from collections import Counter
noise_counts = Counter(noise_types)

print("\nNoise Type Dağılımı:")
for noise, count in noise_counts.most_common():
    print(f"  {noise}: {count} preset")

all_params = [decode_and_sanitize(y_pred[i], encoding_maps) for i in range(len(filenames))]
lp_cutoffs = [p["lp_cutoff_hz"] for p in all_params]
gains = [p["gain_db"] for p in all_params]

print(f"\nParametre Aralıkları:")
print(f"  lp_cutoff_hz: [{min(lp_cutoffs):.0f}, {max(lp_cutoffs):.0f}] Hz")
print(f"  gain_db:      [{min(gains):.1f}, {max(gains):.1f}] dB")

