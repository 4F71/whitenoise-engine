import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
import pickle
import json
import os
from pathlib import Path

features_df = pd.read_csv("v2_ml/features_normalized.csv")
labels_df = pd.read_csv("v2_ml/pseudo_labels.csv")

X = features_df.drop(columns=["filename"]).values

noise_type_map = {
    "brown": 0,
    "pink": 1,
    "white": 2,
    "blue": 3,
    "violet": 4
}
labels_df["noise_type_encoded"] = labels_df["noise_type"].map(noise_type_map)

lfo_target_map = {"amplitude": 0, "pan": 1, "filter_cutoff": 2}
labels_df["lfo_target_encoded"] = labels_df["lfo_target"].map(lfo_target_map)

y_columns = [
    "noise_type_encoded", "gain_db", "lp_cutoff_hz", "hp_cutoff_hz",
    "spectral_tilt_db", "soft_saturation_amount", "normalize_target_db",
    "lfo_rate_hz", "lfo_depth", "lfo_target_encoded", "stereo_width", "reverb_send"
]
y = labels_df[y_columns].values

model = Ridge(alpha=1.0)
model.fit(X, y)

os.makedirs("v2_ml/models", exist_ok=True)
with open("v2_ml/models/ridge_baseline.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model kaydedildi: v2_ml/models/ridge_baseline.pkl")

encoding_maps = {
    "noise_type_map": noise_type_map,
    "lfo_target_map": lfo_target_map,
    "noise_type_inverse": {v: k for k, v in noise_type_map.items()},
    "lfo_target_inverse": {v: k for k, v in lfo_target_map.items()},
}
with open("v2_ml/models/encoding_maps.pkl", "wb") as f:
    pickle.dump(encoding_maps, f)

print("Encoding maps kaydedildi: v2_ml/models/encoding_maps.pkl")

loo = LeaveOneOut()
y_pred_loo = np.zeros_like(y)

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model_loo = Ridge(alpha=1.0)
    model_loo.fit(X_train, y_train)
    y_pred_loo[test_idx] = model_loo.predict(X_test)

mse = mean_squared_error(y, y_pred_loo)
mae = mean_absolute_error(y, y_pred_loo)
r2 = r2_score(y, y_pred_loo)

print(f"\nLOO Cross-Validation Metrikleri:")
print(f"  MSE: {mse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R²:  {r2:.4f}")

param_names = y_columns
print(f"\n{'Parametre':<30s} {'R²':>8s}  {'MAE':>8s}")
print("-" * 50)

param_r2 = {}
for i, param in enumerate(param_names):
    r2_param = r2_score(y[:, i], y_pred_loo[:, i])
    mae_param = mean_absolute_error(y[:, i], y_pred_loo[:, i])
    param_r2[param] = float(r2_param)
    print(f"{param:<30s} {r2_param:>8.3f}  {mae_param:>8.3f}")

metrics = {
    "model": "Ridge",
    "alpha": 1.0,
    "n_samples": 17,
    "n_features": 11,
    "n_params": 12,
    "loo_mse": float(mse),
    "loo_mae": float(mae),
    "loo_r2": float(r2),
    "param_r2": param_r2
}

with open("v2_ml/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nMetrikler kaydedildi: v2_ml/model_metrics.json")

X_test = X[0:1]
y_pred_test = model.predict(X_test)[0]

noise_idx = int(np.clip(round(y_pred_test[0]), 0, 4))
noise_type_pred = encoding_maps["noise_type_inverse"][noise_idx]

lfo_idx = int(np.clip(round(y_pred_test[9]), 0, 2))
lfo_target_pred = encoding_maps["lfo_target_inverse"][lfo_idx]

filename = features_df.iloc[0]["filename"]
print(f"\nTest Prediction: {filename}")
print(f"  noise_type:        {noise_type_pred}")
print(f"  gain_db:           {y_pred_test[1]:.2f} dB")
print(f"  lp_cutoff_hz:      {y_pred_test[2]:.0f} Hz")
print(f"  hp_cutoff_hz:      {y_pred_test[3]:.0f} Hz")
print(f"  spectral_tilt_db:  {y_pred_test[4]:.2f} dB/oct")
print(f"  lfo_rate_hz:       {y_pred_test[7]:.3f} Hz")
print(f"  lfo_depth:         {y_pred_test[8]:.3f}")
print(f"  lfo_target:        {lfo_target_pred}")
print(f"  stereo_width:      {y_pred_test[10]:.2f}")

