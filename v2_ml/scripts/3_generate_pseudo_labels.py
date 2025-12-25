import numpy as np
import pandas as pd
from pathlib import Path

NOMINAL_PARAMS = {
    "noise_type": "pink",
    "gain_db": -6.0,
    "lp_cutoff_hz": 3000.0,
    "hp_cutoff_hz": 100.0,
    "spectral_tilt_db": -3.0,
    "soft_saturation_amount": 0.02,
    "normalize_target_db": -16.0,
    "lfo_rate_hz": 0.05,
    "lfo_depth": 0.1,
    "lfo_target": "amplitude",
    "stereo_width": 1.1,
    "reverb_send": 0.0,
}


def map_noise_type(z_tilt):
    """Spectral_tilt_estimate → noise_type"""
    if z_tilt < -0.5:
        return "brown"
    elif z_tilt < 0.0:
        return "pink"
    elif z_tilt < 0.5:
        return "white"
    else:
        return "blue"


def apply_sanity_gate(params):
    """Sanity gate limits"""
    params["lp_cutoff_hz"] = np.clip(params["lp_cutoff_hz"], 200, 16000)
    params["hp_cutoff_hz"] = np.clip(params["hp_cutoff_hz"], 20, 200)
    params["gain_db"] = np.clip(params["gain_db"], -24, 0)
    params["lfo_rate_hz"] = np.clip(params["lfo_rate_hz"], 0.001, 0.5)
    params["lfo_depth"] = np.clip(params["lfo_depth"], 0.0, 0.3)
    params["stereo_width"] = np.clip(params["stereo_width"], 0.8, 1.5)
    params["soft_saturation_amount"] = np.clip(params["soft_saturation_amount"], 0.0, 0.1)
    params["spectral_tilt_db"] = np.clip(params["spectral_tilt_db"], -6, 0)
    params["normalize_target_db"] = np.clip(params["normalize_target_db"], -24, -6)
    params["reverb_send"] = np.clip(params["reverb_send"], 0.0, 0.1)
    return params


def map_features_to_params(row, nominal):
    """Feature → DSP parameter mapping"""
    params = nominal.copy()
    
    z_rms_mean = row["RMS_mean"]
    z_rms_std_slow = row["RMS_std_slow"]
    z_crest = row["Crest_factor"]
    z_centroid = row["Spectral_centroid_mean"]
    z_rolloff = row["Spectral_rolloff_85_mean"]
    z_tilt = row["Spectral_tilt_estimate"]
    z_flatness = row["Spectral_flatness_mean"]
    z_low_band = row["Low_band_ratio"]
    z_mid_band = row["Mid_band_ratio"]
    z_zcr = row["Zero_crossing_rate_mean"]
    z_am_index = row["Amplitude_modulation_index_slow"]
    
    # 1. noise_type (categorik)
    params["noise_type"] = map_noise_type(z_tilt)
    
    # 2. spectral_tilt_db
    params["spectral_tilt_db"] = nominal["spectral_tilt_db"] + (z_tilt * 2.0)
    
    # 3. lp_cutoff_hz
    params["lp_cutoff_hz"] = nominal["lp_cutoff_hz"] + (z_centroid * 1500.0)
    
    # 4. hp_cutoff_hz (ters oran: yüksek low_band → daha az HP filter)
    params["hp_cutoff_hz"] = nominal["hp_cutoff_hz"] - (z_low_band * 50.0)
    
    # 5. gain_db (ters oran: yüksek RMS → daha düşük gain)
    params["gain_db"] = nominal["gain_db"] - (z_rms_mean * 6.0)
    
    # 6. soft_saturation_amount
    params["soft_saturation_amount"] = nominal["soft_saturation_amount"] + (z_crest * 0.03)
    
    # 7. lfo_depth
    params["lfo_depth"] = nominal["lfo_depth"] + (z_am_index * 0.15)
    
    # 8. lfo_rate_hz
    params["lfo_rate_hz"] = nominal["lfo_rate_hz"] + (z_am_index * 0.03)
    
    # 9. stereo_width
    params["stereo_width"] = nominal["stereo_width"] + (z_mid_band * 0.2)
    
    # 10. reverb_send (flatness etkisi)
    if z_flatness > 0.5 and z_rms_mean < -0.3:
        params["reverb_send"] = 0.02
    else:
        params["reverb_send"] = 0.0
    
    # 11. lfo_target (sabit)
    params["lfo_target"] = "amplitude"
    
    # 12. normalize_target_db (sabit)
    params["normalize_target_db"] = nominal["normalize_target_db"]
    
    params = apply_sanity_gate(params)
    
    return params


def generate_pseudo_labels(features_csv_path, output_csv_path):
    """CSV oku, pseudo-label üret, yaz"""
    df_features = pd.read_csv(features_csv_path)
    
    has_filename = 'filename' in df_features.columns
    
    results = []
    
    for idx, row in df_features.iterrows():
        params = map_features_to_params(row, NOMINAL_PARAMS)
        
        result_row = params.copy()
        if has_filename:
            result_row = {"filename": row["filename"], **params}
        
        results.append(result_row)
    
    df_labels = pd.DataFrame(results)
    
    if has_filename:
        cols = ["filename"] + [k for k in NOMINAL_PARAMS.keys()]
    else:
        cols = list(NOMINAL_PARAMS.keys())
    
    df_labels = df_labels[cols]
    
    df_labels.to_csv(output_csv_path, index=False)
    
    print(f"Pseudo-labels: {output_csv_path}")
    print(f"Samples: {len(df_labels)}")
    print()
    print("Parameter Stats:")
    print("-" * 60)
    
    numeric_params = [k for k in NOMINAL_PARAMS.keys() if k not in ["noise_type", "lfo_target"]]
    
    for param in numeric_params:
        values = df_labels[param].values
        print(f"{param:30s} [{values.min():8.2f}, {values.max():8.2f}]  mean={values.mean():8.2f}")
    
    print()
    print("Categorical Distributions:")
    print("-" * 60)
    print("noise_type:")
    print(df_labels["noise_type"].value_counts().to_string())


if __name__ == "__main__":
    input_path = Path("v2_ml/features_normalized.csv")
    output_path = Path("v2_ml/pseudo_labels.csv")
    
    generate_pseudo_labels(input_path, output_path)

