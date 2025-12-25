import numpy as np
import pandas as pd
from pathlib import Path

EPSILON = 1e-12

CATEGORY_A = ['RMS_mean', 'RMS_std_slow', 'Crest_factor']
CATEGORY_B = ['Spectral_centroid_mean', 'Spectral_rolloff_85_mean']
CATEGORY_C = ['Spectral_flatness_mean', 'Low_band_ratio', 'Mid_band_ratio', 'Zero_crossing_rate_mean']
CATEGORY_D = ['Spectral_tilt_estimate']
CATEGORY_E = ['Amplitude_modulation_index_slow']


def normalize_tanh(f: np.ndarray) -> np.ndarray:
    """Kategori A, D, E: tanh scaling"""
    m = np.median(f)
    MAD = np.median(np.abs(f - m))
    sigma = 1.4826 * MAD
    f_clip = np.clip(f, m - 3 * sigma, m + 3 * sigma)
    u = (f_clip - m) / (sigma + EPSILON)
    z = np.tanh(0.5 * u)
    return z


def normalize_log_tanh(f: np.ndarray) -> np.ndarray:
    """Kategori B: log + tanh"""
    g = np.log(f + EPSILON)
    m = np.median(g)
    MAD = np.median(np.abs(g - m))
    sigma = 1.4826 * MAD
    g_clip = np.clip(g, m - 3 * sigma, m + 3 * sigma)
    u = (g_clip - m) / (sigma + EPSILON)
    z = np.tanh(0.5 * u)
    return z


def normalize_robust_minmax(f: np.ndarray) -> np.ndarray:
    """Kategori C: robust min-max"""
    Q1 = np.percentile(f, 1)
    Q99 = np.percentile(f, 99)
    f_clip = np.clip(f, Q1, Q99)
    z = 2 * (f_clip - Q1) / (Q99 - Q1 + EPSILON) - 1
    return z


def normalize_features(input_csv: Path, output_csv: Path) -> None:
    """CSV oku, normalize et, yaz"""
    df = pd.read_csv(input_csv)
    
    has_filename = 'filename' in df.columns
    
    if has_filename:
        feature_cols = [col for col in df.columns if col != 'filename']
        df_norm = pd.DataFrame({'filename': df['filename']})
    else:
        feature_cols = df.columns.tolist()
        df_norm = pd.DataFrame()
    
    for col in feature_cols:
        f = df[col].values
        
        if col in CATEGORY_A or col in CATEGORY_D or col in CATEGORY_E:
            z = normalize_tanh(f)
        elif col in CATEGORY_B:
            z = normalize_log_tanh(f)
        elif col in CATEGORY_C:
            z = normalize_robust_minmax(f)
        else:
            raise ValueError(f"Feature {col} kategoriye eşleşmiyor")
        
        df_norm[col] = z
    
    df_norm.to_csv(output_csv, index=False)
    print(f"Normalized: {output_csv}")


if __name__ == "__main__":
    input_path = Path("v2_ml/features_raw.csv")
    output_path = Path("v2_ml/features_normalized.csv")
    
    normalize_features(input_path, output_path)

