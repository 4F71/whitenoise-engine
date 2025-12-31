"""
v2_ml/feature_extraction.py

xxxDSP V2 Feature Extraction Pipeline
----------------------------------------
Referanslar:
- docs/02_v2_theory/v2_feature_set.md (11 feature tanımları)
- docs/02_v2_theory/v2_mathematical_lock.md (normalization - sonraki aşama)

Çıktı:
- features_raw.csv: 11 ham feature değeri (normalization YOK)

Kullanım:
    python v2_ml/feature_extraction.py
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# 1. KONFIGÜRASYON
# ============================================================================

SAMPLE_RATE = 48000
DURATION_SEC = 60.0
FRAME_LENGTH = 2048
HOP_LENGTH = 512

# Band tanımları (v2_feature_set.md)
LOW_BAND_HZ = (20, 200)
MID_BAND_HZ = (200, 2000)


# ============================================================================
# 2. AUDIO YÜKLEME
# ============================================================================

def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    """
    Audio dosyasını yükle (mono, 48kHz)
    
    Returns:
        (audio, sample_rate): mono signal, sample rate
    """
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION_SEC
    )
    return audio, sr


# ============================================================================
# 3. FEATURE HESAPLAMA FONKSİYONLARI
# ============================================================================

def compute_rms_features(audio: np.ndarray) -> Dict[str, float]:
    """
    3.1 RMS_mean: ortalama enerji
    3.2 RMS_std_slow: enerji değişkenliği
    """
    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    
    return {
        'RMS_mean': float(np.mean(rms)),
        'RMS_std_slow': float(np.std(rms))
    }


def compute_crest_factor(audio: np.ndarray) -> float:
    """
    3.3 Crest_factor: tepe enerji / ortalama enerji
    """
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Sıfır bölme koruması
    if rms < 1e-12:
        return 0.0
    
    crest = peak / rms
    return float(crest)


def compute_spectral_features(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    3.4 Spectral_centroid_mean: parlaklık
    3.5 Spectral_rolloff_85_mean: %85 enerji frekansı
    3.7 Spectral_flatness_mean: tonal vs noise
    """
    # STFT hesapla
    S = np.abs(librosa.stft(
        audio,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    ))
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        S=S,
        sr=sr,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    
    # Spectral rolloff (85%)
    rolloff = librosa.feature.spectral_rolloff(
        S=S,
        sr=sr,
        roll_percent=0.85,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    
    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(
        S=S,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    
    return {
        'Spectral_centroid_mean': float(np.mean(centroid)),
        'Spectral_rolloff_85_mean': float(np.mean(rolloff)),
        'Spectral_flatness_mean': float(np.mean(flatness))
    }


def compute_spectral_tilt(audio: np.ndarray, sr: int) -> float:
    """
    3.6 Spectral_tilt_estimate: dB/oktav eğim tahmini
    
    Yöntem: log-frequency magnitude spektrumuna lineer regresyon
    """
    # Uzun pencere FFT (tüm klip için temsili spektrum)
    n_fft = 8192
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4))
    S_mean = np.mean(S, axis=1)
    
    # Frekans ekseni (20 Hz - Nyquist)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid_idx = freqs >= 20.0
    freqs = freqs[valid_idx]
    S_mean = S_mean[valid_idx]
    
    # Log-log uzayında lineer regresyon
    log_freqs = np.log2(freqs)
    log_mags = 20 * np.log10(S_mean + 1e-12)  # dB
    
    # Slope hesapla (dB/oktav)
    slope = np.polyfit(log_freqs, log_mags, deg=1)[0]
    
    return float(slope)


def compute_band_ratios(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    3.8 Low_band_ratio: 20-200 Hz / toplam
    3.10 Mid_band_ratio: 200-2000 Hz / toplam
    """
    # FFT ile magnitude spektrum
    n_fft = 8192
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=n_fft // 4))
    S_mean = np.mean(S, axis=1)  # zaman ortalaması
    
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Band maskeleri
    low_mask = (freqs >= LOW_BAND_HZ[0]) & (freqs < LOW_BAND_HZ[1])
    mid_mask = (freqs >= MID_BAND_HZ[0]) & (freqs < MID_BAND_HZ[1])
    
    # Enerji toplamları
    total_energy = np.sum(S_mean ** 2)
    low_energy = np.sum(S_mean[low_mask] ** 2)
    mid_energy = np.sum(S_mean[mid_mask] ** 2)
    
    # Sıfır bölme koruması
    if total_energy < 1e-12:
        return {'Low_band_ratio': 0.0, 'Mid_band_ratio': 0.0}
    
    return {
        'Low_band_ratio': float(low_energy / total_energy),
        'Mid_band_ratio': float(mid_energy / total_energy)
    }


def compute_zero_crossing_rate(audio: np.ndarray) -> float:
    """
    3.11 Zero_crossing_rate_mean: ZCR ortalaması
    """
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    
    return float(np.mean(zcr))


def compute_amplitude_modulation_index(audio: np.ndarray, sr: int) -> float:
    """
    3.9 Amplitude_modulation_index_slow: yavaş genlik modülasyonu
    
    Yöntem:
    1. RMS zarf çıkar (kısa pencere)
    2. Zarfın düşük frekans modülasyonunu ölç (0.1-2 Hz)
    3. Modülasyon derinliği / ortalama oran
    """
    # RMS zarf
    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]
    
    # Zarfın ortalama ve std'si
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # Sıfır bölme koruması
    if rms_mean < 1e-12:
        return 0.0
    
    # Modülasyon indeksi: std / mean
    # (düşük frekans modülasyon derinliği göstergesi)
    am_index = rms_std / rms_mean
    
    return float(am_index)


# ============================================================================
# 4. ANA FEATURE EXTRACTION PIPELINE
# ============================================================================

def extract_features(audio_path: Path) -> Dict[str, float]:
    """
    Tek audio dosyasından 11 feature çıkar
    
    Returns:
        feature dict: {feature_name: value}
    """
    # Audio yükle
    audio, sr = load_audio(audio_path)
    
    # Feature'ları hesapla
    features = {}
    
    # RMS features
    features.update(compute_rms_features(audio))
    
    # Crest factor
    features['Crest_factor'] = compute_crest_factor(audio)
    
    # Spectral features
    features.update(compute_spectral_features(audio, sr))
    
    # Spectral tilt
    features['Spectral_tilt_estimate'] = compute_spectral_tilt(audio, sr)
    
    # Band ratios
    features.update(compute_band_ratios(audio, sr))
    
    # Zero crossing rate
    features['Zero_crossing_rate_mean'] = compute_zero_crossing_rate(audio)
    
    # Amplitude modulation
    features['Amplitude_modulation_index_slow'] = compute_amplitude_modulation_index(audio, sr)
    
    return features


def extract_features_batch(audio_dir: Path, output_csv: Path) -> pd.DataFrame:
    """
    Klasördeki tüm WAV dosyalarından feature çıkar ve CSV'ye yaz
    (Alt klasörler dahil, recursive arama)
    
    Args:
        audio_dir: audio dosyalarının bulunduğu klasör
        output_csv: çıktı CSV dosyası
    
    Returns:
        DataFrame: feature tablosu
    """
    # Recursive arama: alt klasörler dahil
    audio_files = sorted(audio_dir.rglob("*.wav"))
    
    if not audio_files:
        raise ValueError(f"Hiçbir WAV dosyası bulunamadı (alt klasörler dahil): {audio_dir}")
    
    print(f"Toplam {len(audio_files)} audio dosyası bulundu")
    print(f"İşleniyor...")
    
    results = []
    
    for i, audio_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {audio_path.name}")
        
        try:
            features = extract_features(audio_path)
            features['filename'] = audio_path.name
            results.append(features)
        except Exception as e:
            print(f"  ⚠️ HATA: {e}")
            continue
    
    # DataFrame oluştur
    df = pd.DataFrame(results)
    
    # Sütun sırası: filename + 11 feature
    column_order = [
        'filename',
        'RMS_mean',
        'RMS_std_slow',
        'Crest_factor',
        'Spectral_centroid_mean',
        'Spectral_rolloff_85_mean',
        'Spectral_tilt_estimate',
        'Spectral_flatness_mean',
        'Low_band_ratio',
        'Mid_band_ratio',
        'Zero_crossing_rate_mean',
        'Amplitude_modulation_index_slow'
    ]
    
    df = df[column_order]
    
    # CSV'ye yaz
    df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\n✅ Feature extraction tamamlandı")
    print(f"   Çıktı: {output_csv}")
    print(f"   Satır: {len(df)}, Sütun: {len(df.columns)}")
    
    return df


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == "__main__":
    # Klasör yolları
    REPO_ROOT = Path(__file__).parent.parent
    AUDIO_DIR = REPO_ROOT / "v2_dataset" / "processed" / "mono_60s_48k"
    OUTPUT_CSV = REPO_ROOT / "v2_ml" / "features_raw.csv"
    
    # Klasör kontrolü
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio klasörü bulunamadı: {AUDIO_DIR}\n"
            f"Lütfen önce v2_dataset/processed/mono_60s_48k/ klasörünü oluşturun."
        )
    
    # Feature extraction
    df = extract_features_batch(AUDIO_DIR, OUTPUT_CSV)
    
    # Özet istatistikler
    print("\n--- Feature Özeti ---")
    print(df.describe().T[['mean', 'std', 'min', 'max']])

