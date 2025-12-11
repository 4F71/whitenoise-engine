"""
LFO modülü: sinüs ve üçgen üretici, hacim/filter/pan modülasyon yardımcıları.
Tüm fonksiyonlar NumPy tabanlıdır ve basit senaryolara hızlıca entegre edilebilir.
"""

from __future__ import annotations

import numpy as np


def sine_lfo(num_samples: int, freq_hz: float, sample_rate: float, phase_rad: float = 0.0) -> np.ndarray:
    """Sinüs dalgalı LFO üretir (-1..1 aralığında).

    Parametreler:
        num_samples: Üretilecek örnek sayısı.
        freq_hz: LFO frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).
        phase_rad: Başlangıç fazı (radyan cinsinden, varsayılan 0).

    Döndürür:
        np.ndarray: float32 LFO sinyali.
    """
    if num_samples < 0:
        raise ValueError("num_samples negatif olamaz")
    if sample_rate <= 0:
        raise ValueError("sample_rate pozitif olmalı")
    if freq_hz < 0:
        raise ValueError("freq_hz negatif olamaz")

    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    phase = 2.0 * np.pi * freq_hz * t + phase_rad
    return np.sin(phase).astype(np.float32)


def triangle_lfo(num_samples: int, freq_hz: float, sample_rate: float, phase_rad: float = 0.0) -> np.ndarray:
    """Üçgen dalga formunda LFO üretir (-1..1 aralığında).

    Parametreler:
        num_samples: Üretilecek örnek sayısı.
        freq_hz: LFO frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).
        phase_rad: Başlangıç fazı (radyan, sinüse paralel çevrim başlatmak için kullanılır).

    Döndürür:
        np.ndarray: float32 LFO sinyali.
    """
    if num_samples < 0:
        raise ValueError("num_samples negatif olamaz")
    if sample_rate <= 0:
        raise ValueError("sample_rate pozitif olmalı")
    if freq_hz < 0:
        raise ValueError("freq_hz negatif olamaz")

    # Fazı çevrim (0..1) cinsinden hesapla; üçgen formülü çevrim tabanlıdır.
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    phase_cycles = (freq_hz * t + phase_rad / (2.0 * np.pi)) % 1.0
    tri = 1.0 - 4.0 * np.abs(phase_cycles - 0.5)
    return tri.astype(np.float32)


def apply_volume_lfo(signal: np.ndarray, lfo: np.ndarray, depth: float = 1.0) -> np.ndarray:
    """LFO ile hacim modülasyonu (tremolo) uygular.

    depth 0..1 aralığındadır; 0 modülasyon yok, 1 tam genlik salınımıdır.
    Hızlı kullanım için kazanç 1-depth ile 1 arasında tutulur, sinyalin polaritesi değişmez.

    Döndürür:
        Modüle edilmiş sinyal (float32).
    """
    signal = np.asarray(signal, dtype=np.float32)
    lfo = np.asarray(lfo, dtype=np.float32)
    if signal.ndim != 1 or lfo.ndim != 1:
        raise ValueError("signal ve lfo 1B olmalı")
    if signal.shape[0] != lfo.shape[0]:
        raise ValueError("signal ve lfo uzunlukları eşleşmeli")

    depth = float(np.clip(depth, 0.0, 1.0))
    lfo = np.clip(lfo, -1.0, 1.0)
    gain = (1.0 - depth) + depth * (0.5 * (lfo + 1.0))
    return signal * gain


def apply_filter_cutoff_lfo(
    signal: np.ndarray,
    base_cutoff_hz: float,
    lfo: np.ndarray,
    depth_hz: float,
    sample_rate: float,
    min_cutoff_hz: float = 20.0,
    max_cutoff_hz: float = 20_000.0,
) -> np.ndarray:
    """Tek kutuplu alçak geçiren filtrede örnek başına kesim frekansını modüle eder.

    Parametreler:
        signal: 1B giriş sinyali.
        base_cutoff_hz: LFO yokken kullanılacak kesim.
        lfo: -1..1 aralığında LFO sinyali.
        depth_hz: LFO genliğinin kesime çevrileceği aralık (örn. 2000 ise ±2000 Hz oynar).
        sample_rate: Örnekleme hızı.
        min_cutoff_hz: Kesimin düşebileceği alt sınır.
        max_cutoff_hz: Kesimin çıkabileceği üst sınır.

    Döndürür:
        Modüle edilmiş alçak geçiren çıktı (float32).
    """
    signal = np.asarray(signal, dtype=np.float32)
    lfo = np.asarray(lfo, dtype=np.float32)
    if signal.ndim != 1 or lfo.ndim != 1:
        raise ValueError("signal ve lfo 1B olmalı")
    if signal.shape[0] != lfo.shape[0]:
        raise ValueError("signal ve lfo uzunlukları eşleşmeli")
    if sample_rate <= 0:
        raise ValueError("sample_rate pozitif olmalı")

    target_cutoff = base_cutoff_hz + float(abs(depth_hz)) * np.clip(lfo, -1.0, 1.0)
    target_cutoff = np.clip(target_cutoff, min_cutoff_hz, max_cutoff_hz)
    alpha = np.exp(-2.0 * np.pi * target_cutoff / sample_rate).astype(np.float32)

    y = np.empty_like(signal)
    y[0] = (1.0 - alpha[0]) * signal[0]
    for n in range(1, signal.size):
        y[n] = (1.0 - alpha[n]) * signal[n] + alpha[n] * y[n - 1]
    return y


def apply_stereo_pan_lfo(signal: np.ndarray, lfo: np.ndarray, width: float = 1.0) -> np.ndarray:
    """Mono sinyale LFO ile stereo hareket (pan) uygular.

    width 0..1 aralığındadır; 1 tam sağ-sol salınım, 0 ise ortada sabit pan.
    Sabit güç panlama kullanılır; bu nedenle toplam enerji tutarlı kalır.

    Döndürür:
        shape (N, 2) stereo sinyal (float32).
    """
    signal = np.asarray(signal, dtype=np.float32)
    lfo = np.asarray(lfo, dtype=np.float32)
    if signal.ndim != 1 or lfo.ndim != 1:
        raise ValueError("signal ve lfo 1B olmalı")
    if signal.shape[0] != lfo.shape[0]:
        raise ValueError("signal ve lfo uzunlukları eşleşmeli")

    width = float(np.clip(width, 0.0, 1.0))
    pan = np.clip(lfo, -1.0, 1.0) * width
    angles = (pan + 1.0) * (np.pi / 4.0)
    left_gain = np.cos(angles)
    right_gain = np.sin(angles)

    stereo = np.stack((signal * left_gain, signal * right_gain), axis=-1)
    return stereo.astype(np.float32)


if __name__ == "__main__":
    sr = 48_000
    duration = 1.0
    n = int(sr * duration)

    lfo_sine = sine_lfo(n, freq_hz=2.0, sample_rate=sr)
    lfo_tri = triangle_lfo(n, freq_hz=0.5, sample_rate=sr, phase_rad=np.pi / 2)

    carrier = np.ones(n, dtype=np.float32)
    trem = apply_volume_lfo(carrier, lfo_sine, depth=0.8)
    pan_test = apply_stereo_pan_lfo(carrier, lfo_tri, width=1.0)

    print("Tremolo RMS", float(np.sqrt(np.mean(trem ** 2))))
    print("Pan left/right RMS", float(np.sqrt(np.mean(pan_test[:, 0] ** 2))), float(np.sqrt(np.mean(pan_test[:, 1] ** 2))))
