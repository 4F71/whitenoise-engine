"""
UltraGen projesi için temel gürültü üreticileri.

Sağlanan fonksiyonlar:
- Gaussian ve uniform beyaz gürültü
- Voss-McCartney tabanlı pembe gürültü
- Brownian (kırmızı) gürültü
- Blue ve violet gürültü şekillendirme
- Basit spektrum analizi ve kısa bir demo

Tüm çıktılar NumPy dizileridir ve PEP8 + type hint kurallarına uyar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Dışarıdan RNG verilmezse varsayılan Generator döndürür."""
    return rng or np.random.default_rng()


def white_noise_gaussian(
    num_samples: int, mean: float = 0.0, std: float = 1.0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Gaussian dağılımlı beyaz gürültü üretir."""
    generator = _get_rng(rng)
    return generator.normal(loc=mean, scale=std, size=num_samples).astype(np.float32)


def white_noise_uniform(
    num_samples: int, low: float = -1.0, high: float = 1.0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Uniform dağılımlı beyaz gürültü üretir."""
    generator = _get_rng(rng)
    return generator.uniform(low=low, high=high, size=num_samples).astype(np.float32)


def pink_noise_voss_mccartney(
    num_samples: int, num_rows: int = 16, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Voss-McCartney algoritmasıyla pembe gürültü üretir.

    num_rows, spektral yoğunluğu belirler; tipik aralık 8–24 satırdır.
    """
    generator = _get_rng(rng)
    rows = generator.uniform(-1.0, 1.0, size=(num_rows,))
    pink = np.zeros(num_samples, dtype=np.float32)
    counters = np.zeros(num_rows, dtype=np.int64)

    for i in range(num_samples):
        # Rastgele bir satır seçip sadece o satırı güncelleriz (Voss yöntemi)
        idx = generator.integers(0, num_rows)
        rows[idx] = generator.uniform(-1.0, 1.0)
        counters[idx] += 1
        pink[i] = rows.sum()

    # DC ofset ve genlik düzeyi için normalizasyon
    pink -= np.mean(pink)
    max_abs = np.max(np.abs(pink))
    if max_abs > 0:
        pink /= max_abs
    return pink.astype(np.float32)


def brown_noise(
    num_samples: int, step_std: float = 0.02, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Brownian (kırmızı) gürültü üretir.

    step_std, her adımda biriken beyaz gürültünün standart sapmasını kontrol eder.
    """
    generator = _get_rng(rng)
    steps = generator.normal(0.0, step_std, size=num_samples).astype(np.float32)
    brown = np.cumsum(steps)
    brown -= np.mean(brown)
    max_abs = np.max(np.abs(brown))
    if max_abs > 0:
        brown /= max_abs
    return brown.astype(np.float32)


def _highpass_first_difference(x: np.ndarray) -> np.ndarray:
    """Birinci fark yüksek geçişli filtre (yaklaşık +6 dB/oktav eğimi)."""
    diff = np.empty_like(x)
    diff[0] = x[0]
    diff[1:] = x[1:] - x[:-1]
    return diff


def _highpass_second_difference(x: np.ndarray) -> np.ndarray:
    """İkinci fark yüksek geçişli filtre (yaklaşık +12 dB/oktav eğimi)."""
    diff = np.empty_like(x)
    if len(x) == 0:
        return diff
    diff[0] = x[0]
    if len(x) > 1:
        diff[1] = x[1] - x[0]
    if len(x) > 2:
        diff[2:] = x[2:] - 2 * x[1:-1] + x[:-2]
    return diff


def blue_noise(num_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Blue gürültü üretir.

    Beyaz gürültünün birinci farkı alınarak +6 dB/oktav spektral eğim elde edilir.
    """
    white = white_noise_gaussian(num_samples, rng=rng)
    blue = _highpass_first_difference(white)
    max_abs = np.max(np.abs(blue))
    if max_abs > 0:
        blue /= max_abs
    return blue.astype(np.float32)


def violet_noise(num_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Violet (mor) gürültü üretir.

    Beyaz gürültünün ikinci farkı alınarak +12 dB/oktav spektral eğim elde edilir.
    """
    white = white_noise_gaussian(num_samples, rng=rng)
    violet = _highpass_second_difference(white)
    max_abs = np.max(np.abs(violet))
    if max_abs > 0:
        violet /= max_abs
    return violet.astype(np.float32)


def power_spectrum(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tek taraflı güç spektrumu döndürür.

    Returns:
        freqs_hz: Frekans ekseni (Hz)
        power_db: Güç spektrumu (dBFS)
    """
    n = len(signal)
    if n == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    window = np.hanning(n).astype(np.float32)
    windowed = signal * window
    fft = np.fft.rfft(windowed)
    power = np.abs(fft) ** 2
    power /= np.max(power) + 1e-12  # dBFS için normalize
    power_db = 10 * np.log10(power + 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return freqs.astype(np.float32), power_db.astype(np.float32)


@dataclass
class NoiseExample:
    """Demo çıktısını temsil eden küçük veri sınıfı."""
    signal: np.ndarray
    freqs_hz: np.ndarray
    power_db: np.ndarray


def demo_pink_noise(duration_s: float = 1.0, sample_rate: int = 48_000) -> NoiseExample:
    """
    1 saniyelik pembe gürültü üretir ve spektrumunu hesaplar.

    Bu fonksiyon hem kullanım örneği hem de hızlı bir duman testi olarak işlev görür.
    """
    num_samples = int(duration_s * sample_rate)
    signal = pink_noise_voss_mccartney(num_samples=num_samples)
    freqs_hz, power_db = power_spectrum(signal, sample_rate=sample_rate)
    return NoiseExample(signal=signal, freqs_hz=freqs_hz, power_db=power_db)


def _maybe_plot(example: NoiseExample) -> None:
    """
    Matplotlib varsa spektrum grafiğini çizer; yoksa bilgi mesajı basar.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print("Matplotlib bulunamadı; spektrum sadece sayısal olarak hesaplandı.")
        print(
            f"Sinyal RMS: {np.sqrt(np.mean(example.signal ** 2)):.3f}, "
            f"Örnek güç aralığı (dBFS): {example.power_db.min():.1f} .. {example.power_db.max():.1f}"
        )
        return

    plt.figure(figsize=(8, 4))
    plt.semilogx(example.freqs_hz + 1e-12, example.power_db, color="#d57b00")
    plt.title("Pembe Gürültü Güç Spektrumu")
    plt.xlabel("Frekans (Hz)")
    plt.ylabel("Güç (dBFS)")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example = demo_pink_noise()
    _maybe_plot(example)
