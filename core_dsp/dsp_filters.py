import numpy as np


def lpf_one_pole(x: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    """Tek kutuplu alçak geçiren filtre uygular.

    Parametreler:
        x: Giriş sinyali (1B numpy dizisi).
        cutoff_hz: Kesim frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).

    Döndürür:
        Filtrelenmiş sinyal (numpy dizisi).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x 1 boyutlu olmalıdır")
    if cutoff_hz <= 0 or sample_rate <= 0:
        raise ValueError("cutoff_hz ve sample_rate pozitif olmalıdır")

    alpha = np.exp(-2.0 * np.pi * cutoff_hz / sample_rate)
    y = np.empty_like(x)
    y[0] = (1 - alpha) * x[0]
    for n in range(1, x.size):
        y[n] = (1 - alpha) * x[n] + alpha * y[n - 1]
    return y


def hpf_one_pole(x: np.ndarray, cutoff_hz: float, sample_rate: float) -> np.ndarray:
    """Tek kutuplu yüksek geçiren filtre uygular.

    Parametreler:
        x: Giriş sinyali (1B numpy dizisi).
        cutoff_hz: Kesim frekansı (Hz).
        sample_rate: Örnekleme hızı (Hz).

    Döndürür:
        Filtrelenmiş sinyal (numpy dizisi).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x 1 boyutlu olmalıdır")
    if cutoff_hz <= 0 or sample_rate <= 0:
        raise ValueError("cutoff_hz ve sample_rate pozitif olmalıdır")

    alpha = np.exp(-2.0 * np.pi * cutoff_hz / sample_rate)
    y = np.empty_like(x)
    y[0] = 0.0
    for n in range(1, x.size):
        y[n] = alpha * (y[n - 1] + x[n] - x[n - 1])
    return y


def bandpass_simple(x: np.ndarray, low_cut_hz: float, high_cut_hz: float, sample_rate: float) -> np.ndarray:
    """Basit bant geçiren filtre (HPF + LPF) uygular.

    Parametreler:
        x: Giriş sinyali (1B numpy dizisi).
        low_cut_hz: Alt kesim (Hz) için yüksek geçiren filtre.
        high_cut_hz: Üst kesim (Hz) için alçak geçiren filtre.
        sample_rate: Örnekleme hızı (Hz).

    Döndürür:
        Filtrelenmiş sinyal (numpy dizisi).
    """
    if low_cut_hz >= high_cut_hz:
        raise ValueError("low_cut_hz high_cut_hz değerinden küçük olmalıdır")
    high_passed = hpf_one_pole(x, low_cut_hz, sample_rate)
    return lpf_one_pole(high_passed, high_cut_hz, sample_rate)


def eq_3band(
    x: np.ndarray,
    low_gain: float,
    mid_gain: float,
    high_gain: float,
    low_cut_hz: float,
    high_cut_hz: float,
    sample_rate: float,
) -> np.ndarray:
    """3 bantlı basit EQ uygular (low/mid/high kazanç).

    Parametreler:
        x: Giriş sinyali (1B numpy dizisi).
        low_gain: Alçak frekans kazancı (linear).
        mid_gain: Orta frekans kazancı (linear).
        high_gain: Yüksek frekans kazancı (linear).
        low_cut_hz: Low geçişi ayırmak için kesim (Hz).
        high_cut_hz: High geçişi ayırmak için kesim (Hz).
        sample_rate: Örnekleme hızı (Hz).

    Döndürür:
        EQ uygulanmış sinyal (numpy dizisi).
    """
    if not (0 < low_cut_hz < high_cut_hz < sample_rate * 0.5):
        raise ValueError("Kesim frekansları 0 < low < high < Nyquist olmalıdır")

    low = lpf_one_pole(x, low_cut_hz, sample_rate)
    high = hpf_one_pole(x, high_cut_hz, sample_rate)
    mid = x - low - high

    return low * low_gain + mid * mid_gain + high * high_gain


def normalize(x: np.ndarray, target_peak: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """Sinyali verilen tepe değere ölçekler.

    Parametreler:
        x: Giriş sinyali (1B numpy dizisi).
        target_peak: İstenen tepe değeri (varsayılan 1.0).
        eps: Sıfıra bölünmeyi önlemek için küçük değer.

    Döndürür:
        Ölçeklenmiş sinyal (numpy dizisi).
    """
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x))
    if peak < eps:
        return np.zeros_like(x)
    return x * (target_peak / peak)


if __name__ == "__main__":
    # Basit testler
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    x = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 4000 * t)

    lpf_out = lpf_one_pole(x, 500, sr)
    hpf_out = hpf_one_pole(x, 1000, sr)
    bp_out = bandpass_simple(x, 300, 2000, sr)
    eq_out = eq_3band(
        x,
        low_gain=1.2,
        mid_gain=0.8,
        high_gain=1.5,
        low_cut_hz=300,
        high_cut_hz=3000,
        sample_rate=sr,
    )
    norm_out = normalize(eq_out)

    print("LPF ortalama", float(np.mean(lpf_out)))
    print("HPF ortalama", float(np.mean(hpf_out)))
    print("Bandpass max", float(np.max(np.abs(bp_out))))
    print("EQ max", float(np.max(np.abs(eq_out))))
    print("Normalize hedef", float(np.max(np.abs(norm_out))))
