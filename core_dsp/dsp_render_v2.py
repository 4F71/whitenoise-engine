import math
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

FloatArray = np.ndarray
_FT = np.float32
_DENORM_GUARD = _FT(1e-20)


def _to_float32(arr: FloatArray) -> FloatArray:
    """Girişi float32 kopyasına dönüştürür."""
    if arr.dtype != _FT:
        return arr.astype(_FT, copy=False)
    return arr


def _ensure_samples(duration_sec: float, sample_rate: int) -> int:
    """Süreyi örnek sayısına dönüştürür; pozitif olmalıdır."""
    samples = int(round(duration_sec * sample_rate))
    if samples <= 0:
        raise ValueError("Süre ve örnekleme hızı pozitif olmalıdır.")
    return samples


def _dc_block(signal: FloatArray, pole: float = 0.995) -> FloatArray:
    """DC bileşenini bastıran birinci dereceden yüksek geçiren filtre uygular."""
    x = _to_float32(signal)
    y = np.empty_like(x)
    pole_f = _FT(pole)
    x_prev = _FT(0.0)
    y_prev = _FT(0.0)
    for i, sample in enumerate(x):
        y_curr = sample - x_prev + pole_f * y_prev + _DENORM_GUARD
        y[i] = y_curr
        x_prev = sample
        y_prev = y_curr
    return y


def _soft_limit(signal: FloatArray, limit: float = 0.98) -> FloatArray:
    """Arctan tabanlı yumuşak sınırlandırma uygular."""
    lim = _FT(max(1e-6, limit))
    scaled = signal / lim
    return (lim * (_FT(2.0 / math.pi)) * np.arctan(scaled, dtype=_FT)).astype(
        _FT, copy=False
    )


def _fit_length(signal: FloatArray, length: int) -> FloatArray:
    """Sinyali istenen uzunluğa pad eder veya keser."""
    n = signal.shape[0]
    if n == length:
        return signal
    if n > length:
        return signal[:length]
    pad = np.zeros(length - n, dtype=_FT)
    return np.concatenate((signal, pad), axis=0)


def _rms(signal: FloatArray) -> float:
    """RMS değerini float olarak döndürür."""
    x = _to_float32(signal)
    return float(np.sqrt(np.mean(x * x, dtype=_FT)))


def generate_layer(params: Dict) -> FloatArray:
    """
    Verilen parametrelerden tek bir katman sinyali üretir.

    Beklenen anahtarlar:
        - generator: Callable, sinyal üreten fonksiyon
        - args / kwargs: generator argümanları
        - signal: Eğer hazır sinyal verilecekse (alternatif)
        - amplitude: Ölçekleme (varsayılan 1.0)

    Dönüş:
        DC'si bastırılmış ve yumuşak limitli float32 sinyal.
    """
    if "generator" in params:
        gen: Callable = params["generator"]
        args: Iterable = params.get("args", ())
        kwargs: Dict = params.get("kwargs", {})
        signal = gen(*args, **kwargs)
    elif "signal" in params:
        signal = params["signal"]
    else:
        raise ValueError("generator veya signal parametresi gereklidir.")

    amplitude = float(params.get("amplitude", 1.0))
    sig = _to_float32(np.array(signal, dtype=_FT, copy=False)) * _FT(amplitude)
    sig = _dc_block(sig)
    sig = _soft_limit(sig)
    return sig


def mix_layers(layer_list: List[FloatArray]) -> FloatArray:
    """
    Birden fazla katmanı güvenli şekilde toplar.

    Katmanlar farklı uzunlukta ise pad edilerek hizalanır.
    Toplam kazanç, katman sayısına göre ölçeklenir ve yumuşak limit uygulanır.
    """
    if not layer_list:
        return np.zeros(0, dtype=_FT)
    max_len = max(layer.shape[0] for layer in layer_list)
    mix = np.zeros(max_len, dtype=_FT)
    for layer in layer_list:
        l = _fit_length(_to_float32(layer), max_len)
        mix += l
    headroom = _FT(1.0 / max(1.0, math.sqrt(len(layer_list))))
    mix *= headroom
    mix = _soft_limit(mix)
    return mix


def render_sound(
    preset_params: Dict,
    duration_sec: float,
    sample_rate: int,
) -> FloatArray:
    """
    Verilen preset parametreleriyle katmanları oluşturup karıştırır.

    Beklenen preset anahtarları:
        - layers: List[Dict], her biri generate_layer parametreleri
        - target_rms: İstenen RMS seviyesi (opsiyonel, varsayılan 0.1)
    """
    samples = _ensure_samples(duration_sec, sample_rate)
    layers_cfg = preset_params.get("layers", [])
    target_rms = float(preset_params.get("target_rms", 0.1))

    layers = []
    for cfg in layers_cfg:
        layer = generate_layer(cfg)
        layer = _fit_length(layer, samples)
        layers.append(layer)

    mix = mix_layers(layers)
    mix = _dc_block(mix)
    if target_rms > 0.0:
        current_rms = _rms(mix)
        if current_rms > 0.0:
            gain = min(target_rms / current_rms, 2.0)
            mix = (mix * _FT(gain)).astype(_FT, copy=False)
    mix = _soft_limit(mix)
    return mix


if __name__ == "__main__":
    fs = 48000
    duration = 1.0
    t = np.arange(int(fs * duration), dtype=_FT) / _FT(fs)

    def _sine(freq: float, amp: float) -> FloatArray:
        return (_FT(amp) * np.sin(_FT(2.0 * math.pi * freq) * t, dtype=_FT)).astype(
            _FT
        )

    preset = {
        "layers": [
            {"generator": _sine, "args": (220.0, 0.2)},
            {"generator": _sine, "args": (440.0, 0.15)},
        ],
        "target_rms": 0.1,
    }

    out = render_sound(preset, duration_sec=duration, sample_rate=fs)
    print(f"Çıkış min: {float(out.min()):.6f}")
    print(f"Çıkış max: {float(out.max()):.6f}")
    print(f"Çıkış RMS: {_rms(out):.6f}")
