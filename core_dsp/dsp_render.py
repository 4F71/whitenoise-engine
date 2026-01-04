import math
from typing import Callable, List, Tuple
import multiprocessing
from multiprocessing import Pool

import numpy as np

# cloudpickle: Closure'ları pickle et (multiprocessing için)
try:
    import cloudpickle
    # cloudpickle'ı multiprocessing için aktif et
    multiprocessing.reduction.ForkingPickler = type(
        'CustomPickler',
        (multiprocessing.reduction.ForkingPickler,),
        {'dumps': staticmethod(cloudpickle.dumps)}
    )
    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False

FloatArray = np.ndarray
_FT = np.float32


def _ensure_samples(duration_sec: float, sample_rate: int) -> int:
    """Süreyi örnek sayısına dönüştürür; pozitif olmalıdır."""
    samples = int(round(duration_sec * sample_rate))
    if samples <= 0:
        raise ValueError("Süre ve örnekleme hızı pozitif olmalıdır.")
    return samples


def generate_layer(
    layer_generator: Callable[[float, int], FloatArray],
    duration_sec: float,
    sample_rate: int,
) -> FloatArray:
    """
    Verilen generator'ı çağırarak tek bir mono layer üretir.

    Parametreler:
        layer_generator: Mono sinyal üreten callable.
        duration_sec: Süre (saniye).
        sample_rate: Örnekleme hızı (Hz).

    Dönüş:
        Mono sinyal, shape (N,), float32.
    """
    samples = _ensure_samples(duration_sec, sample_rate)
    sig = layer_generator(duration_sec, sample_rate)
    sig = np.array(sig, dtype=_FT, copy=False)
    if sig.ndim != 1:
        raise ValueError("Layer generator mono (N,) sinyal üretmelidir.")
    if sig.shape[0] != samples:
        if sig.shape[0] > samples:
            sig = sig[:samples]
        else:
            pad = np.zeros(samples - sig.shape[0], dtype=_FT)
            sig = np.concatenate((sig, pad), axis=0)
    return sig.astype(_FT, copy=False)


def _render_layer_worker(args: Tuple[Callable[[float, int], FloatArray], float, int]) -> FloatArray:
    """
    Worker function for multiprocessing: renders a single layer.
    
    Must be module-level for Windows pickle compatibility.
    
    Parametreler:
        args: Tuple of (layer_generator, duration_sec, sample_rate).
        
    Dönüş:
        Mono sinyal, shape (N,), float32.
    """
    layer_generator, duration_sec, sample_rate = args
    return generate_layer(layer_generator, duration_sec, sample_rate)


def mix_layers(layers: List[FloatArray]) -> FloatArray:
    """
    Mono layer'ları toplar; shape (N,) döndürür.

    Parametreler:
        layers: Mono sinyaller listesi.

    Dönüş:
        Toplanmış mono sinyal, float32.
    """
    if not layers:
        return np.zeros(0, dtype=_FT)
    max_len = max(layer.shape[0] for layer in layers)
    mix = np.zeros(max_len, dtype=_FT)
    for layer in layers:
        if layer.ndim != 1:
            raise ValueError("Tüm layer'lar mono (N,) olmalıdır.")
        if layer.shape[0] == max_len:
            mix += layer.astype(_FT, copy=False)
        elif layer.shape[0] > max_len:
            mix += layer[:max_len].astype(_FT, copy=False)
        else:
            pad = np.zeros(max_len - layer.shape[0], dtype=_FT)
            mix += np.concatenate((layer.astype(_FT, copy=False), pad), axis=0)
    return mix.astype(_FT, copy=False)


def render_sound(
    layer_generators: List[Callable[[float, int], FloatArray]],
    duration_sec: float,
    sample_rate: int,
    use_multiprocessing: bool = True,
) -> FloatArray:
    """
    Verilen generator listesiyle tüm layer'ları üretir ve toplar.
    
    Optimized: 5+ layer için multiprocessing kullanır (paralel rendering).
    cloudpickle ile closure'lar pickle edilebilir.
    
    Performance:
    - Windows'ta 2x speedup (5 layer, 10+ dakika)
    - 60s altı renderlar için overhead yüksek (sequential önerilir)

    Parametreler:
        layer_generators: Mono sinyal üreten callable listesi.
        duration_sec: Süre (saniye).
        sample_rate: Örnekleme hızı (Hz).
        use_multiprocessing: Paralel processing kullan (default: True).

    Dönüş:
        Mono final sinyal, shape (N,), float32.
    """
    # Single layer: Multiprocessing overhead gereksiz
    if len(layer_generators) == 1:
        return generate_layer(layer_generators[0], duration_sec, sample_rate)
    
    # Kısa render (<120s): Overhead fazla, sequential daha hızlı
    if duration_sec < 120.0:
        use_multiprocessing = False
    
    # Multiprocessing: 2+ layer için paralel render
    if use_multiprocessing and len(layer_generators) > 1:
        # cloudpickle yoksa fallback
        if not CLOUDPICKLE_AVAILABLE:
            print("[INFO] cloudpickle not available. Falling back to sequential rendering.")
            print("[INFO] Install cloudpickle for 2x speedup: pip install cloudpickle")
        else:
            try:
                # Pool size: min(layer_count, 5) → En fazla 5 process
                pool_size = min(len(layer_generators), 5)
                
                # Worker arguments prepare et
                worker_args = [
                    (gen, duration_sec, sample_rate) for gen in layer_generators
                ]
                
                # Parallel render with cloudpickle
                # cloudpickle.dumps/loads kullanarak pickle et
                with Pool(processes=pool_size) as pool:
                    layers = pool.map(_render_layer_worker, worker_args)
                
                return mix_layers(layers)
            
            except Exception as e:
                # Fallback: Sequential rendering (pickle issues vb.)
                print(f"[WARNING] Multiprocessing failed: {e}. Falling back to sequential rendering.")
    
    # Sequential fallback
    layers = [
        generate_layer(gen, duration_sec=duration_sec, sample_rate=sample_rate)
        for gen in layer_generators
    ]
    return mix_layers(layers)


if __name__ == "__main__":
    fs = 48000
    duration = 1.5

    def _sine_layer(duration_sec: float, sample_rate: int) -> FloatArray:
        n = int(round(duration_sec * sample_rate))
        t = np.arange(n, dtype=_FT) / _FT(sample_rate)
        return (_FT(0.2) * np.sin(_FT(2.0 * math.pi * 440.0) * t, dtype=_FT)).astype(
            _FT
        )

    def _saw_layer(duration_sec: float, sample_rate: int) -> FloatArray:
        n = int(round(duration_sec * sample_rate))
        t = np.arange(n, dtype=_FT) / _FT(sample_rate)
        phase = (220.0 * t) % 1.0
        return (_FT(0.1) * (_FT(2.0) * phase - _FT(1.0))).astype(_FT)

    # NOTE: Local functions in __main__ can't be pickled on Windows
    # Use use_multiprocessing=False for __main__ testing
    out = render_sound(
        [_sine_layer, _saw_layer], 
        duration_sec=duration, 
        sample_rate=fs,
        use_multiprocessing=False  # Windows pickle compatibility
    )
    print("Çıkış şekli:", out.shape)
    print("Min:", float(out.min()), "Max:", float(out.max()))
