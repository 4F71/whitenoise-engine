"""
scripts/profile_render.py

Render pipeline profiling - bottleneck tespiti için.

Python cProfile kullanarak render_sound() fonksiyonunu ve 
tüm alt fonksiyon çağrılarını profile eder. En yavaş fonksiyonları raporlar.

Kullanım:
    python scripts/profile_render.py
"""

import cProfile
import pstats
import sys
import os
from pathlib import Path

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core_dsp import dsp_render
from preset_system.preset_library import list_v2_presets, get_v2_preset
from ui_app.preset_to_dsp_adapter import adapt_preset_to_layer_generators


def profile_render():
    """
    1 dakikalık render'ı profile et ve bottleneck'leri raporla.
    """
    print("=" * 70)
    print("RENDER PIPELINE PROFILING")
    print("=" * 70)
    
    # 1. Test setup
    print("\n[1/3] Test Setup...")
    print("  - V2 preset yükleniyor: distant_wind, P0")
    
    v2_presets = list_v2_presets()
    
    # distant_wind P0 preset'i bul
    test_preset = None
    for preset in v2_presets:
        if "distant_wind" in preset["name"] and preset["profile"] == "P0":
            test_preset = preset
            break
    
    if not test_preset:
        print("  ERROR: distant_wind P0 preset bulunamadı!")
        print("  Fallback: İlk P0 preset kullanılıyor")
        for preset in v2_presets:
            if preset["profile"] == "P0":
                test_preset = preset
                break
    
    if not test_preset:
        print("  FATAL: Hiçbir preset bulunamadı!")
        return
    
    print(f"  - Preset: {test_preset['name']}")
    
    preset_config = get_v2_preset(test_preset["path"])
    generators = adapt_preset_to_layer_generators(preset_config)
    
    duration = 60.0  # 1 dakika
    sample_rate = 48000
    
    print(f"  - Duration: {duration} saniye")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Layer sayısı: {len(generators)}")
    
    # 2. Profiling
    print("\n[2/3] Profiling başlıyor...")
    print("  (Bu 1-2 dakika sürebilir)")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # PROFILE EDİLEN KOD
    audio_data = dsp_render.render_sound(
        generators,
        duration_sec=duration,
        sample_rate=sample_rate
    )
    
    profiler.disable()
    
    print(f"  - Render tamamlandı")
    print(f"  - Output shape: {audio_data.shape}")
    print(f"  - Output dtype: {audio_data.dtype}")
    
    # 3. Rapor
    print("\n[3/3] Analiz raporu:")
    print("=" * 70)
    print("TOP 25 FONKSIYON (Cumulative Time - Toplam Süre)")
    print("=" * 70)
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()  # Dosya yollarını kısalt
    stats.sort_stats('cumulative')  # Toplam süreye göre sırala
    stats.print_stats(25)  # İlk 25 fonksiyon
    
    print("\n" + "=" * 70)
    print("BOTTLENECK ÖZET")
    print("=" * 70)
    
    # En yavaş fonksiyonları manual tespit et
    print("\nEn yavaş modüller ve fonksiyonlar:")
    print("(Manual inceleme: yukarıdaki raporda arayın)")
    print("\n  Örnek bottleneck'ler:")
    print("  - dsp_noise.py içindeki fonksiyonlar (noise generation)")
    print("  - dsp_filters.py içindeki fonksiyonlar (filtering)")
    print("  - dsp_lfo.py içindeki fonksiyonlar (LFO generation)")
    print("  - np.* fonksiyonlar (numpy işlemleri)")
    
    print("\n" + "=" * 70)
    print("TAMAMLANDI")
    print("=" * 70)
    print("\nÖneriler:")
    print("  1. 'cumulative time' kolonunda en yüksek değerlere bakın")
    print("  2. core_dsp/ modüllerindeki fonksiyonları not edin")
    print("  3. Bu fonksiyonlar Numba JIT ile optimize edilebilir")
    print("  4. numpy vektörel işlemler genelde hızlıdır")
    print("  5. Loop'lar yavaştır (Numba ile optimize edilebilir)")


if __name__ == "__main__":
    try:
        profile_render()
    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

