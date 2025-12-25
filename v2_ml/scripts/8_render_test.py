import json
import os
import sys
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from preset_system.preset_schema import PresetConfig
from ui_app.preset_to_dsp_adapter import adapt_preset_to_layer_generators
from core_dsp.dsp_render import render_sound

test_presets = [
    "v2_ml/presets/p0/distant_wind__far_airflow_ambient__bbc__v2__60s__mono__48k_p0.json",
    "v2_ml/presets/p2/distant_wind__far_airflow_ambient__bbc__v2__60s__mono__48k_p2.json",
    "v2_ml/presets/p4/distant_wind__far_airflow_ambient__bbc__v2__60s__mono__48k_p4.json",
    
    "v2_ml/presets/p0/room_tone__subtle_room_presence__bbc__v2__60s__mono__48k_p0.json",
    "v2_ml/presets/p1/room_tone__subtle_room_presence__bbc__v2__60s__mono__48k_p1.json",
    
    "v2_ml/presets/p0/mechanical_hum__power_system_steady__bbc__v2__60s__mono__48k_p0.json",
    "v2_ml/presets/p4/mechanical_hum__power_system_steady__bbc__v2__60s__mono__48k_p4.json",
    
    "v2_ml/presets/p0/urban_background__city_far_background__bbc__v2__60s__mono__48k_p0.json",
    "v2_ml/presets/p2/urban_background__city_far_background__bbc__v2__60s__mono__48k_p2.json",
]

output_dir = "v2_ml/render_test"
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("V2 ML PRESET RENDER TESTI")
print("="*70)
print(f"Toplam preset: {len(test_presets)}")
print(f"Render suresi: 10 saniye/preset")
print(f"Cikti klasoru: {output_dir}\n")


def render_preset_to_wav(
    preset_path: str,
    output_path: str,
    duration_sec: float = 10.0,
    sample_rate: int = 48000
) -> dict:
    """Preset JSON'u render edip WAV olarak kaydeder."""
    
    with open(preset_path, "r") as f:
        preset_dict = json.load(f)
    
    preset = PresetConfig(**preset_dict)
    
    generators = adapt_preset_to_layer_generators(preset)
    
    if len(generators) == 0:
        raise ValueError(f"Preset'te aktif layer yok: {preset_path}")
    
    audio = render_sound(
        layer_generators=generators,
        duration_sec=duration_sec,
        sample_rate=sample_rate
    )
    
    peak = np.max(np.abs(audio))
    if peak > 0:
        target_peak = 10 ** (-3 / 20)
        audio = audio * (target_peak / peak)
    
    audio = np.clip(audio, -1.0, 1.0)
    
    audio_int16 = (audio * 32767).astype(np.int16)
    
    wavfile.write(output_path, sample_rate, audio_int16)
    
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
    
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sample_rate)
    magnitude = np.abs(fft)
    centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
    
    return {
        "rms_db": float(rms_db),
        "peak_db": float(peak_db),
        "spectral_centroid_hz": float(centroid),
        "duration_sec": duration_sec,
    }


results = []

print("Render basliyor...\n")

for i, preset_path in enumerate(test_presets):
    preset_name = Path(preset_path).stem
    output_wav = f"{output_dir}/{preset_name}.wav"
    
    print(f"[{i+1}/{len(test_presets)}] {preset_name}")
    
    try:
        analysis = render_preset_to_wav(
            preset_path=preset_path,
            output_path=output_wav,
            duration_sec=10.0,
            sample_rate=48000
        )
        
        with open(preset_path, "r") as f:
            preset_data = json.load(f)
        
        result = {
            "preset_path": preset_path,
            "preset_name": preset_name,
            "output_wav": output_wav,
            "noise_type": preset_data["layers"][0]["noise_type"],
            "lp_cutoff_hz": preset_data["layers"][0]["filter_config"]["cutoff_hz"],
            "lfo_depth": preset_data["layers"][0]["lfo_config"]["depth"],
            "profile": preset_name.split("_")[-1],
            **analysis
        }
        
        results.append(result)
        
        print(f"  RMS: {analysis['rms_db']:.1f} dB, "
              f"Peak: {analysis['peak_db']:.1f} dB, "
              f"Centroid: {analysis['spectral_centroid_hz']:.0f} Hz")
        
    except Exception as e:
        print(f"  HATA: {e}")
        results.append({
            "preset_path": preset_path,
            "preset_name": preset_name,
            "error": str(e)
        })

print(f"\n{len([r for r in results if 'error' not in r])}/{len(test_presets)} basarili")

print("\n" + "="*70)
print("KARSILASTIRMA ANALIZI")
print("="*70)

from collections import defaultdict

by_profile = defaultdict(list)
for r in results:
    if "error" not in r:
        by_profile[r["profile"]].append(r)

print(f"\n{'Profil':<10s} {'Avg RMS (dB)':>15s} {'Avg Centroid (Hz)':>20s} {'Ornek':>10s}")
print("-" * 70)

for profile in sorted(by_profile.keys()):
    items = by_profile[profile]
    avg_rms = np.mean([item["rms_db"] for item in items])
    avg_centroid = np.mean([item["spectral_centroid_hz"] for item in items])
    
    print(f"{profile.upper():<10s} {avg_rms:>15.1f} {avg_centroid:>20.0f} {len(items):>10d}")

print(f"\n{'Noise Type':<15s} {'Avg Centroid (Hz)':>20s} {'Ornek':>10s}")
print("-" * 50)

by_noise = defaultdict(list)
for r in results:
    if "error" not in r:
        by_noise[r["noise_type"]].append(r)

for noise in sorted(by_noise.keys()):
    items = by_noise[noise]
    avg_centroid = np.mean([item["spectral_centroid_hz"] for item in items])
    print(f"{noise:<15s} {avg_centroid:>20.0f} {len(items):>10d}")

report = {
    "test_info": {
        "total_presets": len(test_presets),
        "successful": len([r for r in results if "error" not in r]),
        "duration_per_preset": 10.0,
        "sample_rate": 48000,
        "note": "FX chain (saturation, stereo_width, reverb) uygulanmadi (adapter sinirlama)"
    },
    "results": results,
    "profile_stats": {
        profile: {
            "avg_rms_db": float(np.mean([item["rms_db"] for item in items])),
            "avg_centroid_hz": float(np.mean([item["spectral_centroid_hz"] for item in items])),
            "count": len(items)
        }
        for profile, items in by_profile.items()
    }
}

with open(f"{output_dir}/render_test_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nRapor kaydedildi: {output_dir}/render_test_report.json")
print(f"WAV dosyalari: {output_dir}/*.wav")

