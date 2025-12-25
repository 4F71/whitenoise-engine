import json
import os
import numpy as np
from pathlib import Path
from copy import deepcopy

p0_dir = "v2_ml/presets/p0"
p0_presets = {}

for json_file in sorted(os.listdir(p0_dir)):
    if json_file.endswith(".json"):
        with open(f"{p0_dir}/{json_file}", "r") as f:
            preset = json.load(f)
            base_name = json_file.replace("_p0.json", "")
            p0_presets[base_name] = preset

print(f"Loaded {len(p0_presets)} P0 presets")


def apply_p1_soft(preset_p0):
    """P1 - soft: Uzun dinleme icin guvenli ve yumusak"""
    preset = deepcopy(preset_p0)
    layer = preset["layers"][0]
    
    layer["filter_config"]["cutoff_hz"] *= 0.85
    layer["lfo_config"]["depth"] *= 0.70
    preset["fx_config"]["saturation_amount"] *= 0.50
    
    layer["filter_config"]["cutoff_hz"] = float(np.clip(
        layer["filter_config"]["cutoff_hz"], 200, 16000
    ))
    layer["lfo_config"]["depth"] = float(np.clip(
        layer["lfo_config"]["depth"], 0.0, 0.3
    ))
    preset["fx_config"]["saturation_amount"] = float(np.clip(
        preset["fx_config"]["saturation_amount"], 0.0, 0.1
    ))
    
    preset["name"] = preset["name"].replace("P0", "P1 Soft")
    preset["description"] = "P1 Soft - Gentle and safe for extended listening"
    preset["tags"] = ["v2", "ml-generated", "p1", "soft"]
    
    return preset


def apply_p2_dark(preset_p0):
    """P2 - dark: Koyu, bastirilmis spektral yapi"""
    preset = deepcopy(preset_p0)
    layer = preset["layers"][0]
    
    layer["filter_config"]["cutoff_hz"] *= 0.70
    preset["fx_config"]["stereo_width"] *= 0.90
    
    layer["filter_config"]["cutoff_hz"] = float(np.clip(
        layer["filter_config"]["cutoff_hz"], 200, 16000
    ))
    preset["fx_config"]["stereo_width"] = float(np.clip(
        preset["fx_config"]["stereo_width"], 0.8, 1.5
    ))
    
    preset["name"] = preset["name"].replace("P0", "P2 Dark")
    preset["description"] = "P2 Dark - Deep and muted spectral character"
    preset["tags"] = ["v2", "ml-generated", "p2", "dark"]
    
    return preset


def apply_p3_calm(preset_p0):
    """P3 - calm: Statik, az hareketli ambience"""
    preset = deepcopy(preset_p0)
    layer = preset["layers"][0]
    
    layer["lfo_config"]["rate_hz"] *= 0.80
    layer["lfo_config"]["depth"] *= 0.50
    layer["gain"] *= 0.90
    
    layer["lfo_config"]["rate_hz"] = float(np.clip(
        layer["lfo_config"]["rate_hz"], 0.001, 0.5
    ))
    layer["lfo_config"]["depth"] = float(np.clip(
        layer["lfo_config"]["depth"], 0.0, 0.3
    ))
    layer["gain"] = float(np.clip(layer["gain"], 0.0, 1.0))
    
    preset["name"] = preset["name"].replace("P0", "P3 Calm")
    preset["description"] = "P3 Calm - Static and tranquil ambience"
    preset["tags"] = ["v2", "ml-generated", "p3", "calm"]
    
    return preset


def apply_p4_airy(preset_p0):
    """P4 - airy: Acik, nefes alan karakter"""
    preset = deepcopy(preset_p0)
    layer = preset["layers"][0]
    
    layer["filter_config"]["cutoff_hz"] *= 1.15
    preset["fx_config"]["stereo_width"] *= 1.10
    
    layer["filter_config"]["cutoff_hz"] = float(np.clip(
        layer["filter_config"]["cutoff_hz"], 200, 16000
    ))
    preset["fx_config"]["stereo_width"] = float(np.clip(
        preset["fx_config"]["stereo_width"], 0.8, 1.5
    ))
    
    preset["name"] = preset["name"].replace("P0", "P4 Airy")
    preset["description"] = "P4 Airy - Open and breathable character"
    preset["tags"] = ["v2", "ml-generated", "p4", "airy"]
    
    return preset


def apply_p5_dense(preset_p0):
    """P5 - dense: Dolgun ve yogun ses"""
    preset = deepcopy(preset_p0)
    layer = preset["layers"][0]
    
    layer["gain"] *= 1.10
    preset["fx_config"]["saturation_amount"] *= 1.50
    
    layer["gain"] = float(np.clip(layer["gain"], 0.0, 1.0))
    preset["fx_config"]["saturation_amount"] = float(np.clip(
        preset["fx_config"]["saturation_amount"], 0.0, 0.1
    ))
    
    preset["name"] = preset["name"].replace("P0", "P5 Dense")
    preset["description"] = "P5 Dense - Full and perceptually thick sound"
    preset["tags"] = ["v2", "ml-generated", "p5", "dense"]
    
    return preset


profiles = {
    "p1": apply_p1_soft,
    "p2": apply_p2_dark,
    "p3": apply_p3_calm,
    "p4": apply_p4_airy,
    "p5": apply_p5_dense,
}

print("\nP1-P5 Karakter Profili Preset Uretimi Basliyor...")
print(f"Toplam: {len(p0_presets)} ses x 5 profil = {len(p0_presets)*5} preset\n")

for profile_name, profile_func in profiles.items():
    os.makedirs(f"v2_ml/presets/{profile_name}", exist_ok=True)
    
    print(f"[{profile_name.upper()}] Profil uygulanıyor...")
    
    for base_name, preset_p0 in p0_presets.items():
        preset_variant = profile_func(preset_p0)
        
        json_filename = f"{base_name}_{profile_name}.json"
        json_path = f"v2_ml/presets/{profile_name}/{json_filename}"
        
        with open(json_path, "w") as f:
            json.dump(preset_variant, f, indent=2)
    
    print(f"  {len(p0_presets)} {profile_name.upper()} preset olusturuldu\n")

print(f"TOPLAM PRESET:")
print(f"   P0:    {len(p0_presets)} (zaten mevcut)")
print(f"   P1-P5: {len(p0_presets)*5} (yeni)")
print(f"   TOPLAM: {len(p0_presets)*6} preset (preset family)")

print("\n" + "="*70)
print("P0-P5 Karakter Profili Karsilastirmasi (Ilk Preset)")
print("="*70)

first_base = list(p0_presets.keys())[0]
print(f"Referans: {first_base}\n")

all_variants = {
    "P0": p0_presets[first_base],
    "P1": apply_p1_soft(p0_presets[first_base]),
    "P2": apply_p2_dark(p0_presets[first_base]),
    "P3": apply_p3_calm(p0_presets[first_base]),
    "P4": apply_p4_airy(p0_presets[first_base]),
    "P5": apply_p5_dense(p0_presets[first_base]),
}

print(f"{'Profil':<10s} {'LP Cutoff':>12s} {'LFO Depth':>12s} {'Stereo W':>12s} {'Saturation':>12s}")
print("-" * 70)

for profile, preset in all_variants.items():
    layer = preset["layers"][0]
    lp = layer["filter_config"]["cutoff_hz"]
    lfo_depth = layer["lfo_config"]["depth"]
    stereo = preset["fx_config"]["stereo_width"]
    sat = preset["fx_config"]["saturation_amount"]
    
    print(f"{profile:<10s} {lp:>10.0f} Hz {lfo_depth:>12.3f} {stereo:>12.2f} {sat:>12.3f}")

print("\nYorum:")
print("  P1 (soft):  LP-down, LFO-down, Sat-down")
print("  P2 (dark):  LP-down-down, Stereo-down")
print("  P3 (calm):  LFO-down-down, Gain-down")
print("  P4 (airy):  LP-up, Stereo-up")
print("  P5 (dense): Gain-up, Sat-up")

