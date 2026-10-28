"""
=============================================================================
 AgriVest -- Agro-Climatic Advisory Engine
 File   : src/weather_service.py
 Purpose: Automated climate detection from NOAA ONI data + historical district
          baselines, and biologically-grounded agronomic suitability scoring.

 This module replaces manual rainfall/temperature/ENSO inputs with an
 intelligent advisory layer that:
   1. Reads the latest ENSO anomaly from oni_climate_index.csv
   2. Looks up district/season historical medians from master_cleaned_dataset.csv
   3. Adjusts baselines for the active El Niño / La Niña regime
   4. Scores each crop against season compatibility, water requirements,
      and thermal tolerance -- returning a penalty multiplier [0.05, 1.0]
      plus plain-English diagnostic reasons for the explainability panel.

 Run standalone for a quick demo:
   python src/weather_service.py
=============================================================================
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "master_cleaned_dataset.csv")
ONI_CSV       = os.path.join(PROJECT_ROOT, "data", "raw",       "oni_climate_index.csv")

# =============================================================================
# AGRONOMIC RULES & BIOLOGICAL THRESHOLDS  (CROP_AGRONOMY)
# =============================================================================
# Sources:
#   - ICAR Handbook of Agriculture (2023 edition)
#   - FAO Crop Water Requirement Manual (Doorenbos & Pruitt, 1977)
#   - CRIDA (Central Research Institute for Dryland Agriculture) guidelines

CROP_AGRONOMY: dict = {
    "Rice": {
        "allowed_seasons"   : ["Kharif", "Summer", "Zaid"],
        "min_water_mm"      : 900,     # Flooded paddy requires high water
        "max_temp_c"        : 38.0,    # Spikelet sterility above 38°C
        "drought_resilience": 0.20,    # Very low -- collapses under water stress
        "max_yield_cap"     : 6.0,     # t/ha biological ceiling (HYV paddy)
    },
    "Wheat": {
        "allowed_seasons"   : ["Rabi", "Winter"],
        "min_water_mm"      : 400,
        "max_temp_c"        : 32.0,    # Grain-filling sensitive to heat
        "drought_resilience": 0.55,
        "max_yield_cap"     : 5.0,
    },
    "Maize": {
        "allowed_seasons"   : ["Kharif", "Rabi", "Summer", "Zaid"],
        "min_water_mm"      : 500,
        "max_temp_c"        : 40.0,
        "drought_resilience": 0.50,
        "max_yield_cap"     : 5.5,
    },
    "Jowar": {
        "allowed_seasons"   : ["Kharif", "Rabi", "Summer"],
        "min_water_mm"      : 300,     # Highly drought tolerant
        "max_temp_c"        : 42.0,
        "drought_resilience": 0.90,    # Excellent water-stress retention
        "max_yield_cap"     : 3.5,
    },
    "Bajra": {
        "allowed_seasons"   : ["Kharif", "Summer", "Zaid"],
        "min_water_mm"      : 250,
        "max_temp_c"        : 44.0,
        "drought_resilience": 0.85,
        "max_yield_cap"     : 2.5,
    },
    "Gram": {
        "allowed_seasons"   : ["Rabi", "Winter"],
        "min_water_mm"      : 280,
        "max_temp_c"        : 30.0,    # Flowers abort above 30°C
        "drought_resilience": 0.65,
        "max_yield_cap"     : 2.5,
    },
    "Groundnut": {
        "allowed_seasons"   : ["Kharif", "Summer", "Rabi"],
        "min_water_mm"      : 450,
        "max_temp_c"        : 40.0,
        "drought_resilience": 0.55,
        "max_yield_cap"     : 3.5,
    },
    "Cotton": {
        "allowed_seasons"   : ["Kharif", "Whole Year"],
        "min_water_mm"      : 600,     # Needs deep moisture at boll-setting
        "max_temp_c"        : 43.0,    # Relatively heat-tolerant
        "drought_resilience": 0.40,    # Poor without monsoon reliability
        "max_yield_cap"     : 4.0,     # Seed cotton t/ha
    },
    "Soyabean": {
        "allowed_seasons"   : ["Kharif"],
        "min_water_mm"      : 450,
        "max_temp_c"        : 38.0,
        "drought_resilience": 0.45,
        "max_yield_cap"     : 3.5,
    },
    "Sugarcane": {
        "allowed_seasons"   : ["Kharif", "Whole Year"],
        "min_water_mm"      : 1500,    # Extremely water intensive
        "max_temp_c"        : 40.0,
        "drought_resilience": 0.30,
        "max_yield_cap"     : 80.0,    # Cane biomass, not refined output
    },
    "Mustard": {
        "allowed_seasons"   : ["Rabi", "Winter"],
        "min_water_mm"      : 300,
        "max_temp_c"        : 30.0,
        "drought_resilience": 0.60,
        "max_yield_cap"     : 2.5,
    },
    "Sunflower": {
        "allowed_seasons"   : ["Kharif", "Rabi", "Summer"],
        "min_water_mm"      : 400,
        "max_temp_c"        : 38.0,
        "drought_resilience": 0.60,
        "max_yield_cap"     : 2.0,
    },
    "Arhar/Tur": {
        "allowed_seasons"   : ["Kharif", "Whole Year"],
        "min_water_mm"      : 350,
        "max_temp_c"        : 40.0,
        "drought_resilience": 0.70,
        "max_yield_cap"     : 2.0,
    },
    "Urad": {
        "allowed_seasons"   : ["Kharif", "Summer", "Rabi"],
        "min_water_mm"      : 350,
        "max_temp_c"        : 39.0,
        "drought_resilience": 0.60,
        "max_yield_cap"     : 1.5,
    },
    "Moong(Green Gram)": {
        "allowed_seasons"   : ["Kharif", "Summer", "Zaid"],
        "min_water_mm"      : 300,
        "max_temp_c"        : 40.0,
        "drought_resilience": 0.65,
        "max_yield_cap"     : 1.5,
    },
    "Potato": {
        "allowed_seasons"   : ["Rabi", "Winter"],
        "min_water_mm"      : 450,
        "max_temp_c"        : 30.0,    # Tuber quality deteriorates >30°C
        "drought_resilience": 0.45,
        "max_yield_cap"     : 25.0,
    },
    "Onion": {
        "allowed_seasons"   : ["Rabi", "Kharif", "Summer"],
        "min_water_mm"      : 350,
        "max_temp_c"        : 40.0,
        "drought_resilience": 0.50,
        "max_yield_cap"     : 20.0,
    },
    "Tomato": {
        "allowed_seasons"   : ["Kharif", "Rabi", "Summer"],
        "min_water_mm"      : 400,
        "max_temp_c"        : 38.0,
        "drought_resilience": 0.45,
        "max_yield_cap"     : 40.0,
    },
    "Banana": {
        "allowed_seasons"   : ["Whole Year", "Kharif"],
        "min_water_mm"      : 1200,
        "max_temp_c"        : 38.0,
        "drought_resilience": 0.30,
        "max_yield_cap"     : 35.0,
    },
    "Coconut": {
        "allowed_seasons"   : ["Whole Year"],
        "min_water_mm"      : 1500,
        "max_temp_c"        : 40.0,
        "drought_resilience": 0.35,
        "max_yield_cap"     : 15_000.0,   # nuts/ha converted to ~15 t/ha copra equiv
    },
}

# Default agronomy for crops not in the table
DEFAULT_AGRONOMY = {
    "allowed_seasons"   : ["Kharif", "Rabi", "Summer", "Whole Year", "Zaid", "Winter", "Autumn"],
    "min_water_mm"      : 400,
    "max_temp_c"        : 40.0,
    "drought_resilience": 0.55,
    "max_yield_cap"     : 5.0,
}

# Irrigation credit: extra effective water (mm equivalent per season)
IRRIGATION_CREDITS: dict = {
    "Rainfed"      : 0,
    "Well_Borewell": 350,
    "Canal_River"  : 700,
}

# ENSO adjustment factors applied to historical district baselines
ENSO_RAINFALL_ADJUSTMENT: dict = {
    "El_Nino": -0.15,    # -15% rainfall under El Niño
    "La_Nina":  0.15,    # +15% rainfall under La Niña
    "Normal" :  0.00,
}
ENSO_TEMP_ADJUSTMENT: dict = {
    "El_Nino": +0.8,     # +0.8°C warmer under El Niño
    "La_Nina":  0.0,
    "Normal" :  0.0,
}


# =============================================================================
# MODULE-LEVEL DATA CACHE  (loaded once per process)
# =============================================================================

_MASTER_DF_CACHE: Optional[pd.DataFrame] = None


def _get_master_df() -> pd.DataFrame:
    """Lazily load and cache the master cleaned dataset."""
    global _MASTER_DF_CACHE
    if _MASTER_DF_CACHE is None:
        if not os.path.exists(MASTER_CSV):
            raise FileNotFoundError(
                f"Master dataset not found: {MASTER_CSV}\n"
                "Run data_preprocessing.py first."
            )
        _MASTER_DF_CACHE = pd.read_csv(
            MASTER_CSV,
            usecols=["State", "District", "Season", "Rainfall", "Temperature"],
        )
    return _MASTER_DF_CACHE


# =============================================================================
# 1. AUTOMATED CLIMATE DETECTION
# =============================================================================

def auto_detect_climate_and_weather(
    district: str,
    season  : str,
    state   : Optional[str] = None,
) -> dict:
    """
    Automatically detect the current climate state and estimate seasonal
    weather parameters for a given district and season.

    Pipeline
    --------
    1. Read the latest valid ONI anomaly from oni_climate_index.csv.
       Filter out placeholder values (< -90).
       Classify:
         ONI >= +0.5  ->  El_Nino
         ONI <= -0.5  ->  La_Nina
         else         ->  Normal

    2. Query the master dataset for the historical median rainfall and
       temperature for the given District + Season combination.
       Fallback hierarchy:
         a. District + Season match  (primary)
         b. State  + Season match    (state-level median)
         c. Season-only match        (national seasonal median)
         d. Grand median             (last resort)

    3. Adjust baseline by ENSO regime:
         El Niño: rainfall × 0.85, temp + 0.8°C
         La Niña: rainfall × 1.15

    Args:
        district : District name (title-case, must match training vocabulary).
        season   : Season string ('Kharif', 'Rabi', etc.)
        state    : Optional state name for fallback lookup.

    Returns:
        {
          "rainfall_mm"    : float,
          "temperature_c"  : float,
          "climate_regime" : str,   # 'El_Nino' | 'La_Nina' | 'Normal'
          "oni_value"      : float,
          "baseline_source": str,   # which fallback level was used
        }
    """
    # ── Step 1: Classify ENSO regime from latest ONI ─────────────────────────
    oni_value = 0.0
    climate_regime = "Normal"

    if os.path.exists(ONI_CSV):
        try:
            oni_df  = pd.read_csv(ONI_CSV)
            oni_col = oni_df.columns[1]
            valid   = oni_df[oni_df[oni_col] > -90.0]
            if not valid.empty:
                oni_value = float(valid.iloc[-1][oni_col])
                if oni_value >= 0.5:
                    climate_regime = "El_Nino"
                elif oni_value <= -0.5:
                    climate_regime = "La_Nina"
                else:
                    climate_regime = "Normal"
        except Exception:
            pass   # silently fall back to Normal if ONI file has issues

    # ── Step 2: Query historical district/season medians ─────────────────────
    df = _get_master_df()
    baseline_source = "district+season"

    # (a) District + Season
    subset = df[
        (df["District"].str.lower() == district.lower()) &
        (df["Season"].str.lower()   == season.lower())
    ]

    # (b) State + Season fallback
    if subset.empty and state:
        subset = df[
            (df["State"].str.lower() == state.lower()) &
            (df["Season"].str.lower() == season.lower())
        ]
        baseline_source = "state+season"

    # (c) Season-only fallback
    if subset.empty:
        subset = df[df["Season"].str.lower() == season.lower()]
        baseline_source = "season_national"

    # (d) Grand fallback
    if subset.empty:
        subset = df
        baseline_source = "grand_median"

    baseline_rain = float(subset["Rainfall"].median())
    baseline_temp = float(subset["Temperature"].median())

    # ── Step 3: Apply ENSO adjustments ───────────────────────────────────────
    rain_adj = ENSO_RAINFALL_ADJUSTMENT.get(climate_regime, 0.0)
    temp_adj = ENSO_TEMP_ADJUSTMENT.get(climate_regime, 0.0)

    adjusted_rain = round(max(0.0, baseline_rain * (1.0 + rain_adj)), 1)
    adjusted_temp = round(baseline_temp + temp_adj, 2)

    return {
        "rainfall_mm"    : adjusted_rain,
        "temperature_c"  : adjusted_temp,
        "climate_regime" : climate_regime,
        "oni_value"      : round(oni_value, 3),
        "baseline_source": baseline_source,
        "baseline_rain"  : round(baseline_rain, 1),
        "baseline_temp"  : round(baseline_temp, 2),
        "enso_rain_adj_pct": rain_adj * 100,
        "enso_temp_adj_c"  : temp_adj,
    }


# =============================================================================
# 2. AGRONOMIC SUITABILITY MULTIPLIER
# =============================================================================

def calculate_agronomic_penalty(
    crop           : str,
    season         : str,
    temp_c         : float,
    rainfall_mm    : float,
    irrigation_type: str = "Rainfed",
) -> tuple:
    """
    Calculate an agronomic suitability multiplier [0.05, 1.0] based on
    biological season compatibility, water availability, and thermal tolerance.

    The multiplier is applied to the ML-predicted expected yield to produce
    a biologically-bounded, contextually-realistic yield estimate.

    Args:
        crop            : Crop name (must match CROP_AGRONOMY keys; fallback used otherwise).
        season          : Season string.
        temp_c          : Average seasonal temperature (°C).
        rainfall_mm     : Effective seasonal rainfall (mm).
        irrigation_type : One of 'Rainfed', 'Well_Borewell', 'Canal_River'.

    Returns:
        (multiplier: float, reasons: list[str])
        - multiplier : Final penalty factor in [0.05, 1.0].  1.0 = fully suitable.
        - reasons    : Plain-English list of agronomic diagnostics for the
                       explainability panel.
    """
    agro = CROP_AGRONOMY.get(crop, DEFAULT_AGRONOMY)
    reasons: list = []
    multiplier = 1.0

    # ── Irrigation credit ─────────────────────────────────────────────────────
    irrigation_credit = IRRIGATION_CREDITS.get(irrigation_type, 0)
    total_water = rainfall_mm + irrigation_credit

    if irrigation_credit > 0:
        reasons.append(
            f"Irrigation ({irrigation_type}) adds {irrigation_credit} mm equivalent → "
            f"total effective water: {total_water:.0f} mm."
        )

    # ── 1. Season Mismatch Penalty ────────────────────────────────────────────
    allowed = agro["allowed_seasons"]
    if season not in allowed:
        season_penalty = 0.15
        multiplier *= season_penalty
        reasons.append(
            f"SEASON MISMATCH ⚠: {crop} is biologically suited to "
            f"{'/'.join(allowed)}, but '{season}' was selected. "
            f"Severe yield penalty applied (×{season_penalty}). "
            f"Crop viability is very low outside its natural growing season."
        )
    else:
        reasons.append(
            f"Season '{season}' is compatible with {crop}'s biological calendar ✓."
        )

    # ── 2. Water Deficit Penalty ──────────────────────────────────────────────
    min_water = agro["min_water_mm"]
    drought_r = agro["drought_resilience"]

    if total_water < min_water:
        water_ratio = total_water / min_water
        water_penalty = max(water_ratio * drought_r, 0.05)
        multiplier *= water_penalty
        deficit = min_water - total_water
        reasons.append(
            f"WATER DEFICIT ⚠: {crop} requires ≥{min_water} mm; "
            f"effective water is {total_water:.0f} mm (deficit: {deficit:.0f} mm). "
            f"Drought resilience factor: {drought_r:.2f}. "
            f"Yield penalty: ×{water_penalty:.2f}."
        )
    else:
        surplus = total_water - min_water
        reasons.append(
            f"Water availability ({total_water:.0f} mm) meets {crop}'s requirement "
            f"of {min_water} mm (surplus: {surplus:.0f} mm) ✓."
        )

    # ── 3. Heat Stress Penalty ────────────────────────────────────────────────
    max_temp = agro["max_temp_c"]

    if temp_c > max_temp:
        overheat = temp_c - max_temp
        heat_penalty = max(1.0 - overheat * 0.15, 0.05)
        multiplier *= heat_penalty
        reasons.append(
            f"HEAT STRESS ⚠: Temperature {temp_c:.1f}°C exceeds {crop}'s "
            f"thermal limit of {max_temp}°C (excess: +{overheat:.1f}°C). "
            f"Cellular stress causes yield collapse (×{heat_penalty:.2f})."
        )
    else:
        headroom = max_temp - temp_c
        reasons.append(
            f"Temperature {temp_c:.1f}°C is within {crop}'s safe thermal range "
            f"(limit: {max_temp}°C, headroom: {headroom:.1f}°C) ✓."
        )

    # ── Clamp final multiplier ────────────────────────────────────────────────
    multiplier = float(np.clip(multiplier, 0.05, 1.0))

    # ── Summary reason ────────────────────────────────────────────────────────
    if multiplier >= 0.85:
        summary = f"Overall suitability for {crop}: EXCELLENT ({multiplier:.2f})"
    elif multiplier >= 0.60:
        summary = f"Overall suitability for {crop}: GOOD ({multiplier:.2f})"
    elif multiplier >= 0.35:
        summary = f"Overall suitability for {crop}: MODERATE ({multiplier:.2f}) — constraints exist"
    else:
        summary = f"Overall suitability for {crop}: POOR ({multiplier:.2f}) — major agronomic constraints"

    reasons.insert(0, summary)

    return multiplier, reasons


def get_yield_cap(crop: str) -> float:
    """Return the biological maximum yield cap (t/ha) for a crop."""
    return CROP_AGRONOMY.get(crop, DEFAULT_AGRONOMY)["max_yield_cap"]


# =============================================================================
# STANDALONE DEMO
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  AgriVest Weather Service -- Standalone Demo")
    print("=" * 70)

    # --- Climate detection
    print("\n[1] Auto-detecting climate for Pune, Kharif, Maharashtra:")
    climate = auto_detect_climate_and_weather("Pune", "Kharif", "Maharashtra")
    for k, v in climate.items():
        print(f"    {k:<25}: {v}")

    # --- Agronomic suitability for a few crops
    print("\n[2] Agronomic suitability assessment:")
    test_cases = [
        ("Cotton",   "Kharif", climate["temperature_c"], climate["rainfall_mm"], "Rainfed"),
        ("Wheat",    "Kharif", climate["temperature_c"], climate["rainfall_mm"], "Rainfed"),   # wrong season
        ("Soyabean", "Kharif", climate["temperature_c"], climate["rainfall_mm"], "Well_Borewell"),
        ("Rice",     "Summer", 36.0,                     150.0,                  "Canal_River"),
    ]
    for crop, season, temp, rain, irr in test_cases:
        mult, reasons = calculate_agronomic_penalty(crop, season, temp, rain, irr)
        print(f"\n  {crop} ({season}, {irr})  →  multiplier = {mult:.3f}")
        for r in reasons:
            print(f"    · {r}")
