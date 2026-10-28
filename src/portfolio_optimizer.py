"""
=============================================================================
 AgriVest -- Risk-Optimized Crop Portfolio Optimizer
 File   : src/portfolio_optimizer.py
 Purpose: Combines ML yield predictions with Markowitz-style Mean-Variance
          optimization and PMFBY parametric insurance to recommend the
          risk-optimal allocation of land across multiple crops.

 Mathematical framework
 ----------------------
  Maximize  : sum_i( w_i * E[yield_i] * price_i * area ) - total_premium
  Subject to:
    C1 (Land)    : sum(w) == 1.0
    C2 (Bounds)  : 0 <= w_i <= 1.0  for all i
    C3 (Survival): worst_case_revenue >= survival_budget

  where w_i is the fraction of total land allocated to crop i,
        E[yield_i] is the ML-predicted expected yield (t/ha),
        worst_case_i = max(0, E[yield_i] - confidence_margin)
        PMFBY payout kicks in when worst_case_i < 0.20 * E[yield_i]

 Outputs (when run as __main__)
 -------
   Terminal  : Optimal portfolio weights + financial summary
   reports/figures/07_portfolio_allocation.png

 Run:
   python src/portfolio_optimizer.py
=============================================================================
"""

import json
import os
import sys
import warnings
from typing import Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# Import the Agro-Climatic Advisory Engine
# (handles climate auto-detection and biological suitability scoring)
try:
    from src.weather_service import (
        auto_detect_climate_and_weather,
        calculate_agronomic_penalty,
        get_yield_cap,
    )
except ImportError:
    # Allow running from project root without src/ as a package
    _ws_dir = os.path.dirname(os.path.abspath(__file__))
    if _ws_dir not in sys.path:
        sys.path.insert(0, _ws_dir)
    from weather_service import (
        auto_detect_climate_and_weather,
        calculate_agronomic_penalty,
        get_yield_cap,
    )

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data",    "processed")
RAW_DIR       = os.path.join(PROJECT_ROOT, "data",    "raw")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR   = os.path.join(PROJECT_ROOT, "reports", "figures")

PKL_MODEL      = os.path.join(MODELS_DIR,    "yield_model.pkl")
PKL_SCALER     = os.path.join(PROCESSED_DIR, "scaler.pkl")
PKL_FEAT_NAMES = os.path.join(PROCESSED_DIR, "feature_names.pkl")
JSON_METRICS       = os.path.join(MODELS_DIR,    "model_metrics.json")
JSON_RISK_PROFILES = os.path.join(MODELS_DIR,    "crop_risk_profiles.json")
CSV_PMFBY          = os.path.join(RAW_DIR,       "pmfby_rules.csv")

# =============================================================================
# MARKET PRICES (INR per Tonne) -- reference: MSP + market premium, FY2024
# =============================================================================
# These represent approximate prevailing rates used for expected revenue
# calculation.  In the live dashboard these can be overridden by the user.

MARKET_PRICES = {
    "Rice"      : 25_000,
    "Wheat"     : 22_000,
    "Maize"     : 20_000,
    "Jowar"     : 30_000,
    "Bajra"     : 23_500,
    "Gram"      : 50_000,
    "Groundnut" : 55_000,
    "Cotton"    : 60_000,
    "Soyabean"  : 45_000,
    "Sugarcane" : 3_400,    # priced per tonne of cane, not refined sugar
    "Mustard"   : 55_500,
    "Sunflower" : 48_000,
    "Arhar/Tur" : 70_000,
    "Urad"      : 68_000,
    "Moong(Green Gram)": 84_000,
    "Potato"    : 15_000,
    "Onion"     : 18_000,
    "Tomato"    : 22_000,
    "Banana"    : 12_000,
    "Coconut"   : 15,       # priced per nut (yield is recorded in nuts/ha, ~8,000 nuts/ha)
}

# Fallback price used for crops not in the dictionary above
DEFAULT_MARKET_PRICE = 25_000   # INR/tonne

# PMFBY-inspired protection model: in this academic implementation, an indemnity payout
# triggers when severe disaster/drought causes worst-case yield to fall below 20% of expected yield.
PMFBY_TRIGGER_THRESHOLD = 0.20   # worst_case < 20% of expected -> PMFBY kicks in


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _banner(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _step(msg: str, detail: str = "") -> None:
    line = f"  >>  {msg}"
    if detail:
        line += f"  ->  {detail}"
    print(line)


# =============================================================================
# MODULE-LEVEL ARTEFACT LOADING  (loaded once, reused for every prediction)
# =============================================================================

def _load_artefacts() -> tuple:
    """
    Load all inference artefacts on first call.  Returns a 3-tuple:
        (model, scaler, feature_names)

    Exits with a clear error if any required file is missing.
    """
    required = {
        "yield_model"  : PKL_MODEL,
        "scaler"       : PKL_SCALER,
        "feature_names": PKL_FEAT_NAMES,
    }
    for label, path in required.items():
        if not os.path.exists(path):
            print(f"\n  ERROR: Required artefact not found -> {path}")
            print("  Please run yield_predictor.py and eda_and_engineering.py first.")
            sys.exit(1)

    model         = joblib.load(PKL_MODEL)
    scaler        = joblib.load(PKL_SCALER)
    feature_names = joblib.load(PKL_FEAT_NAMES)
    return model, scaler, feature_names


# Lazy-load cache (populated on first call to predict_crop_yield)
_ARTEFACT_CACHE: dict = {}


def _get_artefacts() -> tuple:
    """Return cached artefacts; load them if this is the first call."""
    if not _ARTEFACT_CACHE:
        model, scaler, feature_names = _load_artefacts()
        _ARTEFACT_CACHE["model"]         = model
        _ARTEFACT_CACHE["scaler"]        = scaler
        _ARTEFACT_CACHE["feature_names"] = feature_names
    return (
        _ARTEFACT_CACHE["model"],
        _ARTEFACT_CACHE["scaler"],
        _ARTEFACT_CACHE["feature_names"],
    )


# =============================================================================
# INFERENCE HELPER -- predict_crop_yield
# =============================================================================

def predict_crop_yield(
    crop           : str,
    state          : str,
    district       : str,
    season         : str,
    rainfall       : float,
    temperature    : float,
    climate_regime : str,
) -> float:
    """
    Predict the expected yield (t/ha) for a single crop under given conditions
    using the trained Multiple Linear Regression model.

    OHE reconstruction
    ------------------
    The model was trained on a 816-column feature vector built by
    pd.get_dummies(..., drop_first=True).  The reference (dropped) categories
    are:
        State          -> 'Andaman And Nicobar Island'
        District       -> 'Adilabad'
        Crop           -> 'Arecanut'
        Season         -> 'Autumn'
        Climate_Regime -> 'El_Nino'

    For an input value that equals the reference category, all OHE columns
    for that group remain 0.  For any other value, exactly one column in the
    group is set to 1.  This function reconstructs that vector exactly.

    Args:
        crop           : Crop name matching the training vocabulary (title-case).
        state          : State name (title-case, e.g. 'Maharashtra').
        district       : District name (title-case, e.g. 'Pune').
        season         : Season string ('Kharif', 'Rabi', 'Summer', etc.).
        rainfall       : Total seasonal rainfall in mm.
        temperature    : Average temperature in degrees C.
        climate_regime : ENSO label -- 'El_Nino', 'La_Nina', or 'Normal'.

    Returns:
        Predicted yield in tonnes per hectare (floored at 0).
    """
    model, scaler, feature_names = _get_artefacts()

    # ── Build a zeroed feature vector ────────────────────────────────────────
    vec = np.zeros(len(feature_names), dtype=np.float32)

    # Index lookup for speed
    col_index = {name: idx for idx, name in enumerate(feature_names)}

    # ── Numeric features (positions 0 and 1) -- apply StandardScaler ─────────
    # The scaler was fit on [Rainfall, Temperature] in that order.
    rainfall_scaled, temp_scaled = scaler.transform(
        [[rainfall, temperature]]
    )[0]
    vec[col_index["Rainfall"]]    = rainfall_scaled
    vec[col_index["Temperature"]] = temp_scaled

    # ── Categorical OHE columns ───────────────────────────────────────────────
    # Helper: set the column to 1 only if the column exists in feature_names
    # (it won't exist for the reference/dropped category -- that's correct OHE).
    def _set_ohe(prefix: str, value: str) -> None:
        col = f"{prefix}_{value}"
        if col in col_index:
            vec[col_index[col]] = 1.0
        # else: value is the reference category -> all zeros for this group (correct)

    _set_ohe("State",          state)
    _set_ohe("District",       district)
    _set_ohe("Crop",           crop)
    _set_ohe("Season",         season)
    _set_ohe("Climate_Regime", climate_regime)

    # ── Predict ───────────────────────────────────────────────────────────────
    prediction = float(model.predict(vec.reshape(1, -1))[0])

    # If the Multiple Linear Regression model extrapolates unrealistically
    # (e.g. negative or near-zero yield due to OHE dummy coefficient interactions on small crops),
    # fall back to the historical regime/crop baseline for that crop.
    if prediction <= 0.10:
        risk_stat = get_crop_risk_stat(crop, climate_regime)
        prediction = float(risk_stat.get("yield_mean", 1.0))

    # Cap at biological maximum yield -- the OLS model can extrapolate
    # unrealistically (e.g., Cotton at 184 t/ha due to OHE coefficient
    # interactions). We clip to the agronomic ceiling defined in
    # weather_service.CROP_AGRONOMY.
    cap = get_yield_cap(crop)
    return min(max(0.05, prediction), cap)


# =============================================================================
# FINANCIAL MATH HELPERS
# =============================================================================

_RISK_PROFILES_CACHE: Optional[dict] = None


def _get_risk_profiles() -> dict:
    """Load and cache the crop risk profiles and covariance matrices."""
    global _RISK_PROFILES_CACHE
    if _RISK_PROFILES_CACHE is None:
        if not os.path.exists(JSON_RISK_PROFILES):
            return {"crop_stats": {}, "covariance_matrices": {}}
        try:
            with open(JSON_RISK_PROFILES, "r") as fh:
                _RISK_PROFILES_CACHE = json.load(fh)
        except Exception:
            _RISK_PROFILES_CACHE = {"crop_stats": {}, "covariance_matrices": {}}
    return _RISK_PROFILES_CACHE


def get_crop_risk_stat(crop: str, climate_regime: str) -> dict:
    """
    Return regime-specific yield CV and revenue standard deviation for a given crop.
    Falls back gracefully if the exact crop/regime combination is not recorded.
    """
    rp = _get_risk_profiles()
    crop_stats = rp.get("crop_stats", {})
    c_data = crop_stats.get(crop) or crop_stats.get(crop.title(), {})
    reg_data = c_data.get(climate_regime) or c_data.get("Overall", {})
    if not reg_data:
        return {"yield_cv": 0.50, "revenue_std": 20000.0}
    return reg_data


def get_confidence_margin() -> float:
    """Load the global Confidence Margin from model_metrics.json (legacy fallback)."""
    if not os.path.exists(JSON_METRICS):
        return 50.0
    with open(JSON_METRICS) as fh:
        metrics = json.load(fh)
    return float(metrics.get("confidence_margin", 50.0))


def get_pmfby_terms(crop: str, season: str) -> dict:
    """
    Look up the PMFBY insurance terms for a given crop and season.

    Returns a dict with:
        sum_insured_per_ha   : INR -- maximum payout per hectare
        premium_rate_pct     : float -- farmer's share of premium (e.g. 2.0 = 2%)
        threshold_yield_pct  : float -- yield shortfall threshold that triggers payout
    If the crop is not in the PMFBY table, returns a conservative fallback.
    """
    pmfby = pd.read_csv(CSV_PMFBY)

    # Match by crop name (case-insensitive); season match is secondary
    crop_lower = crop.lower()
    pmfby["Crop_lower"] = pmfby["Crop"].str.lower()

    # Try exact crop + season match first
    match = pmfby[
        (pmfby["Crop_lower"] == crop_lower) &
        (pmfby["Season"].str.lower() == season.lower())
    ]

    # Fallback: any season for this crop
    if match.empty:
        match = pmfby[pmfby["Crop_lower"] == crop_lower]

    if not match.empty:
        row = match.iloc[0]
        return {
            "sum_insured_per_ha" : float(row["Sum_Insured_per_Ha"]),
            "premium_rate_pct"   : float(row["Premium_Rate_Farmer_pct"]),
            "threshold_yield_pct": float(row["Threshold_Yield_pct"]),
        }

    # No PMFBY data available -- use safe defaults
    return {
        "sum_insured_per_ha" : 25_000.0,
        "premium_rate_pct"   :     2.0,
        "threshold_yield_pct":    70.0,
    }


def get_market_price(crop: str) -> float:
    """Return the market price (INR/tonne) for a crop, with fallback."""
    return MARKET_PRICES.get(crop, DEFAULT_MARKET_PRICE)


def compute_crop_financials(
    crop             : str,
    state            : str,
    district         : str,
    season           : str,
    rainfall         : float,
    temperature      : float,
    climate_regime   : str,
    area_ha          : float,
    confidence_margin: Optional[float] = None,
    irrigation_type  : str = "Rainfed",
) -> dict:
    """
    Compute all financial quantities for a single crop holding given a specific
    land area allocation, incorporating:
      • ML yield prediction (capped at biological max)
      • Agronomic suitability penalty (season / water / heat)
      • Regime-conditional crop CV for realistic worst-case scenario
      • PMFBY parametric insurance

    Returns a dict with:
        expected_yield      (t/ha)  -- agronomic-adjusted ML prediction
        raw_ml_yield        (t/ha)  -- raw (pre-penalty) model prediction
        agronomic_multiplier        -- penalty factor applied [0.05, 1.0]
        agronomic_reasons   (list)  -- plain-English diagnostics
        worst_case_yield    (t/ha)  -- crop-calibrated downside yield
        market_price        (INR/t)
        expected_revenue    (INR)   -- expected_yield * price * area_ha
        worst_case_revenue  (INR)   -- worst_case_yield * price * area_ha
        pmfby_payout        (INR)   -- insurance compensation on worst-case
        premium_cost        (INR)   -- farmer's PMFBY premium
        net_worst_case      (INR)   -- worst_case_revenue + pmfby_payout - premium
    """
    # ── Raw ML prediction (with biological ceiling cap) ───────────────────────
    raw_ml_yield = predict_crop_yield(
        crop, state, district, season, rainfall, temperature, climate_regime
    )

    # ── Agronomic suitability penalty ─────────────────────────────────────────
    agro_mult, agro_reasons = calculate_agronomic_penalty(
        crop, season, temperature, rainfall, irrigation_type
    )

    # Apply penalty to get the biologically-bounded expected yield
    expected_yield = raw_ml_yield * agro_mult

    # ── Crop-Specific & Regime-Conditional Downside (Residual-Based Downside Estimate)
    # We compute a crop-calibrated residual-based downside estimate utilizing the crop's
    # regime-specific historical yield Coefficient of Variation (CV).
    # This corresponds approximately to a 1-sigma (~84% one-sided normal equivalent) buffer,
    # rather than an uncalibrated flat constant.
    risk_stat = get_crop_risk_stat(crop, climate_regime)
    base_cv   = float(risk_stat.get("yield_cv", 0.50))

    # Agronomic stress (agro_mult < 1) amplifies downside shortfall:
    # A drought/heat-stressed crop suffers a wider downside collapse.
    shortfall_factor = min(0.92, max(0.15, base_cv / np.sqrt(max(0.05, agro_mult))))
    worst_case_yield = max(0.05 * expected_yield, expected_yield * (1.0 - shortfall_factor))

    market_price      = get_market_price(crop)
    pmfby             = get_pmfby_terms(crop, season)

    sum_insured_per_ha = pmfby["sum_insured_per_ha"]
    premium_rate_pct   = pmfby["premium_rate_pct"]

    # PMFBY premium cost for this crop holding
    premium_cost = (premium_rate_pct / 100.0) * sum_insured_per_ha * area_ha

    # Revenue calculations
    expected_revenue   = expected_yield   * market_price * area_ha
    worst_case_revenue = worst_case_yield * market_price * area_ha

    # PMFBY parametric trigger:
    # If worst-case yield falls below PMFBY_TRIGGER_THRESHOLD * expected yield,
    # the insurance policy pays out the full sum insured for that area.
    pmfby_payout = 0.0
    if expected_yield > 0:
        shortfall_ratio = worst_case_yield / expected_yield
        if shortfall_ratio < PMFBY_TRIGGER_THRESHOLD:
            pmfby_payout = sum_insured_per_ha * area_ha

    net_worst_case = worst_case_revenue + pmfby_payout - premium_cost

    # Agmarknet price-risk profile (calibrated from Agmarknet modal price volatility benchmarks)
    # utilizing the Bohrnstedt-Goldberger (1969) variance expansion for Var(P * Y)
    rp = _get_risk_profiles()
    pr_prof = rp.get("agmarknet_price_risk", {}).get(crop, {})
    price_cv = float(pr_prof.get("price_cv", 0.15))
    price_yield_corr = float(pr_prof.get("rho_py", -0.15))
    price_policy = str(pr_prof.get("policy", "Market Mandi"))
    price_risk_mult = float(pr_prof.get("revenue_vol_multiplier", 1.0))

    return {
        "expected_yield"       : round(expected_yield,    4),
        "raw_ml_yield"         : round(raw_ml_yield,      4),
        "agronomic_multiplier" : round(agro_mult,         4),
        "agronomic_reasons"    : agro_reasons,
        "worst_case_yield"     : round(worst_case_yield,  4),
        "market_price"         : market_price,
        "expected_revenue"     : round(expected_revenue,  2),
        "worst_case_revenue"   : round(worst_case_revenue,2),
        "pmfby_payout"         : round(pmfby_payout,      2),
        "premium_cost"         : round(premium_cost,      2),
        "net_worst_case"       : round(net_worst_case,    2),
        "price_cv"             : price_cv,
        "price_yield_corr"     : price_yield_corr,
        "price_policy"         : price_policy,
        "price_risk_mult"      : price_risk_mult,
    }


# =============================================================================
# SCIPY OPTIMIZER ENGINE -- optimize_portfolio
# =============================================================================

def optimize_portfolio(
    state           : str,
    district        : str,
    season          : str,
    crops           : list,
    total_area_ha   : float,
    survival_budget : float,
    rainfall        : Optional[float] = None,
    temperature     : Optional[float] = None,
    climate_regime  : Optional[str]   = None,
    irrigation_type : str             = "Rainfed",
    risk_aversion   : float           = 1.0,
    verbose         : bool            = True,
) -> dict:
    """
    Find the Markowitz-optimal land allocation across the given crops using
    scipy.optimize.minimize with the SLSQP method.

    Markowitz Mean-Variance framework
    ---------------------------------
    Maximizes: Expected Profit - lambda * Portfolio Volatility (INR)
    Subject to:
      C1: sum(w) == 1.0
      C2: 0 <= w_i <= 1.0
      C3: worst_case_portfolio_revenue(w) >= survival_budget

    Regime-conditional covariance
    -----------------------------
    The crop covariance matrix Sigma is loaded from models/crop_risk_profiles.json
    conditioned on the active ENSO regime (El Niño, La Niña, Normal).
    Under agronomic stress (water/thermal deficit), crop variance is amplified
    proportionally (Sigma_ij / sqrt(alpha_i * alpha_j)), preventing high-margin
    crops from overpowering severe climate risks.
    """
    # ── Auto-detect climate if not supplied ───────────────────────────────────
    climate_info: dict = {}
    if rainfall is None or temperature is None or climate_regime is None:
        try:
            detected = auto_detect_climate_and_weather(district, season, state)
            climate_info = detected
            if rainfall       is None: rainfall       = detected["rainfall_mm"]
            if temperature    is None: temperature    = detected["temperature_c"]
            if climate_regime is None: climate_regime = detected["climate_regime"]
            if verbose:
                _step("Auto-detected climate",
                      f"{climate_regime} | {rainfall} mm | {temperature}°C "
                      f"(ONI={detected['oni_value']}, source={detected['baseline_source']})")
        except Exception as exc:
            # Safe defaults if detection fails
            if rainfall       is None: rainfall       = 600.0
            if temperature    is None: temperature    = 27.0
            if climate_regime is None: climate_regime = "Normal"
            if verbose:
                _step(f"Climate auto-detect failed ({exc}); using safe defaults")
    else:
        climate_info = {
            "rainfall_mm"    : rainfall,
            "temperature_c"  : temperature,
            "climate_regime" : climate_regime,
            "baseline_source": "manual_override",
        }

    if verbose:
        _banner("PORTFOLIO OPTIMIZER -- Markowitz SLSQP (Regime-Conditional)")
        _step("State", state)
        _step("District", district)
        _step("Season", season)
        _step("Irrigation", irrigation_type)
        _step("Crops", str(crops))
        _step("Total area", f"{total_area_ha} ha")
        _step("Survival budget", f"INR {survival_budget:,.0f}")
        _step("Risk aversion (lambda)", f"{risk_aversion:.2f}")
        _step("Rainfall / Temperature / Regime",
              f"{rainfall} mm | {temperature} C | {climate_regime}")

    confidence_margin = get_confidence_margin()
    n_crops = len(crops)

    # ── Pre-compute per-hectare financials & PMFBY terms (cached) ────────────
    if verbose:
        _step("Pre-computing per-hectare yield predictions for each crop...")

    per_ha: dict = {}
    pmfby_cache: dict = {}
    agronomic_diagnostics: dict = {}

    for crop in crops:
        fin = compute_crop_financials(
            crop, state, district, season,
            rainfall, temperature, climate_regime,
            area_ha=1.0,                  # per-hectare baseline
            confidence_margin=confidence_margin,
            irrigation_type=irrigation_type,
        )
        per_ha[crop]                = fin
        pmfby_cache[crop]           = get_pmfby_terms(crop, season)
        agronomic_diagnostics[crop] = fin.get("agronomic_reasons", [])
        if verbose:
            _step(f"  {crop:<20}",
                  f"E[yield]={fin['expected_yield']:.2f} t/ha "
                  f"(raw={fin['raw_ml_yield']:.2f}, agro×{fin['agronomic_multiplier']:.2f}) | "
                  f"WC={fin['worst_case_yield']:.2f} t/ha | "
                  f"Price=INR {fin['market_price']:,}/t")

    # ── Regime-Conditional Covariance Matrix & Stress Scaling ───────────────
    rp = _get_risk_profiles()
    cov_regime = (
        rp.get("covariance_matrices", {}).get(climate_regime)
        or rp.get("covariance_matrices", {}).get("Overall", {})
    )

    Sigma = np.zeros((n_crops, n_crops))
    for i, c1 in enumerate(crops):
        for j, c2 in enumerate(crops):
            val = cov_regime.get(c1, {}).get(c2)
            if val is None:
                val = cov_regime.get(c2, {}).get(c1)
            if val is None:
                s1 = per_ha[c1]["expected_revenue"] * 0.4
                s2 = per_ha[c2]["expected_revenue"] * 0.4
                val = (1.0 if i == j else 0.25) * s1 * s2
            Sigma[i, j] = float(val)

    # ── Agmarknet Price-Risk Layer (Bohrnstedt-Goldberger Multipliers) ─────
    price_risk_dict = rp.get("agmarknet_price_risk", {})
    kappas = np.ones(n_crops)
    for i, c in enumerate(crops):
        prof = price_risk_dict.get(c, {})
        kappas[i] = float(prof.get("revenue_vol_multiplier", 1.0))

    Sigma_stochastic = np.zeros((n_crops, n_crops))
    for i in range(n_crops):
        for j in range(n_crops):
            Sigma_stochastic[i, j] = Sigma[i, j] * kappas[i] * kappas[j]

    # Inflate covariance matrix with agronomic stress multipliers:
    # Stressed crops experience higher volatility: Sigma_ij / sqrt(alpha_i * alpha_j)
    Sigma_scaled = np.zeros((n_crops, n_crops))
    for i, c1 in enumerate(crops):
        a1 = per_ha[c1]["agronomic_multiplier"]
        for j, c2 in enumerate(crops):
            a2 = per_ha[c2]["agronomic_multiplier"]
            Sigma_scaled[i, j] = Sigma_stochastic[i, j] / np.sqrt(max(0.05, a1 * a2))

    # Regularize diagonal for numerical stability (ensures positive definiteness)
    diag_mean = max(100.0, float(np.mean(np.diag(Sigma_scaled))))
    Sigma_scaled += np.eye(n_crops) * (diag_mean * 0.01)

    # Reference scales for quadratic risk penalty
    ref_var = float(np.mean(np.diag(Sigma_scaled))) * (total_area_ha ** 2)
    ref_ret = max(1000.0, float(np.mean([per_ha[c]["expected_revenue"] for c in crops])) * total_area_ha)

    # ── Objective function ────────────────────────────────────────────────────
    def _objective(
        w        : np.ndarray,
        _ph      = per_ha,
        _pf      = pmfby_cache,
        _crops   = crops,
        _area    = total_area_ha,
        _sig     = Sigma_scaled,
        _ref_var = ref_var,
        _ref_ret = ref_ret,
        _lam     = risk_aversion,
    ) -> float:
        """
        Markowitz Mean-Variance Utility:
            Maximize:  Expected Profit - (lambda / (2 * ref_var)) * Portfolio Variance * ref_ret
            Minimize: -Utility
        """
        tot_exp_rev = 0.0
        tot_premium = 0.0
        for i, crop in enumerate(_crops):
            area_i = w[i] * _area
            fin    = _ph[crop]
            pf     = _pf[crop]
            tot_exp_rev += fin["expected_yield"] * fin["market_price"] * area_i
            tot_premium += (pf["premium_rate_pct"] / 100.0 * pf["sum_insured_per_ha"] * area_i)

        exp_profit = tot_exp_rev - tot_premium

        # Markowitz quadratic variance across total farm area (w is fractions summing to 1.0)
        port_var = float(np.dot(w, np.dot(_sig, w))) * (_area ** 2)

        # Dimensionless quadratic risk penalty scaled to monetary return units
        risk_penalty = 0.5 * _lam * (port_var / max(1e-6, _ref_var)) * _ref_ret
        utility      = exp_profit - risk_penalty
        return -(utility / max(1000.0, _ref_ret))

    # ── Worst-case revenue (used for constraint C3) ───────────────────────────
    def _worst_case_portfolio_revenue(
        w        : np.ndarray,
        _ph      = per_ha,
        _pf      = pmfby_cache,
        _crops   = crops,
        _area    = total_area_ha,
        _trigger = PMFBY_TRIGGER_THRESHOLD,
    ) -> float:
        """
        Compute the guaranteed floor revenue across all crops:
          WC_net_i = WC_yield_i * price_i * area_i
                   + PMFBY_payout_i (if triggered)
                   - premium_i
        Uses only pre-cached per_ha + pmfby_cache -- no file I/O.
        """
        total_wc = 0.0
        for i, crop in enumerate(_crops):
            area_i = w[i] * _area
            fin    = _ph[crop]
            pf     = _pf[crop]

            wc_rev_i   = fin["worst_case_yield"] * fin["market_price"] * area_i
            premium_i  = pf["premium_rate_pct"] / 100.0 * pf["sum_insured_per_ha"] * area_i

            # PMFBY payout fires when shortfall > (1 - PMFBY_TRIGGER_THRESHOLD)
            pmfby_payout_i = 0.0
            if fin["expected_yield"] > 0:
                shortfall_ratio = fin["worst_case_yield"] / fin["expected_yield"]
                if shortfall_ratio < _trigger:
                    pmfby_payout_i = pf["sum_insured_per_ha"] * area_i
            elif fin["expected_yield"] == 0:
                # Zero expected yield -> PMFBY payout cannot be justified
                pmfby_payout_i = 0.0

            total_wc += wc_rev_i + pmfby_payout_i - premium_i

        return total_wc

    # ── Bounds: each weight in [0, 1] ─────────────────────────────────────────────────
    bounds = [(0.0, 1.0)] * n_crops

    # ── Pre-feasibility check ───────────────────────────────────────────────────
    w_eq = np.array([1.0 / n_crops] * n_crops)
    wc_at_equal = _worst_case_portfolio_revenue(w_eq)
    survival_feasible = wc_at_equal >= survival_budget

    if not survival_feasible:
        # Check each pure single-crop allocation
        for i in range(n_crops):
            w_single = np.zeros(n_crops); w_single[i] = 1.0
            if _worst_case_portfolio_revenue(w_single) >= survival_budget:
                survival_feasible = True
                break

    if verbose and not survival_feasible:
        _step("NOTE: No single-crop allocation meets the survival budget.")
        _step("      PMFBY payouts cover worst-case losses; optimizer will")
        _step("      maximise income subject to best achievable worst-case.")

    # ── Initial guess: unbiased uniform allocation ───────────────────────────
    w0 = np.array([1.0 / n_crops] * n_crops)

    # ── Constraints dictionary for SLSQP ─────────────────────────────────────
    norm_budget = max(1000.0, survival_budget)
    constraints = [
        # C1: weights must sum to exactly 1.0 (equality)
        {
            "type": "eq",
            "fun" : lambda w: np.sum(w) - 1.0,
        },
        # C3: worst-case portfolio revenue >= survival_budget (normalized inequality)
        {
            "type": "ineq",
            "fun" : lambda w: (_worst_case_portfolio_revenue(w) - survival_budget) / norm_budget,
        },
    ]

    # ── Run optimizer ────────────────────────────────────────────────────────────────
    if verbose:
        _step("Running scipy SLSQP optimizer ...")


    result = minimize(
        _objective,
        w0,
        method     = "SLSQP",
        bounds     = bounds,
        constraints= constraints,
        options    = {
            "ftol"   : 1e-9,
            "maxiter": 1000,
            "disp"   : False,
        },
    )

    # ── Handle convergence failure gracefully ─────────────────────────────────
    if not result.success:
        # Attempt a relaxed retry without the survival-budget constraint
        # to at least return a best-effort allocation.
        if verbose:
            _step("WARNING: Optimizer did not converge with survival constraint.")
            _step("         Retrying without survival constraint (best-effort mode)...")

        result_relaxed = minimize(
            _objective,
            w0,
            method     = "SLSQP",
            bounds     = bounds,
            constraints= [constraints[0]],    # only C1 (sum = 1)
            options    = {"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )

        if result_relaxed.success:
            result  = result_relaxed
            message = (
                "PARTIAL SUCCESS: Optimal allocation found, but the survival "
                "budget constraint could not be satisfied. Consider lowering "
                "the survival budget or expanding the crop list."
            )
            success = False
        else:
            # Both attempts failed -- return equal allocation with actual financials
            eq_w = np.array([1.0 / n_crops] * n_crops)
            eq_details = {}
            eq_exp = eq_wc = eq_prem = 0.0
            for i, crop in enumerate(crops):
                area_i = eq_w[i] * total_area_ha
                fin = compute_crop_financials(
                    crop, state, district, season,
                    rainfall, temperature, climate_regime,
                    area_ha=area_i, confidence_margin=confidence_margin,
                    irrigation_type=irrigation_type,
                )
                eq_details[crop] = fin
                eq_exp  += fin["expected_revenue"]
                eq_wc   += fin["net_worst_case"]
                eq_prem += fin["premium_cost"]
            return {
                "success"              : False,
                "budget_met"           : False,
                "message"              : (
                    f"Optimizer failed: {result.message}. "
                    "Try adjusting the survival budget or crop list."
                ),
                "weights"                  : {c: 1.0/n_crops for c in crops},
                "area_ha"                  : {c: total_area_ha/n_crops for c in crops},
                "expected_income"          : round(eq_exp,  2),
                "worst_case_income"        : round(eq_wc,   2),
                "total_premium"            : round(eq_prem, 2),
                "portfolio_volatility_inr" : 0.0,
                "sharpe_ratio"             : 0.0,
                "risk_aversion"            : risk_aversion,
                "crop_details"             : eq_details,
                "optimizer_status"         : result.message,
                "climate_info"             : climate_info,
                "agronomic_diagnostics"    : agronomic_diagnostics,
                "agmarknet_price_risk"     : {c: price_risk_dict.get(c, {}) for c in crops},
            }
    else:
        success = True
        message = "Optimization converged successfully."

    # ── Extract results ───────────────────────────────────────────────────────
    opt_weights = np.clip(result.x, 0.0, 1.0)
    opt_weights /= opt_weights.sum()            # re-normalise to sum = 1.0

    # Compute final financials at optimal weights
    total_expected_rev  = 0.0
    total_worst_case    = 0.0
    total_premium       = 0.0
    crop_details_final  = {}

    for i, crop in enumerate(crops):
        area_i = opt_weights[i] * total_area_ha
        fin    = compute_crop_financials(
            crop, state, district, season,
            rainfall, temperature, climate_regime,
            area_ha           = area_i,
            confidence_margin = confidence_margin,
            irrigation_type   = irrigation_type,
        )
        crop_details_final[crop]  = fin
        total_expected_rev       += fin["expected_revenue"]
        total_worst_case         += fin["net_worst_case"]
        total_premium            += fin["premium_cost"]

    budget_met = total_worst_case >= survival_budget

    port_var_final = float(np.dot(opt_weights, np.dot(Sigma_scaled, opt_weights))) * (total_area_ha ** 2)
    port_vol_final = float(np.sqrt(max(0.0, port_var_final)))
    sharpe = round((total_expected_rev - total_premium - survival_budget) / max(1.0, port_vol_final), 2)

    output = {
        "success"                  : success,
        "budget_met"               : budget_met,
        "message"                  : message,
        "weights"                  : {c: round(float(opt_weights[i]), 6)
                                      for i, c in enumerate(crops)},
        "area_ha"                  : {c: round(float(opt_weights[i]) * total_area_ha, 4)
                                      for i, c in enumerate(crops)},
        "expected_income"          : round(total_expected_rev,  2),
        "worst_case_income"        : round(total_worst_case,    2),
        "total_premium"            : round(total_premium,       2),
        "portfolio_volatility_inr" : round(port_vol_final,      2),
        "sharpe_ratio"             : sharpe,
        "risk_aversion"            : risk_aversion,
        "crop_details"             : crop_details_final,
        "optimizer_status"         : result.message,
        "climate_info"             : climate_info,
        "agronomic_diagnostics"    : agronomic_diagnostics,
        "agmarknet_price_risk"     : {c: price_risk_dict.get(c, {}) for c in crops},
    }

    return output

# =============================================================================
# RESULT PRINTER
# =============================================================================

def print_portfolio_results(result: dict, crops: list, total_area_ha: float) -> None:
    """Print a formatted portfolio summary to the terminal."""
    _banner("PORTFOLIO OPTIMIZATION RESULTS")

    status_icon = "SUCCESS" if result["success"] else "WARNING"
    print(f"\n  Status  : [{status_icon}] {result['message']}")
    print(f"  Solver  : {result['optimizer_status']}")

    print("\n  " + "-" * 64)
    print("  CROP ALLOCATION")
    print("  " + "-" * 64)
    print(f"  {'Crop':<22} {'Weight':>8} {'Area(ha)':>10} {'E[Yield]':>10} "
          f"{'E[Revenue]':>14} {'WC Revenue':>14} {'Premium':>12}")
    print("  " + "-" * 64)

    for crop in crops:
        w   = result["weights"].get(crop, 0.0)
        ha  = result["area_ha"].get(crop, 0.0)
        fin = result["crop_details"].get(crop, {})
        print(
            f"  {crop:<22} {w*100:>7.2f}% {ha:>10.2f} "
            f"{fin.get('expected_yield', 0.0):>10.3f} "
            f"INR {fin.get('expected_revenue', 0.0):>12,.0f} "
            f"INR {fin.get('net_worst_case', 0.0):>12,.0f} "
            f"INR {fin.get('premium_cost', 0.0):>10,.0f}"
        )

    print("  " + "-" * 64)
    print(f"  {'TOTAL':<22} {'100.00%':>8} {total_area_ha:>10.2f} "
          f"{'':>10} "
          f"INR {result['expected_income']:>12,.0f} "
          f"INR {result['worst_case_income']:>12,.0f} "
          f"INR {result['total_premium']:>10,.0f}")
    print("  " + "-" * 64)
    print(f"\n  Expected Portfolio Income : INR {result['expected_income']:>15,.2f}")
    print(f"  Worst-Case Income (insured): INR {result['worst_case_income']:>15,.2f}")
    print(f"  Total PMFBY Premium        : INR {result['total_premium']:>15,.2f}")
    print(f"  Net Expected Profit        : INR {result['expected_income'] - result['total_premium']:>15,.2f}")
    print()


# =============================================================================
# PORTFOLIO PIE CHART
# =============================================================================

def plot_portfolio_allocation(result: dict, crops: list,
                               total_area_ha: float,
                               save_path: str) -> None:
    """
    Save a two-panel portfolio visualisation:
      Left  : Pie chart of optimal land allocation.
      Right : Grouped bar chart comparing expected vs worst-case revenue per crop.
    """
    import matplotlib.patches as mpatches

    weights  = [result["weights"].get(c, 0.0) for c in crops]
    exp_rev  = [result["crop_details"][c]["expected_revenue"]  / 1_000
                for c in crops]       # in thousands INR
    wc_rev   = [result["crop_details"][c]["net_worst_case"]    / 1_000
                for c in crops]

    # Filter out zero-weight crops from the pie
    nonzero  = [(c, w) for c, w in zip(crops, weights) if w > 0.001]
    pie_labels  = [f"{c}\n({w*100:.1f}%)" for c, w in nonzero]
    pie_sizes   = [w for _, w in nonzero]

    palette = plt.cm.Set2(np.linspace(0, 1, len(crops)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "Risk-Optimized Crop Portfolio -- Markowitz + PMFBY Insurance\n"
        f"Total Land: {total_area_ha} ha | "
        f"Expected Income: INR {result['expected_income']:,.0f} | "
        f"Worst-Case: INR {result['worst_case_income']:,.0f}",
        fontsize=12, fontweight="bold", color="#2C2C2C", y=1.01,
    )

    # ── Pie chart ─────────────────────────────────────────────────────────────
    wedge_props = dict(width=0.55, edgecolor="white", linewidth=2)
    ax1.pie(
        pie_sizes,
        labels     = pie_labels,
        colors     = palette[:len(pie_sizes)],
        startangle = 140,
        wedgeprops = wedge_props,
        textprops  = {"fontsize": 9},
        pctdistance= 0.75,
    )
    ax1.set_title("Optimal Land Allocation (% of total area)", fontsize=11, pad=12)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    x     = np.arange(len(crops))
    width = 0.38

    bars_exp = ax2.bar(x - width/2, exp_rev, width, label="Expected Revenue",
                       color="#2A9D8F", edgecolor="white", linewidth=0.6)
    bars_wc  = ax2.bar(x + width/2, wc_rev,  width, label="Worst-Case (insured)",
                       color="#E76F51", edgecolor="white", linewidth=0.6, alpha=0.85)

    # Value labels on bars
    for bar in bars_exp:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h + max(exp_rev)*0.01,
                     f"{h:.0f}K", ha="center", va="bottom",
                     fontsize=7.5, color="#2C2C2C")

    for bar in bars_wc:
        h = bar.get_height()
        y_pos = h + max(exp_rev)*0.01 if h >= 0 else h - max(exp_rev)*0.03
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f"{h:.0f}K", ha="center",
                 va="bottom" if h >= 0 else "top",
                 fontsize=7.5, color="#2C2C2C")

    ax2.axhline(0, color="#888888", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(crops, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Revenue (INR thousands)", labelpad=8)
    ax2.set_title("Expected vs Worst-Case Revenue per Crop", fontsize=11, pad=12)
    ax2.legend(loc="upper right", framealpha=0.85, fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#EEEEEE")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    _step(f"Portfolio chart saved -> {os.path.basename(save_path)}")


# =============================================================================
# ENTRY POINT -- DEMO / SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    print("\n")
    print("+------------------------------------------------------------------+")
    print("|  AgriVest -- Portfolio Optimizer  (Demo Run)                    |")
    print("+------------------------------------------------------------------+")

    # ── Test scenario ─────────────────────────────────────────────────────────
    # A farmer in Pune, Maharashtra with 5 hectares wants to plant
    # Rice, Cotton, and Soyabean in the Kharif season.
    # He needs at least INR 1,00,000 in the worst case to cover living costs.

    TEST_STATE    = "Maharashtra"
    TEST_DISTRICT = "Pune"
    TEST_SEASON   = "Kharif"
    TEST_CROPS    = ["Rice", "Cotton", "Soyabean"]
    TEST_AREA     = 5.0          # hectares
    TEST_BUDGET   = 100_000      # INR survival budget (worst-case floor)
    TEST_RAINFALL = 700.0        # mm (typical Kharif Pune)
    TEST_TEMP     = 28.5         # degC
    TEST_REGIME   = "Normal"

    # ── Run optimization ──────────────────────────────────────────────────────
    result = optimize_portfolio(
        state          = TEST_STATE,
        district       = TEST_DISTRICT,
        season         = TEST_SEASON,
        crops          = TEST_CROPS,
        total_area_ha  = TEST_AREA,
        survival_budget= TEST_BUDGET,
        rainfall       = TEST_RAINFALL,
        temperature    = TEST_TEMP,
        climate_regime = TEST_REGIME,
        verbose        = True,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print_portfolio_results(result, TEST_CROPS, TEST_AREA)

    # ── Save portfolio chart ──────────────────────────────────────────────────
    FIG_PATH = os.path.join(FIGURES_DIR, "07_portfolio_allocation.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_portfolio_allocation(result, TEST_CROPS, TEST_AREA, FIG_PATH)

    print("  Portfolio optimization complete.")
    print(f"  Chart saved -> {FIG_PATH}")
    print("  Next step: python src/app.py\n")
