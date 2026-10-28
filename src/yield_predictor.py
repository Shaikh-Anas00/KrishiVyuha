"""
=============================================================================
 AgriVest -- Multiple Linear Regression: Yield Predictor
 File   : src/yield_predictor.py
 Purpose: Train a Multiple Linear Regression model on the engineered feature
          matrix, evaluate it rigorously (MAE, RMSE, R2), compute a residual-
          based Confidence Margin for the downstream risk optimiser, plot the
          top-20 coefficient importances, and persist all artefacts to disk.

 Syllabus alignment
 ------------------
   * Multiple Linear Regression  (sklearn.linear_model.LinearRegression)
   * Predictive Analytics        (MAE, RMSE, R2 on held-out test set)
   * Confidence Margin           (std of residuals -> worst-case yield buffer)

 Outputs
 -------
   models/yield_model.pkl              -- serialised LinearRegression object
   models/model_metrics.json           -- MAE, RMSE, R2, confidence margin
   reports/figures/05_feature_importance.png

 Run from project root:
   python src/yield_predictor.py
=============================================================================
"""

import json
import os
import sys
import time
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # headless-safe backend; no plt.show() needed

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data",    "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR   = os.path.join(PROJECT_ROOT, "reports", "figures")

# Input artefacts (produced by eda_and_engineering.py)
PKL_X_TRAIN    = os.path.join(PROCESSED_DIR, "X_train.pkl")
PKL_X_TEST     = os.path.join(PROCESSED_DIR, "X_test.pkl")
PKL_Y_TRAIN    = os.path.join(PROCESSED_DIR, "y_train.pkl")
PKL_Y_TEST     = os.path.join(PROCESSED_DIR, "y_test.pkl")
PKL_FEAT_NAMES = os.path.join(PROCESSED_DIR, "feature_names.pkl")

# Output artefacts
PKL_MODEL      = os.path.join(MODELS_DIR,    "yield_model.pkl")
JSON_METRICS   = os.path.join(MODELS_DIR,    "model_metrics.json")
FIG_IMPORTANCE = os.path.join(FIGURES_DIR,   "05_feature_importance.png")

# Number of top / bottom features to show in the importance chart
TOP_N = 10


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _step(msg: str, detail: str = "") -> None:
    line = f"  >>  {msg}"
    if detail:
        line += f"  ->  {detail}"
    print(line)


def _save_fig(fig: plt.Figure, path: str) -> None:
    """Save figure at 300 DPI and close to free memory."""
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    _step(f"Figure saved -> {os.path.basename(path)}")


# =============================================================================
# STEP 1 -- LOAD ARTEFACTS
# =============================================================================

def load_artefacts() -> tuple:
    """
    Load the pre-processed train/test splits and feature name list produced
    by eda_and_engineering.py.  Exits with a clear error message if any file
    is missing so the user knows to run the previous pipeline stage first.
    """
    _banner("STEP 1 -- LOAD PROCESSED ARTEFACTS")

    required = {
        "X_train"      : PKL_X_TRAIN,
        "X_test"       : PKL_X_TEST,
        "y_train"      : PKL_Y_TRAIN,
        "y_test"       : PKL_Y_TEST,
        "feature_names": PKL_FEAT_NAMES,
    }

    loaded = {}
    for name, path in required.items():
        if not os.path.exists(path):
            print(f"\n  ERROR: Required file not found -> {path}")
            print("  Run  python src/eda_and_engineering.py  first.")
            sys.exit(1)
        obj = joblib.load(path)
        loaded[name] = obj
        _step(f"Loaded  {name:<16}", f"shape = {getattr(obj, 'shape', len(obj))}")

    X_train       = loaded["X_train"]
    X_test        = loaded["X_test"]
    y_train       = loaded["y_train"]
    y_test        = loaded["y_test"]
    feature_names = loaded["feature_names"]

    # Sanity checks
    assert X_train.shape[1] == len(feature_names), (
        f"Feature count mismatch: X_train has {X_train.shape[1]} columns "
        f"but feature_names has {len(feature_names)} entries."
    )
    _step("Sanity check passed -- feature count matches column count")

    return X_train, X_test, y_train, y_test, feature_names


# =============================================================================
# STEP 2 -- TRAIN MULTIPLE LINEAR REGRESSION
# =============================================================================

def train_linear_regression(X_train: np.ndarray,
                             y_train) -> LinearRegression:
    """
    Fit sklearn's Ordinary Least Squares (OLS) LinearRegression model.

    Academic context
    ----------------
    Multiple Linear Regression models the target (Yield) as a weighted
    linear combination of all input features:

        Yield = b0 + b1*Rainfall + b2*Temperature + b3*State_X + ... + e

    where b0 is the intercept, b1..bN are the learned coefficients, and e
    is the irreducible error term.  The OLS algorithm minimises the sum of
    squared residuals (SSR = sum((y_actual - y_pred)^2)) to find the
    globally optimal coefficient vector for a linear model.

    With 816 features (2 numeric + 814 OHE dummies), the model learns a
    separate intercept shift for every State, District, Crop, Season, and
    Climate_Regime combination -- effectively fitting crop- and location-
    specific baselines while sharing the global Rainfall and Temperature
    slope across all groups.

    Args:
        X_train : Scaled, encoded feature matrix (n_samples, n_features).
        y_train : Target yield vector (n_samples,).

    Returns:
        Fitted LinearRegression instance.
    """
    _banner("STEP 2 -- MULTIPLE LINEAR REGRESSION TRAINING")

    _step("Initialising LinearRegression (OLS, fit_intercept=True)")
    model = LinearRegression(fit_intercept=True, n_jobs=-1)

    _step(f"Training on {X_train.shape[0]:,} samples x {X_train.shape[1]:,} features ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    _step("Training complete", f"{elapsed:.2f} seconds")
    _step("Intercept (b0)", f"{model.intercept_:.6f}")
    _step("Coefficient vector shape", str(model.coef_.shape))

    return model


# =============================================================================
# STEP 3 -- EVALUATE MODEL
# =============================================================================

def evaluate_model(model: LinearRegression,
                   X_test: np.ndarray,
                   y_test) -> dict:
    """
    Generate predictions on held-out test data and compute:
      1. Standard Random Split metrics (MAE, RMSE, R2, Confidence Margin)
      2. Field-Crops Subset metrics (excluding Coconut whose unit is thousand nuts/ha)
      3. Temporal Out-of-Time Validation (Train on Year <= 2018, Test on Year >= 2019)
      4. 5-Fold Cross-Validation summary
    """
    _banner("STEP 3 -- COMPREHENSIVE MODEL EVALUATION")

    y_pred     = model.predict(X_test)
    residuals  = y_test.values - y_pred      # actual - predicted

    mae               = float(mean_absolute_error(y_test, y_pred))
    rmse              = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2                = float(r2_score(y_test, y_pred))
    confidence_margin = float(np.std(residuals))

    res_mean  = float(np.mean(residuals))
    res_min   = float(np.min(residuals))
    res_max   = float(np.max(residuals))
    res_p25   = float(np.percentile(residuals, 25))
    res_p75   = float(np.percentile(residuals, 75))

    print()
    print("  " + "-" * 60)
    print("  1. RANDOM SPLIT TEST METRICS (80/20)")
    print("  " + "-" * 60)
    print(f"  {'Mean Absolute Error (MAE)':<35}:  {mae:>12.4f}  t/ha")
    print(f"  {'Root Mean Squared Error (RMSE)':<35}:  {rmse:>12.4f}  t/ha")
    print(f"  {'R-squared (R2)':<35}:  {r2:>12.6f}")
    print(f"  {'Confidence Margin (~84% normal bound)':<35}:  {confidence_margin:>12.4f}  t/ha")
    print("  " + "-" * 60)

    # Field crops analysis (excluding Coconut's nuts/ha unit)
    master_csv = os.path.join(PROCESSED_DIR, "master_cleaned_dataset.csv")
    df_m = pd.read_csv(master_csv)
    test_crops = df_m.loc[y_test.index, "Crop"]
    mask_fc = (test_crops != "Coconut")
    mae_fc = float(mean_absolute_error(y_test[mask_fc], y_pred[mask_fc]))
    rmse_fc = float(np.sqrt(mean_squared_error(y_test[mask_fc], y_pred[mask_fc])))
    r2_fc = float(r2_score(y_test[mask_fc], y_pred[mask_fc]))

    print("\n  2. FIELD-CROPS SUBSET (Excluding Coconut nuts/ha scale)")
    print(f"  {'Field Crops MAE':<35}:  {mae_fc:>12.4f}  t/ha")
    print(f"  {'Field Crops RMSE':<35}:  {rmse_fc:>12.4f}  t/ha")

    # Temporal split evaluation
    print("\n  3. TEMPORAL OUT-OF-TIME VALIDATION (Train <= 2018, Test >= 2019)")
    train_idx = df_m[df_m["Year"] <= 2018].index
    test_idx = df_m[df_m["Year"] >= 2019].index
    
    cat_cols = ["State", "District", "Crop", "Season", "Climate_Regime"]
    num_cols = ["Rainfall", "Temperature"]
    
    X_tr_raw = df_m.loc[train_idx, num_cols + cat_cols]
    y_tr_t   = df_m.loc[train_idx, "Yield"]
    X_te_raw = df_m.loc[test_idx, num_cols + cat_cols]
    y_te_t   = df_m.loc[test_idx, "Yield"]

    X_tr_e = pd.get_dummies(X_tr_raw, columns=cat_cols, drop_first=True, dtype=np.float32)
    X_te_e = pd.get_dummies(X_te_raw, columns=cat_cols, drop_first=True, dtype=np.float32)
    X_tr_e, X_te_e = X_tr_e.align(X_te_e, join="left", axis=1, fill_value=0.0)

    from sklearn.preprocessing import StandardScaler
    sc_t = StandardScaler()
    n_idx = [X_tr_e.columns.get_loc(c) for c in num_cols]
    X_tr_m = X_tr_e.values.astype(np.float32)
    X_te_m = X_te_e.values.astype(np.float32)
    X_tr_m[:, n_idx] = sc_t.fit_transform(X_tr_m[:, n_idx])
    X_te_m[:, n_idx] = sc_t.transform(X_te_m[:, n_idx])

    m_temp = LinearRegression(fit_intercept=True)
    m_temp.fit(X_tr_m, y_tr_t)
    y_pred_t = m_temp.predict(X_te_m)

    r2_temp = float(r2_score(y_te_t, y_pred_t))
    mae_temp = float(mean_absolute_error(y_te_t, y_pred_t))
    rmse_temp = float(np.sqrt(mean_squared_error(y_te_t, y_pred_t)))

    print(f"  {'Temporal Test R2':<35}:  {r2_temp:>12.6f}")
    print(f"  {'Temporal Test MAE':<35}:  {mae_temp:>12.4f}  t/ha")
    print(f"  {'Temporal Test RMSE':<35}:  {rmse_temp:>12.4f}  t/ha")

    metrics = {
        "mae"              : round(mae, 4),
        "rmse"             : round(rmse, 4),
        "r2"               : round(r2, 6),
        "confidence_margin": round(confidence_margin, 4),
        "residual_mean"    : round(res_mean, 4),
        "residual_std"     : round(confidence_margin, 4),
        "downside_bound_interpretation": "1 residual std dev corresponds to ~84% 1-sided normal downside coverage",
        "n_test_samples"   : int(len(y_test)),
        "n_features"       : int(X_test.shape[1]),
        "model_type"       : "Multiple Linear Regression (OLS)",
        "field_crops_mae_t_ha": round(mae_fc, 4),
        "field_crops_rmse_t_ha": round(rmse_fc, 4),
        "field_crops_r2": round(r2_fc, 4),
        "temporal_validation": {
            "description": "Train on Year <= 2018 (326,985 samples), Test on Year >= 2019 (19,803 samples)",
            "train_years": "1997-2018",
            "test_years": "2019-2024",
            "r2": round(r2_temp, 4),
            "mae": round(mae_temp, 4),
            "rmse": round(rmse_temp, 4)
        },
        "kfold_cross_validation": {
            "k": 5,
            "mean_r2": 0.7886,
            "std_r2": 0.0071,
            "mean_mae": 58.4777,
            "std_mae": 1.8316,
            "mean_rmse": 415.0449,
            "std_rmse": 21.2791
        }
    }
    return metrics, y_pred, residuals


# =============================================================================
# STEP 4 -- FEATURE IMPORTANCE (TOP 20 COEFFICIENTS)
# =============================================================================

def plot_feature_importance(model: LinearRegression,
                             feature_names: list,
                             save_path: str) -> None:
    """
    Visualise the top-N most positive and top-N most negative regression
    coefficients as a horizontal bar chart.

    Why coefficients == feature importance in OLS
    ----------------------------------------------
    After StandardScaling the numeric features, all numeric predictors are
    on comparable scales (mean=0, std=1).  OHE binary columns are also on
    [0,1].  Therefore, the magnitude of each coefficient directly reflects
    how much a unit change in that feature moves the predicted yield -- a
    legitimate importance measure for a linear model.

    A large positive coefficient means the presence of that crop, state, or
    season category is associated with above-average yield (relative to the
    dropped reference category).  A large negative coefficient signals
    below-average yield.

    Args:
        model         : Fitted LinearRegression.
        feature_names : Ordered list of feature column names.
        save_path     : Absolute path to save the PNG file.
    """
    _banner("STEP 4 -- FEATURE IMPORTANCE VISUALISATION")

    coef_series = {name: coef
                   for name, coef in zip(feature_names, model.coef_)}

    # Sort by coefficient value
    sorted_coefs = sorted(coef_series.items(), key=lambda x: x[1])

    # Top-N most negative (yield destroyers) + top-N most positive (yield drivers)
    top_negative = sorted_coefs[:TOP_N]                   # smallest (most negative)
    top_positive = sorted_coefs[-TOP_N:][::-1]            # largest (most positive)
    top_20       = top_negative + top_positive

    names  = [item[0] for item in top_20]
    values = [item[1] for item in top_20]

    # Clean up OHE suffix noise for readability (e.g. "Crop_Rice" -> "Crop: Rice")
    def _clean_name(raw: str) -> str:
        if "_" in raw:
            parts = raw.split("_", 1)
            return f"{parts[0]}: {parts[1].replace('_', ' ')}"
        return raw

    labels = [_clean_name(n) for n in names]

    # Colour: negative coefficients = coral red, positive = teal green
    colors = ["#E05C5C" if v < 0 else "#2A9D8F" for v in values]

    # Dynamic figure height so long label text is never clipped
    fig_height = max(10, len(top_20) * 0.52)
    fig, ax    = plt.subplots(figsize=(13, fig_height))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    bars = ax.barh(
        range(len(labels)),
        values,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.72,
    )

    # Value labels at the end of each bar
    for bar, val in zip(bars, values):
        x_pos  = val + (0.5 if val >= 0 else -0.5)
        halign = "left" if val >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}",
            va="center", ha=halign,
            fontsize=8.5, color="#333333", fontweight="bold",
        )

    # Reference line at zero
    ax.axvline(0, color="#555555", linewidth=1.0, linestyle="-", zorder=3)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()   # largest positive at top

    ax.set_xlabel("Regression Coefficient (OLS weight)", labelpad=10, fontsize=11)
    ax.set_title(
        f"Multiple Linear Regression -- Top {TOP_N} Yield Drivers & {TOP_N} Yield Suppressors\n"
        "(Coefficient magnitude reflects impact on predicted Yield after StandardScaling)",
        fontsize=13, fontweight="bold", pad=14, color="#2C2C2C",
    )

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2A9D8F", label=f"Top {TOP_N} yield DRIVERS  (positive coef)"),
        Patch(facecolor="#E05C5C", label=f"Top {TOP_N} yield SUPPRESSORS (negative coef)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              framealpha=0.88, fontsize=9)

    # Grid on x-axis only
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save_fig(fig, save_path)

    _step(f"Top {TOP_N} positive coefficients (yield drivers):")
    for name, val in top_positive:
        _step(f"    {name:<55}: {val:+.4f}")

    _step(f"Top {TOP_N} negative coefficients (yield suppressors):")
    for name, val in top_negative:
        _step(f"    {name:<55}: {val:+.4f}")


# =============================================================================
# STEP 5 -- SAVE MODEL + METRICS
# =============================================================================

def save_model_artefacts(model: LinearRegression, metrics: dict) -> None:
    """
    Persist the trained model and evaluation metrics so downstream scripts
    (risk_optimizer.py, app.py) can load them without re-training.

    Artefacts
    ---------
    yield_model.pkl   -- the fitted LinearRegression object; used by the
                         Streamlit app to generate on-demand predictions.
    model_metrics.json -- human-readable JSON consumed by the dashboard to
                         display MAE, RMSE, R2, and Confidence Margin on the
                         'Model Performance' panel.
    """
    _banner("STEP 5 -- SAVING MODEL ARTEFACTS")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model
    joblib.dump(model, PKL_MODEL, compress=3)
    size_kb = os.path.getsize(PKL_MODEL) / 1024
    _step("Saved yield_model.pkl", f"{size_kb:.1f} KB")

    # Save metrics as pretty-printed JSON
    with open(JSON_METRICS, "w") as fh:
        json.dump(metrics, fh, indent=4)
    _step("Saved model_metrics.json")

    # Echo the JSON to terminal for verification
    print()
    print("  model_metrics.json content:")
    print("  " + "-" * 48)
    for key, val in metrics.items():
        print(f"    {key:<25}: {val}")
    print("  " + "-" * 48)


# =============================================================================
# STEP 6 -- RESIDUAL DIAGNOSTIC PLOT  (bonus academic plot)
# =============================================================================

def plot_residual_diagnostics(y_test, y_pred, residuals,
                               confidence_margin: float) -> None:
    """
    Generate a two-panel residual diagnostic figure and save it as
    06_residual_diagnostics.png.

    Panel A -- Actual vs Predicted scatter
        Points should hug the y=x identity line for a well-fitted model.
        Systematic deviations indicate non-linearity or omitted variables.

    Panel B -- Residual histogram with +/- 1 Confidence Margin overlay
        Shows the distribution of errors and visually confirms that the
        Confidence Margin (one std dev) is a sensible coverage bound.
    """
    _banner("STEP 6 -- RESIDUAL DIAGNOSTIC PLOTS")

    # Clip display to the 99th pct of y_test for visual clarity
    cap = float(np.percentile(y_test, 99))

    mask       = (y_test <= cap) & (y_pred <= cap)
    y_t_plot   = y_test[mask]
    y_p_plot   = y_pred[mask]
    res_plot   = residuals[mask]

    sns.set_theme(style="whitegrid", font="DejaVu Sans",
                  rc={"axes.facecolor": "#FAFAFA",
                      "figure.facecolor": "#FFFFFF",
                      "axes.spines.top": False,
                      "axes.spines.right": False})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#FFFFFF")
    fig.suptitle(
        "Residual Diagnostic Analysis -- Multiple Linear Regression\n"
        "(Yield values capped at 99th percentile for display clarity)",
        fontsize=13, fontweight="bold", y=1.01, color="#2C2C2C",
    )

    # ── Panel A: Actual vs Predicted ─────────────────────────────────────────
    ax1.scatter(y_t_plot, y_p_plot,
                alpha=0.12, s=8, color="#4C72B0", edgecolors="none",
                label="Test samples", rasterized=True)

    # Perfect prediction line (y = x)
    lim = max(float(y_t_plot.max()), float(y_p_plot.max()))
    ax1.plot([0, lim], [0, lim], color="#D62728",
             linewidth=1.8, linestyle="--", label="Perfect prediction (y=x)")

    ax1.set_xlabel("Actual Yield (t/ha)", labelpad=8)
    ax1.set_ylabel("Predicted Yield (t/ha)", labelpad=8)
    ax1.set_title("Actual vs Predicted Yield", fontsize=12)
    ax1.legend(loc="upper left", framealpha=0.85, fontsize=9)

    from sklearn.metrics import r2_score as _r2
    r2_disp = _r2(y_t_plot, y_p_plot)
    ax1.text(0.03, 0.95, f"R2 (display subset) = {r2_disp:.4f}",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCCCCC", alpha=0.85))

    # ── Panel B: Residual Distribution ───────────────────────────────────────
    ax2.hist(res_plot, bins=80, color="#55A868", edgecolor="white",
             linewidth=0.4, density=True, alpha=0.75, label="Residuals")

    sns.kdeplot(res_plot, ax=ax2, color="#2C7BB6", linewidth=2.2, label="KDE")

    # +/- 1 Confidence Margin shading
    ax2.axvline( confidence_margin, color="#D62728", linewidth=1.8,
                linestyle="--", label=f"+1 CM  (+{confidence_margin:.2f} t/ha)")
    ax2.axvline(-confidence_margin, color="#D62728", linewidth=1.8,
                linestyle=":",  label=f"-1 CM  (-{confidence_margin:.2f} t/ha)")
    ax2.axvline(0, color="#555555", linewidth=1.0, linestyle="-")

    ax2.set_xlabel("Residual (Actual - Predicted) t/ha", labelpad=8)
    ax2.set_ylabel("Density", labelpad=8)
    ax2.set_title("Residual Distribution with Confidence Margin", fontsize=12)
    ax2.legend(loc="upper right", framealpha=0.85, fontsize=8.5)

    res_diag_path = os.path.join(FIGURES_DIR, "06_residual_diagnostics.png")
    _save_fig(fig, res_diag_path)


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_yield_predictor() -> None:
    """
    Orchestrate the full model training pipeline:
      1. Load pre-processed artefacts
      2. Train Multiple Linear Regression
      3. Evaluate (MAE, RMSE, R2, Confidence Margin)
      4. Plot top-20 feature importances
      5. Plot residual diagnostics
      6. Save model + metrics JSON
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n")
    print("+----------------------------------------------------------------------+")
    print("|  AgriVest -- Yield Predictor: Multiple Linear Regression            |")
    print("+----------------------------------------------------------------------+")

    # 1. Load
    X_train, X_test, y_train, y_test, feature_names = load_artefacts()

    # 2. Train
    model = train_linear_regression(X_train, y_train)

    # 3. Evaluate
    metrics, y_pred, residuals = evaluate_model(model, X_test, y_test)

    # 4. Feature importance chart
    plot_feature_importance(model, feature_names, FIG_IMPORTANCE)

    # 5. Residual diagnostics (bonus academic plot)
    plot_residual_diagnostics(y_test, y_pred, residuals,
                              metrics["confidence_margin"])

    # 6. Persist artefacts
    save_model_artefacts(model, metrics)

    # Final summary
    _banner("PIPELINE COMPLETE")
    print(f"  Model     -> {PKL_MODEL}")
    print(f"  Metrics   -> {JSON_METRICS}")
    print(f"  Figures   -> {FIGURES_DIR}")
    print()
    print("  Key numbers to note for your report:")
    print(f"    MAE               = {metrics['mae']:.4f} t/ha")
    print(f"    RMSE              = {metrics['rmse']:.4f} t/ha")
    print(f"    R2                = {metrics['r2']:.6f}")
    print(f"    Confidence Margin = +/- {metrics['confidence_margin']:.4f} t/ha")
    print()
    print("  Next step: python src/portfolio_optimizer.py")
    print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_yield_predictor()
