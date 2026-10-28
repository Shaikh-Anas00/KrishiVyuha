"""
=============================================================================
 AgriVest -- EDA & Feature Engineering Pipeline
 File   : src/eda_and_engineering.py
 Purpose: Exploratory Data Analysis (visualisations) followed by a complete
          scikit-learn-compatible feature-engineering pipeline that produces
          ready-to-model train/test arrays persisted to disk.

 Outputs
 -------
   reports/figures/01_correlation_heatmap.png
   reports/figures/02_distributions.png
   reports/figures/03_scatter_trends.png
   reports/figures/04_climate_impact_boxplot.png
   data/processed/X_train.pkl
   data/processed/X_test.pkl
   data/processed/y_train.pkl
   data/processed/y_test.pkl
   data/processed/feature_names.pkl   <- column names after encoding
   data/processed/scaler.pkl          <- fitted StandardScaler (reuse at inference)

 Run from project root:
   python src/eda_and_engineering.py
=============================================================================
"""

import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Use the non-interactive Agg backend so plt.savefig() works without a
# display server (e.g. on a headless CI machine or Windows without a GUI loop)
# ---------------------------------------------------------------------------
matplotlib.use("Agg")

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR    = os.path.join(PROJECT_ROOT, "reports", "figures")

MASTER_CSV     = os.path.join(PROCESSED_DIR, "master_cleaned_dataset.csv")

# Output artefact paths
FIG_HEATMAP    = os.path.join(FIGURES_DIR, "01_correlation_heatmap.png")
FIG_DIST       = os.path.join(FIGURES_DIR, "02_distributions.png")
FIG_SCATTER    = os.path.join(FIGURES_DIR, "03_scatter_trends.png")
FIG_BOX        = os.path.join(FIGURES_DIR, "04_climate_impact_boxplot.png")

PKL_X_TRAIN    = os.path.join(PROCESSED_DIR, "X_train.pkl")
PKL_X_TEST     = os.path.join(PROCESSED_DIR, "X_test.pkl")
PKL_Y_TRAIN    = os.path.join(PROCESSED_DIR, "y_train.pkl")
PKL_Y_TEST     = os.path.join(PROCESSED_DIR, "y_test.pkl")
PKL_FEAT_NAMES = os.path.join(PROCESSED_DIR, "feature_names.pkl")
PKL_SCALER     = os.path.join(PROCESSED_DIR, "scaler.pkl")

# Reproducibility seed
RANDOM_STATE   = 42
TEST_SIZE      = 0.20          # 80/20 train-test split

# Yield is extremely right-skewed (max ~44k, median ~1).
# We cap the visualisation window at the 99th percentile for readability.
YIELD_VIZ_CAP_PCT = 99

# ---------------------------------------------------------------------------
# Global seaborn / matplotlib theme  (applied once, inherited by all plots)
# ---------------------------------------------------------------------------
PALETTE      = "viridis"
ACCENT_COLOR = "#4C72B0"       # muted-blue default line colour
GRID_COLOR   = "#E5E5E5"

sns.set_theme(
    style="whitegrid",
    palette=PALETTE,
    font="DejaVu Sans",
    rc={
        "axes.facecolor"  : "#FAFAFA",
        "figure.facecolor": "#FFFFFF",
        "axes.edgecolor"  : "#CCCCCC",
        "axes.grid"       : True,
        "grid.color"      : GRID_COLOR,
        "grid.linewidth"  : 0.7,
        "axes.spines.top" : False,
        "axes.spines.right": False,
        "xtick.labelsize" : 10,
        "ytick.labelsize" : 10,
        "axes.titlesize"  : 13,
        "axes.labelsize"  : 11,
        "legend.fontsize" : 9,
        "figure.dpi"      : 150,
    },
)


# =============================================================================
# UTILITY HELPERS
# =============================================================================

def _banner(title: str) -> None:
    """Print a section banner to stdout for pipeline progress tracking."""
    sep = "=" * 72
    print(f"\n{sep}\n  {title}\n{sep}")


def _step(msg: str, detail: str = "") -> None:
    """Print a single pipeline step with an optional value."""
    line = f"  >>  {msg}"
    if detail:
        line += f"  ->  {detail}"
    print(line)


def _save_fig(fig: plt.Figure, path: str, tight: bool = True) -> None:
    """
    Save a matplotlib figure to *path* at 300 DPI and close it immediately
    to release memory.  Using savefig instead of show() makes the script
    headless-safe and produces reproducible, archive-quality images.
    """
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    _step(f"Saved  ->  {os.path.basename(path)}")


# =============================================================================
# STEP 1 -- LOAD DATA
# =============================================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Load the master cleaned dataset and perform a quick sanity check."""
    _banner("STEP 1 -- LOAD MASTER DATASET")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except FileNotFoundError:
        print(f"\n  ERROR: File not found -- {csv_path}")
        print("  Run data_preprocessing.py first to generate it.")
        sys.exit(1)

    _step("Loaded", csv_path)
    _step("Shape", str(df.shape))
    _step("Columns", str(list(df.columns)))

    # Quick sanity -- confirm zero NaN in modelling columns
    core = ["Rainfall", "Temperature", "Yield", "ONI_Annual_Mean",
            "State", "District", "Crop", "Season", "Climate_Regime"]
    nan_counts = df[core].isna().sum()
    if nan_counts.any():
        print("\n  WARNING -- NaN values detected in core columns:")
        print(nan_counts[nan_counts > 0].to_string())
    else:
        _step("NaN check", "All core columns are clean")

    return df


# =============================================================================
# STEP 2 -- EXPLORATORY DATA ANALYSIS  (4 publication-quality plots)
# =============================================================================

# ---------------------------------------------------------------------------
# Plot 1: Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str) -> None:
    """
    Pearson correlation heatmap for the four key numeric features.

    Why: Correlation analysis reveals linear relationships between predictors
    and the target (Yield) and flags multicollinearity risks before modelling.
    """
    _banner("PLOT 1 -- CORRELATION HEATMAP")

    numeric_cols = ["Rainfall", "Temperature", "Yield", "ONI_Annual_Mean"]

    # Compute Pearson correlation matrix
    corr_matrix = df[numeric_cols].corr(method="pearson")
    _step("Correlation matrix computed")

    # Build a triangle mask -- show only the lower triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#FFFFFF")

    cmap = sns.diverging_palette(220, 20, as_cmap=True)   # blue-white-red

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.8,
        linecolor="#CCCCCC",
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        annot_kws={"size": 12, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(
        "Pearson Correlation Matrix\n"
        "Rainfall | Temperature | Yield | ONI Annual Mean",
        pad=16, fontsize=14, fontweight="bold", color="#2C2C2C",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Annotation footnote
    fig.text(
        0.01, -0.02,
        "Note: Values range from -1 (perfect negative) to +1 (perfect positive) correlation.",
        fontsize=8, color="#777777",
    )

    _save_fig(fig, save_path)

    # Print interpretation hints
    _step("Top correlations with Yield:")
    yield_corr = corr_matrix["Yield"].drop("Yield").sort_values(key=abs, ascending=False)
    for feat, val in yield_corr.items():
        _step(f"    {feat:<22}: r = {val:+.4f}")


# ---------------------------------------------------------------------------
# Plot 2: Distribution Analysis (Histograms + KDE)
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, save_path: str) -> None:
    """
    Subplot grid of histograms with kernel-density overlays for the three
    primary numeric features (Yield, Rainfall, Temperature).

    Why: Understanding the shape of each feature's distribution is essential
    before applying a linear model.  Skewed targets may need log-transformation.
    Yield is capped at its 99th percentile for this visualisation because a
    handful of extreme values (sugarcane, banana) would otherwise compress
    the rest of the distribution into a single spike.
    """
    _banner("PLOT 2 -- FEATURE DISTRIBUTIONS")

    yield_cap = df["Yield"].quantile(YIELD_VIZ_CAP_PCT / 100)
    _step(f"Yield visualisation cap (p{YIELD_VIZ_CAP_PCT})", f"{yield_cap:.2f} t/ha")

    # Create a capped view -- we do NOT modify the original dataframe
    df_viz = df.copy()
    df_viz["Yield_Capped"] = df_viz["Yield"].clip(upper=yield_cap)

    features = {
        "Yield_Capped": {
            "label": f"Yield (t/ha, capped at p{YIELD_VIZ_CAP_PCT} = {yield_cap:.0f})",
            "color": "#4C72B0",
            "bins" : 60,
        },
        "Rainfall": {
            "label": "Rainfall (mm)",
            "color": "#55A868",
            "bins" : 60,
        },
        "Temperature": {
            "label": "Temperature (degC)",
            "color": "#C44E52",
            "bins" : 50,
        },
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Feature Distribution Analysis -- Histograms with Kernel Density Estimate (KDE)",
        fontsize=14, fontweight="bold", color="#2C2C2C", y=1.02,
    )

    for ax, (col, cfg) in zip(axes, features.items()):
        data = df_viz[col].dropna()

        # Histogram
        ax.hist(
            data,
            bins=cfg["bins"],
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.75,
            density=True,
            label="Histogram (density)",
            zorder=2,
        )

        # KDE overlay using seaborn
        sns.kdeplot(data, ax=ax, color=cfg["color"], linewidth=2.2,
                    label="KDE", zorder=3)

        # Vertical lines for mean and median
        mean_val   = data.mean()
        median_val = data.median()
        ax.axvline(mean_val,   color="#FF7F0E", linestyle="--",
                   linewidth=1.5, label=f"Mean   {mean_val:.2f}")
        ax.axvline(median_val, color="#9467BD", linestyle=":",
                   linewidth=1.5, label=f"Median {median_val:.2f}")

        ax.set_xlabel(cfg["label"], labelpad=8)
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", framealpha=0.85)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

        # Skewness annotation
        skew = data.skew()
        ax.set_title(f"{col.replace('_Capped', '')}\n(Skewness = {skew:.2f})",
                     fontsize=11, pad=8)

    _save_fig(fig, save_path)
    _step("Distributions plotted for Yield (capped), Rainfall, Temperature")


# ---------------------------------------------------------------------------
# Plot 3: Scatter Plots -- Rainfall vs Yield & Temperature vs Yield
# ---------------------------------------------------------------------------

def plot_scatter_trends(df: pd.DataFrame, save_path: str) -> None:
    """
    Two scatter plots coloured by Climate_Regime with a linear-regression
    trend line overlaid, revealing how climate modulates the relationship
    between weather variables and crop yield.

    A random 8,000-row sample is used for performance -- full 346k points
    would produce an unreadable solid blob and a very slow render.
    """
    _banner("PLOT 3 -- SCATTER TREND ANALYSIS")

    yield_cap   = df["Yield"].quantile(YIELD_VIZ_CAP_PCT / 100)
    sample_size = min(8000, len(df))

    df_sample = (
        df[df["Yield"] <= yield_cap]
        .sample(n=sample_size, random_state=RANDOM_STATE)
        .copy()
    )
    _step(f"Sample size for scatter ({YIELD_VIZ_CAP_PCT}th-pct yield cap)", f"{len(df_sample):,} rows")

    # Colour palette: one colour per climate regime
    regime_palette = {
        "Normal" : "#4C72B0",
        "El_Nino": "#DD8452",
        "La_Nina": "#55A868",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    fig.suptitle(
        "Scatter Trend Analysis -- Weather Variables vs Crop Yield\n"
        "(Coloured by Climate Regime | Trend line = OLS regression)",
        fontsize=14, fontweight="bold", color="#2C2C2C", y=1.02,
    )

    scatter_pairs = [
        ("Rainfall",    "Rainfall (mm)",       axes[0]),
        ("Temperature", "Temperature (degC)",   axes[1]),
    ]

    for x_col, x_label, ax in scatter_pairs:

        # Scatter per regime for a colour-coded legend
        for regime, colour in regime_palette.items():
            subset = df_sample[df_sample["Climate_Regime"] == regime]
            ax.scatter(
                subset[x_col],
                subset["Yield"],
                c=colour,
                alpha=0.35,
                s=14,
                label=regime,
                edgecolors="none",
                zorder=2,
            )

        # Overall OLS trend line via numpy polyfit
        x_vals = df_sample[x_col].values
        y_vals = df_sample["Yield"].values
        valid  = ~(np.isnan(x_vals) | np.isnan(y_vals))
        m, b   = np.polyfit(x_vals[valid], y_vals[valid], 1)
        x_line = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 300)
        ax.plot(x_line, m * x_line + b, color="#D62728",
                linewidth=2.2, linestyle="-", label=f"OLS trend (slope={m:.3f})",
                zorder=5)

        ax.set_xlabel(x_label, labelpad=8)
        ax.set_ylabel("Yield (t/ha)")
        ax.set_title(f"{x_col} vs Yield", fontsize=12)
        ax.legend(loc="upper right", framealpha=0.85, markerscale=2)

        # Pearson r annotation inside the plot
        r_val = np.corrcoef(x_vals[valid], y_vals[valid])[0, 1]
        ax.text(
            0.03, 0.96,
            f"Pearson r = {r_val:.4f}",
            transform=ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.85),
        )

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 4: Box Plot -- Yield across Climate Regime
# ---------------------------------------------------------------------------

def plot_climate_boxplot(df: pd.DataFrame, save_path: str) -> None:
    """
    Box-and-whisker plot comparing Yield distribution across the three ENSO
    climate regimes (Normal, El_Nino, La_Nina).

    Why: Box plots expose the median, inter-quartile range (IQR), and outlier
    density for each group simultaneously -- a key visual for understanding
    whether climate regime is a meaningful predictor of yield.

    Yield is log1p-transformed on the y-axis because the raw scale is
    dominated by a few high-value crops (e.g. sugarcane at ~43,000 t/ha)
    which would hide all variation in the lower 95% of crops.
    """
    _banner("PLOT 4 -- CLIMATE REGIME IMPACT ON YIELD (BOX PLOT)")

    # Log1p-transform yield so the plot is readable across crops
    df_plot          = df.copy()
    df_plot["Yield"] = np.log1p(df_plot["Yield"])

    regime_order   = ["La_Nina", "Normal", "El_Nino"]
    regime_palette = {
        "Normal" : "#4C72B0",
        "El_Nino": "#DD8452",
        "La_Nina": "#55A868",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#FFFFFF")

    sns.boxplot(
        data=df_plot,
        x="Climate_Regime",
        y="Yield",
        order=regime_order,
        palette=regime_palette,
        width=0.55,
        fliersize=2.5,
        flierprops=dict(marker="o", markerfacecolor="#888888",
                        markersize=2.5, alpha=0.4, linewidth=0),
        linewidth=1.2,
        ax=ax,
    )

    # Overlay individual data points (violin-style jitter strip)
    # Use a small subsample so the plot does not become a solid block
    strip_sample = df_plot.sample(n=min(3000, len(df_plot)),
                                  random_state=RANDOM_STATE)
    sns.stripplot(
        data=strip_sample,
        x="Climate_Regime",
        y="Yield",
        order=regime_order,
        color="#2C2C2C",
        alpha=0.08,
        size=2.5,
        jitter=True,
        ax=ax,
        zorder=1,
    )

    # Median value annotation above each box
    medians = (
        df_plot.groupby("Climate_Regime")["Yield"]
        .median()
        .reindex(regime_order)
    )
    for tick_idx, (regime, med) in enumerate(medians.items()):
        ax.text(
            tick_idx,
            med + 0.05,
            f"Median\n{np.expm1(med):.2f} t/ha",
            ha="center", va="bottom",
            fontsize=8.5, color="#222222", fontweight="bold",
        )

    ax.set_title(
        "Crop Yield Distribution by ENSO Climate Regime\n"
        "(Y-axis: log1p-transformed Yield for readability)",
        fontsize=14, fontweight="bold", pad=14, color="#2C2C2C",
    )
    ax.set_xlabel("Climate Regime (ENSO Classification)", labelpad=10)
    ax.set_ylabel("log1p(Yield) -- t/ha", labelpad=10)
    ax.set_xticklabels(["La Nina", "Normal", "El Nino"], fontsize=11)

    # Sample size annotation per group
    counts = df_plot["Climate_Regime"].value_counts().reindex(regime_order)
    for tick_idx, (regime, cnt) in enumerate(counts.items()):
        ax.text(
            tick_idx,
            ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            f"n = {cnt:,}",
            ha="center", va="top",
            fontsize=8, color="#555555",
        )

    fig.text(
        0.01, -0.03,
        "Note: Raw yield values are log1p-transformed to compress the extreme "
        "right tail caused by high-value crops (sugarcane, banana).",
        fontsize=8, color="#777777",
    )

    _save_fig(fig, save_path)

    # Print group statistics
    _step("Group statistics (original yield scale):")
    stats = df.groupby("Climate_Regime")["Yield"].agg(
        Count="count", Median="median", Mean="mean", Std="std"
    ).reindex(regime_order).round(3)
    print(stats.to_string())


# =============================================================================
# STEP 3 -- FEATURE ENGINEERING
# =============================================================================

def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Construct X (feature matrix) and y (target vector) ready for modelling.

    Feature engineering choices explained
    --------------------------------------
    1. Feature Selection
       We keep Rainfall, Temperature (direct climate drivers), plus State,
       District, Crop, Season, and Climate_Regime (context / categorical
       modifiers).  ONI_Annual_Mean is intentionally excluded here because
       Climate_Regime is its ordinal distillation -- including both would
       introduce mild multicollinearity.

    2. One-Hot Encoding (pd.get_dummies with drop_first=True)
       Linear regression requires purely numerical input.  Nominal categories
       like 'Crop' or 'State' have no intrinsic ordering, so ordinal encoding
       would impose a false numeric distance.  OHE creates a binary indicator
       column for every level.
       drop_first=True drops one dummy per group (the reference category) to
       avoid the Dummy Variable Trap: with K dummies and an intercept in the
       model, the design matrix would be rank-deficient (perfect
       multicollinearity) if all K columns were kept.

    3. StandardScaler on numeric features
       Ridge / OLS regression optimises coefficients that are in the same
       unit space.  Rainfall ranges 0-1,462 mm while Temperature spans
       3-51 degC.  Without scaling, the model penalises large-magnitude
       features differently from small-magnitude ones, distorting coefficient
       interpretation.  StandardScaler (z-score) centres each numeric feature
       to mean=0 and scales it to std=1.
       IMPORTANT: The scaler is FIT only on X_train, then applied (transform)
       to both X_train and X_test.  Fitting on the full dataset would cause
       data leakage -- the model would have seen test-set statistics.

    Returns
    -------
    X_train_final, X_test_final : np.ndarray
    y_train, y_test             : pd.Series
    feature_names               : list[str]
    scaler                      : fitted StandardScaler
    """
    _banner("STEP 3 -- FEATURE ENGINEERING")

    # ------------------------------------------------------------------
    # 3-A  Feature / Target Selection
    # ------------------------------------------------------------------
    _step("3-A  Feature & target selection")

    CATEGORICAL_FEATURES = ["State", "District", "Crop", "Season", "Climate_Regime"]
    NUMERIC_FEATURES     = ["Rainfall", "Temperature"]
    TARGET               = "Yield"

    # Drop any rows with NaN in our selected columns (safety guard)
    modelling_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    before         = len(df)
    df_model       = df[modelling_cols].dropna()
    _step("Rows after NaN drop", f"{len(df_model):,}  (removed {before - len(df_model):,})")

    X_raw = df_model[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y     = df_model[TARGET].copy()

    _step("X shape (pre-encoding)", str(X_raw.shape))
    _step("y shape", str(y.shape))

    # ------------------------------------------------------------------
    # 3-B  Train-Test Split  (must happen BEFORE encoding to prevent leakage)
    # ------------------------------------------------------------------
    _step("3-B  Train-test split (80/20, stratified on Climate_Regime)")

    # We stratify on Climate_Regime so El_Nino / La_Nina minority classes
    # appear proportionally in both train and test sets.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = X_raw["Climate_Regime"],   # preserve regime proportions
    )

    _step("X_train rows", f"{len(X_train_raw):,}")
    _step("X_test  rows", f"{len(X_test_raw):,}")

    # ------------------------------------------------------------------
    # 3-C  One-Hot Encoding of categorical features
    # ------------------------------------------------------------------
    _step("3-C  One-Hot Encoding categorical features")

    # pd.get_dummies is applied separately to train and test so that the
    # test set can ONLY see categories that existed in training.
    X_train_enc = pd.get_dummies(
        X_train_raw,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,       # drop reference category -> avoids dummy trap
        dtype=np.float32,      # float32 saves ~50% memory vs float64
    )

    X_test_enc = pd.get_dummies(
        X_test_raw,
        columns=CATEGORICAL_FEATURES,
        drop_first=True,
        dtype=np.float32,
    )

    # Align columns: test set may be missing categories unseen during training
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join="left", axis=1, fill_value=0.0
    )

    _step("Columns after encoding", f"{X_train_enc.shape[1]:,}")

    feature_names = list(X_train_enc.columns)

    # ------------------------------------------------------------------
    # 3-D  StandardScaler on numeric features only
    # ------------------------------------------------------------------
    _step("3-D  StandardScaler on numeric features (fit on train only)")

    numeric_col_indices = [
        X_train_enc.columns.get_loc(col) for col in NUMERIC_FEATURES
    ]

    scaler = StandardScaler()

    # Fit scaler ONLY on training data to avoid data leakage into test set
    X_train_np = X_train_enc.values.astype(np.float32)
    X_test_np  = X_test_enc.values.astype(np.float32)

    X_train_np[:, numeric_col_indices] = scaler.fit_transform(
        X_train_np[:, numeric_col_indices]
    )
    X_test_np[:, numeric_col_indices] = scaler.transform(
        X_test_np[:, numeric_col_indices]
    )

    _step("Rainfall  (train) -- mean after scaling",
          f"{X_train_np[:, numeric_col_indices[0]].mean():.6f}  (should be ~0)")
    _step("Temperature (train) -- std after scaling",
          f"{X_train_np[:, numeric_col_indices[1]].std():.6f}  (should be ~1)")

    return X_train_np, X_test_np, y_train, y_test, feature_names, scaler


# =============================================================================
# STEP 4 -- PERSIST ARTEFACTS TO DISK
# =============================================================================

def save_artefacts(
    X_train, X_test, y_train, y_test,
    feature_names, scaler,
) -> None:
    """
    Persist all engineered data splits and preprocessing objects.

    Why joblib?
    -----------
    joblib.dump / joblib.load is the standard way to serialise large numpy
    arrays in scikit-learn workflows.  It is significantly faster than pickle
    for large arrays because it uses memory-mapped file I/O.  The saved .pkl
    files can be loaded directly in the next pipeline stage (model_training.py)
    without re-running the expensive encoding step.
    """
    _banner("STEP 4 -- SAVING ARTEFACTS")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    artefacts = {
        PKL_X_TRAIN   : X_train,
        PKL_X_TEST    : X_test,
        PKL_Y_TRAIN   : y_train,
        PKL_Y_TEST    : y_test,
        PKL_FEAT_NAMES: feature_names,
        PKL_SCALER    : scaler,
    }

    for path, obj in artefacts.items():
        joblib.dump(obj, path, compress=3)
        size_kb = os.path.getsize(path) / 1024
        _step(f"Saved {os.path.basename(path)}", f"{size_kb:.1f} KB")


# =============================================================================
# STEP 5 -- FINAL SUMMARY REPORT
# =============================================================================

def print_summary(X_train, X_test, y_train, y_test, feature_names) -> None:
    """Print a concise summary table of the engineered dataset."""
    _banner("PIPELINE SUMMARY")

    print(f"\n  {'Metric':<40}  {'Value':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Total features (post-encoding)':<40}  {len(feature_names):>15,}")
    print(f"  {'Training samples':<40}  {X_train.shape[0]:>15,}")
    print(f"  {'Test samples':<40}  {X_test.shape[0]:>15,}")
    print(f"  {'Train/Test ratio':<40}  {100*(1-TEST_SIZE):.0f}/{100*TEST_SIZE:.0f}")
    print(f"  {'y_train median yield (t/ha)':<40}  {y_train.median():>15.4f}")
    print(f"  {'y_test  median yield (t/ha)':<40}  {y_test.median():>15.4f}")
    print(f"  {'y_train range':<40}  {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"  {'-'*55}")
    print(f"\n  Figures saved to  ->  reports/figures/")
    print(f"  Artefacts saved to->  data/processed/")
    print()


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_eda_and_engineering() -> None:
    """
    Master function that orchestrates all pipeline stages in sequence:
      1. Load master_cleaned_dataset.csv
      2. Generate 4 EDA visualisations
      3. Build feature matrix (encode + scale)
      4. Persist train/test splits to disk
      5. Print summary report
    """
    # Ensure output directories exist before any file I/O
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("\n")
    print("+----------------------------------------------------------------------+")
    print("|  AgriVest -- EDA & Feature Engineering Pipeline                     |")
    print("+----------------------------------------------------------------------+")

    # ── Stage 1: Data loading ─────────────────────────────────────────────────
    df = load_data(MASTER_CSV)

    # ── Stage 2: EDA Visualisations ──────────────────────────────────────────
    _banner("STAGE 2 -- EDA VISUALISATIONS (4 plots)")

    plot_correlation_heatmap(df, FIG_HEATMAP)
    plot_distributions(df, FIG_DIST)
    plot_scatter_trends(df, FIG_SCATTER)
    plot_climate_boxplot(df, FIG_BOX)

    print(f"\n  All 4 figures saved to {FIGURES_DIR}")

    # ── Stage 3: Feature Engineering ─────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, scaler = \
        build_feature_matrix(df)

    # ── Stage 4: Persist artefacts ────────────────────────────────────────────
    save_artefacts(X_train, X_test, y_train, y_test, feature_names, scaler)

    # ── Stage 5: Summary ──────────────────────────────────────────────────────
    print_summary(X_train, X_test, y_train, y_test, feature_names)

    print("  EDA & Feature Engineering pipeline completed successfully.")
    print(f"  Next step: run  python src/model_training.py\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_eda_and_engineering()
