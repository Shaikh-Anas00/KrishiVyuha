import os

script_path = "src/data_preprocessing.py"
with open(script_path, "r", encoding="utf-8") as f:
    code = f.read()

# Normalize CRLF
code_norm = code.replace("\r\n", "\n")

target_start = "def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:"
target_end = "    return df\n\n\ndef cap_rainfall_outliers"

idx_start = code_norm.find(target_start)
idx_end = code_norm.find(target_end)

if idx_start == -1 or idx_end == -1:
    print(f"Error: target boundaries not found (start={idx_start}, end={idx_end})")
    exit(1)

new_code = '''def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing numerical values using hierarchical group-wise medians:
      - Rainfall & Temperature: grouped by [District, Season] median -> [State, Season] median -> global median.
      - Yield: grouped by [Crop, District, Season] median -> [Crop, State] median -> [Crop] median -> global median.
    
    This preserves local agro-climatic and crop-specific baselines, conforming
    to the syllabus requirements for robust central tendency measures in skewed distributions.
    """
    _print_banner("STEP 4a - HIERARCHICAL GROUP-WISE MEDIAN IMPUTATION")

    # 1. Weather features: Rainfall and Temperature
    for col in ["Rainfall", "Temperature"]:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            # Level 1: District x Season median
            d_s_median = df.groupby(["District", "Season"])[col].transform("median")
            df[col] = df[col].fillna(d_s_median)

            # Level 2: State x Season median
            s_s_median = df.groupby(["State", "Season"])[col].transform("median")
            df[col] = df[col].fillna(s_s_median)

            # Level 3: Global median
            global_med = df[col].median()
            df[col] = df[col].fillna(global_med)

            _print_step(f"Imputed {n_missing:,} missing values in '{col}'",
                        f"hierarchical group-wise median (fallback: {global_med:.2f})")
        else:
            _print_step(f"No missing values in '{col}'")

    # 2. Target feature: Yield
    n_yield_missing = df["Yield"].isna().sum()
    if n_yield_missing > 0:
        # Level 1: Crop x District x Season median
        c_d_s_median = df.groupby(["Crop", "District", "Season"])["Yield"].transform("median")
        df["Yield"] = df["Yield"].fillna(c_d_s_median)

        # Level 2: Crop x State median
        c_s_median = df.groupby(["Crop", "State"])["Yield"].transform("median")
        df["Yield"] = df["Yield"].fillna(c_s_median)

        # Level 3: Crop global median
        c_median = df.groupby("Crop")["Yield"].transform("median")
        df["Yield"] = df["Yield"].fillna(c_median)

        # Level 4: Global median
        g_median = df["Yield"].median()
        df["Yield"] = df["Yield"].fillna(g_median)

        _print_step(f"Imputed {n_yield_missing:,} missing values in 'Yield'",
                    "hierarchical crop-district median")
    else:
        _print_step("No missing values in 'Yield'")

    # Also impute Year if it happened to be missing
    n_year_missing = df["Year"].isna().sum()
    if n_year_missing > 0:
        _print_step(f"Dropping {n_year_missing:,} rows with missing Year")
        df.dropna(subset=["Year"], inplace=True)

    # Drop rows with NaN in categorical keys
    cat_cols = ["State", "District", "Crop", "Season"]
    before = len(df)
    df.dropna(subset=cat_cols, inplace=True)
    dropped = before - len(df)
    if dropped:
        _print_step(f"Dropped {dropped:,} rows with NaN in categorical columns")

    _print_step("Shape after imputation", str(df.shape))
    return df


def audit_and_clean_yield_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit yield units and handle extreme reporting discrepancies:
      - Field Crops (Cereals, Pulses, Oilseeds): Standard unit is Tonnes/Hectare.
        Records with Yield >= 500 t/ha are known kg/ha reporting errors in district returns;
        these are normalized by dividing by 1,000 (kg -> tonnes).
        Yields between 100 and 500 t/ha for field crops are capped at the 99.5th crop percentile.
      - Sugarcane: High-biomass stalk crop (normal yield 40-120 t/ha); capped at 150 t/ha.
      - Coconut: Measured in 'Thousand Nuts/Ha' (normal yield 2,000-15,000);
        capped at 25,000 nuts/ha to remove anomalous extreme recording artifacts.
    """
    _print_banner("STEP 4b - YIELD UNIT AUDIT & OUTLIER NORMALISATION")

    # Unit correction for field crops mistakenly entered in kg/ha (>= 500)
    high_kg_mask = (df["Yield"] >= 500) & (~df["Crop"].isin(["Coconut", "Sugarcane"]))
    n_kg_corrected = high_kg_mask.sum()
    if n_kg_corrected > 0:
        df.loc[high_kg_mask, "Yield"] = df.loc[high_kg_mask, "Yield"] / 1000.0
        _print_step(f"Normalized {n_kg_corrected:,} kg/ha entries to tonnes/ha (Yield >= 500)")

    # Crop-specific capping for field crops with anomalous yields (> 100 t/ha)
    field_crop_mask = (df["Yield"] > 100) & (~df["Crop"].isin(["Coconut", "Sugarcane"]))
    n_field_capped = field_crop_mask.sum()
    if n_field_capped > 0:
        for crop_name in df.loc[field_crop_mask, "Crop"].unique():
            c_mask = (df["Crop"] == crop_name)
            p99_5 = df.loc[c_mask, "Yield"].quantile(0.995)
            excess_mask = c_mask & (df["Yield"] > max(50.0, p99_5))
            df.loc[excess_mask, "Yield"] = max(50.0, p99_5)
        _print_step(f"Capped {n_field_capped:,} extreme field crop yields at crop-specific 99.5th percentile")

    # Sugarcane capping at 150 t/ha
    sc_excess = (df["Crop"] == "Sugarcane") & (df["Yield"] > 150)
    n_sc = sc_excess.sum()
    if n_sc > 0:
        df.loc[sc_excess, "Yield"] = 150.0
        _print_step(f"Capped {n_sc:,} extreme Sugarcane entries at 150.0 t/ha")

    # Coconut capping at 25,000 nuts/ha
    coco_excess = (df["Crop"] == "Coconut") & (df["Yield"] > 25000)
    n_coco = coco_excess.sum()
    if n_coco > 0:
        df.loc[coco_excess, "Yield"] = 25000.0
        _print_step(f"Capped {n_coco:,} extreme Coconut entries at 25,000 nuts/ha")

    _print_step("Yield range after audit",
                f"Min: {df['Yield'].min():.2f}, Median: {df['Yield'].median():.2f}, Max: {df['Yield'].max():.2f}")
    return df
'''

code_updated = code_norm[:idx_start] + new_code + "\n\n" + code_norm[idx_end:]

# Also update run_pipeline() to call audit_and_clean_yield_outliers
pipe_target = "    #    5: Missing values"
pipe_idx = code_updated.find(pipe_target)
if pipe_idx != -1:
    pipe_end = code_updated.find("    #    6: Outlier capping", pipe_idx)
    if pipe_end != -1:
        pipe_replacement = """    #    5: Missing values & Yield audit
    master_df = handle_missing_values(master_df)
    master_df = audit_and_clean_yield_outliers(master_df)

"""
        code_updated = code_updated[:pipe_idx] + pipe_replacement + code_updated[pipe_end:]

with open(script_path, "w", encoding="utf-8") as f:
    f.write(code_updated)

print("SUCCESS: src/data_preprocessing.py updated cleanly.")
