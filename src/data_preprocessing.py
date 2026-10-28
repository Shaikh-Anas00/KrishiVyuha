"""
                                                                                
   AgriVest   Data Preprocessing Pipeline                                      
   File   : src/data_preprocessing.py                                          
   Purpose: Schema harmonisation, dataset stitching, missing-value handling,   
            outlier capping, ONI climate-index merge, and regime labelling.     
                                                                                
   Output : data/processed/master_cleaned_dataset.csv                          
                                                                                

Run from project root:
    python src/data_preprocessing.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

#                                                                              
# PATH CONFIGURATION  (paths are relative to the project root)
#                                                                              

# Resolve the project root regardless of where the script is called from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

PATH_345K  = os.path.join(RAW_DIR, "crop_production_weather_345k.csv")
PATH_7K    = os.path.join(RAW_DIR, "crop_production_weather_7k.csv")
PATH_ONI   = os.path.join(RAW_DIR, "oni_climate_index.csv")
PATH_OUT   = os.path.join(PROCESSED_DIR, "master_cleaned_dataset.csv")

# ONI thresholds for El-Ni o / La-Ni a regime classification
ONI_EL_NINO_THRESH =  0.5
ONI_LA_NINA_THRESH = -0.5

# Rainfall outlier cap percentile
RAINFALL_CAP_PERCENTILE = 99

# Sentinel / placeholder values used in the ONI file to mark missing entries
ONI_MISSING_SENTINELS = {-9999.0, -99.9}


#                                                                              
# UTILITY HELPERS
#                                                                              

def _print_banner(title: str) -> None:
    """Print a clearly visible section banner to stdout."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_step(step: str, detail: str = "") -> None:
    """Print a single pipeline step with an optional detail message."""
    msg = f"  >>  {step}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)


def _clean_text_column(series: pd.Series) -> pd.Series:
    """
    Normalise a string Series:
      1. Cast to str and strip leading/trailing whitespace.
      2. Apply title-case so 'RICE', 'rice', 'Rice' all become 'Rice'.
    """
    return series.astype(str).str.strip().str.title()


#                                                                              
# STEP 1   LOAD RAW FILES
#                                                                              

def load_345k_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the large historical dataset ( 345 000 rows, 1997-2020).

    Raw columns  : State, District , Crop, Crop_Year, Season,
                   Area , Production, Yield, Rainfall, Temperature
    Target schema: State, District, Crop, Year, Season,
                   Rainfall, Temperature, Yield

    Note: 'Area' and 'Production' are dropped because the 7 k dataset does
          not carry equivalent columns and they are not primary model features.
    """
    _print_step("Loading 345k dataset", filepath)
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"\n     ERROR: File not found   {filepath}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n     ERROR loading 345k dataset: {exc}")
        sys.exit(1)

    _print_step("345k raw shape", str(df.shape))

    #    Strip any accidental whitespace from column names themselves           
    df.columns = df.columns.str.strip()

    #    Column rename map (raw   harmonised)                                  
    rename_map = {
        "Crop_Year"   : "Year",
        # All other target columns already carry the correct names.
    }
    df.rename(columns=rename_map, inplace=True)

    #    Select only the harmonised columns we care about                      
    target_cols = ["State", "District", "Crop", "Year", "Season",
                   "Rainfall", "Temperature", "Yield"]
    df = df[target_cols].copy()

    _print_step("345k after column selection", str(df.shape))
    return df


def load_7k_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the recent dataset ( 7 000 rows, 2014-2024).

    Raw columns  : Record_ID, State, District, Year, Season, Crop_Type,
                   Rainfall_mm, Avg_Temperature_C, Humidity_percent,
                   Soil_Type, Irrigation_Type, Fertilizer_Used_kg,
                   Pesticide_Used_kg, Cultivated_Area_hectares,
                   Yield_ton_per_hectare, Market_Price_per_ton,
                   Farmer_Profit_INR
    Target schema: State, District, Crop, Year, Season,
                   Rainfall, Temperature, Yield
    """
    _print_step("Loading 7k dataset", filepath)
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        print(f"\n     ERROR: File not found   {filepath}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n     ERROR loading 7k dataset: {exc}")
        sys.exit(1)

    _print_step("7k raw shape", str(df.shape))

    #    Column rename map (raw   harmonised)                                  
    rename_map = {
        "Crop_Type"              : "Crop",
        "Rainfall_mm"            : "Rainfall",
        "Avg_Temperature_C"      : "Temperature",
        "Yield_ton_per_hectare"  : "Yield",
    }
    df.rename(columns=rename_map, inplace=True)

    #    Select only harmonised columns                                        
    target_cols = ["State", "District", "Crop", "Year", "Season",
                   "Rainfall", "Temperature", "Yield"]
    df = df[target_cols].copy()

    _print_step("7k after column selection", str(df.shape))
    return df


#                                                                              
# STEP 2   SCHEMA HARMONISATION  (text normalisation)
#                                                                              

def harmonise_schema(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Apply uniform text normalisation to all categorical / string columns so
    that values like 'RICE', 'rice', 'Rice' are all merged under 'Rice'.

    Args:
        df           : DataFrame with the harmonised column set.
        source_label : Short label for logging (e.g. '345k' or '7k').

    Returns:
        Cleaned DataFrame with a 'Source' tag appended.
    """
    _print_step(f"Harmonising text columns [{source_label}]")

    text_columns = ["State", "District", "Crop", "Season"]
    for col in text_columns:
        if col in df.columns:
            df[col] = _clean_text_column(df[col])

    # Ensure Year is integer (some CSVs may read it as float)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    # Ensure numeric columns are float
    for col in ["Rainfall", "Temperature", "Yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tag the originating dataset for traceability
    df["Source"] = source_label

    return df


#                                                                              
# STEP 3   DEDUPLICATION via YEAR-RANGE PARTITIONING
#                                                                              

def split_by_year(df_345k: pd.DataFrame,
                  df_7k:   pd.DataFrame) -> pd.DataFrame:
    """
    Prevent overlap between the two datasets:
        345k   keep rows where Year <= 2020  (its natural range is 1997-2020)
        7k     keep rows where Year >  2020  (the 2021-2024 extension)

    The 2014-2020 window present in both datasets is resolved by keeping the
    large, more authoritative 345k file for those years and using the 7k file
    only for years beyond the 345k's coverage.
    """
    _print_banner("STEP 3   DEDUPLICATION (Year-range Partitioning)")

    # Filter each dataset to its non-overlapping year window
    hist_df   = df_345k[df_345k["Year"] <= 2020].copy()
    recent_df = df_7k[df_7k["Year"] > 2020].copy()

    _print_step("345k after year filter (  2020)", str(hist_df.shape))
    _print_step("7k   after year filter (> 2020)", str(recent_df.shape))

    # Vertical concatenation (axis=0)   schemas are now identical
    master_df = pd.concat([hist_df, recent_df], axis=0, ignore_index=True)

    _print_step("Master dataset after concatenation", str(master_df.shape))
    print(f"\n  Year coverage: {master_df['Year'].min()}   {master_df['Year'].max()}")

    return master_df


#                                                                              
# STEP 4   MISSING VALUE IMPUTATION & OUTLIER CAPPING
#                                                                              

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
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


    return df


def cap_rainfall_outliers(df: pd.DataFrame,
                          percentile: int = RAINFALL_CAP_PERCENTILE) -> pd.DataFrame:
    """
    Cap extreme Rainfall values at the specified percentile to prevent
    individual weather events from distorting model training.

    Values above the cap are clipped to the cap ceiling; values below zero
    (physically impossible) are zeroed out.

    Args:
        df         : Master DataFrame with a 'Rainfall' column.
        percentile : Upper cap percentile (default 99).

    Returns:
        DataFrame with Rainfall column capped in-place.
    """
    _print_banner("STEP 4b   RAINFALL OUTLIER CAPPING")

    cap_value = df["Rainfall"].quantile(percentile / 100)
    n_above   = (df["Rainfall"] > cap_value).sum()
    n_below   = (df["Rainfall"] < 0).sum()

    _print_step(f"Rainfall {percentile}th-percentile cap", f"{cap_value:.2f} mm")
    _print_step(f"Rows above cap (to be clipped)", f"{n_above:,}")
    _print_step(f"Rows below 0   (to be zeroed)",  f"{n_below:,}")

    df["Rainfall"] = df["Rainfall"].clip(lower=0, upper=cap_value)

    _print_step("Rainfall range after capping",
                f"{df['Rainfall'].min():.2f}   {df['Rainfall'].max():.2f} mm")

    return df


#                                                                              
# STEP 5   ONI CLIMATE INDEX LOADING & PROCESSING
#                                                                              

def load_oni_index(filepath: str) -> pd.DataFrame:
    """
    Load and process the NOAA Oceanic Ni o Index (ONI) CSV.

    The raw file has a messy header and uses -99.9 as a placeholder for
    missing values.  This function:
      1. Reads the file, forcing exactly two columns: ['Date', 'ONI'].
      2. Drops rows where ONI is a sentinel placeholder (-9999.0 or -99.9).
      3. Extracts the year from the Date column.
      4. Aggregates monthly readings to a single annual mean ONI value.

    Returns:
        DataFrame with columns ['Year', 'ONI_Annual_Mean'].
    """
    _print_banner("STEP 5   ONI CLIMATE INDEX")
    _print_step("Loading ONI file", filepath)

    try:
        # The header line contains a long URL in the second column name.
        # We read with header=0 and rename immediately.
        oni_raw = pd.read_csv(filepath, header=0)
    except FileNotFoundError:
        print(f"\n     ERROR: ONI file not found   {filepath}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n     ERROR loading ONI file: {exc}")
        sys.exit(1)

    # Standardise to exactly two columns regardless of the messy header text
    oni_raw.columns = ["Date", "ONI"]
    _print_step("ONI raw shape", str(oni_raw.shape))

    #    Coerce types                                                          
    oni_raw["ONI"]  = pd.to_numeric(oni_raw["ONI"],  errors="coerce")
    oni_raw["Date"] = pd.to_datetime(oni_raw["Date"], errors="coerce")

    #    Remove sentinel / placeholder values                                  
    sentinel_mask = oni_raw["ONI"].isin(ONI_MISSING_SENTINELS)
    n_sentinels   = sentinel_mask.sum()
    if n_sentinels:
        _print_step(f"Removing {n_sentinels:,} sentinel placeholder rows")
    oni_raw = oni_raw[~sentinel_mask].dropna(subset=["ONI", "Date"])

    #    Extract year from Date                                                 
    oni_raw["Year"] = oni_raw["Date"].dt.year

    #    Aggregate to annual mean ONI                                          
    oni_annual = (
        oni_raw
        .groupby("Year", as_index=False)["ONI"]
        .mean()
        .rename(columns={"ONI": "ONI_Annual_Mean"})
    )

    _print_step("Annual ONI records", str(len(oni_annual)))
    print(f"  Year coverage: {oni_annual['Year'].min()}   {oni_annual['Year'].max()}")

    return oni_annual


#                                                                              
# STEP 6   MERGE ONI INTO MASTER & CREATE Climate_Regime LABEL
#                                                                              

def merge_oni_and_label(master_df: pd.DataFrame,
                        oni_df:    pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the annual ONI values onto the master crop DataFrame by Year.
    Rows with no corresponding ONI entry (years outside NOAA coverage) will
    have NaN in ONI_Annual_Mean   these are filled with 0 (neutral) for safety.

    Then derive a 'Climate_Regime' categorical column:
        ONI >= +0.5     'El_Nino'   (warm Pacific event)
        ONI <= -0.5     'La_Nina'   (cool Pacific event)
        otherwise      'Normal'

    Args:
        master_df : Deduplicated, cleaned crop DataFrame.
        oni_df    : Annual ONI summary with columns ['Year', 'ONI_Annual_Mean'].

    Returns:
        Merged DataFrame with 'ONI_Annual_Mean' and 'Climate_Regime' appended.
    """
    _print_banner("STEP 6   ONI MERGE & CLIMATE REGIME LABELLING")

    # Ensure Year dtype alignment before the join
    master_df["Year"] = master_df["Year"].astype(int)
    oni_df["Year"]    = oni_df["Year"].astype(int)

    merged = master_df.merge(oni_df, on="Year", how="left")

    n_unmatched = merged["ONI_Annual_Mean"].isna().sum()
    _print_step("Rows without an ONI match (filled with 0.0)", f"{n_unmatched:,}")
    merged["ONI_Annual_Mean"].fillna(0.0, inplace=True)

    #    Climate Regime classification                                          
    def _classify_regime(oni_val: float) -> str:
        if oni_val >= ONI_EL_NINO_THRESH:
            return "El_Nino"
        elif oni_val <= ONI_LA_NINA_THRESH:
            return "La_Nina"
        else:
            return "Normal"

    merged["Climate_Regime"] = merged["ONI_Annual_Mean"].apply(_classify_regime)

    #    Regime distribution summary                                            
    regime_counts = merged["Climate_Regime"].value_counts()
    _print_step("Climate_Regime distribution:")
    for regime, count in regime_counts.items():
        pct = 100 * count / len(merged)
        print(f"      {regime:<12}: {count:>8,}  ({pct:.1f}%)")

    return merged


#                                                                              
# STEP 7   FINAL VALIDATION & SAVE
#                                                                              

def validate_and_save(df: pd.DataFrame, output_path: str) -> None:
    """
    Run a final sanity-check on the processed DataFrame and write it to disk.

    Checks performed:
        No duplicate rows (State + District + Crop + Year + Season).
        No remaining NaN values in the core feature columns.
        Yield values are non-negative.
    """
    _print_banner("STEP 7   FINAL VALIDATION & SAVE")

    #    Duplicate check                                                        
    key_cols  = ["State", "District", "Crop", "Year", "Season"]
    n_dupes   = df.duplicated(subset=key_cols).sum()
    _print_step("Duplicate rows (same crop-season-year per location)", f"{n_dupes:,}")
    if n_dupes > 0:
        df.drop_duplicates(subset=key_cols, keep="first", inplace=True)
        _print_step("Duplicates removed; new shape", str(df.shape))

    #    NaN check on core features                                             
    core_features = ["Rainfall", "Temperature", "Yield",
                     "ONI_Annual_Mean", "Climate_Regime"]
    nan_summary = df[core_features].isna().sum()
    if nan_summary.any():
        print("\n     Remaining NaN values detected:")
        print(nan_summary[nan_summary > 0].to_string())
    else:
        _print_step("NaN check passed", "No missing values in core features  ")

    #    Non-negative Yield check                                               
    n_neg_yield = (df["Yield"] < 0).sum()
    if n_neg_yield > 0:
        _print_step(f"Clipping {n_neg_yield:,} negative Yield values to 0")
        df["Yield"] = df["Yield"].clip(lower=0)
    else:
        _print_step("Yield non-negativity check passed  ")

    #    Column order for readability                                           
    ordered_cols = [
        "State", "District", "Crop", "Year", "Season",
        "Rainfall", "Temperature", "Yield",
        "ONI_Annual_Mean", "Climate_Regime",
        "Source",
    ]
    df = df[ordered_cols]

    #    Save                                                                   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    _print_step("Output saved", output_path)
    _print_step("Final shape", str(df.shape))

    #    Summary statistics                                                     
    print("\n" + "-" * 72)
    print("  FINAL DATASET SUMMARY")
    print("-" * 72)
    print(f"  Total rows          : {len(df):,}")
    print(f"  Year range          : {df['Year'].min()}   {df['Year'].max()}")
    print(f"  Unique States       : {df['State'].nunique()}")
    print(f"  Unique Districts    : {df['District'].nunique()}")
    print(f"  Unique Crops        : {df['Crop'].nunique()}")
    print(f"  Unique Seasons      : {df['Season'].nunique()}")
    print(f"  Columns             : {list(df.columns)}")
    print("-" * 72)
    print()


#                                                                              
# MAIN PIPELINE ORCHESTRATOR
#                                                                              

def run_pipeline() -> pd.DataFrame:
    """
    Orchestrate the full end-to-end preprocessing pipeline:

      1. Load raw 345k dataset            (load_345k_dataset)
      2. Load raw 7k dataset              (load_7k_dataset)
      3. Harmonise schemas                (harmonise_schema   2)
      4. Deduplicate by year range        (split_by_year)
      5. Impute missing values            (handle_missing_values)
      6. Cap rainfall outliers            (cap_rainfall_outliers)
      7. Load & aggregate ONI index       (load_oni_index)
      8. Merge ONI + label regime         (merge_oni_and_label)
      9. Validate & persist to disk       (validate_and_save)

    Returns:
        The final cleaned master DataFrame.
    """
    print("\n")
    print("+" + "-" * 70 + "+")
    print("|  AgriVest - Data Preprocessing Pipeline                            |")
    print("+" + "-" * 70 + "+")

    #    1 & 2: Load                                                            
    _print_banner("STEP 1 & 2   LOAD RAW DATASETS")
    df_345k = load_345k_dataset(PATH_345K)
    df_7k   = load_7k_dataset(PATH_7K)

    #    3: Harmonise                                                           
    _print_banner("STEP 3   SCHEMA HARMONISATION")
    df_345k = harmonise_schema(df_345k, source_label="345k")
    df_7k   = harmonise_schema(df_7k,   source_label="7k")

    #    4: Deduplication                                                       
    master_df = split_by_year(df_345k, df_7k)

    #    5: Missing values & Yield audit
    master_df = handle_missing_values(master_df)
    master_df = audit_and_clean_yield_outliers(master_df)

    #    6: Outlier capping                                                     
    master_df = cap_rainfall_outliers(master_df, percentile=RAINFALL_CAP_PERCENTILE)

    #    7: ONI                                                                 
    oni_annual = load_oni_index(PATH_ONI)

    #    8: Merge & label                                                       
    master_df = merge_oni_and_label(master_df, oni_annual)

    #    9: Validate & save                                                     
    validate_and_save(master_df, PATH_OUT)

    return master_df


#                                                                              
# ENTRY POINT
#                                                                              

if __name__ == "__main__":
    final_df = run_pipeline()
    print("  Pipeline completed successfully.  ")
    print(f"  Output   {PATH_OUT}\n")
