import nbformat as nbf
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# AgriVest: Foundations of Data Science in Python
### Comprehensive Curriculum Demonstration Notebook (Semester 5)

This notebook explicitly demonstrates the foundational programming and statistical concepts specified in the Data Science syllabus:
1. **Python Lists & Dictionaries**: Native data structures applied to tabular data analysis.
2. **NumPy Vectors & 2D Matrices**: Indexing, slicing, vectorization, and matrix operations ($w^T\mu$, $w^T\Sigma w$).
3. **Pandas DataFrames**: `.loc[]`, `.iloc[]`, multi-level grouping, reshaping, and categorical summaries.
4. **Statistical Descriptions of Data**: Central tendency (Mean, Median, Mode) and dispersion (Range, Quartiles, Variance, Std Dev, IQR).
5. **Data Cleaning & Outlier Detection**: IQR-based outlier boundaries and missing value handling.
6. **Multiple Linear Regression Modeling**: Fitting, residual computation, and model evaluation metrics.
"""))

# Section 1: Python Lists & Dictionaries
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Python Lists & Dictionaries (Data Science Applications)
Demonstrating how raw agricultural records can be processed using native Python lists and dictionaries without external libraries.
"""))

cells.append(nbf.v4.new_code_cell("""# Sample agricultural records represented as a list of dictionaries
crop_records = [
    {"state": "Maharashtra", "district": "Pune", "crop": "Soyabean", "yield": 1.85, "rainfall": 720.0},
    {"state": "Maharashtra", "district": "Pune", "crop": "Cotton",   "yield": 1.42, "rainfall": 680.0},
    {"state": "Maharashtra", "district": "Nashik", "crop": "Onion",  "yield": 14.50, "rainfall": 610.0},
    {"state": "Maharashtra", "district": "Satara", "crop": "Soyabean", "yield": 2.10, "rainfall": 810.0},
    {"state": "Maharashtra", "district": "Pune", "crop": "Rice",     "yield": 2.75, "rainfall": 950.0},
]

# Extracting values using list comprehensions
crops = [record["crop"] for record in crop_records]
yields = [record["yield"] for record in crop_records]

print("List of crops:", crops)
print("List of yields (t/ha):", yields)

# Aggregation using a dictionary
crop_yield_dict = {}
for r in crop_records:
    c = r["crop"]
    crop_yield_dict.setdefault(c, []).append(r["yield"])

print("\\nGrouped yield dictionary:")
for crop, y_list in crop_yield_dict.items():
    avg_y = sum(y_list) / len(y_list)
    print(f"  {crop:<10} -> Recorded: {y_list}, Mean Yield: {avg_y:.2f} t/ha")
"""))

# Section 2: NumPy Vectors & Matrices
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Vectors, 2D Matrices & Vectorized Operations (NumPy)
Demonstrating vector creation, slicing, reshaping, matrix multiplication, and the Markowitz quadratic form $w^T\Sigma w$.
"""))

cells.append(nbf.v4.new_code_cell("""import numpy as np

# 1. 1D Vector creation & indexing
yield_vector = np.array([1.85, 1.42, 14.50, 2.10, 2.75], dtype=np.float64)
print("1D Yield Vector:", yield_vector)
print("Vector Indexing (first 3):", yield_vector[:3])
print("Vector Slicing (step=2):", yield_vector[::2])

# 2. 2D Matrix creation & sub-setting (array[row_slice, col_slice])
feature_matrix = np.array([
    [720.0, 26.5, 1.85],
    [680.0, 27.1, 1.42],
    [610.0, 25.8, 14.50],
    [810.0, 24.9, 2.10],
    [950.0, 25.2, 2.75],
], dtype=np.float64)

print("\\n2D Feature Matrix (Rainfall, Temp, Yield):\\n", feature_matrix)
print("\\nMatrix Subsetting - Rows 1-3, Columns 0-1 (Rainfall & Temp):\\n", feature_matrix[1:4, 0:2])

# 3. Vectorized Operations (Mean-Variance Quadratic Form)
# Crop portfolio allocation weights: w = [Soyabean, Cotton, Rice]
w = np.array([0.45, 0.35, 0.20], dtype=np.float64)
expected_returns = np.array([45000.0, 58000.0, 38000.0])  # Expected profit INR/ha

# 3x3 Covariance matrix of crop returns
Sigma = np.array([
    [1.2e7, 0.5e7, 0.2e7],
    [0.5e7, 2.8e7, 0.4e7],
    [0.2e7, 0.4e7, 0.8e7],
])

# Vectorized Expected Portfolio Return: w^T * mu
portfolio_return = np.dot(w, expected_returns)

# Quadratic Portfolio Risk (Variance): w^T * Sigma * w
portfolio_variance = np.dot(w.T, np.dot(Sigma, w))
portfolio_volatility = np.sqrt(portfolio_variance)

print(f"\\nAllocated Weights w : {w} (Sum = {w.sum():.2f})")
print(f"Expected Portfolio Profit w^T * mu : INR {portfolio_return:,.2f}")
print(f"Portfolio Volatility sqrt(w^T * Sigma * w): INR {portfolio_volatility:,.2f}")
"""))

# Section 3: Pandas DataFrames & Slicing
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Pandas DataFrames: `.loc[]`, `.iloc[]`, and Grouped Summaries
Reading the cleaned master dataset, selecting subsets with label-based `.loc` and position-based `.iloc`, and summarizing across categories.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd

# Load master cleaned dataset
df = pd.read_csv("../data/processed/master_cleaned_dataset.csv")
print("Dataset Shape:", df.shape)
print("Columns:", list(df.columns))

# 1. Label-based selection with .loc[]
maharashtra_kharif = df.loc[(df["State"] == "Maharashtra") & (df["Season"] == "Kharif"), ["District", "Crop", "Rainfall", "Yield"]]
print("\\n--- .loc[] Selection (Maharashtra Kharif, first 5 rows) ---")
print(maharashtra_kharif.head())

# 2. Position-based selection with .iloc[]
subset_iloc = df.iloc[10:15, [0, 1, 2, 7]]
print("\\n--- .iloc[] Selection (Rows 10-14, Columns 0, 1, 2, 7) ---")
print(subset_iloc)

# 3. Grouped summary statistics
print("\\n--- Crop-Wise Yield Summary (Top 8 Crops by Count) ---")
crop_summary = df.groupby("Crop")["Yield"].agg(
    Count="count",
    Mean="mean",
    Median="median",
    StdDev="std",
    Min="min",
    Max="max"
).sort_values("Count", ascending=False).head(8)
print(crop_summary)
"""))

# Section 4: Statistical Descriptions of Data
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Basic Statistical Descriptions of Data
Calculating Central Tendency (Mean, Median, Mode) and Dispersion (Range, Quartiles, Variance, Standard Deviation, IQR).
"""))

cells.append(nbf.v4.new_code_cell("""# Extract Rainfall series
rainfall = df["Rainfall"]

# Central Tendency
mean_rf = rainfall.mean()
median_rf = rainfall.median()
mode_rf = rainfall.mode()[0]

# Dispersion
min_rf = rainfall.min()
max_rf = rainfall.max()
range_rf = max_rf - min_rf
q1_rf = rainfall.quantile(0.25)
q3_rf = rainfall.quantile(0.75)
iqr_rf = q3_rf - q1_rf
var_rf = rainfall.var()
std_rf = rainfall.std()

print("==================================================")
print("  STATISTICAL DESCRIPTION OF RAINFALL (mm)")
print("==================================================")
print(f"  Central Tendency:")
print(f"    Mean               : {mean_rf:10.2f} mm")
print(f"    Median (50th pct)  : {median_rf:10.2f} mm")
print(f"    Mode               : {mode_rf:10.2f} mm")
print(f"  Dispersion:")
print(f"    Minimum            : {min_rf:10.2f} mm")
print(f"    Maximum            : {max_rf:10.2f} mm")
print(f"    Range              : {range_rf:10.2f} mm")
print(f"    First Quartile (Q1): {q1_rf:10.2f} mm")
print(f"    Third Quartile (Q3): {q3_rf:10.2f} mm")
print(f"    IQR (Q3 - Q1)      : {iqr_rf:10.2f} mm")
print(f"    Variance           : {var_rf:10.2f} mm^2")
print(f"    Standard Deviation : {std_rf:10.2f} mm")
print("==================================================")
"""))

# Section 5: Data Cleaning & Outlier Detection
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Data Cleaning: IQR Outlier Detection & Group-wise Median Imputation
Implementing the $1.5 \\times IQR$ Tukey outlier rule and verifying group-wise median imputation.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. IQR-based outlier threshold computation
lower_bound = max(0, q1_rf - 1.5 * iqr_rf)
upper_bound = q3_rf + 1.5 * iqr_rf

outliers = df[(df["Rainfall"] < lower_bound) | (df["Rainfall"] > upper_bound)]
print(f"Rainfall IQR Outlier Boundaries: [{lower_bound:.2f} mm, {upper_bound:.2f} mm]")
print(f"Number of statistical outliers identified: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")

# 2. Demonstration of Group-wise Median Imputation
# Inject synthetic NaNs into a sample copy
sample_df = df[["District", "Crop", "Season", "Yield"]].head(100).copy()
sample_df.iloc[[2, 5, 12], 3] = np.nan

print(f"\\nInjected NaNs in sample: {sample_df['Yield'].isna().sum()}")

# Group-wise median imputation by Crop x Season
sample_df["Yield"] = sample_df["Yield"].fillna(
    sample_df.groupby(["Crop", "Season"])["Yield"].transform("median")
).fillna(sample_df["Yield"].median())

print(f"Remaining NaNs after group-wise median imputation: {sample_df['Yield'].isna().sum()}")
"""))

# Section 6: Multiple Linear Regression Evaluation
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Multiple Linear Regression & Residual Diagnostics
Demonstrating OLS regression training on encoded features, residual calculation, and model performance metrics.
"""))

cells.append(nbf.v4.new_code_cell("""from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load preprocessed arrays
X_train = joblib.load("../data/processed/X_train.pkl")
X_test = joblib.load("../data/processed/X_test.pkl")
y_train = joblib.load("../data/processed/y_train.pkl")
y_test = joblib.load("../data/processed/y_test.pkl")

# Train model
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)

# Predict on held-out test set
y_pred = lr.predict(X_test)
residuals = y_test.values - y_pred

# Compute Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
residual_std = np.std(residuals)

print("--- MULTIPLE LINEAR REGRESSION TEST EVALUATION ---")
print(f"  R-Squared (R2)                   : {r2:.4f}")
print(f"  Mean Absolute Error (MAE)        : {mae:.4f} t/ha")
print(f"  Root Mean Squared Error (RMSE)   : {rmse:.4f} t/ha")
print(f"  Residual Mean (Bias)             : {np.mean(residuals):.4f} (centered near 0)")
print(f"  Residual Std (Confidence Margin) : {residual_std:.4f} t/ha")
print(f"  Interpretation: 1 std dev corresponds to ~84% 1-sided downside bound.")
"""))

nb["cells"] = cells

nb_path = os.path.join(PROJECT_ROOT, "notebooks", "data_science_foundations.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"SUCCESS: Created notebook at {nb_path}")
