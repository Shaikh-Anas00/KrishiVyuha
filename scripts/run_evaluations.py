import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MASTER_CSV = os.path.join(PROCESSED_DIR, "master_cleaned_dataset.csv")

print("Loading preprocessed arrays...")
X_train = joblib.load(os.path.join(PROCESSED_DIR, "X_train.pkl"))
X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(PROCESSED_DIR, "y_train.pkl"))
y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
feature_names = joblib.load(os.path.join(PROCESSED_DIR, "feature_names.pkl"))

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# 1. Train primary model
print("Training primary OLS model...")
model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2_rand = r2_score(y_test, y_pred)
mae_rand = mean_absolute_error(y_test, y_pred)
rmse_rand = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Random Split -> R2: {r2_rand:.4f}, MAE: {mae_rand:.4f}, RMSE: {rmse_rand:.4f}")

# 2. Field crops subset evaluation (excluding Coconut)
df_master = pd.read_csv(MASTER_CSV)
test_crops = df_master.loc[y_test.index, "Crop"]
mask_field = (test_crops != "Coconut")

mae_field = mean_absolute_error(y_test[mask_field], y_pred[mask_field])
rmse_field = np.sqrt(mean_squared_error(y_test[mask_field], y_pred[mask_field]))
r2_field = r2_score(y_test[mask_field], y_pred[mask_field])

print(f"Field Crops (excluding Coconut) -> R2: {r2_field:.4f}, MAE: {mae_field:.4f} t/ha, RMSE: {rmse_field:.4f} t/ha")

# 3. Temporal Out-of-Time Validation (Train <= 2018, Test >= 2019)
print("\nRunning Temporal Out-of-Time Validation...")
CATEGORICAL_FEATURES = ["State", "District", "Crop", "Season", "Climate_Regime"]
NUMERIC_FEATURES = ["Rainfall", "Temperature"]

train_idx = df_master[df_master["Year"] <= 2018].index
test_idx = df_master[df_master["Year"] >= 2019].index

# Build temporal matrices aligned with feature_names
X_train_raw = df_master.loc[train_idx, NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train_temp = df_master.loc[train_idx, "Yield"]
X_test_raw = df_master.loc[test_idx, NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test_temp = df_master.loc[test_idx, "Yield"]

X_train_enc = pd.get_dummies(X_train_raw, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=np.float32)
X_test_enc = pd.get_dummies(X_test_raw, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=np.float32)

X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0.0)

# Scale numeric features
from sklearn.preprocessing import StandardScaler
scaler_temp = StandardScaler()
num_indices = [X_train_enc.columns.get_loc(c) for c in NUMERIC_FEATURES]

X_tr_mat = X_train_enc.values.astype(np.float32)
X_te_mat = X_test_enc.values.astype(np.float32)
X_tr_mat[:, num_indices] = scaler_temp.fit_transform(X_tr_mat[:, num_indices])
X_te_mat[:, num_indices] = scaler_temp.transform(X_te_mat[:, num_indices])

model_temp = LinearRegression(fit_intercept=True)
model_temp.fit(X_tr_mat, y_train_temp)
y_pred_temp = model_temp.predict(X_te_mat)

r2_temp = r2_score(y_test_temp, y_pred_temp)
mae_temp = mean_absolute_error(y_test_temp, y_pred_temp)
rmse_temp = np.sqrt(mean_squared_error(y_test_temp, y_pred_temp))

mask_temp_field = (df_master.loc[test_idx, "Crop"] != "Coconut")
mae_temp_field = mean_absolute_error(y_test_temp[mask_temp_field], y_pred_temp[mask_temp_field])
rmse_temp_field = np.sqrt(mean_squared_error(y_test_temp[mask_temp_field], y_pred_temp[mask_temp_field]))
r2_temp_field = r2_score(y_test_temp[mask_temp_field], y_pred_temp[mask_temp_field])

print(f"Temporal Split (Train <= 2018, Test >= 2019) -> R2: {r2_temp:.4f}, MAE: {mae_temp:.4f}, RMSE: {rmse_temp:.4f}")
print(f"Temporal Split Field Crops -> R2: {r2_temp_field:.4f}, MAE: {mae_temp_field:.4f} t/ha, RMSE: {rmse_temp_field:.4f} t/ha")

# 4. 5-Fold Cross Validation on 100,000 sample slice
print("\nRunning 5-Fold Cross-Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []
cv_mae = []
cv_rmse = []

# Evaluate on X_train folds (277k samples)
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train)):
    m_cv = LinearRegression()
    m_cv.fit(X_train[trn_idx], y_train.iloc[trn_idx])
    preds = m_cv.predict(X_train[val_idx])
    y_true = y_train.iloc[val_idx]
    cv_r2.append(r2_score(y_true, preds))
    cv_mae.append(mean_absolute_error(y_true, preds))
    cv_rmse.append(np.sqrt(mean_squared_error(y_true, preds)))

print(f"5-Fold CV Mean R2: {np.mean(cv_r2):.4f} (+/- {np.std(cv_r2):.4f})")
print(f"5-Fold CV Mean MAE: {np.mean(cv_mae):.4f} (+/- {np.std(cv_mae):.4f})")
print(f"5-Fold CV Mean RMSE: {np.mean(cv_rmse):.4f} (+/- {np.std(cv_rmse):.4f})")

# 5. Persist comprehensive metrics to model_metrics.json
metrics_dict = {
    "random_split": {
        "mae": round(mae_rand, 4),
        "rmse": round(rmse_rand, 4),
        "r2": round(r2_rand, 4),
        "field_crops_mae_t_ha": round(mae_field, 4),
        "field_crops_rmse_t_ha": round(rmse_field, 4),
        "field_crops_r2": round(r2_field, 4),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_test.shape[1])
    },
    "temporal_split": {
        "description": "Train on Year <= 2018 (326,985 rows), Test on Year >= 2019 (19,803 rows)",
        "train_years": "1997-2018",
        "test_years": "2019-2024",
        "mae": round(mae_temp, 4),
        "rmse": round(rmse_temp, 4),
        "r2": round(r2_temp, 4),
        "field_crops_mae_t_ha": round(mae_temp_field, 4),
        "field_crops_rmse_t_ha": round(rmse_temp_field, 4),
        "field_crops_r2": round(r2_temp_field, 4),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx))
    },
    "kfold_cv": {
        "k": 5,
        "mean_r2": round(float(np.mean(cv_r2)), 4),
        "std_r2": round(float(np.std(cv_r2)), 4),
        "mean_mae": round(float(np.mean(cv_mae)), 4),
        "std_mae": round(float(np.std(cv_mae)), 4),
        "mean_rmse": round(float(np.mean(cv_rmse)), 4),
        "std_rmse": round(float(np.std(cv_rmse)), 4)
    },
    "confidence_margin": round(float(np.std(y_test.values - y_pred)), 4),
    "residual_mean": round(float(np.mean(y_test.values - y_pred)), 4),
    "residual_std": round(float(np.std(y_test.values - y_pred)), 4),
    "downside_bound_interpretation": "1 residual std dev corresponds to ~84% 1-sided normal downside coverage",
    "model_type": "Multiple Linear Regression (OLS)"
}

out_metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
with open(out_metrics_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"\nPersisted comprehensive metrics to {out_metrics_path}")

# Save the primary model as yield_model.pkl
joblib.dump(model, os.path.join(MODELS_DIR, "yield_model.pkl"))
print("Saved yield_model.pkl")
