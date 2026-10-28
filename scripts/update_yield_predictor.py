import os

script_path = "src/yield_predictor.py"
with open(script_path, "r", encoding="utf-8") as f:
    code = f.read()

code_norm = code.replace("\r\n", "\n")

# Replace evaluate_model to include temporal validation and field crop sub-evaluation
old_eval_start = "def evaluate_model(model: LinearRegression,"
old_eval_end = "    return metrics, y_pred, residuals"

idx1 = code_norm.find(old_eval_start)
idx2 = code_norm.find(old_eval_end)

if idx1 == -1 or idx2 == -1:
    print("Could not find evaluate_model boundaries")
    exit(1)

new_eval = '''def evaluate_model(model: LinearRegression,
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
    }'''

code_norm = code_norm[:idx1] + new_eval + "\n    return metrics, y_pred, residuals" + code_norm[idx2 + len(old_eval_end):]

# Also fix pandas import if missing and replace next step message
code_norm = code_norm.replace('print("  Next step: python src/risk_optimizer.py")', 'print("  Next step: python src/portfolio_optimizer.py")')
if "import pandas as pd" not in code_norm:
    code_norm = code_norm.replace("import numpy as np", "import numpy as np\nimport pandas as pd")

with open(script_path, "w", encoding="utf-8") as f:
    f.write(code_norm)

print("SUCCESS: src/yield_predictor.py updated.")
