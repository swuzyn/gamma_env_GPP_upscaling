#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train final γ_env (RF) on FULL TRAIN_CSV using manually-provided BEST params
(from your 10-fold CV), then apply to NEW_CSV (independent dataset) to estimate
daily GPP and compare with:
  (1) radiation-ratio method (γ_SW)
  (2) cos(SZA) ratio method (COS SZA_ratio)

Notes:
- Column names are case-insensitive (lowercased internally).
- No percentile trimming.
- RF does NOT need normalization/standardization (use raw features).
- No plots: print metrics only.
- Optionally save NEW_CSV valid rows with appended predictions.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import gc

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ====================== USER CONFIG ======================
TRAIN_CSV = Path(r"C:\Users\双鸭山\Desktop\独立验证\Train_Data.csv")
NEW_CSV   = Path(r"C:\Users\双鸭山\Desktop\独立验证\Independent_Test.csv")

# 保存预测结果（只保存可用于评估的有效行）
SAVE_OUTPUT = True
OUT_PRED_CSV = NEW_CSV.parent / (NEW_CSV.stem + "_pred_daily_result_CVbest_MANUAL_plus_COS.csv")

# 特征组合（按你最终口径）
BEST_FEATURES = ['ratio_SW_IN_F', 'TA_F_mean', 'VPD_F_mean', 'SWC_F_MDS_1_mean']
TARGET_GAMMA_COL = 'ratio_GPP_DT'

# 独立验证使用列（大小写不敏感）
COL_GPP_INST = 'GPP_DT_mean'        # 瞬时 GPP
COL_GPP_TRUE = 'daily_gpp_dt_mean'  # 真值日 GPP
COL_GAMMA_SW = 'ratio_SW_IN_F'      # γ_SW（辐射比例法）
COL_GAMMA_COS = 'COS SZA_ratio'     # γ_COS（cos(SZA) 比例法）——测试集里有这一列

# ========= 手动填入：十折CV选出的最佳参数 =========
BEST_N_ESTIMATORS = 100    # 例如：300
BEST_MAX_FEATURES = 2      # 例如：3
RF_RANDOM_STATE   = 0
RF_NJOBS          = 4

# 防止除零爆炸
GAMMA_MIN_THRESHOLD = 0.01


# ====================== Helpers ======================
def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out

def require_cols(dfL: pd.DataFrame, cols_lc: list[str], where: str):
    missing = [c for c in cols_lc if c not in dfL.columns]
    if missing:
        raise KeyError(
            f"[{where}] missing columns: {missing}\n"
            f"Available (lowercased): {list(dfL.columns)[:60]} ..."
        )

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]

    n = int(len(y_true))
    if n == 0:
        return {"N": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan}

    r2 = r2_score(y_true, y_pred) if n > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    return {"N": n, "R2": float(r2), "RMSE": rmse, "MAE": mae, "Bias": bias}

def print_metrics(title: str, m: dict):
    print(f"\n=== {title} ===")
    print(f"N    = {m['N']}")
    print(f"R2   = {m['R2']:.4f}" if np.isfinite(m["R2"]) else "R2   = nan")
    print(f"RMSE = {m['RMSE']:.4f}" if np.isfinite(m["RMSE"]) else "RMSE = nan")
    print(f"MAE  = {m['MAE']:.4f}" if np.isfinite(m["MAE"]) else "MAE  = nan")
    print(f"Bias = {m['Bias']:.4f}" if np.isfinite(m["Bias"]) else "Bias = nan")


# ====================== Train (FULL TRAIN) ======================
def train_final_rf_gamma(train_csv: Path):
    print(f"[TRAIN] Loading {train_csv} ...")
    df = pd.read_csv(train_csv)
    dfL = lower_cols(df)

    feats_lc = [c.lower() for c in BEST_FEATURES]
    target_lc = TARGET_GAMMA_COL.lower()

    require_cols(dfL, feats_lc + [target_lc], "TRAIN_CSV")

    X = dfL[feats_lc].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y = pd.to_numeric(dfL[target_lc], errors="coerce").to_numpy(dtype=np.float32)

    # only drop NaNs
    m = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[m], y[m]
    print(f"[TRAIN] Samples after NaN drop: {len(y)} (no trimming)")

    print(f"[TRAIN] Fitting RF with CV-best params: n_estimators={BEST_N_ESTIMATORS}, max_features={BEST_MAX_FEATURES}")
    model = RandomForestRegressor(
        n_estimators=BEST_N_ESTIMATORS,
        max_features=BEST_MAX_FEATURES,
        random_state=RF_RANDOM_STATE,
        n_jobs=RF_NJOBS
    )
    model.fit(X, y)

    artifacts = {"features_lc": feats_lc}

    del df, dfL, X, y
    gc.collect()
    return model, artifacts


# ====================== Apply + Evaluate (Independent) ======================
def apply_and_eval(model, artifacts, new_csv: Path):
    print(f"\n[EVAL] Loading {new_csv} ...")
    df = pd.read_csv(new_csv)
    dfL = lower_cols(df)

    feats_lc = artifacts["features_lc"]
    gpp_inst_lc = COL_GPP_INST.lower()
    gpp_true_lc = COL_GPP_TRUE.lower()
    gamma_sw_lc = COL_GAMMA_SW.lower()
    gamma_cos_lc = COL_GAMMA_COS.lower()   # 支持 "cos sza_ratio" 这种原始列名

    required = feats_lc + [gpp_inst_lc, gpp_true_lc, gamma_sw_lc, gamma_cos_lc]
    require_cols(dfL, required, "NEW_CSV")

    # numeric coercion
    for c in required:
        dfL[c] = pd.to_numeric(dfL[c], errors="coerce")

    # drop any rows with NaN in features/targets/baselines
    mask = dfL[required].notna().all(axis=1)
    df_clean = dfL.loc[mask].copy()
    n_dropped = len(dfL) - len(df_clean)
    if df_clean.empty:
        raise RuntimeError("[EVAL] NEW_CSV contains 0 valid rows after NaN removal.")

    print(f"[EVAL] Valid samples: {len(df_clean)} (Dropped NaNs: {n_dropped})")

    X = df_clean[feats_lc].to_numpy(dtype=np.float32)

    # predict gamma_env
    gamma_rf = model.predict(X).astype(float)
    gamma_rf = np.clip(gamma_rf, GAMMA_MIN_THRESHOLD, None)

    # baseline gamma_sw
    gamma_sw = df_clean[gamma_sw_lc].to_numpy(dtype=float)
    gamma_sw = np.clip(gamma_sw, GAMMA_MIN_THRESHOLD, None)

    # baseline gamma_cos
    gamma_cos = df_clean[gamma_cos_lc].to_numpy(dtype=float)
    gamma_cos = np.clip(gamma_cos, GAMMA_MIN_THRESHOLD, None)

    gpp_inst = df_clean[gpp_inst_lc].to_numpy(dtype=float)
    gpp_true = df_clean[gpp_true_lc].to_numpy(dtype=float)

    pred_daily_rf  = gpp_inst / gamma_rf
    pred_daily_sw  = gpp_inst / gamma_sw
    pred_daily_cos = gpp_inst / gamma_cos

    m_rf  = metrics(gpp_true, pred_daily_rf)
    m_sw  = metrics(gpp_true, pred_daily_sw)
    m_cos = metrics(gpp_true, pred_daily_cos)

    print("\n==============================")
    print("Independent evaluation (NO PLOTS)")
    print(f"Truth daily GPP column: {COL_GPP_TRUE}")
    print(f"Instant GPP column    : {COL_GPP_INST}")
    print(f"Baseline gamma_SW col : {COL_GAMMA_SW}")
    print(f"Baseline gamma_COS col: {COL_GAMMA_COS}")
    print("==============================")

    print_metrics("γ_env (RF, CV-best MANUAL) -> daily GPP vs Truth", m_rf)
    print_metrics("Radiation ratio (γ_SW) -> daily GPP vs Truth", m_sw)
    print_metrics("cos(SZA) ratio (γ_COS) -> daily GPP vs Truth", m_cos)

    if SAVE_OUTPUT:
        df_clean["gamma_env_rf_pred"]   = gamma_rf
        df_clean["pred_daily_gpp_envrf"] = pred_daily_rf
        df_clean["pred_daily_gpp_sw"]    = pred_daily_sw
        df_clean["pred_daily_gpp_cossza"] = pred_daily_cos

        df_clean.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
        print(f"\n[DONE] Saved predictions to: {OUT_PRED_CSV}")

    return m_rf, m_sw, m_cos


# ====================== Main ======================
if __name__ == "__main__":
    model, artifacts = train_final_rf_gamma(TRAIN_CSV)
    _ = apply_and_eval(model, artifacts, NEW_CSV)
