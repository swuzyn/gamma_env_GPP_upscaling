#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 1
Site-based 10-fold CV (strict by Site_Num) + hyperparameter search
(n_estimators step search + max_features recursive trial),
then select BEST config by ratio_R2 on concatenated OOF predictions.

Key outputs:
1) cv_performance_summary.csv      : metrics ONLY for ratio (true_ratio vs pred_ratio)
2) cv_oof_predictions_best.csv     : OOF rows for best config (Sample_Num, Site_Num, true_ratio, pred_ratio)
3) train_with_best_oof_and_daily.csv:
   ORIGINAL training table + best OOF ratio prediction + derived daily GPP predictions:
     - pred_daily_gpp_envrf_oof = GPP_DT_mean / pred_ratio_best_oof
     - pred_daily_gpp_sw        = GPP_DT_mean / ratio_SW_IN_F
     - pred_daily_gpp_cos       = GPP_DT_mean / COS SZA_ratio

Notes:
- Summary metrics DO NOT include any daily GPP performance (per your request).
- Column names are handled case-insensitively (and separator-insensitively).
- NO trimming, NO normalization (RF does not require scaling).
"""

from __future__ import annotations

import os
import json
import time
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =========================
# USER CONFIG (edit here)
# =========================
DATA_PATH = Path(r"C:\Users\双鸭山\Desktop\独立验证\Train_Data.csv")
OUTPUT_DIR = Path(r"C:\Users\双鸭山\Desktop\独立验证\rf_gamma_env_cv_outputs")

N_FOLDS = 10
RANDOM_SEED = 0
N_JOBS = 4

# Feature set (fixed as you requested)
FEATURE_SET_FIXED = ['ratio_SW_IN_F', 'TA_F_mean', 'VPD_F_mean', 'SWC_F_MDS_1_mean']

# Hyperparameter search (your original rule)
N_ESTIMATORS_LIST = list(range(100, 501, 50))  # 100..500 step 50
# max_features will be tried as 1..n_features (recursive trial)

# Required core columns
SAMPLE_COL = 'Sample_Num'
SITE_COL   = 'Site_Num'
TARGET_COL = 'ratio_GPP_DT'       # gamma target (ratio)
GPP_INST_COL = 'GPP_DT_mean'      # instantaneous GPP used to derive daily predictions in the final merged output
GAMMA_SW_COL = 'ratio_SW_IN_F'    # baseline gamma_SW (also a feature)
GAMMA_COS_COL = 'COS SZA_ratio'   # baseline gamma_COS (must exist if you want pred_daily_gpp_cos)


# Safety clip to avoid division explosion
GAMMA_MIN = 0.01

# =========================
# Case-insensitive helpers
# =========================
def _normalize_colname(s: str) -> str:
    if s is None:
        return ""
    s2 = str(s).strip().lower()
    for ch in [" ", "\t", "\n", "\r", "-", ".", "/", "\\"]:
        s2 = s2.replace(ch, "")
    return s2

def standardize_columns_case_insensitive(df: pd.DataFrame, canonical_cols: List[str]) -> pd.DataFrame:
    """
    Rename df columns to canonical names by robust matching:
    - case-insensitive
    - separator-insensitive (space, -, ., /, \\)
    """
    norm_to_original: Dict[str, str] = {}
    for c in df.columns:
        key = _normalize_colname(c)
        if key not in norm_to_original:
            norm_to_original[key] = c

    rename_map: Dict[str, str] = {}
    for canon in canonical_cols:
        canon_key = _normalize_colname(canon)
        if canon_key in norm_to_original:
            orig = norm_to_original[canon_key]
            if orig != canon:
                rename_map[orig] = canon

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

def require_cols(df: pd.DataFrame, cols: List[str], where: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        preview = ", ".join(list(map(str, df.columns[:60])))
        raise KeyError(
            f"[{where}] Missing required columns: {missing}\n"
            f"First columns in loaded CSV: {preview}"
        )

def safe_gamma(x: np.ndarray, gamma_min: float = GAMMA_MIN) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, gamma_min, None)
    return x

def ratio_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]
    n = int(len(y_true))
    if n == 0:
        return {"N": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan}

    r2 = r2_score(y_true, y_pred) if n > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"N": n, "R2": float(r2), "RMSE": rmse, "MAE": mae, "Bias": bias}


# =========================
# Data class
# =========================
@dataclass
class ModelConfig:
    n_trees: int
    max_features: int


# =========================
# Main
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    log_file = OUTPUT_DIR / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()]
    )

    checkpoint_file = OUTPUT_DIR / "checkpoints" / "checkpoint.json"

    def load_checkpoint() -> Dict[str, Dict]:
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_checkpoint(obj: Dict[str, Dict]):
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    t0 = time.time()
    logging.info("Start: strict site-based 10-fold CV + param search (ratio only metrics).")
    logging.info(f"DATA_PATH  = {DATA_PATH}")
    logging.info(f"OUTPUT_DIR = {OUTPUT_DIR}")

    # ---- Load raw, then standardize columns case-insensitively
    data_raw = pd.read_csv(DATA_PATH)
    canonical_needed = [SAMPLE_COL, SITE_COL, TARGET_COL, GPP_INST_COL, GAMMA_SW_COL, GAMMA_COS_COL] + FEATURE_SET_FIXED
    data = standardize_columns_case_insensitive(data_raw, canonical_needed)

    # ---- Required columns (for running + for final merged daily prediction columns)
    required_cols = [SAMPLE_COL, SITE_COL, TARGET_COL, GPP_INST_COL, GAMMA_SW_COL, GAMMA_COS_COL] + FEATURE_SET_FIXED
    require_cols(data, required_cols, "DATA_PATH")

    # ---- Type coercion
    for c in required_cols:
        if c in [SAMPLE_COL, SITE_COL]:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        else:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    # drop NaNs in essential
    before_n = len(data)
    data = data.dropna(subset=required_cols).copy()
    logging.info(f"Loaded rows: {before_n}, after NaN-drop: {len(data)}")

    # force ints
    data[SAMPLE_COL] = data[SAMPLE_COL].astype(np.int64)
    data[SITE_COL] = data[SITE_COL].astype(np.int64)

    # ---- site-based splits (KEEP your logic)
    site_numbers = data[SITE_COL].unique()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(site_numbers)
    site_splits = np.array_split(site_numbers, N_FOLDS)

    # ---- build configs by your rule:
    # n_estimators step list + max_features recursive 1..n_features
    n_features = len(FEATURE_SET_FIXED)
    configs: List[ModelConfig] = []
    for n_trees in N_ESTIMATORS_LIST:
        for mf in range(1, n_features + 1):
            configs.append(ModelConfig(n_trees=n_trees, max_features=mf))

    completed = load_checkpoint()  # config_key -> saved metrics
    all_rows: List[Dict] = []      # for performance summary

    best_key: Optional[str] = None
    best_r2: float = -np.inf

    # ============= run each config =============
    for idx, cfg in enumerate(configs, 1):
        config_key = f"{'_'.join(FEATURE_SET_FIXED)}__n{cfg.n_trees}__mf{cfg.max_features}"
        pred_path = OUTPUT_DIR / "predictions" / f"oof_{config_key}.csv"

        if config_key in completed and pred_path.exists():
            # reuse checkpoint
            m = completed[config_key]
            row = {
                "features": "_".join(FEATURE_SET_FIXED),
                "n_trees": cfg.n_trees,
                "max_features": cfg.max_features,
                **m
            }
            all_rows.append(row)
            if np.isfinite(m.get("R2", np.nan)) and m["R2"] > best_r2:
                best_r2 = float(m["R2"])
                best_key = config_key
            logging.info(f"[{idx}/{len(configs)}] Skip completed: {config_key} (R2={m.get('R2')})")
            continue

        logging.info(f"[{idx}/{len(configs)}] Running: n_trees={cfg.n_trees}, max_features={cfg.max_features}")

        # ---- OOF containers
        oof_parts: List[pd.DataFrame] = []

        # ---- 10-fold by site splits
        for fold_id in range(N_FOLDS):
            valid_sites = site_splits[fold_id]
            valid_mask = data[SITE_COL].isin(valid_sites)
            train_idx = data.index[~valid_mask]
            valid_idx = data.index[valid_mask]

            # X, y
            X_train = data.loc[train_idx, FEATURE_SET_FIXED].to_numpy(dtype=np.float32)
            y_train = data.loc[train_idx, TARGET_COL].to_numpy(dtype=np.float32)

            X_valid = data.loc[valid_idx, FEATURE_SET_FIXED].to_numpy(dtype=np.float32)
            y_valid = data.loc[valid_idx, TARGET_COL].to_numpy(dtype=np.float32)

            # ---- train RF
            model = RandomForestRegressor(
                n_estimators=cfg.n_trees,
                max_features=cfg.max_features,
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
                verbose=0
            )
            model.fit(X_train, y_train)

            # ---- predict ratio
            y_pred = model.predict(X_valid).astype(np.float32)

            part = pd.DataFrame({
                SAMPLE_COL: data.loc[valid_idx, SAMPLE_COL].values,
                SITE_COL: data.loc[valid_idx, SITE_COL].values,
                "true_ratio": y_valid,
                "pred_ratio": y_pred,
            })
            oof_parts.append(part)

            del model, X_train, y_train, X_valid, y_valid, y_pred
            gc.collect()

        oof_df = pd.concat(oof_parts, axis=0, ignore_index=True)

        # ---- overall OOF ratio metrics (concatenated OOF predictions)
        m = ratio_metrics(oof_df["true_ratio"].values, oof_df["pred_ratio"].values)

        # save per-config oof preds
        oof_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        # update checkpoint
        completed[config_key] = m
        save_checkpoint(completed)

        # summary row
        row = {
            "features": "_".join(FEATURE_SET_FIXED),
            "n_trees": cfg.n_trees,
            "max_features": cfg.max_features,
            **m
        }
        all_rows.append(row)

        if np.isfinite(m["R2"]) and m["R2"] > best_r2:
            best_r2 = float(m["R2"])
            best_key = config_key

        logging.info(f"  => OOF ratio metrics: R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, Bias={m['Bias']:.4f}")

    # ============= write performance summary (ratio only) =============
    summary_df = pd.DataFrame(all_rows).sort_values(by="R2", ascending=False)
    summary_csv = OUTPUT_DIR / "cv_performance_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    logging.info(f"Saved: {summary_csv}")

    if best_key is None:
        raise RuntimeError("No best config selected (this should not happen).")

    # ============= load best OOF predictions and write best files =============
    best_oof_path = OUTPUT_DIR / "predictions" / f"oof_{best_key}.csv"
    best_oof = pd.read_csv(best_oof_path)

    best_by_r2_csv = OUTPUT_DIR / "best_by_ratioR2.csv"
    best_row = summary_df.iloc[0:1].copy()
    best_row["best_config_key"] = best_key
    best_row.to_csv(best_by_r2_csv, index=False, encoding="utf-8-sig")
    logging.info(f"Best config: {best_key} (R2={best_r2:.4f}). Saved: {best_by_r2_csv}")

    # standardized best oof file name
    best_oof_out = OUTPUT_DIR / "cv_oof_predictions_best.csv"
    best_oof.to_csv(best_oof_out, index=False, encoding="utf-8-sig")
    logging.info(f"Saved: {best_oof_out}")

    # ============= merge best OOF ratio back to ORIGINAL raw table =============
    # Important: merge by Sample_Num into the original loaded raw table (data_raw), but use standardized columns
    raw2 = standardize_columns_case_insensitive(data_raw.copy(), canonical_needed)
    require_cols(raw2, [SAMPLE_COL], "RAW_FOR_MERGE")
    raw2[SAMPLE_COL] = pd.to_numeric(raw2[SAMPLE_COL], errors="coerce").astype("Int64")

    best_oof_small = best_oof[[SAMPLE_COL, "pred_ratio"]].copy()
    best_oof_small = best_oof_small.rename(columns={"pred_ratio": "pred_ratio_best_oof"})
    best_oof_small[SAMPLE_COL] = pd.to_numeric(best_oof_small[SAMPLE_COL], errors="coerce").astype("Int64")

    merged = raw2.merge(best_oof_small, on=SAMPLE_COL, how="left")

    # derive daily predictions columns (for later grouped analysis)
    # Need GPP_DT_mean, ratio_SW_IN_F, COS_SZA_ratio
    merged = standardize_columns_case_insensitive(merged, [GPP_INST_COL, GAMMA_SW_COL, GAMMA_COS_COL])
    require_cols(merged, [GPP_INST_COL, GAMMA_SW_COL, GAMMA_COS_COL, "pred_ratio_best_oof"], "MERGED_DERIVE_DAILY")

    merged[GPP_INST_COL]  = pd.to_numeric(merged[GPP_INST_COL], errors="coerce")
    merged[GAMMA_SW_COL]  = pd.to_numeric(merged[GAMMA_SW_COL], errors="coerce")
    merged[GAMMA_COS_COL] = pd.to_numeric(merged[GAMMA_COS_COL], errors="coerce")
    merged["pred_ratio_best_oof"] = pd.to_numeric(merged["pred_ratio_best_oof"], errors="coerce")

    gpp_inst = merged[GPP_INST_COL].to_numpy(dtype=float)
    gamma_env_oof = safe_gamma(merged["pred_ratio_best_oof"].to_numpy(dtype=float), GAMMA_MIN)
    gamma_sw = safe_gamma(merged[GAMMA_SW_COL].to_numpy(dtype=float), GAMMA_MIN)
    gamma_cos = safe_gamma(merged[GAMMA_COS_COL].to_numpy(dtype=float), GAMMA_MIN)

    merged["pred_daily_gpp_envrf_oof"] = gpp_inst / gamma_env_oof
    merged["pred_daily_gpp_sw"] = gpp_inst / gamma_sw
    merged["pred_daily_gpp_cos"] = gpp_inst / gamma_cos

    out_merged_csv = OUTPUT_DIR / "train_with_best_oof_and_daily.csv"
    merged.to_csv(out_merged_csv, index=False, encoding="utf-8-sig")
    logging.info(f"Saved merged training table: {out_merged_csv}")

    logging.info(f"All done. Total time: {(time.time() - t0)/60:.2f} min")


if __name__ == "__main__":
    main()
