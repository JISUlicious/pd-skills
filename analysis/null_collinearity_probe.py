"""Probe how the data-analyst skill (linear + ML methods) handles
nulls and multicollinearity. Three tests:

  1. Compute VIF for the 10 factors used in the stress analysis.
  2. Inject 5% NaN into two columns and re-run linear analysis.
  3. Same NaN injection, re-run XGBoost + SHAP analysis.

Reports what each method silently tolerates, what crashes, and what
gives wrong answers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/student_mental_health_burnout_1M.csv")
TARGET = "stress_level"
FACTORS = [
    "study_hours_per_day", "exam_pressure", "academic_performance",
    "sleep_hours", "physical_activity", "social_support",
    "screen_time", "internet_usage", "financial_stress",
    "family_expectation",
]
RNG = np.random.default_rng(42)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R²_j) where R²_j is from regressing column j on the others."""
    rows = []
    for j, col in enumerate(X.columns):
        y = X[col].to_numpy(dtype=np.float64)
        Xrest = X.drop(columns=col).to_numpy(dtype=np.float64)
        Xd = np.column_stack([np.ones(len(Xrest)), Xrest])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        yhat = Xd @ beta
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        vif = np.inf if r2 >= 0.9999 else 1 / (1 - r2)
        rows.append({"feature": col, "R²_on_others": round(r2, 4), "VIF": round(vif, 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def linear_fit(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """Standardized OLS — returns dict with R², top-3 by std-β, and how many rows survived."""
    n_before = len(X)
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[mask], y.loc[mask]
    n_after = len(X)
    Xv = X.to_numpy(dtype=np.float64)
    yv = y.to_numpy(dtype=np.float64)
    Xz = (Xv - Xv.mean(axis=0)) / Xv.std(axis=0, ddof=0)
    yz = (yv - yv.mean()) / yv.std(ddof=0)
    Xd = np.column_stack([np.ones(len(Xz)), Xz])
    beta, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
    yhat = Xd @ beta
    r2 = 1 - ((yz - yhat) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    coefs = pd.Series(beta[1:], index=X.columns)
    top3 = coefs.abs().sort_values(ascending=False).head(3).index.tolist()
    print(f"\n[{label}]  rows: {n_before:,} → {n_after:,} (dropped {n_before-n_after:,})")
    print(f"          R² = {r2:.4f}  top-3 by |std β| = {top3}")
    return {"r2": r2, "n_after": n_after, "top3": top3, "coefs": coefs}


def ml_fit(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    """XGBoost + SHAP on a 200k subsample. Returns top-3 by SHAP."""
    try:
        import xgboost as xgb
        import shap
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
    except Exception as e:
        print(f"[{label}] ML stack unavailable: {e}")
        return {}

    sample_idx = RNG.choice(len(X), size=min(200_000, len(X)), replace=False)
    Xs, ys = X.iloc[sample_idx], y.iloc[sample_idx]

    n_x_nan = int(Xs.isna().sum().sum())
    n_y_nan = int(ys.isna().sum())
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, tree_method="hist", n_jobs=-1,
    )
    try:
        model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        pred = model.predict(Xte)
        # XGBoost returns NaN predictions if y has NaN — drop those for r2 score
        ok = ~np.isnan(pred) & ~np.isnan(yte.values)
        r2 = r2_score(yte.values[ok], pred[ok]) if ok.sum() > 0 else float("nan")
        explainer = shap.TreeExplainer(model)
        Xte_shap = Xte.sample(n=min(20_000, len(Xte)), random_state=42)
        shap_vals = explainer.shap_values(Xte_shap)
        shap_mean = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
        top3 = shap_mean.sort_values(ascending=False).head(3).index.tolist()
        print(f"[{label}]  X_NaN={n_x_nan:,}  y_NaN={n_y_nan:,}  holdout R²={r2:.4f}  SHAP top-3={top3}")
        return {"r2": r2, "top3": top3, "shap_mean": shap_mean}
    except Exception as e:
        print(f"[{label}] CRASHED: {type(e).__name__}: {e}")
        return {"crashed": str(e)}


def main() -> None:
    df = pd.read_csv(DATA, low_memory=False)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols\n")
    X_clean = df[FACTORS].copy()
    y_clean = df[TARGET].copy()

    # ── Test 1: VIF on the factors ──────────────────────────────────────
    print("=" * 72)
    print("[Test 1] VIF — multicollinearity among the 10 factors")
    print("=" * 72)
    print("VIF interpretation: 1=none, 1–5=mild, 5–10=moderate, >10=severe")
    print(vif_table(X_clean).to_string(index=False))

    # ── Test 2: Linear with vs. without injected nulls ──────────────────
    print("\n" + "=" * 72)
    print("[Test 2] LINEAR — clean vs. 5% NaN in two columns")
    print("=" * 72)
    base = linear_fit(X_clean, y_clean, "linear / clean")

    X_nan = X_clean.copy()
    for col in ["financial_stress", "exam_pressure"]:
        idx = RNG.choice(len(X_nan), size=int(0.05 * len(X_nan)), replace=False)
        X_nan.iloc[idx, X_nan.columns.get_loc(col)] = np.nan
    print(f"Injected NaN: financial_stress={X_nan['financial_stress'].isna().sum():,},"
          f" exam_pressure={X_nan['exam_pressure'].isna().sum():,}")
    nan_drop = linear_fit(X_nan, y_clean, "linear / NaN dropped")

    # Naive imputation: fill with column mean
    X_imp = X_nan.fillna(X_nan.mean())
    naive = linear_fit(X_imp, y_clean, "linear / mean-imputed")

    # ── Test 3: ML with same scenarios ──────────────────────────────────
    print("\n" + "=" * 72)
    print("[Test 3] XGBoost + SHAP — clean vs. NaN passed through (no fillna)")
    print("=" * 72)
    ml_clean = ml_fit(X_clean, y_clean, "ML / clean")
    ml_nan   = ml_fit(X_nan,   y_clean, "ML / NaN passed through")
    ml_imp   = ml_fit(X_imp,   y_clean, "ML / mean-imputed")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("- VIF audit: see Test 1.  Any factor with VIF > 5 is multicollinear.")
    print("- Linear with NaN-drop: rows lost, R² may shift, top-3 may flip if NaN bias is informative.")
    print("- Linear with mean-imputed: keeps rows, but biases slope toward 0 for imputed columns.")
    print("- ML with NaN: XGBoost handles natively (learns missing-direction); SHAP works.")
    print("- ML with mean-imputed: comparable to clean if data is missing-at-random.")


if __name__ == "__main__":
    main()
