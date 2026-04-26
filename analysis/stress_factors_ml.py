"""Verify feature-importance.md recipes on the student-stress dataset.

Cross-checks the linear analysis (stress_factors.py) using XGBoost gain,
permutation importance, and SHAP. Top-3 should match the linear top-3:
financial_stress, exam_pressure, family_expectation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_absolute_error
    HAS_ML = True
except Exception as e:
    HAS_ML = False
    print(f"ML stack unavailable ({type(e).__name__}: {e}).")
    print("Install with: uv pip install xgboost shap scikit-learn")
    print("On macOS, xgboost also needs OpenMP: brew install libomp")
    raise SystemExit(0)

DATA = Path("data/student_mental_health_burnout_1M.csv")
TARGET = "stress_level"
FACTORS = [
    "study_hours_per_day", "exam_pressure", "academic_performance",
    "sleep_hours", "physical_activity", "social_support",
    "screen_time", "internet_usage", "financial_stress",
    "family_expectation",
]


def main() -> None:
    df = pd.read_csv(DATA, low_memory=False)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols")

    # Sample for speed — full 1M is overkill for importance ranking
    sample = df.sample(n=200_000, random_state=42)
    X = sample[FACTORS].astype(np.float32)
    y = sample[TARGET].astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # ── XGBoost ───────────────────────────────────────────────────────
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, tree_method="hist", n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    print(f"\nHoldout R²:  {r2_score(y_test, pred):.4f}")
    print(f"Holdout MAE: {mean_absolute_error(y_test, pred):.4f}")

    # ── 1. Gain importance ────────────────────────────────────────────
    gain = pd.Series(model.feature_importances_, index=X.columns)

    # ── 2. Permutation importance ─────────────────────────────────────
    print("\nComputing permutation importance...")
    X_perm = X_test.iloc[:20_000]
    y_perm = y_test.iloc[:20_000]
    perm = permutation_importance(
        model, X_perm, y_perm, n_repeats=5, random_state=42,
        n_jobs=-1, scoring="r2",
    )
    perm_imp = pd.Series(perm.importances_mean, index=X.columns)

    # ── 3. SHAP ───────────────────────────────────────────────────────
    print("Computing SHAP values on 30k sample...")
    X_shap = X_test.sample(n=30_000, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap_mean = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)

    # ── Spearman with target (from linear analysis) ───────────────────
    spearman = sample[FACTORS].corrwith(sample[TARGET], method="spearman")

    # ── Rank-comparison table ─────────────────────────────────────────
    table = pd.DataFrame({
        "spearman_|rho|": spearman.abs(),
        "xgb_gain":       gain,
        "perm_importance": perm_imp,
        "shap_|mean|":     shap_mean,
    })
    ranks = table.rank(ascending=False, method="min").astype(int)
    ranks.columns = [f"r_{c}" for c in table.columns]
    out = pd.concat([table.round(4), ranks], axis=1).sort_values("r_shap_|mean|")

    print("\n" + "=" * 80)
    print("Rank-comparison table (sorted by SHAP rank)")
    print("=" * 80)
    print(out.to_string())

    # ── Method agreement ──────────────────────────────────────────────
    agree = ranks.corr(method="spearman").round(3)
    print("\nRank-method agreement (Spearman):")
    print(agree.to_string())

    # ── Top-3 consensus ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Top-3 by each method:")
    print("=" * 80)
    for col in ["spearman_|rho|", "xgb_gain", "perm_importance", "shap_|mean|"]:
        top3 = table[col].sort_values(ascending=False).head(3).index.tolist()
        print(f"  {col:<18s}: {top3}")

    # ── Verification assertion ────────────────────────────────────────
    expected_top3 = {"financial_stress", "exam_pressure", "family_expectation"}
    shap_top3 = set(shap_mean.sort_values(ascending=False).head(3).index)
    perm_top3 = set(perm_imp.sort_values(ascending=False).head(3).index)
    print("\n" + "=" * 80)
    print(f"Expected top-3 (from linear β):  {sorted(expected_top3)}")
    print(f"SHAP top-3:                      {sorted(shap_top3)}")
    print(f"Permutation top-3:               {sorted(perm_top3)}")
    assert shap_top3 == expected_top3, (
        f"SHAP top-3 differs from linear: {shap_top3} vs {expected_top3}"
    )
    assert perm_top3 == expected_top3, (
        f"Permutation top-3 differs from linear: {perm_top3} vs {expected_top3}"
    )
    print("\n✓ Cross-method agreement confirmed: ML cross-check matches linear analysis.")


if __name__ == "__main__":
    main()
