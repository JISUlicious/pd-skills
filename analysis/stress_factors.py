"""Quantify the most influential factors on student stress_level.

Approach:
  1. Split predictors into (a) plausible upstream factors and (b) concurrent
     symptoms/outcomes (anxiety, depression, burnout, MHI, dropout_risk).
  2. Report Spearman ρ for both groups.
  3. Run a multiple OLS on standardized upstream factors → standardized stress.
     Standardized coefficients (β) are directly comparable across predictors
     and account for collinearity between factors.
  4. Cross-check with non-parametric Kruskal–Wallis using quintile bins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("data/student_mental_health_burnout_1M.csv")
TARGET = "stress_level"

# Concurrent psychological state — same time as stress, likely co-determined
SYMPTOMS = [
    "anxiety_score", "depression_score", "burnout_score",
    "mental_health_index", "dropout_risk",
]
# Plausibly upstream / modifiable lifestyle & situation factors
FACTORS = [
    "study_hours_per_day", "exam_pressure", "academic_performance",
    "sleep_hours", "physical_activity", "social_support",
    "screen_time", "internet_usage", "financial_stress",
    "family_expectation",
]
# Demographics (kept as controls, not "factors")
DEMOG = ["age", "academic_year", "gender"]


def main() -> None:
    df = pd.read_csv(
        DATA,
        dtype={"gender": "category", "risk_level": "category"},
        low_memory=False,
    )
    n = len(df)
    print(f"Loaded {n:,} rows × {df.shape[1]} cols\n")

    # ── Spearman correlations with target ───────────────────────────────
    print("=" * 70)
    print("[1] Spearman correlation with stress_level")
    print("=" * 70)

    rho_factors = (
        df[FACTORS].corrwith(df[TARGET], method="spearman")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    rho_sympt = (
        df[SYMPTOMS].corrwith(df[TARGET], method="spearman")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    print("\nUpstream factors (lifestyle / situational):")
    for k, v in rho_factors.items():
        print(f"  {k:<24s}  ρ = {v:+.3f}")
    print("\nConcurrent symptoms (NOT upstream — co-determined with stress):")
    for k, v in rho_sympt.items():
        print(f"  {k:<24s}  ρ = {v:+.3f}")

    # ── Standardized OLS: factors only ──────────────────────────────────
    print("\n" + "=" * 70)
    print("[2] Standardized OLS — factors only (β = std-coef, β² = unique R²)")
    print("=" * 70)

    X = df[FACTORS].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.float64)

    Xz = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    yz = (y - y.mean()) / y.std(ddof=0)

    Xd = np.column_stack([np.ones(n), Xz])
    beta, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
    intercept, coefs = beta[0], beta[1:]

    yhat = Xd @ beta
    r2_full = 1.0 - ((yz - yhat) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()

    # Per-predictor unique contribution: drop column k, refit, ΔR²
    deltas = []
    for k in range(len(FACTORS)):
        Xd_k = np.delete(Xd, k + 1, axis=1)
        b_k, *_ = np.linalg.lstsq(Xd_k, yz, rcond=None)
        yhat_k = Xd_k @ b_k
        r2_k = 1.0 - ((yz - yhat_k) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
        deltas.append(r2_full - r2_k)

    table = (
        pd.DataFrame({
            "factor": FACTORS,
            "spearman_rho": rho_factors.reindex(FACTORS).values,
            "std_beta": coefs,
            "delta_R2": deltas,
        })
        .assign(abs_beta=lambda d: d["std_beta"].abs())
        .sort_values("abs_beta", ascending=False)
        .drop(columns="abs_beta")
        .reset_index(drop=True)
    )
    print(f"\nFull-model R² (factors only) = {r2_full:.4f}")
    print(table.round(4).to_string(index=False))

    # ── Effect size via group means (top 5 factors only) ────────────────
    print("\n" + "=" * 70)
    print("[3] Effect size: stress at low vs high quintile of each factor")
    print("=" * 70)
    rows = []
    for f in table["factor"].head(8):
        s = df[f]
        bins = pd.qcut(s, q=5, labels=False, duplicates="drop")
        g_lo = df.loc[bins == bins.min(), TARGET]
        g_hi = df.loc[bins == bins.max(), TARGET]
        h, p = stats.kruskal(*[df.loc[bins == b, TARGET].values for b in bins.dropna().unique()])
        rows.append({
            "factor": f,
            "stress @ Q1 mean": g_lo.mean(),
            "stress @ Q5 mean": g_hi.mean(),
            "Δ (Q5 − Q1)": g_hi.mean() - g_lo.mean(),
            "kruskal_H": h,
            "p < 1e-300?": p < 1e-300,
        })
    eff = pd.DataFrame(rows).round(3)
    print(eff.to_string(index=False))

    # ── Headline ────────────────────────────────────────────────────────
    top = table.iloc[0]
    print("\n" + "=" * 70)
    print("HEADLINE")
    print("=" * 70)
    print(
        f"\nMost influential upstream factor on stress_level:\n"
        f"   → {top['factor']}\n"
        f"     standardized β = {top['std_beta']:+.3f}\n"
        f"     unique ΔR²     = {top['delta_R2']:.4f}\n"
        f"     Spearman ρ     = {top['spearman_rho']:+.3f}\n"
    )


if __name__ == "__main__":
    main()
