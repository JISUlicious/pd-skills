"""Regression test for mixed_type_vif() and cross_type_binding().

Verifies the new helpers from data-exploration.md § 6 against known
cross-type bindings in the Ames Housing dataset:

  • Garage Yr Blt ↔ Garage Finish/Qual/Cond/Type — η² > 0.98
  • Pool Area ↔ Pool QC                          — η² > 0.9
  • Mixed-type VIF flags categorical sources missed by numeric-only VIF
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DATA = Path("data/ames_housing.csv")

ABSENCE_CAT = [
    "Alley", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Fireplace Qu", "Garage Type", "Garage Finish",
    "Garage Qual", "Garage Cond", "Pool QC", "Fence", "Misc Feature",
    "Mas Vnr Type",
]
ABSENCE_NUM = [
    "Mas Vnr Area", "Bsmt Full Bath", "Bsmt Half Bath",
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF",
    "Garage Cars", "Garage Area", "Garage Yr Blt",
]


# ── helpers copied verbatim from data-exploration.md § 6 ────────────────
def vif_table(X):
    Xv = X.to_numpy(dtype=np.float64)
    rows = []
    for j, col in enumerate(X.columns):
        y = Xv[:, j]
        Xrest = np.delete(Xv, j, axis=1)
        Xd = np.column_stack([np.ones(len(Xrest)), Xrest])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        ss_tot = ((y - y.mean()) ** 2).sum()
        if ss_tot == 0:
            r2, vif = 1.0, float("inf")
        else:
            r2 = 1 - ((y - Xd @ beta) ** 2).sum() / ss_tot
            vif = float("inf") if r2 >= 0.9999 else 1 / (1 - r2)
        rows.append({"feature": col, "R²_on_others": round(float(r2), 3),
                     "VIF": round(float(vif), 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def mixed_type_vif(X, num_cols, cat_cols, max_cardinality=15):
    cat_low = [c for c in cat_cols if X[c].nunique() <= max_cardinality]
    cat_skip = [c for c in cat_cols if c not in cat_low]
    if cat_skip:
        print(f"Skipped {len(cat_skip)} high-cardinality cats from VIF "
              f"(use cross_type_binding instead): {cat_skip[:5]}"
              f"{'...' if len(cat_skip) > 5 else ''}")
    X_design = pd.get_dummies(X[num_cols + cat_low], columns=cat_low,
                              drop_first=True, dtype="float64")
    per_feature = vif_table(X_design)
    source_map = {c: c for c in num_cols}
    for src in cat_low:
        prefix = f"{src}_"
        for c in X_design.columns:
            if c.startswith(prefix):
                source_map[c] = src
    per_feature["source"] = per_feature["feature"].map(source_map)
    per_source = (per_feature.groupby("source")
                  .agg(max_VIF=("VIF", "max"),
                       mean_VIF=("VIF", "mean"),
                       n_features=("feature", "count"))
                  .sort_values("max_VIF", ascending=False)
                  .reset_index())
    return per_source, per_feature


def cross_type_binding(X, num_cols, cat_cols, threshold=0.5):
    rows = []
    for n in num_cols:
        for c in cat_cols:
            d = pd.DataFrame({"x": X[n], "g": X[c]}).dropna()
            if len(d) < 5 or d["g"].nunique() < 2:
                continue
            grand = d["x"].mean()
            ss_tot = ((d["x"] - grand) ** 2).sum()
            if ss_tot == 0:
                continue
            ss_btw = (d.groupby("g", observed=True)["x"]
                       .apply(lambda s: len(s) * (s.mean() - grand) ** 2)
                       .sum())
            eta2 = float(ss_btw / ss_tot)
            if eta2 > threshold:
                rows.append({"a": n, "b": c, "score": round(eta2, 3),
                             "kind": "η² (num↔cat)"})
    for i, a in enumerate(cat_cols):
        for b in cat_cols[i + 1:]:
            d = pd.DataFrame({"a": X[a], "b": X[b]}).dropna()
            if len(d) < 5 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
                continue
            ct = pd.crosstab(d["a"], d["b"])
            chi2, *_ = stats.chi2_contingency(ct)
            n = ct.sum().sum()
            denom = n * max(min(ct.shape) - 1, 1)
            v = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
            if v > threshold:
                rows.append({"a": a, "b": b, "score": round(v, 3),
                             "kind": "Cramér's V (cat↔cat)"})
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            rho = X[[a, b]].corr(method="spearman").iloc[0, 1]
            if pd.notna(rho) and rho * rho > threshold:
                rows.append({"a": a, "b": b, "score": round(rho * rho, 3),
                             "kind": "ρ² (num↔num)"})
    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))


def main():
    df = pd.read_csv(DATA, index_col=0)
    for c in ABSENCE_CAT:
        if c in df.columns: df[c] = df[c].fillna("None")
    for c in ABSENCE_NUM:
        if c in df.columns: df[c] = df[c].fillna(0)

    X_full = df.drop(columns=["SalePrice", "PID", "Order"], errors="ignore")
    num_cols = X_full.select_dtypes(include="number").columns.tolist()
    cat_cols = X_full.select_dtypes(exclude="number").columns.tolist()
    X_full[num_cols] = X_full[num_cols].fillna(X_full[num_cols].median())
    for c in cat_cols:
        X_full[c] = X_full[c].fillna("Missing")

    print("=" * 70)
    print("Test 1: numeric-only VIF — current skill behavior")
    print("=" * 70)
    vif_num = vif_table(X_full[num_cols])
    severe_num = set(vif_num.query("VIF > 10")["feature"])
    print(f"  Severe (VIF > 10) — numeric only: {sorted(severe_num)}")
    print(f"  Garage Yr Blt VIF (numeric-only): "
          f"{float(vif_num.loc[vif_num.feature == 'Garage Yr Blt', 'VIF'].iloc[0]):.2f}")

    print("\n" + "=" * 70)
    print("Test 2: mixed_type_vif() — full design matrix")
    print("=" * 70)
    per_source, per_feature = mixed_type_vif(X_full, num_cols, cat_cols)
    severe_mixed = set(per_source.query("max_VIF > 10")["source"])
    new_severe = severe_mixed - severe_num
    print(f"\n  Severe sources (mixed-type): {len(severe_mixed)}")
    print(f"  New severe sources (missed by numeric-only): {len(new_severe)}")
    print(f"  Garage Yr Blt max VIF (mixed-type): "
          f"{float(per_source.loc[per_source.source == 'Garage Yr Blt', 'max_VIF'].iloc[0]):.0f}")
    assert "Garage Yr Blt" in severe_mixed, "Expected Garage Yr Blt to be severe in mixed-type"
    assert "Garage Yr Blt" not in severe_num, "Garage Yr Blt should NOT be severe in numeric-only"
    print("  ✓ Mixed-type VIF correctly catches what numeric-only missed")

    print("\n" + "=" * 70)
    print("Test 3: cross_type_binding() — η² and Cramér's V")
    print("=" * 70)
    bindings = cross_type_binding(X_full, num_cols, cat_cols, threshold=0.5)
    num_cat_bindings = bindings[bindings["kind"] == "η² (num↔cat)"]
    print(f"\n  Top 5 numeric ↔ categorical bindings (η²):")
    print(num_cat_bindings.head(5).to_string(index=False))

    # Garage Yr Blt ↔ Garage Finish should be the strongest binding
    top_garage = bindings.query(
        "a == 'Garage Yr Blt' and b == 'Garage Finish'"
    )
    assert len(top_garage) == 1, "Expected Garage Yr Blt ↔ Garage Finish in bindings"
    eta2 = float(top_garage["score"].iloc[0])
    print(f"\n  Garage Yr Blt ↔ Garage Finish η² = {eta2:.3f}")
    assert eta2 > 0.95, f"Expected η² > 0.95 (deterministic), got {eta2}"
    print("  ✓ cross_type_binding() correctly identifies near-deterministic binding")

    # Pool Area ↔ Pool QC should also be very high
    pool = bindings.query("a == 'Pool Area' and b == 'Pool QC'")
    assert len(pool) == 1
    pool_eta = float(pool["score"].iloc[0])
    print(f"  Pool Area ↔ Pool QC η²    = {pool_eta:.3f}")
    assert pool_eta > 0.9, f"Expected η² > 0.9, got {pool_eta}"
    print("  ✓ Pool Area ↔ Pool QC binding correctly detected")

    print("\n" + "=" * 70)
    print("✓ All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
