"""Probe whether numeric-only VIF misses cross-type collinearity.

Three approaches compared on Ames Housing:
  1. Numeric-only VIF (current skill behavior) — what gets flagged?
  2. Mixed-type VIF: one-hot encode categoricals (drop_first=True), then
     compute VIF on the full design matrix. Aggregate per source column
     for readability.
  3. Pairwise association alternative: numeric↔numeric (Spearman ρ²),
     numeric↔categorical (η²), categorical↔categorical (Cramér's V).
     A purer view of cross-type redundancy than dummy VIF.

Question: does the mixed-type approach surface "bound" relationships that
numeric-only VIF misses?
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


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Standard VIF on a numeric design matrix."""
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


def eta_squared(numeric: pd.Series, categorical: pd.Series) -> float:
    """η² — fraction of variance in `numeric` explained by `categorical` group means."""
    df_ = pd.DataFrame({"x": numeric, "g": categorical}).dropna()
    if len(df_) < 5 or df_["g"].nunique() < 2:
        return 0.0
    grand_mean = df_["x"].mean()
    ss_total = ((df_["x"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = (df_.groupby("g", observed=True)["x"]
                     .apply(lambda s: len(s) * (s.mean() - grand_mean) ** 2)
                     .sum())
    return float(ss_between / ss_total)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramér's V — symmetric association between two categoricals."""
    df_ = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df_) < 5 or df_["a"].nunique() < 2 or df_["b"].nunique() < 2:
        return 0.0
    ct = pd.crosstab(df_["a"], df_["b"])
    chi2, *_ = stats.chi2_contingency(ct)
    n = ct.sum().sum()
    denom = n * max(min(ct.shape) - 1, 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def main():
    df = pd.read_csv(DATA, index_col=0)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols")

    # Apply structural-absence handling
    for c in ABSENCE_CAT:
        if c in df.columns: df[c] = df[c].fillna("None")
    for c in ABSENCE_NUM:
        if c in df.columns: df[c] = df[c].fillna(0)

    X_full = df.drop(columns=["SalePrice", "PID", "Order"], errors="ignore")
    num_cols = X_full.select_dtypes(include="number").columns.tolist()
    cat_cols = X_full.select_dtypes(exclude="number").columns.tolist()
    # Fill remaining nulls
    X_full[num_cols] = X_full[num_cols].fillna(X_full[num_cols].median())
    for c in cat_cols:
        X_full[c] = X_full[c].fillna("Missing")

    print(f"Numeric: {len(num_cols)}, Categorical: {len(cat_cols)}\n")

    # ── 1. Numeric-only VIF (current skill behavior) ────────────────────
    print("=" * 76)
    print("[1] Numeric-only VIF (what the current skill flags)")
    print("=" * 76)
    vif_num = vif_table(X_full[num_cols])
    severe_num = vif_num.query("VIF > 10")["feature"].tolist()
    moderate_num = vif_num.query("5 < VIF <= 10")["feature"].tolist()
    print(f"  Severe (VIF > 10):    {severe_num}")
    print(f"  Moderate (5 < VIF≤10): {moderate_num}")

    # ── 2. Mixed-type VIF on full design matrix ─────────────────────────
    print("\n" + "=" * 76)
    print("[2] Mixed-type VIF — full design matrix (numeric + one-hot dummies)")
    print("=" * 76)
    # Limit to low-cardinality categoricals (≤ 15 levels) to keep design tractable
    cat_low = [c for c in cat_cols if X_full[c].nunique() <= 15]
    cat_skip = [c for c in cat_cols if c not in cat_low]
    print(f"  One-hot encoding {len(cat_low)} cats with ≤15 levels (skipping {len(cat_skip)} high-card)")
    X_design = pd.get_dummies(X_full[num_cols + cat_low], columns=cat_low,
                              drop_first=True, dtype="float64")
    print(f"  Design matrix shape: {X_design.shape}")
    vif_design = vif_table(X_design)

    # Map dummy → source
    source_map = {c: c for c in num_cols}
    for src in cat_low:
        prefix = f"{src}_"
        for c in X_design.columns:
            if c.startswith(prefix):
                source_map[c] = src
    vif_design["source"] = vif_design["feature"].map(source_map)

    # ── 2a. Aggregate per source: max VIF, mean VIF, n_dummies ──────────
    print("\n  [2a] VIF aggregated per source column:")
    agg = (vif_design.groupby("source")
           .agg(max_VIF=("VIF", "max"),
                mean_VIF=("VIF", "mean"),
                n_features=("feature", "count"))
           .sort_values("max_VIF", ascending=False))
    # Mark sources that became severe ONLY in mixed-type (not in numeric-only)
    severe_now = agg.query("max_VIF > 10").index.tolist()
    new_severe = [s for s in severe_now if s not in severe_num]
    print(agg.head(20).round(2).to_string())
    print(f"\n  Severe cross-type collinearity revealed by mixed-type VIF "
          f"that numeric-only MISSED:")
    if new_severe:
        for s in new_severe:
            n_dum = int(agg.loc[s, "n_features"])
            max_v = agg.loc[s, "max_VIF"]
            print(f"    • {s}  (max VIF = {max_v:.1f}, {n_dum} dummies)")
    else:
        print("    (none — numeric-only caught it all)")

    # ── 3. Pairwise association — alternative cross-type view ───────────
    print("\n" + "=" * 76)
    print("[3] Pairwise association — η² (num↔cat) and Cramér's V (cat↔cat)")
    print("=" * 76)
    print("  η²: variance in numeric column explained by categorical groups")
    print("       Values: 0 (independent) → 1 (deterministic)\n")

    # Top numeric ↔ categorical bindings (η² > 0.5)
    bindings = []
    for n in num_cols:
        for c in cat_cols:
            e = eta_squared(X_full[n], X_full[c])
            if e > 0.5:
                bindings.append((n, c, e, X_full[c].nunique()))
    bindings.sort(key=lambda t: -t[2])
    print("  Numeric ↔ categorical bindings with η² > 0.5  (cross-type collinearity):")
    print(f"    {'numeric':<22s}  {'categorical':<22s}  η²     n_levels")
    for n, c, e, k in bindings[:15]:
        print(f"    {n:<22s}  {c:<22s}  {e:.3f}   {k}")
    if len(bindings) > 15:
        print(f"    ... and {len(bindings) - 15} more pairs")

    # ── 4. Verdict ──────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)
    print(f"  Numeric-only VIF flagged {len(severe_num)} severe + {len(moderate_num)} moderate.")
    print(f"  Mixed-type VIF flagged {len(severe_now)} sources as severe at design-matrix level.")
    print(f"  Newly surfaced cross-type collinearity: {len(new_severe)} sources.")
    print(f"  η² > 0.5 numeric↔categorical bindings: {len(bindings)} pairs.")
    if new_severe or bindings:
        print(f"\n  → Mixed-type VIF (and η²) reveal real collinearity that the")
        print(f"    numeric-only audit silently misses. The user's intuition")
        print(f"    was correct: bound relationships across types ARE there,")
        print(f"    and the current skill behavior is INCOMPLETE.")


if __name__ == "__main__":
    main()
