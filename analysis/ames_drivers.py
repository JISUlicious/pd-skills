"""Find the most influential feature on Ames Housing SalePrice.

Refined from the initial run. Five corrections:
  1. MAE-in-dollars formula uses correct log-space conversion.
  2. `Sale Condition` removed (encodes post-sale information → leakage).
  3. De Cock's documented 5 outliers removed (per data dictionary).
  4. Linear baseline one-hot encodes categoricals so the linear-vs-ML
     comparison is apples-to-apples.
  5. Permutation importance uses n_repeats=5 on the full holdout.

Also exercises the new skill features:
  - target-skew check + log-transform recommendation
  - dtype-aware null_audit (Spearman / point-biserial / Cramér's V)
  - rating-tautology surfacing: re-run with `Overall Qual` removed.
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA = Path("data/ames_housing.csv")
TARGET = "SalePrice"

# Per the Ames data dictionary: NaN means "no pool"/"no garage"/etc.
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

# De Cock (2011) §5: the 5 instructional outliers.
# Three Partial Sales (Sale Condition=Partial) with Gr Liv Area > 4000
# plus two unusual non-Partial sales (Gr Liv Area > 4000).
# Practical filter:
DE_COCK_OUTLIER_FILTER = "`Gr Liv Area` > 4000"

# Post-sale leakage columns (drop before modeling)
LEAKAGE_COLS = ["Sale Condition", "Sale Type", "Mo Sold", "Yr Sold"]


# ── functions copied verbatim from feature-importance.md ────────────────
def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xv = X.to_numpy(dtype=np.float64)
    rows = []
    for j, col in enumerate(X.columns):
        y = Xv[:, j]
        Xrest = np.delete(Xv, j, axis=1)
        Xd = np.column_stack([np.ones(len(Xrest)), Xrest])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        yhat = Xd @ beta
        ss_tot = ((y - y.mean()) ** 2).sum()
        if ss_tot == 0:
            r2, vif = 1.0, float("inf")
        else:
            r2 = 1 - ((y - yhat) ** 2).sum() / ss_tot
            vif = float("inf") if r2 >= 0.9999 else 1 / (1 - r2)
        rows.append({"feature": col, "R²_on_others": round(float(r2), 3),
                     "VIF": round(float(vif), 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def null_audit(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """dtype-aware: Spearman ρ for numeric y, point-biserial for binary,
    Cramér's V for multiclass."""
    is_num = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    is_bin = pd.api.types.is_bool_dtype(y) or (is_num and y.nunique() == 2)
    rows = []
    for col in X.columns:
        miss_pct = float(X[col].isna().mean())
        if miss_pct == 0:
            rows.append({"feature": col, "null_pct": 0.0, "miss_target_assoc": 0.0})
            continue
        is_miss = X[col].isna().astype(int)
        if is_bin:
            r, _ = stats.pointbiserialr(is_miss, pd.Series(y).astype(int))
            assoc = float(r)
        elif is_num:
            assoc = float(is_miss.corr(y, method="spearman"))
        else:
            ct = pd.crosstab(is_miss, y)
            chi2, *_ = stats.chi2_contingency(ct)
            n = ct.sum().sum()
            denom = n * max(min(ct.shape) - 1, 1)
            assoc = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
        rows.append({"feature": col, "null_pct": round(miss_pct, 4),
                     "miss_target_assoc": round(assoc, 4)})
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False)


def null_action(null_pct: float, assoc: float) -> str:
    if null_pct == 0: return "none"
    if null_pct < 0.01: return "drop_rows"
    if abs(assoc) >= 0.05:
        return "indicator+impute (informative)"
    if null_pct < 0.10: return "median-impute"
    if null_pct < 0.50: return "indicator+impute"
    return "remove_column"


def fit_linear(X: pd.DataFrame, y: pd.Series):
    """Standardized OLS — returns (R², coefs Series)."""
    Xv = X.to_numpy(dtype=np.float64)
    yv = y.to_numpy(dtype=np.float64)
    sd = Xv.std(axis=0, ddof=0)
    Xz = (Xv - Xv.mean(axis=0)) / np.where(sd > 0, sd, 1)
    yz = (yv - yv.mean()) / yv.std(ddof=0)
    Xd = np.column_stack([np.ones(len(Xz)), Xz])
    beta, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
    yhat = Xd @ beta
    r2 = 1 - ((yz - yhat) ** 2).sum() / ((yz - yz.mean()) ** 2).sum()
    return float(r2), pd.Series(beta[1:], index=X.columns)


def aggregate_onehot_beta(coefs: pd.Series, source_map: dict[str, str]) -> pd.Series:
    """Sum |β| across one-hot dummies back to the source column name."""
    out = {}
    for col, beta in coefs.items():
        src = source_map.get(col, col)
        out[src] = out.get(src, 0.0) + abs(beta)
    return pd.Series(out)


def main() -> None:
    df = pd.read_csv(DATA, index_col=0)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols\n")

    # ── 1. Outlier removal (De Cock §5) ─────────────────────────────────
    print("=" * 76)
    print("[1] Remove De Cock's 5 documented outliers")
    print("=" * 76)
    n_before = len(df)
    df = df.query(f"~({DE_COCK_OUTLIER_FILTER})").reset_index(drop=True)
    print(f"  {n_before:,} → {len(df):,} rows ({n_before - len(df)} removed)")

    # ── 2. Drop leakage columns (post-sale info) ────────────────────────
    print(f"\n  Dropping post-sale leakage columns: {LEAKAGE_COLS}")
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    # ── 3. Structural-absence handling ──────────────────────────────────
    pre_nulls = int(df.isna().sum().sum())
    for c in ABSENCE_CAT:
        if c in df.columns: df[c] = df[c].fillna("None")
    for c in ABSENCE_NUM:
        if c in df.columns: df[c] = df[c].fillna(0)
    post_nulls = int(df.isna().sum().sum())
    print(f"\n  Structural-absence fills: {pre_nulls - post_nulls:,} cells")
    print(f"  Remaining true-nulls:     {post_nulls}")

    # ── 4. Target skew + log transform ──────────────────────────────────
    print("\n" + "=" * 76)
    print("[2] Target skew check")
    print("=" * 76)
    y_raw = df[TARGET]
    print(f"  SalePrice skew (raw):     {y_raw.skew():+.2f}")
    if y_raw.skew() > 1 and (y_raw > 0).all():
        y = np.log1p(y_raw)
        print(f"  → applying log1p(y); new skew: {y.skew():+.2f}")
    else:
        y = y_raw.copy()

    X_full = df.drop(columns=[TARGET, "PID", "Order"], errors="ignore")
    num_cols = X_full.select_dtypes(include="number").columns.tolist()
    cat_cols = X_full.select_dtypes(exclude="number").columns.tolist()
    print(f"  Numeric features: {len(num_cols)}, Categorical: {len(cat_cols)}")

    # ── 5. null_audit (dtype-aware) ─────────────────────────────────────
    print("\n" + "=" * 76)
    print("[3] null_audit() — dtype-aware association measure")
    print("=" * 76)
    audit = null_audit(X_full, y)
    nz = audit.query("null_pct > 0").copy()
    nz["action"] = [null_action(p, a) for p, a in zip(nz["null_pct"], nz["miss_target_assoc"])]
    print(nz.to_string(index=False) if len(nz) else "  No remaining nulls.")

    # apply actions
    for _, row in nz.iterrows():
        col = row["feature"]
        if col not in num_cols:
            X_full[col] = X_full[col].fillna("Missing")
            continue
        if row["action"].startswith("indicator"):
            X_full[f"{col}_is_missing"] = X_full[col].isna().astype("int8")
        X_full[col] = X_full[col].fillna(X_full[col].median())
    assert X_full.isna().sum().sum() == 0
    num_cols = X_full.select_dtypes(include="number").columns.tolist()
    cat_cols = X_full.select_dtypes(exclude="number").columns.tolist()

    # ── 6. VIF on numeric features ──────────────────────────────────────
    print("\n" + "=" * 76)
    print("[4] vif_table()")
    print("=" * 76)
    vif = vif_table(X_full[num_cols])
    sev = vif.query("VIF > 10")["feature"].tolist()
    mod = vif.query("5 < VIF <= 10")["feature"].tolist()
    print(f"  Severe (VIF > 10):  {len(sev)} cols  →  {sev[:5]}{'...' if len(sev)>5 else ''}")
    print(f"  Moderate (5–10):    {mod}")
    print("  → β on severe features is uninterpretable; use SHAP/permutation rankings.")

    # ── 7. Linear baseline (FAIR — one-hot encoded) ─────────────────────
    print("\n" + "=" * 76)
    print("[5] Fair linear baseline (numeric + one-hot categoricals)")
    print("=" * 76)
    X_lin = pd.get_dummies(X_full, columns=cat_cols, dtype="int8", drop_first=True)
    # Map each one-hot dummy back to its source column
    source_map = {}
    for src in cat_cols:
        prefix = f"{src}_"
        for c in X_lin.columns:
            if c.startswith(prefix): source_map[c] = src
    r2_lin_full, coefs_lin = fit_linear(X_lin, y)
    print(f"  Linear (numeric + one-hot): R² = {r2_lin_full:.4f}")
    print(f"  Linear (numeric only):      R² = {fit_linear(X_full[num_cols], y)[0]:.4f}")
    src_imp_lin = aggregate_onehot_beta(coefs_lin, source_map).sort_values(ascending=False)
    print("  Top 10 by aggregated |β| (one-hot dummies summed back to source):")
    print(src_imp_lin.head(10).round(3).to_string())

    # ── 8. ML: XGBoost native categorical + permutation + SHAP ──────────
    print("\n" + "=" * 76)
    print("[6] XGBoost + permutation + SHAP")
    print("=" * 76)
    X_ml = X_full.copy()
    for c in cat_cols:
        X_ml[c] = X_ml[c].astype("category")
    X_train, X_test, y_train, y_test = train_test_split(
        X_ml, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, tree_method="hist", n_jobs=-1,
        enable_categorical=True,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    r2_ml = r2_score(y_test, pred)
    mae_log = mean_absolute_error(y_test, pred)
    mae_dollars = y_raw.median() * (np.exp(mae_log) - 1)            # FIXED formula
    print(f"  Holdout R² = {r2_ml:.4f}, MAE(log) = {mae_log:.4f}")
    print(f"  → ~{mae_log:.1%} typical relative error  (~${mae_dollars:,.0f} on median home)")

    # permutation: full holdout, n_repeats=5
    print("  Computing permutation importance (full holdout, n_repeats=5)...")
    perm = permutation_importance(model, X_test, y_test, n_repeats=5,
                                  random_state=42, n_jobs=-1, scoring="r2")
    perm_imp = pd.Series(perm.importances_mean, index=X_ml.columns)
    gain = pd.Series(model.feature_importances_, index=X_ml.columns)

    # SHAP on full holdout (~580 rows is fine)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test)
    shap_mean = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_ml.columns)

    # ── 9. Cross-method ranking ─────────────────────────────────────────
    print("\n" + "=" * 76)
    print("[7] Cross-method ranking (top 15 by SHAP)")
    print("=" * 76)
    table = pd.DataFrame({
        "spearman_|rho|": X_full.select_dtypes("number").corrwith(y, method="spearman").abs(),
        "lin_|β|_agg": src_imp_lin,
        "xgb_gain": gain,
        "perm_importance": perm_imp,
        "shap_|mean|": shap_mean,
    })
    out = table.sort_values("shap_|mean|", ascending=False).head(15)
    print(out.round(4).to_string())

    # ── 10. Rating-tautology check: drop Overall Qual, re-fit ───────────
    print("\n" + "=" * 76)
    print("[8] Rating-tautology check: refit without 'Overall Qual'")
    print("=" * 76)
    if "Overall Qual" in X_ml.columns:
        X_train2 = X_train.drop(columns="Overall Qual")
        X_test2  = X_test.drop(columns="Overall Qual")
        model2 = xgb.XGBRegressor(**model.get_params()).fit(
            X_train2, y_train, eval_set=[(X_test2, y_test)], verbose=False
        )
        r2_drop = r2_score(y_test, model2.predict(X_test2))
        print(f"  R² with Overall Qual:    {r2_ml:.4f}")
        print(f"  R² without Overall Qual: {r2_drop:.4f}")
        print(f"  Marginal contribution:   ΔR² = {r2_ml - r2_drop:+.4f}")
        # What replaces it?
        explainer2 = shap.TreeExplainer(model2)
        shap2 = pd.Series(np.abs(explainer2.shap_values(X_test2)).mean(axis=0),
                          index=X_test2.columns).sort_values(ascending=False)
        print("\n  Top 5 SHAP without Overall Qual (vs. previous top 5):")
        for f, v in shap2.head(5).items():
            prev_rank = list(shap_mean.sort_values(ascending=False).index).index(f) + 1 \
                        if f in shap_mean.index else None
            print(f"    {f:<20s}  SHAP={v:.3f}  (was rank {prev_rank})")

    # ── 11. Headline ────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("HEADLINE")
    print("=" * 76)
    top5 = shap_mean.sort_values(ascending=False).head(5)
    print(f"\n  Holdout R²: {r2_ml:.3f}  (linear-baseline R²: {r2_lin_full:.3f})")
    print(f"  MAE: ~{mae_log:.1%} relative error  (~${mae_dollars:,.0f} on median home)")
    print(f"\n  Top 5 drivers of log(SalePrice) by SHAP:")
    for i, (feat, val) in enumerate(top5.items(), 1):
        rho = table["spearman_|rho|"].get(feat)
        rho_str = f"|ρ|={rho:.2f}" if not pd.isna(rho) else "(categorical)"
        print(f"    {i}. {feat:<24s}  SHAP={val:.3f}   {rho_str}")


if __name__ == "__main__":
    main()
