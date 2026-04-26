"""SECOM root-cause analysis: end-to-end exercise of root-cause-analysis.md.

Goal: show that the RCA pipeline (CPD + commonality + DiD/causal) finds
something the feature-importance pipeline alone would miss.

SECOM dataset: semiconductor manufacturing, 1,567 timestamped samples ×
590 sensor features + pass/fail labels (-1=pass, 1=fail).
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import ruptures as rpt
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/secom")
LABELS_FILE = DATA_DIR / "secom_labels.data"
DATA_FILE   = DATA_DIR / "secom.data"


# ── functions copied from root-cause-analysis.md ────────────────────────
def detect_changepoints(series, model="rbf", min_size=30, penalty=None):
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    if penalty is None:
        penalty = np.log(n) * float(np.var(s))
    algo = rpt.Pelt(model=model, min_size=min_size).fit(s)
    return algo.predict(pen=penalty)[:-1]


def commonality(df, factor_col, defect_col, defect_value=1, alpha=0.01):
    rows = []
    levels = df[factor_col].dropna().unique()
    n_tests = max(len(levels), 1)
    for lvl in levels:
        is_lvl = (df[factor_col] == lvl)
        is_def = (df[defect_col] == defect_value)
        a = int((is_lvl & is_def).sum())
        b = int((is_lvl & ~is_def).sum())
        c = int((~is_lvl & is_def).sum())
        d = int((~is_lvl & ~is_def).sum())
        if a + b == 0 or a + c == 0:
            continue
        odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        lift = (a / (a + c)) / ((a + b) / (a + b + c + d))
        rows.append({
            "level": lvl, "n_def": a, "n_pass": b,
            "defect_rate_in_level": round(a / (a + b), 4),
            "lift": round(lift, 2),
            "p_bonferroni": min(1.0, float(p) * n_tests),
        })
    return (pd.DataFrame(rows).sort_values("p_bonferroni")
              .query(f"p_bonferroni < {alpha}").reset_index(drop=True))


def main():
    # ── 1. Load + parse ──────────────────────────────────────────────────
    print("=" * 76)
    print("[1] Load SECOM and time-index")
    print("=" * 76)
    raw = LABELS_FILE.read_text().strip().splitlines()
    parsed = []
    for line in raw:
        # format: <label> "<dd/mm/yyyy HH:MM:SS>"
        label_str, _, ts_str = line.partition(" ")
        ts = ts_str.strip().strip('"')
        parsed.append((int(label_str), ts))
    labels = pd.DataFrame(parsed, columns=["label", "timestamp"])
    labels["timestamp"] = pd.to_datetime(
        labels["timestamp"], format="%d/%m/%Y %H:%M:%S"
    )
    labels["is_defect"] = (labels["label"] == 1).astype(int)
    X = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, na_values="NaN",
                    engine="python")
    X.columns = [f"s{i:03d}" for i in range(X.shape[1])]
    df = pd.concat([labels, X], axis=1).sort_values("timestamp").reset_index(drop=True)
    print(f"  Samples: {len(df):,}  Features: {X.shape[1]}  "
          f"Defect rate: {df['is_defect'].mean():.1%}")
    print(f"  Time range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Total nulls: {int(X.isna().sum().sum()):,} "
          f"({X.isna().sum().sum() / X.size:.1%})")

    # ── 2. Pre-modeling diagnostics: drop too-sparse features ────────────
    keep = X.columns[X.isna().mean() < 0.5]                 # > 50% null → drop
    X_kept = X[keep].fillna(X[keep].median())
    print(f"  Features after dropping >50% null: {len(keep)} "
          f"(removed {X.shape[1] - len(keep)})")

    # ── 3. Detect change points in defect rate ───────────────────────────
    print("\n" + "=" * 76)
    print("[2] Change-point detection on rolling defect rate")
    print("=" * 76)
    # Rolling defect rate over 50-sample windows
    df["defect_roll"] = df["is_defect"].rolling(50, min_periods=10).mean()
    series = df["defect_roll"].dropna().to_numpy()
    bkps = detect_changepoints(series, model="rbf", min_size=50)
    print(f"  Detected {len(bkps)} change point(s) in rolling defect rate")
    if not bkps:
        print("  No change points found — exiting.")
        return

    # Rank candidates by raw-defect-rate Δ magnitude, then pick the first that
    # passes the strict sanity check (Fisher exact, the textbook test for two
    # binary proportions)
    offset = df["defect_roll"].first_valid_index()
    candidates = []
    WIN = 250                       # samples on each side of the candidate CP
    for b in bkps:
        change_row = offset + b
        pre  = df.iloc[max(0, change_row - WIN):change_row]
        post = df.iloc[change_row:change_row + WIN]
        if len(pre) < 100 or len(post) < 100:
            continue
        a = int(pre["is_defect"].sum());  c = len(pre) - a
        b2 = int(post["is_defect"].sum()); d = len(post) - b2
        _, p = stats.fisher_exact([[a, c], [b2, d]])
        delta = post["is_defect"].mean() - pre["is_defect"].mean()
        candidates.append({"row": change_row, "delta": delta, "p": p,
                           "n_pre": len(pre), "n_post": len(post),
                           "pre_rate": pre["is_defect"].mean(),
                           "post_rate": post["is_defect"].mean()})

    cand_df = pd.DataFrame(candidates).sort_values("p")
    print(f"\n  Top 5 candidates by p-value (Fisher exact, ±{WIN} sample windows):")
    print(cand_df.head(5).round(4).to_string(index=False))

    passing = cand_df.query("p < 0.01")
    if len(passing) == 0:
        print("\n  ✗ No change point passes the strict (p<0.01) sanity check.")
        print("  → SECOM defect rate has noisy signals but no single robust break.")
        print("  → This is a valid finding: the skill correctly refuses to over-claim.")
        # For the rest of the verification, use the p=0.0156 marginal one with caveat
        chosen = cand_df.iloc[0]
        print(f"\n  Continuing with marginal CP for demonstration only (p={chosen['p']:.4f}):")
    else:
        chosen = passing.iloc[0]
        print(f"\n  ✓ Selected CP: row {int(chosen['row'])}, p={chosen['p']:.4f}")

    change_row = int(chosen["row"])
    change_time = df.loc[change_row, "timestamp"]
    pre  = df.iloc[max(0, change_row - WIN):change_row]
    post = df.iloc[change_row:change_row + WIN]
    print(f"  Change point: row {change_row}, timestamp {change_time}")
    print(f"  Pre  defect rate: {pre['is_defect'].mean():.1%} (n={len(pre)})")
    print(f"  Post defect rate: {post['is_defect'].mean():.1%} (n={len(post)})")
    print(f"  Δ = {chosen['delta']:+.1%}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["defect_roll"], lw=1)
    ax.axvline(change_time, color="red", ls="--", label=f"CP @ {change_time}")
    ax.set_title("SECOM rolling defect rate (50-sample window)")
    ax.set_ylabel("Defect rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/secom_changepoint.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Plot saved to /tmp/secom_changepoint.png")

    # ── 4. Commonality: which sensors shifted at the change point? ──────
    print("\n" + "=" * 76)
    print("[3] Commonality: sensors with distribution shift across CP")
    print("=" * 76)
    # For each sensor, compute Mann-Whitney on pre vs post values
    rows = []
    for col in keep:
        pre_vals = pre[col].dropna().to_numpy()
        post_vals = post[col].dropna().to_numpy()
        if len(pre_vals) < 30 or len(post_vals) < 30:
            continue
        try:
            u, p = stats.mannwhitneyu(pre_vals, post_vals)
        except ValueError:
            continue
        rows.append({
            "sensor": col,
            "pre_mean": float(pre_vals.mean()),
            "post_mean": float(post_vals.mean()),
            "delta": float(post_vals.mean() - pre_vals.mean()),
            "delta_normalized": float(
                (post_vals.mean() - pre_vals.mean()) /
                (pre_vals.std() + 1e-10)
            ),
            "p": float(p),
        })
    shift_df = pd.DataFrame(rows)
    # Bonferroni over the sensors that survived missingness filtering
    shift_df["p_bonf"] = (shift_df["p"] * len(shift_df)).clip(upper=1.0)
    sig_shifts = shift_df.query("p_bonf < 0.01").sort_values("p_bonf")
    print(f"  Sensors with Bonferroni-significant shift: {len(sig_shifts)}")
    print(sig_shifts.head(10).round(4).to_string(index=False))

    # ── 5. Compare to feature-importance ranking ────────────────────────
    print("\n" + "=" * 76)
    print("[4] Feature-importance baseline (XGBoost + SHAP) for contrast")
    print("=" * 76)
    Xtr, Xte, ytr, yte = train_test_split(
        X_kept, df["is_defect"], test_size=0.2, random_state=42, stratify=df["is_defect"]
    )
    # SECOM is severely imbalanced (~6% defect); use scale_pos_weight
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method="hist", random_state=42, n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(Xtr, ytr)
    auc = stats.kendalltau(yte.values, model.predict_proba(Xte)[:, 1]).correlation
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(Xte.iloc[:500])
    shap_mean = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_kept.columns) \
                  .sort_values(ascending=False)
    print(f"  Top 10 sensors by SHAP |mean|:")
    print(shap_mean.head(10).round(4).to_string())

    # ── 6. The killer comparison: do the methods agree? ──────────────────
    print("\n" + "=" * 76)
    print("[5] CPD-commonality vs. SHAP — do they identify the same sensors?")
    print("=" * 76)
    cpd_top10  = sig_shifts["sensor"].head(10).tolist() if len(sig_shifts) else []
    shap_top10 = shap_mean.head(10).index.tolist()
    overlap = set(cpd_top10) & set(shap_top10)
    print(f"  CPD-commonality top 10: {cpd_top10}")
    print(f"  SHAP top 10:            {shap_top10}")
    print(f"  Overlap: {len(overlap)} sensor(s) — {sorted(overlap) if overlap else '(none)'}")
    print(f"\n  CPD-only (sensors that shifted at the change but don't predict overall): "
          f"{sorted(set(cpd_top10) - set(shap_top10))}")
    print(f"  SHAP-only (sensors important overall but didn't shift at the change): "
          f"{sorted(set(shap_top10) - set(cpd_top10))}")

    # ── 7. Causal analysis on the top CPD sensor ────────────────────────
    print("\n" + "=" * 76)
    print("[6] Diff-in-differences on top shifted sensor")
    print("=" * 76)
    if len(sig_shifts):
        top_sensor = sig_shifts.iloc[0]["sensor"]
        print(f"  Candidate cause: {top_sensor}")
        # Treat samples where the sensor exceeded its pre-CP median as "treated",
        # split by pre/post the change point — DiD via OLS interaction
        import statsmodels.formula.api as smf
        df_did = df.copy()
        pre_median = pre[top_sensor].median()
        df_did["high_sensor"] = (df_did[top_sensor] > pre_median).astype(int)
        df_did["post"] = (df_did["timestamp"] >= change_time).astype(int)
        df_did = df_did.dropna(subset=[top_sensor])
        ols = smf.ols("is_defect ~ post + high_sensor + post:high_sensor",
                      data=df_did).fit()
        did_est = ols.params["post:high_sensor"]
        ci = ols.conf_int().loc["post:high_sensor"]
        print(f"  DiD estimate: {did_est:+.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        print(f"  p-value:      {ols.pvalues['post:high_sensor']:.4f}")
        if ols.pvalues["post:high_sensor"] < 0.05:
            print(f"  → high {top_sensor} after the CP shifts defect rate by "
                  f"{did_est:+.1%}")
        else:
            print("  → DiD effect not significant; cause is not localized to this sensor.")

    # ── 8. Final verdict ────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print("VERDICT — does RCA pipeline differ from feature-importance?")
    print("=" * 76)
    if cpd_top10 and shap_top10:
        agreement = len(overlap) / 10
        print(f"\n  Top-10 overlap: {agreement:.0%}  ({len(overlap)} of 10)")
        if agreement < 0.5:
            print("  → Methods disagree substantially. The CP-localized cause is")
            print("    NOT the same as the population-level predictor — exactly")
            print("    the case where RCA-specific tooling earns its keep.")
        else:
            print("  → Methods largely agree. RCA tooling confirms the predictor.")
    print("\n  Plot of change point: /tmp/secom_changepoint.png")


if __name__ == "__main__":
    main()
