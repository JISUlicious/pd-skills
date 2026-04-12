"""One-shot EDA report for a pandas DataFrame.

Usage as a library:
    from eda import eda_report
    findings = eda_report(df, target="my_target")

Usage as a script (two-pass load with category inference):
    python eda.py path/to/data.csv [target_column]

The report runs all steps from data-exploration.md and returns a dict of
findings. It is defensive about zero-inflated / clipped columns (skips IQR
where it would be misleading) and uses Spearman correlation by default.
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd


def eda_report(df: pd.DataFrame, target: Optional[str] = None) -> dict:
    """Run the full EDA workflow and return a dict of findings."""
    findings: dict = {}
    n, k = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"{'='*72}\nEDA Report: {n:,} rows × {k} cols, {mem_mb:.1f} MB\n{'='*72}")
    findings["shape"] = (n, k)
    findings["memory_mb"] = round(mem_mb, 1)

    # ── [1] Structure ──────────────────────────────────────────────────
    print("\n[1] Dtypes")
    print(df.dtypes.to_string())

    # ── [2] Missing ────────────────────────────────────────────────────
    missing = df.isnull().sum()
    missing_pct = (missing / n * 100).round(2)
    mreport = (pd.DataFrame({"count": missing, "pct": missing_pct})
               .query("count > 0")
               .sort_values("pct", ascending=False))
    print(f"\n[2] Missing: {int(missing.sum()):,} cells across {len(mreport)} columns")
    if len(mreport):
        print(mreport.to_string())
    findings["missing_cols"] = mreport.index.tolist()

    # ── [3] Duplicates ─────────────────────────────────────────────────
    n_dupes = int(df.duplicated().sum())
    print(f"\n[3] Duplicate rows: {n_dupes:,} ({n_dupes/n*100:.2f}%)")
    findings["duplicates"] = n_dupes

    # ── [3.5] Data-quality signature (neutral) ─────────────────────────
    num = df.select_dtypes(include="number")
    indicators = {
        "no missing values":       int(missing.sum()) == 0,
        "no duplicates":           n_dupes == 0,
        "no datetime column":      len(df.select_dtypes(include=["datetime64", "datetimetz"]).columns) == 0,
        "integer-bounded numerics": False,
        "unique-per-row numerics": False,
    }
    if len(num.columns):
        indicators["integer-bounded numerics"] = bool(
            (num.min() % 1 == 0).all() and (num.max() % 1 == 0).all()
        )
        indicators["unique-per-row numerics"] = any(df[c].nunique() == n for c in num.columns)
    print("\n[3.5] Data-quality signature")
    for key, val in indicators.items():
        print(f"  [{'x' if val else ' '}] {key}")
    if sum(indicators.values()) >= 3:
        print("  → Multiple indicators present. Verify data provenance before")
        print("    generalizing any statistical conclusions.")
    findings["quality_signature"] = indicators

    # ── [4] Describe ───────────────────────────────────────────────────
    print("\n[4] describe()")
    if len(num.columns):
        print(df.describe().round(3).to_string())
    for col in df.select_dtypes(include=["object", "str", "category"]).columns:
        vc = df[col].value_counts(dropna=False).head(10)
        print(f"\n  {col} (unique={df[col].nunique()})")
        print(vc.to_string())

    # ── [5a] Point-mass detection ──────────────────────────────────────
    # Real and synthetic datasets both produce clipped/zero-inflated columns
    # (revenue, counts, ReLU-style generators). IQR and moment-based stats
    # mislead when > ~1% of rows sit at the min or max.
    print("\n[5a] Point-mass check (>1% of rows at min or max)")
    point_mass: dict = {}
    for col in num.columns:
        vmin, vmax = df[col].min(), df[col].max()
        at_min = float((df[col] == vmin).mean())
        at_max = float((df[col] == vmax).mean())
        if at_min > 0.01 or at_max > 0.01:
            point_mass[col] = {"at_min": at_min, "min": float(vmin),
                                "at_max": at_max, "max": float(vmax)}
            print(f"  {col}: {at_min:.1%} @ min({vmin:.3g}), "
                  f"{at_max:.1%} @ max({vmax:.3g})")
    if not point_mass:
        print("  (none)")
    findings["point_mass"] = point_mass

    # ── [5b] Skew / kurtosis ───────────────────────────────────────────
    if len(num.columns):
        print("\n[5b] Skew / kurtosis (sorted by |skew|)")
        dist = pd.DataFrame({"skew": num.skew(), "kurt": num.kurt()})
        dist["abs"] = dist["skew"].abs()
        print(dist.sort_values("abs", ascending=False)
                  .drop(columns="abs").round(3).to_string())

    # ── [5c] IQR outliers (skipping point-mass columns) ────────────────
    print("\n[5c] IQR outliers  (point-mass columns skipped — IQR misleading)")
    iqr_rows = []
    for col in num.columns:
        if col in point_mass:
            continue
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_out:
            iqr_rows.append((col, n_out, n_out / n * 100))
    if iqr_rows:
        for col, n_out, pct in sorted(iqr_rows, key=lambda x: -x[1]):
            print(f"  {col}: {n_out:,} ({pct:.2f}%)")
    else:
        skipped = len(point_mass)
        print(f"  (none outside point-mass columns; {skipped} skipped)")
    findings["iqr_outliers"] = {c: n for c, n, _ in iqr_rows}

    # ── [6] Correlation (Spearman — robust to clipping/skew) ───────────
    if len(num.columns) >= 2:
        corr = num.corr(method="spearman")
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        pairs = (corr.where(mask).stack()
                     .reset_index()
                     .rename(columns={"level_0": "a", "level_1": "b", 0: "rho"}))
        strong = pairs.loc[pairs["rho"].abs() > 0.3].sort_values(
            "rho", key=lambda s: s.abs(), ascending=False
        )
        print("\n[6] Spearman pairs with |rho| > 0.3")
        if len(strong):
            print(strong.round(3).to_string(index=False))
        else:
            print("  (none)")
        findings["high_corr"] = strong.round(3).to_dict("records")

        # Neutral "loners" report — columns uncorrelated with all others
        self_masked = corr.abs().where(~np.eye(len(corr), dtype=bool))
        max_abs = self_masked.max()
        loners = max_abs[max_abs < 0.05].index.tolist()
        if loners:
            print(f"\n  Columns with |rho| < 0.05 against all others: {loners}")
            print("  → may be IDs, independent factors, the target itself, or noise.")
            print("    Verify before drawing conclusions.")
        findings["loners"] = loners

    # ── [6.5] Derived-column / target-leakage check ────────────────────
    print("\n[6.5] Derived-column check (categoricals that are a function of a numeric)")
    cat_cols = df.select_dtypes(include=["category", "object", "str"]).columns
    derived = []
    for cat in cat_cols:
        card = df[cat].nunique()
        if card < 2 or card > 20:
            continue
        for nc in num.columns:
            try:
                bins = pd.qcut(df[nc], q=card, labels=False, duplicates="drop")
            except ValueError:
                continue
            purity = float(pd.crosstab(bins, df[cat], normalize="index")
                               .max(axis=1).mean())
            if purity > 0.90:
                derived.append((cat, nc, round(purity, 3)))
    if derived:
        for cat, nc, p in sorted(derived, key=lambda x: -x[2])[:10]:
            print(f"  {cat} ≈ f({nc})   row-purity={p}")
        print("  → For supervised tasks, treat these as potential target leakage.")
    else:
        print("  (none)")
    findings["derived_columns"] = derived

    # ── [7] Cardinality ────────────────────────────────────────────────
    card = df.nunique().sort_values()
    print("\n[7] Cardinality")
    constants = card[card == 1].index.tolist()
    ids = card[card == n].index.tolist()
    print(f"  Constant cols:  {constants}")
    print(f"  Unique-per-row: {ids}")
    findings["constant_cols"] = constants
    findings["id_cols"] = ids

    # ── [8] Target view (if specified) ─────────────────────────────────
    if target is not None:
        if target not in df.columns:
            print(f"\n[8] Target {target!r} not found in columns")
        else:
            print(f"\n[8] Target: {target}")
            t = df[target]
            if pd.api.types.is_numeric_dtype(t):
                print(t.describe().round(3).to_string())
                others = num.drop(columns=[target], errors="ignore")
                corrs = others.corrwith(t, method="spearman").round(3)
                corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
                print("\n  Spearman correlation with target:")
                print(corrs.to_string())
                findings["target_corr"] = corrs.to_dict()
            else:
                vc = t.value_counts(dropna=False)
                print(vc.to_string())
                findings["target_counts"] = vc.to_dict()

    # ── Checklist ──────────────────────────────────────────────────────
    checks = {
        "shape/memory/dtypes":    True,
        "missing quantified":     True,
        "duplicates detected":    True,
        "descriptive stats":      len(num.columns) > 0,
        "point-mass reported":    True,
        "outliers reported":      True,
        "skew/kurtosis reported": len(num.columns) > 0,
        "correlations reported":  len(num.columns) >= 2,
        "cardinality reported":   True,
        "target analyzed":        target is not None and target in df.columns,
    }
    print("\n[Checklist]")
    for key, val in checks.items():
        print(f"  [{'x' if val else ' '}] {key}")
    unchecked = [k for k, v in checks.items() if not v]
    if unchecked:
        print(f"  → Unchecked: {unchecked}")
    findings["checklist"] = checks
    return findings


def _two_pass_load(path: str) -> pd.DataFrame:
    """Load a CSV with category inference from a 1000-row sample."""
    sample = pd.read_csv(path, nrows=1000)
    str_cols = sample.select_dtypes(include=["object", "str"]).columns
    cat_cols = [c for c in str_cols if sample[c].nunique() < 50]
    dtype_map = {c: "category" for c in cat_cols}
    return pd.read_csv(path, dtype=dtype_map, low_memory=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]
    tgt = sys.argv[2] if len(sys.argv) > 2 else None
    eda_report(_two_pass_load(path), target=tgt)
