"""Regression test for select_cluster_representative() and find_aggregate().

Verifies the priority-ordered selector from data-exploration.md § 6 against
known cases in the Ames Housing dataset:

  1. Sum decomposition (Gr Liv Area = 1st Flr + 2nd Flr + Low Qual Fin SF)
     → Priority 1, with the largest-mean tiebreaker that resolves the
     ambiguity in mutually predictable clusters.
  2. Sum decomposition for basement (Total Bsmt SF = parts).
  3. Quality-rating cluster (Overall Qual + per-component qualities)
     → Priority 2 (summary-name match).
  4. Garage cluster (Cars vs Area) → default Priority 3 |ρ(target)|.
  5. RCA-style score_fn override on a synthetic shift.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data/ames_housing.csv")
SUMMARY_PATTERNS = ("overall", "total", "index", "score", "rating", "summary")


def find_aggregate(X, cluster_cols, threshold=0.99):
    Xc = X[cluster_cols].dropna()
    if len(Xc) < len(cluster_cols) + 1:
        return None
    candidates = []
    for col in cluster_cols:
        others = [c for c in cluster_cols if c != col]
        Xrest = Xc[others].to_numpy(dtype=np.float64)
        y = Xc[col].to_numpy(dtype=np.float64)
        Xd = np.column_stack([np.ones(len(Xrest)), Xrest])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        ss_tot = ((y - y.mean()) ** 2).sum()
        if ss_tot == 0:
            continue
        r2 = 1 - ((y - Xd @ beta) ** 2).sum() / ss_tot
        if r2 >= threshold:
            candidates.append((col, abs(float(Xc[col].mean()))))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[1])
    return candidates[0][0]


def select_cluster_representative(X, y, cluster_cols, *, fold_mask=None, score_fn=None):
    Xf = X.loc[fold_mask] if fold_mask is not None else X
    yf = (y.loc[fold_mask] if (y is not None and fold_mask is not None)
          else (y if y is not None else None))
    agg = find_aggregate(Xf, cluster_cols)
    if agg is not None:
        return ([c for c in cluster_cols if c != agg], [agg],
                f"aggregate dropped: {agg}")
    summaries = [c for c in cluster_cols
                 if any(p in c.lower() for p in SUMMARY_PATTERNS)]
    components = [c for c in cluster_cols if c not in summaries]
    if summaries and components:
        return components, summaries, f"summary names dropped: {summaries}"
    if score_fn is None:
        score_fn = ((lambda Xf, _y, c: float(Xf[c].std())) if yf is None
                    else (lambda Xf, yf, c: abs(
                        float(Xf[c].corr(yf, method="spearman"))
                    )))
    scores = (pd.Series({c: score_fn(Xf, yf, c) for c in cluster_cols})
                .sort_values(ascending=False))
    keep = scores.index[0]
    return ([keep], [c for c in cluster_cols if c != keep],
            f"kept {keep} (score={scores[keep]:.3f})")


def main():
    df = pd.read_csv(DATA, index_col=0)
    y = np.log1p(df["SalePrice"])
    print(f"Loaded {len(df):,} rows × {df.shape[1]} cols\n")

    print("=" * 70)
    print("Cluster-representative selector tests")
    print("=" * 70)

    # 1, 2 — sum decompositions
    for name, cluster, expected_drop in [
        ("Floor-SF sum decomposition",
         ["1st Flr SF", "2nd Flr SF", "Low Qual Fin SF", "Gr Liv Area"],
         ["Gr Liv Area"]),
        ("Basement-SF sum decomposition",
         ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Total Bsmt SF"],
         ["Total Bsmt SF"]),
    ]:
        keep, drop, reason = select_cluster_representative(df, y, cluster)
        ok = drop == expected_drop
        print(f"\n[{'✓' if ok else '✗'}] {name}")
        print(f"     cluster: {cluster}")
        print(f"     reason:  {reason}")
        print(f"     dropped: {drop}  (expected {expected_drop})")
        assert ok

    # 3 — summary-name match
    qual_map = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5, "None": 0}
    df_q = df.copy()
    for c in ("Exter Qual", "Bsmt Qual", "Kitchen Qual"):
        df_q[c] = df_q[c].fillna("None").map(qual_map)
    qcluster = ["Overall Qual", "Exter Qual", "Bsmt Qual", "Kitchen Qual"]
    keep3, drop3, reason3 = select_cluster_representative(df_q, y, qcluster)
    ok = "Overall Qual" in drop3
    print(f"\n[{'✓' if ok else '✗'}] Quality-rating summary")
    print(f"     reason:  {reason3}")
    print(f"     dropped: {drop3}")
    assert ok

    # 4 — Garage Cars vs Area (Priority 3 default)
    gcluster = ["Garage Cars", "Garage Area"]
    keep4, drop4, reason4 = select_cluster_representative(df, y, gcluster)
    rho_cars = df["Garage Cars"].corr(y, method="spearman")
    rho_area = df["Garage Area"].corr(y, method="spearman")
    expected_keep = "Garage Cars" if abs(rho_cars) > abs(rho_area) else "Garage Area"
    ok = keep4[0] == expected_keep
    print(f"\n[{'✓' if ok else '✗'}] Garage cluster (Priority 3 ρ-target)")
    print(f"     ρ(target): Cars={rho_cars:.3f}, Area={rho_area:.3f}")
    print(f"     reason:  {reason4}")
    print(f"     kept:    {keep4}")
    assert ok

    # 5 — RCA-style score_fn override
    pre = df.iloc[:1500]
    post = df.iloc[1500:].copy()
    post["1st Flr SF"] = post["1st Flr SF"] + 200
    full = pd.concat([pre, post])
    rcluster = ["1st Flr SF", "2nd Flr SF", "Low Qual Fin SF"]

    def shift_score(_Xf, _y, col):
        a = pre[col].dropna()
        b = post[col].dropna()
        return abs(b.mean() - a.mean()) / max(a.std(), 1e-9)

    keep5, drop5, reason5 = select_cluster_representative(
        full, None, rcluster, score_fn=shift_score
    )
    ok = keep5[0] == "1st Flr SF"
    print(f"\n[{'✓' if ok else '✗'}] RCA-style override (synthetic shift)")
    print(f"     reason:  {reason5}")
    print(f"     kept:    {keep5}  (expected '1st Flr SF' — engineered to shift)")
    assert ok

    print("\n" + "=" * 70)
    print("✓ All tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
