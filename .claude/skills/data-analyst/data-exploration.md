# Pandas Data Exploration & EDA — Expert Skill

You are an expert data analyst. Apply this structured EDA methodology whenever
exploring an unfamiliar dataset with pandas >= 2.3.

## Step 0 — First Contact: Schema Discovery

When loading an unknown dataset, do a two-pass load: sample first to discover
the schema, then reload with optimized dtypes.

```python
# Pass 1: load a small sample with no dtype to discover schema
sample = pd.read_csv("file.csv", nrows=1000)
print(sample.dtypes)
print(sample.nunique())

# Decide categoricals (low-cardinality string columns)
str_cols = sample.select_dtypes(include=["object", "str"]).columns
cat_cols = [c for c in str_cols if sample[c].nunique() < 50]

# Pass 2: reload the full file with optimized dtypes
dtype_map = {c: "category" for c in cat_cols}
df = pd.read_csv("file.csv", dtype=dtype_map, low_memory=False)
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols, "
      f"{df.memory_usage(deep=True).sum()/1e6:.1f} MB")
```

For very large files (> 1GB), use `chunksize=` and aggregate iteratively
(see `data-loading.md`).

## Step 1 — Structural Overview

```python
# Shape and memory
print(f"Shape: {df.shape}")                  # (rows, cols)
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Column types
print(df.dtypes)
print(df.info(memory_usage="deep"))          # compact full overview

# First / last rows
df.head(10)
df.tail(5)
df.sample(5, random_state=42)               # random rows (reproducible)
```

## Step 2 — Missing Data Audit

```python
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    "count": missing,
    "pct": missing_pct
}).query("count > 0").sort_values("pct", ascending=False)
print(missing_report)

# Visual pattern (requires missingno library, optional)
# import missingno as msno; msno.matrix(df)

# Which rows have ANY missing value?
df[df.isnull().any(axis=1)]
```

## Step 3 — Duplicates

```python
n_dupes = df.duplicated().sum()
print(f"Duplicate rows: {n_dupes} ({n_dupes/len(df)*100:.1f}%)")

# Duplicates on key subset
df.duplicated(subset=["user_id", "date"]).sum()

# Inspect duplicates
df[df.duplicated(keep=False)].sort_values(["user_id", "date"])
```

## Step 3.5 — Data-Quality Signature

Report a neutral readout of quality signals. These are **informative**, not a
binary synthetic-or-real verdict: well-curated real datasets (cleaned research
data, fact tables, feature stores) can hit several of these. When multiple
indicators fire, just verify provenance before generalizing conclusions.

```python
num = df.select_dtypes(include="number")
indicators = {
    "no missing values":       df.isnull().sum().sum() == 0,
    "no duplicates":           df.duplicated().sum() == 0,
    "no datetime column":      len(df.select_dtypes(include=["datetime64","datetimetz"]).columns) == 0,
    "integer-bounded numerics": len(num.columns) > 0 and (num.min() % 1 == 0).all() and (num.max() % 1 == 0).all(),
    "unique-per-row numerics": any(df[c].nunique() == len(df) for c in num.columns),
}
for key, val in indicators.items():
    print(f"  [{'x' if val else ' '}] {key}")
if sum(indicators.values()) >= 3:
    print("  → Verify provenance before generalizing statistical conclusions.")
```

## Step 4 — Descriptive Statistics

```python
# Numeric columns
df.describe()                    # count, mean, std, min, quartiles, max
df.describe(percentiles=[.05, .25, .5, .75, .95])  # custom percentiles

# Categorical / object columns
df.describe(include=["object", "str", "category"])

# All columns
df.describe(include="all")

# Per-column stats
df["amount"].agg(["mean", "median", "std", "skew", "kurt"])

# Value counts for categoricals
for col in df.select_dtypes(include=["object", "str", "category"]).columns:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(10))
    print(f"  Unique: {df[col].nunique()}")
```

## Step 5 — Distribution Analysis

### Step 5a — Point-mass / clipping detection (run BEFORE outlier checks)

Many real columns — revenue (non-purchasers), counts, time-censored values,
bounded surveys — have large fractions of rows pinned at min or max. On these
columns **IQR and moment-based outlier stats are misleading** (they flag the
right tail of a truncated distribution as "outliers"). Always check for point
masses first and skip IQR where present.

```python
numeric_cols = df.select_dtypes(include="number").columns
point_mass = {}
for col in numeric_cols:
    vmin, vmax = df[col].min(), df[col].max()
    at_min = (df[col] == vmin).mean()
    at_max = (df[col] == vmax).mean()
    if at_min > 0.01 or at_max > 0.01:
        point_mass[col] = {"at_min": at_min, "at_max": at_max}
        print(f"  {col}: {at_min:.1%} at min({vmin:.3g}), "
              f"{at_max:.1%} at max({vmax:.3g})")
```

### Step 5b — Skew & kurtosis

```python
import numpy as np
dist = pd.DataFrame({
    "skew": df[numeric_cols].skew(),
    "kurt": df[numeric_cols].kurt(),
})
print(dist.sort_values("skew", key=lambda s: s.abs(), ascending=False).round(3))
```

### Step 5c — IQR outliers (skipping point-mass columns)

```python
def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in numeric_cols:
    if col in point_mass:
        continue   # IQR misleading on clipped distributions
    lo, hi = iqr_bounds(df[col])
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out:
        print(f"{col}: {n_out} outliers (IQR method)")

# Z-score outliers (same caveat — skip point-mass columns)
from scipy import stats
ok_cols = [c for c in numeric_cols if c not in point_mass]
z = np.abs(stats.zscore(df[ok_cols].dropna()))
print(f"Rows with |z| > 3 in any non-clipped column: {(z > 3).any(axis=1).sum()}")
```

For distribution-shape testing (Shapiro, KS, D'Agostino-Pearson), see
`statistical-analysis.md`.

## Step 6 — Correlation Analysis

**Prefer Spearman by default** for EDA. It's rank-based, so it handles
skewed, clipped, or zero-inflated columns without distortion. Use Pearson
only when you have evidence the relationships are linear.

```python
# Spearman pairs with |rho| > 0.3
corr = df[numeric_cols].corr(method="spearman")
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
pairs = (corr.where(mask).stack()
    .reset_index()
    .rename(columns={"level_0": "a", "level_1": "b", 0: "rho"}))
strong = pairs.loc[pairs["rho"].abs() > 0.3].sort_values(
    "rho", key=lambda s: s.abs(), ascending=False
)
print(strong.round(3))

# Correlation with a target variable (when the user specified one)
df[numeric_cols].corrwith(df["target"], method="spearman").sort_values()
```

### Uncorrelated columns (neutral report)

Columns that show |ρ| < 0.05 against **every** other numeric column deserve a
second look. They may be IDs, independent factors, the target itself, or
noise. Report them — do **not** auto-drop.

```python
self_masked = corr.abs().where(~np.eye(len(corr), dtype=bool))
loners = self_masked.max()[self_masked.max() < 0.05].index.tolist()
if loners:
    print(f"Uncorrelated with all others: {loners}")
    print("  → verify whether these are IDs, independent factors, or targets")
```

## Step 6.5 — Derived-Column / Target-Leakage Check

A categorical column that is strictly a function of some numeric column is a
common form of **target leakage** in supervised pipelines — and a common
source of silly model-accuracy claims. Detect it by binning each numeric
column to match the categorical's cardinality and measuring row-level purity
of the resulting cross-tab.

```python
cat_cols = df.select_dtypes(include=["category", "object", "str"]).columns
derived = []
for cat in cat_cols:
    card = df[cat].nunique()
    if card < 2 or card > 20:
        continue
    for nc in numeric_cols:
        try:
            bins = pd.qcut(df[nc], q=card, labels=False, duplicates="drop")
        except ValueError:
            continue
        purity = (pd.crosstab(bins, df[cat], normalize="index")
                  .max(axis=1).mean())
        if purity > 0.90:
            derived.append((cat, nc, round(float(purity), 3)))

for cat, nc, p in sorted(derived, key=lambda x: -x[2]):
    print(f"  {cat} ≈ f({nc})   row-purity={p}")
```

If any pair shows row-purity > 0.9, flag it: the categorical is (nearly) a
deterministic function of the numeric — do not use it as a feature alongside
the numeric when predicting anything downstream of that numeric.

## Step 7 — Temporal Overview (if dates present)

```python
# Detect date columns
date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns

for col in date_cols:
    print(f"\n{col}:")
    print(f"  Range: {df[col].min()} → {df[col].max()}")
    print(f"  Span:  {df[col].max() - df[col].min()}")
    print(f"  NaT:   {df[col].isnull().sum()}")

# Time-based record counts
if len(date_cols) > 0:
    date_col = date_cols[0]
    print(df.set_index(date_col).resample("ME").size())   # monthly counts
```

## Step 8 — Cardinality & Type Audit

```python
cardinality = df.nunique().sort_values()
print(cardinality)

# Columns where all values are the same (useless)
constant_cols = cardinality[cardinality == 1].index.tolist()
print(f"Constant columns (drop candidates): {constant_cols}")

# Columns with very high cardinality (likely IDs)
id_like = cardinality[cardinality == len(df)].index.tolist()
print(f"Unique-per-row columns (ID candidates): {id_like}")
```

## Step 8.5 — Memory Optimization

After profiling, optimize dtypes before heavier analysis. This pays off
substantially on million-row datasets.

```python
mem_before = df.memory_usage(deep=True).sum() / 1e6

# 1. Convert low-cardinality string columns to category
for col in df.select_dtypes(include=["object", "str"]).columns:
    if df[col].nunique() / len(df) < 0.05:        # < 5% unique
        df[col] = df[col].astype("category")

# 2. Downcast numeric columns where range permits
for col in df.select_dtypes(include="integer").columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")
for col in df.select_dtypes(include="float").columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

mem_after = df.memory_usage(deep=True).sum() / 1e6
print(f"Memory: {mem_before:.1f} MB → {mem_after:.1f} MB "
      f"({(1 - mem_after/mem_before):.0%} reduction)")
```

## Step 9 — One-Shot EDA Helper

The skill ships a runnable helper at `eda.py` (same directory as this file)
that executes every step above in order, guards IQR on point-mass columns,
runs Spearman correlations, performs the derived-column check, and returns
a `findings` dict plus a printed checklist. **Prefer this over cherry-picking
steps by hand** — it prevents the "forgot to run outliers" failure mode.

```python
# Library usage
from eda import eda_report
findings = eda_report(df, target="burnout_score")   # target is optional

# CLI usage (two-pass load with automatic category inference)
#   python eda.py data/file.csv [target_column]
```

Ask the user for the target column when it isn't obvious. Do **not** guess
it from column names — domain-specific targets (`conversion`, `churned`,
`nps`) defeat name heuristics and guessing wastes a pass.

For an HTML report with histograms, the optional `ydata-profiling` library
still works: `ProfileReport(df).to_file("eda.html")`.

## Pandas Built-in Plotting for EDA

```python
import matplotlib.pyplot as plt

# Distribution of all numeric columns
df[numeric_cols].hist(bins=30, figsize=(14, 8))
plt.tight_layout(); plt.show()

# Correlation heatmap
import seaborn as sns
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)

# Box plots for outlier detection
df[numeric_cols].plot(kind="box", subplots=True, figsize=(14, 6))

# Value counts bar chart
df["category"].value_counts().plot(kind="bar")
```

## EDA Checklist (Runnable)

Call `eda_report(df, target=...)` and it returns a `findings["checklist"]`
dict. For manual runs, verify every item yourself before handing results
back to the user:

```python
def eda_checklist(findings: dict) -> list[str]:
    """Return the list of unchecked items from an eda_report findings dict."""
    return [k for k, v in findings["checklist"].items() if not v]

unchecked = eda_checklist(findings)
assert not unchecked, f"EDA incomplete: {unchecked}"
```

The checklist items enforced by `eda_report`:

- Shape / memory / dtypes reported
- Missing values quantified per column
- Duplicate rows counted
- Descriptive stats for numeric columns
- **Point-mass columns reported (before outlier detection)**
- Outliers reported (IQR, skipping point-mass columns)
- Skew / kurtosis reported
- Pairwise Spearman correlations > 0.3 reported
- Cardinality checked (constants and ID-like columns)
- Target variable analyzed (if one was provided)
