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

## Step 3.5 — Data Provenance Check (Synthetic Data Detection)

Before diving into analysis, sanity-check whether the data looks **real** or
**synthetic/simulated**. Flag as likely synthetic if **3 or more** of these
are true:

- Zero missing values across all columns
- Zero duplicate rows
- Numeric columns have suspiciously round min/max bounds (e.g., exactly 0–100)
- Continuous columns have cardinality == row count (every value unique)
- Distributions are perfectly uniform or perfectly normal
- No date/timestamp column (real-world surveys/transactions almost always have one)
- Categorical columns have suspiciously balanced class frequencies

```python
flags = []
if df.isnull().sum().sum() == 0:
    flags.append("zero missing values")
if df.duplicated().sum() == 0:
    flags.append("zero duplicates")
num = df.select_dtypes(include="number")
if len(num) and ((num.min() % 1 == 0).all() and (num.max() % 1 == 0).all()):
    flags.append("round numeric bounds")
if any(df[c].nunique() == len(df) for c in num.columns):
    flags.append("continuous column with unique-per-row values")
if not len(df.select_dtypes(include=["datetime64", "datetimetz"]).columns):
    flags.append("no datetime column")

if len(flags) >= 3:
    print(f"⚠️  LIKELY SYNTHETIC DATA — flags: {flags}")
    print("   Statistical inferences may not generalize to real populations.")
```

If flagged: **report this prominently in your findings**. Statistical tests
and model results on synthetic data do not transfer to real populations.

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

```python
import numpy as np

# Skewness & kurtosis for all numeric columns
numeric_cols = df.select_dtypes(include="number").columns
print(df[numeric_cols].skew().sort_values())
print(df[numeric_cols].kurt().sort_values())

# Outlier detection via IQR
def iqr_bounds(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in numeric_cols:
    lo, hi = iqr_bounds(df[col])
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out > 0:
        print(f"{col}: {n_out} outliers (IQR method)")

# Z-score outliers
from scipy import stats
z_scores = np.abs(stats.zscore(df[numeric_cols].dropna()))
outlier_mask = (z_scores > 3).any(axis=1)
print(f"Rows with |z| > 3 in any column: {outlier_mask.sum()}")
```

## Step 6 — Correlation Analysis

```python
# Pearson correlation matrix
corr = df[numeric_cols].corr()

# High correlations (> 0.8, excluding diagonal)
high_corr = (corr.abs()
    .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
    .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "corr"})
    .query("corr > 0.8")
    .sort_values("corr", ascending=False)
)
print(high_corr)

# Spearman (rank-based, handles non-linear)
df[numeric_cols].corr(method="spearman")

# Correlation with a target variable
df[numeric_cols].corrwith(df["target"]).sort_values()
```

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

## Step 9 — Quick EDA Report (one-liner)

```python
# Using ydata-profiling (formerly pandas-profiling)
# pip install ydata-profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="EDA Report", explorative=True)
profile.to_file("eda_report.html")
```

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

## EDA Checklist

- [ ] Shape, memory, dtypes confirmed
- [ ] Missing values quantified per column
- [ ] Duplicate rows detected
- [ ] Summary statistics reviewed (mean, median, std, quartiles)
- [ ] Outliers identified (IQR or Z-score)
- [ ] Skewness / kurtosis noted for modeling considerations
- [ ] Correlations above 0.8 flagged
- [ ] Cardinality checked (constants and ID-like columns)
- [ ] Date range and coverage verified
- [ ] Target variable distribution examined (if supervised task)
