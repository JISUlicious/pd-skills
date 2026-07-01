# Pandas 데이터 탐색 및 EDA — 전문가 스킬

당신은 전문 데이터 분석가입니다. pandas >= 2.3으로 익숙하지 않은 데이터셋을
탐색할 때마다 이 구조화된 EDA 방법론을 적용하십시오.

## 목차

- Step 0 — 스키마 발견
- Step 1 — 구조 개요
- Step 2 — 결측 데이터 감사
- Step 3 / 3.5 — 중복 + 데이터 품질 시그니처
- Step 4 — 기술 통계
- Step 5 — 분포 분석 (포인트 매스, 왜도, 이상치)
- Step 6 — 상관계수 분석  →  공선성/VIF 기계장치는
  `collinearity-diagnostics.md`
- Step 6.5 — 파생 칼럼 / 타겟 누수 검사
- Step 7 — 시간적 개요
- Step 8 / 8.5 — 카디널리티 + 메모리 최적화
- Step 9 — 원샷 EDA 스크립트
- EDA 체크리스트 (Explore 후 모든 항목 검증)

## Step 0 — 첫 접촉: 스키마 발견

알 수 없는 데이터셋을 로드할 때는 두 단계 로드를 수행합니다. 먼저 샘플을
로드하여 스키마를 파악한 뒤, 최적화된 dtype으로 다시 로드합니다.

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

매우 큰 파일(> 1GB)의 경우 `chunksize=`를 사용하여 반복적으로 집계합니다
(자세한 내용은 `data-loading.md` 참조).

## Step 1 — 구조 개요

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

## Step 2 — 결측 데이터 감사

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

## Step 3 — 중복

```python
n_dupes = df.duplicated().sum()
print(f"Duplicate rows: {n_dupes} ({n_dupes/len(df)*100:.1f}%)")

# Duplicates on key subset
df.duplicated(subset=["user_id", "date"]).sum()

# Inspect duplicates
df[df.duplicated(keep=False)].sort_values(["user_id", "date"])
```

## Step 3.5 — 데이터 품질 시그니처

품질 신호에 대한 중립적인 판독 결과를 보고합니다. 이는 **정보를 제공하는**
것이지, 합성/실제 데이터 여부에 대한 이진 판정이 아닙니다. 잘 큐레이션된 실제
데이터셋(정제된 연구 데이터, 팩트 테이블, 피처 스토어)도 이 중 여러 항목에
해당할 수 있습니다. 여러 지표가 동시에 발화하면, 결론을 일반화하기 전에 출처를
검증하면 됩니다.

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

## Step 4 — 기술 통계

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

## Step 5 — 분포 분석

### Step 5a — 포인트 매스 / 클리핑 탐지 (이상치 검사 BEFORE 실행)

많은 실제 칼럼들 — 매출(비구매자), 카운트, 시간 검열 값, 경계가 있는 설문조사
— 은 최솟값 또는 최댓값에 큰 비율의 행이 고정되어 있습니다. 이러한 칼럼에서는
**IQR과 모멘트 기반의 이상치 통계가 오해를 불러일으킵니다** (잘린 분포의 오른쪽
꼬리를 "이상치"로 표시함). 항상 포인트 매스를 먼저 확인하고, 존재하는 경우
IQR을 건너뛰십시오.

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

### Step 5b — 왜도 & 첨도

```python
import numpy as np
dist = pd.DataFrame({
    "skew": df[numeric_cols].skew(),
    "kurt": df[numeric_cols].kurt(),
})
print(dist.sort_values("skew", key=lambda s: s.abs(), ascending=False).round(3))
```

### Step 5c — IQR 이상치 (포인트 매스 칼럼은 건너뛰기)

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

분포 형태 검정(Shapiro, KS, D'Agostino-Pearson)에 대해서는
`statistical-analysis.md`를 참조하십시오.

## Step 6 — 상관계수 분석

EDA에서는 **기본적으로 Spearman을 선호하십시오**. 순위 기반이므로 왜곡된,
클리핑된, 또는 영-팽창 칼럼을 왜곡 없이 처리합니다. 관계가 선형임을 입증할 수
있을 때만 Pearson을 사용하십시오.

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

### 무상관 칼럼 (중립 보고)

**모든** 다른 숫자 칼럼에 대해 |ρ| < 0.05를 보이는 칼럼은 한 번 더 살펴볼 필요가
있습니다. ID, 독립적인 요인, 타겟 자체, 또는 노이즈일 수 있습니다. 이들을
보고하되 — 자동으로 제거하지 **마십시오**.

```python
self_masked = corr.abs().where(~np.eye(len(corr), dtype=bool))
loners = self_masked.max()[self_masked.max() < 0.05].index.tolist()
if loners:
    print(f"Uncorrelated with all others: {loners}")
    print("  → verify whether these are IDs, independent factors, or targets")
```

### 다중공선성, 혼합 타입 결합 & 클러스터 선택

`collinearity-diagnostics.md`로 이동했습니다 (VIF 감사, 혼합 타입
η² / Cramér's V, 5-우선순위 `select_cluster_representative()` 선택기).
여기 Step 6에서 실행하세요 — 이는 Explore의 일부이며 모델링으로 미루는
단계가 아닙니다.

## Step 6.5 — 파생 칼럼 / 타겟 누수 검사

어떤 숫자 칼럼의 엄격한 함수인 범주형 칼럼은 지도 학습 파이프라인에서 흔한
형태의 **타겟 누수**이며 — 어이없는 모델 정확도 주장의 흔한 원인입니다. 각 숫자
칼럼을 범주형의 카디널리티에 맞추어 비닝하고 결과 교차표의 행 수준 순도를
측정하여 탐지하십시오.

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

어떤 쌍이 행 순도 > 0.9를 보이면 표시하십시오. 범주형이 (거의) 숫자형의 결정론적
함수라는 의미입니다 — 그 숫자에 의해 후속으로 결정되는 무언가를 예측할 때 이
범주형을 숫자형과 함께 피처로 사용하지 마십시오.

## Step 7 — 시간적 개요 (날짜가 있는 경우)

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

## Step 8 — 카디널리티 & 타입 감사

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

## Step 8.5 — 메모리 최적화

프로파일링 후, 더 무거운 분석에 앞서 dtype을 최적화하십시오. 백만 행 단위
데이터셋에서는 이것이 상당히 효과적입니다.

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

## Step 9 — 원샷 EDA 스크립트 생성

새로운 데이터셋을 마주할 때, 위의 모든 단계를 순서대로 실행하는 **맞춤형 EDA
스크립트를 생성하십시오**. 단계를 골라서 빼지 마십시오. 가장 흔한 실패 모드는
이상치를 계산하기 전에 포인트 매스 검사를 잊거나, 타겟 분포 검증을 잊는
것입니다.

재사용 가능한 템플릿:

```python
def eda_report(df: pd.DataFrame, target: str | None = None) -> dict:
    """Run the full EDA workflow above. Returns dict with shape, memory,
    missing_cols, duplicates, point_mass, iqr_outliers, spearman_pairs,
    cardinality, target_analysis, and a checklist of completed steps."""
    findings = {}
    # 1. Structure (Step 1)
    findings["shape"] = df.shape
    findings["memory_mb"] = df.memory_usage(deep=True).sum() / 1e6
    # 2. Missing (Step 2) — store columns with > 0 nulls
    findings["missing_cols"] = df.columns[df.isnull().any()].tolist()
    # 3. Duplicates (Step 3)
    findings["duplicates"] = int(df.duplicated().sum())
    # 3.5. Quality signature (Step 3.5)
    # 4. describe() (Step 4)
    # 5a. Point-mass detection (CRITICAL — must run before 5c)
    # 5b. Skew/kurtosis (Step 5b)
    # 5c. IQR outliers (Step 5c — skip point_mass columns!)
    # 6. Spearman correlations > 0.3 (Step 6)
    # 6.5. Derived-column check (Step 6.5)
    # 7. Temporal overview if datetime present (Step 7)
    # 8. Cardinality + ID detection (Step 8)
    # Target-specific analysis when provided
    if target and target in df.columns:
        findings["target_corr"] = (
            df.select_dtypes("number")
              .corrwith(df[target], method="spearman")
              .sort_values(key=abs, ascending=False)
        )
    findings["checklist"] = {...}        # boolean per step
    return findings

findings = eda_report(df, target="my_target_column")
```

타겟 칼럼이 명확하지 않을 때는 사용자에게 물어보십시오. 칼럼 이름에서 추측하지
**마십시오** — 도메인별 타겟(`conversion`, `churned`, `nps`)은 이름 휴리스틱을
무력화하며, 추측은 한 번의 패스를 낭비합니다.

히스토그램이 포함된 HTML 리포트의 경우, 선택적 `ydata-profiling` 라이브러리가
여전히 작동합니다: `ProfileReport(df).to_file("eda.html")`.

## EDA를 위한 Pandas 내장 플로팅

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

## EDA 체크리스트

각 단계를 수동으로 실행했든 생성된 `eda_report`를 통해 실행했든, 결과를
사용자에게 다시 전달하기 전에 아래의 모든 항목을 검증하십시오. 가장 흔한 EDA
실패 모드는 포인트 매스 검사 건너뛰기(IQR 이상치를 오해 소지가 있게 만듦)와
타겟의 분포를 분석하는 것을 잊는 것입니다.

- [ ] 모양 / 메모리 / dtype 보고됨
- [ ] 칼럼별 결측값 정량화됨
- [ ] 중복 행 카운트됨
- [ ] 숫자 칼럼에 대한 기술 통계
- [ ] **포인트 매스 칼럼 보고됨 (이상치 탐지 전)**
- [ ] 이상치 보고됨 (IQR, 포인트 매스 칼럼은 건너뜀)
- [ ] 왜도 / 첨도 보고됨
- [ ] 쌍별 Spearman 상관계수 > 0.3 보고됨
- [ ] **수치형 피처에 VIF 감사 실행; VIF > 5 표시, VIF > 10 심각으로 적시**
- [ ] **범주형이 존재하면 혼합 타입 다중공선성 감사** — `mixed_type_vif()` 소스별 max VIF, `cross_type_binding()` η²/Cramér's V; η² > 0.5 또는 점수 > 0.7인 쌍 보고
- [ ] 카디널리티 검사됨 (상수 및 ID 유사 칼럼)
- [ ] 타겟 변수 분석됨 (제공된 경우)

체크되지 않은 항목이 있으면, 결과를 보고하기 전에 해당 단계로 돌아가십시오.
