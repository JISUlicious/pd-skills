# Pandas 데이터 탐색 및 EDA — 전문가 스킬

당신은 전문 데이터 분석가입니다. pandas >= 2.3으로 익숙하지 않은 데이터셋을
탐색할 때마다 이 구조화된 EDA 방법론을 적용하십시오.

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

### 다중공선성 감사 (VIF) — 모든 데이터셋에서 실행

VIF(분산팽창인수)는 **선택 사항이 아니며**, **모델링 시점으로 미루지
않습니다**. 모든 Explore 단계의 일부로, 어떠한 상관계수 해석이나 드라이버
분석보다 먼저 실행하십시오. 쌍별 ρ는 여러 변수에 걸친 중복성을 놓칩니다.
피처 A가 다른 모든 피처와 개별적으로 ρ < 0.3을 보일 수 있지만, 사실은
다른 세 피처의 완벽한 선형 결합일 수 있습니다(VIF = ∞). Ames Housing
예시: `Gr Liv Area`는 적당한 쌍별 ρ 값을 가졌지만 VIF = ∞였는데, 이는
정확히 `1st Flr SF + 2nd Flr SF + Low Qual Fin SF`와 일치하기 때문입니다.

```python
import numpy as np

def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R²_j), R²_j from regressing column j on the others."""
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

vif = vif_table(df[numeric_cols].dropna())
print(vif.to_string(index=False))
severe   = vif.query("VIF > 10")["feature"].tolist()
moderate = vif.query("5 < VIF <= 10")["feature"].tolist()
if severe:
    print(f"⚠ SEVERE multicollinearity: {severe}")
    print("  → drop one of each redundant pair OR plan to use RidgeCV / LassoCV")
elif moderate:
    print(f"⚠ Moderate multicollinearity: {moderate}")
    print("  → trust SHAP/permutation rankings over OLS β if you model later")
```

모델링이 후속으로 이어지지 않더라도 항상 **VIF를 EDA 결과의 일부로
보고하십시오** — VIF는 이해관계자에게 그들이 별개라고 생각하는 두 메트릭이
실제로는 중복인지 알려줍니다. `feature-importance.md`의 모델링 사전 진단 섹션은
Ridge/Lasso 폴백 패턴과 VIF 등급에 대한 전체 결정 규칙을 다룹니다.

### 혼합 타입 다중공선성 — 범주형이 존재할 때

수치형만 사용하는 VIF는 **실제 데이터셋에서는 불완전합니다**. 많은
수치형 칼럼이 범주형 칼럼에 *결합(bound)*되어 있습니다. 범주형이 "이
피처가 존재하는가?"를 인코딩하고 관련된 수치형이 "피처 크기"를
인코딩할 때, 쌍별 ρ가 중간 정도로 보여도 두 칼럼은 같은 이진 신호를
공유합니다.

Ames Housing 사례: `Garage Yr Blt` (수치형)과 `Garage Finish`
(범주형)는 **η² = 0.998** — 거의 결정론적입니다. 차고가 없을 때 둘 다
"부재" 상태가 됩니다. 수치형만 사용하는 VIF는 `Garage Yr Blt`를
VIF ≈ 1로 평가하여 이 결합을 완전히 놓칩니다. 원-핫 확장과 함께
설계 행렬 VIF를 계산하면 1614까지 치솟습니다.

이 격차를 메우기 위해 **두 가지 점검**을 실행합니다:

#### A. 혼합 타입 VIF — 원-핫 더미를 포함한 설계 행렬

```python
def mixed_type_vif(X, num_cols, cat_cols, max_cardinality=15):
    """전체 설계 행렬(수치형 + 범주형의 원-핫 더미)에 대한 VIF.

    drop_first=True 필수 — 그렇지 않으면 한 범주형의 더미들이 합 = 1로
    구성상 완벽히 공선입니다.

    높은 카디널리티 범주형(> max_cardinality 수준)은 설계 행렬을
    과도하게 키우고 더미별 VIF의 해석이 어렵습니다. 이런 경우는
    cross_type_binding()을 사용하세요.

    반환:
      per_source:  [source, max_VIF, mean_VIF, n_features]
      per_feature: [feature, source, R²_on_others, VIF]
    """
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

per_source, per_feature = mixed_type_vif(df, numeric_cols, categorical_cols)
print(per_source.head(20).round(2).to_string(index=False))
new_severe = per_source.query("max_VIF > 10")["source"].tolist()
print(f"\nSources with severe design-matrix VIF: {new_severe}")
```

**유의사항 — 공유된 구조적 부재(structural-absence) 수준에서 오는
VIF=∞:** 여러 범주형이 같은 물리적 부재에 대해 "None" 수준을 공유할
때(예: Ames의 `Bsmt Qual_None`, `Bsmt Cond_None`, `BsmtFin Type 1_None`은
지하실이 없는 부동산에서 모두 1), 그 더미들은 구성상 완벽히
공선이 됩니다. 이는 N개의 독립적 다중공선성 소스가 아니라 "X개의
피처가 같은 '부재' 신호를 가진다"로 보고하세요.

#### B. 교차 타입 결합(cross-type binding) — η²와 Cramér's V

VIF에서 제외된 높은 카디널리티 범주형, 그리고 타입 간 해석이 더 쉬운
보완책으로 쌍별 연관성을 계산합니다:

```python
from scipy import stats

def cross_type_binding(X, num_cols, cat_cols, threshold=0.5):
    """타입 경계를 넘는 강한 결합 탐지.

    임계값을 초과하는 쌍에 대해 [a, b, score, kind] 칼럼의 DataFrame 반환:
      - num↔cat:  score = η²  (수치형 분산 중 범주 그룹 평균이 설명하는 부분)
      - cat↔cat:  score = Cramér's V
      - num↔num:  score = ρ²  (Spearman, 임계값 초과 시에만)
    """
    rows = []

    # 수치형 ↔ 범주형: η²
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

    # 범주형 ↔ 범주형: Cramér's V
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

    # 수치형 ↔ 수치형: ρ² (임계값 초과 시에만)
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            rho = X[[a, b]].corr(method="spearman").iloc[0, 1]
            if pd.notna(rho) and rho * rho > threshold:
                rows.append({"a": a, "b": b, "score": round(rho * rho, 3),
                             "kind": "ρ² (num↔num)"})

    return (pd.DataFrame(rows)
              .sort_values("score", ascending=False)
              .reset_index(drop=True))

bindings = cross_type_binding(df, numeric_cols, categorical_cols, threshold=0.5)
print(bindings.head(20).to_string(index=False))
```

#### 결정 규칙

| 상황 | 사용 |
|---|---|
| 수치형만 있는 데이터셋 | `vif_table(X)`로 충분 |
| 수치형 + 낮은 카디널리티 범주형 혼합 | `mixed_type_vif()`로 소스 단위 VIF |
| 높은 카디널리티 범주형 존재 (>15 수준) | 그 칼럼들에는 `cross_type_binding()` |
| "타입 간 중복이 있는가?" 빠른 점검 | `cross_type_binding(threshold=0.5)` 한 번 호출 |
| 프로덕션 진단 | 셋 다; max VIF > 10인 소스 OR 결합 점수 > 0.7인 모든 소스를 보고 |

**경험칙:** 수치형과 범주형 사이의 η² > 0.5는 그 수치형이 범주형 그룹
평균을 넘어 어떤 신호도 가지지 않음을 뜻합니다 — 모델링 전에 한쪽을
제거하세요. η² > 0.9는 결정론적 결합을 의미합니다 (예: Ames `Pool Area`
↔ `Pool QC` η² = 0.94 — Pool QC가 "None"일 때 Pool Area는 정확히 0).

### 클러스터 대표 선택 — 하나를 제거해야 할 때

VIF가 다중공선 클러스터를 표시하고 후속 방법론이 이를 허용할 수 없을 때(OLS
계수 해석, Lasso 피처 선택, RCA 공통성 보고), 하나의 대표를 선택하고 나머지를
제거해야 합니다. **임의로 선택하는 것이 가장 흔한 무언의 오류입니다** —
"유지된" 피처가 분석이 어떤 이야기를 들려줄지를 결정합니다.

다음 우선순위 순서를 적용하십시오:

| 우선순위 | 규칙 | 이유 |
|---|---|---|
| 1 | 한 칼럼 ≈ Σ(나머지)이고 R² ≥ 0.99이면, 그 집계 칼럼을 제거하고 구성 칼럼을 유지 | 구성 칼럼이 더 세분화된 정보를 엄격히 더 많이 담고 있음 |
| 2 | 칼럼 이름이 요약 패턴(`overall`, `total`, `index`, `score`, `rating`, `summary`)과 일치하고 구성 요소별 피처가 존재하면, 요약 칼럼을 제거 | 등급 동어반복 함정 회피 (Ames Overall Qual 사례) |
| 3 | **컨텍스트별(아래 오버라이드):** 훈련 폴드에서 `|ρ(target)|`로 순위 매김, 가장 높은 것을 유지 | 분석 목표에 가장 유용 |
| 4 | 동점자 처리: 결측값이 더 적은 쪽 | 대체 노이즈가 적음 |
| 5 | 최종 동점자 처리: 변동계수가 더 높은 쪽 | 더 많은 동적 범위, 더 많은 정보 |
| 6 | 5% 이내에서 여전히 동점이면, 인간 검토를 위해 FLAG — 자동 결정하지 말 것 | 도메인 지식이 필요 |

**우선순위 3은 분석 컨텍스트에 따라 변경됩니다:**

| 컨텍스트 | 점수 기준 |
|---|---|
| 피처 중요도 / 드라이버 분석 | 훈련 폴드에서의 `|ρ(target)|` (기본값) |
| RCA / 변경 시점 / 공통성 | 사전 기간의 σ 단위에서의 `|Δ at change point|` |
| 타겟 없는 순수 EDA | 표준화 후의 분산 |

```python
import numpy as np
import pandas as pd

SUMMARY_PATTERNS = ("overall", "total", "index", "score", "rating", "summary")


def find_aggregate(X, cluster_cols, threshold=0.99):
    """Return the cluster member best predicted by the rest (R² ≥ threshold).

    When multiple members satisfy the threshold (e.g. the perfect-sum case
    `agg = a + b + c` where any of the four can be predicted from the
    other three), prefer the one with the **largest mean magnitude** —
    aggregates tend to be sums and are numerically larger than their
    components. This produces the more interpretable choice for human
    readers.
    """
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
    candidates.sort(key=lambda t: -t[1])             # largest magnitude first
    return candidates[0][0]


def select_cluster_representative(
    X, y, cluster_cols, *,
    fold_mask=None,
    score_fn=None,
):
    """Returns (keep_list, drop_list, reason).

    score_fn(X_fold, y_fold, col) → float — higher is better.
    Default: |ρ(target)| via Spearman; falls back to variance if y is None.
    """
    Xf = X.loc[fold_mask] if fold_mask is not None else X
    yf = (y.loc[fold_mask] if (y is not None and fold_mask is not None)
          else (y if y is not None else None))

    # Priority 1: aggregate-vs-components
    agg = find_aggregate(Xf, cluster_cols)
    if agg is not None:
        components = [c for c in cluster_cols if c != agg]
        return components, [agg], f"aggregate dropped: {agg} ≈ Σ({components})"

    # Priority 2: summary names
    summaries = [c for c in cluster_cols
                 if any(p in c.lower() for p in SUMMARY_PATTERNS)]
    components = [c for c in cluster_cols if c not in summaries]
    if summaries and components:
        return components, summaries, f"summary names dropped: {summaries}"

    # Priority 3: context-aware score
    if score_fn is None:
        if yf is None:
            score_fn = lambda Xf, _y, c: float(Xf[c].std())
        else:
            score_fn = lambda Xf, yf, c: abs(
                float(Xf[c].corr(yf, method="spearman"))
            )

    scores = (pd.Series({c: score_fn(Xf, yf, c) for c in cluster_cols})
                .sort_values(ascending=False))
    if len(scores) == 1:
        return [scores.index[0]], [], "single member"

    top = scores.iloc[0]
    runner_up = scores.iloc[1]

    # Priority 4-5: tiebreak when top-2 within 5%
    if top > 0 and (top - runner_up) / top < 0.05:
        candidates = scores[scores >= runner_up * 0.95].index.tolist()
        tie = pd.Series({
            c: (1 - X[c].isna().mean())
               * (X[c].std() / (abs(X[c].mean()) + 1e-9))
            for c in candidates
        }).sort_values(ascending=False)
        if len(tie) > 1 and tie.iloc[0] > 0 \
           and (tie.iloc[0] - tie.iloc[1]) / tie.iloc[0] < 0.05:
            return None, None, f"AMBIGUOUS — manual review needed: {candidates}"
        keep = tie.index[0]
    else:
        keep = scores.index[0]

    drop_list = [c for c in cluster_cols if c != keep]
    return [keep], drop_list, f"kept {keep} (score={scores[keep]:.3f})"
```

**워크플로** — 클러스터 식별(쌍별 |ρ| ≥ 0.85인 피처를 그룹화), 클러스터별로
선택자 호출, 그런 다음 남은 것에 대해 VIF 재감사:

```python
from scipy.cluster.hierarchy import linkage, fcluster

def find_clusters(X, rho_threshold=0.85):
    """Hierarchical clustering on |corr|. Returns list of column lists."""
    corr = X.corr(method="spearman").abs().fillna(0)
    dist = 1 - corr
    Z = linkage(dist.values[np.triu_indices_from(dist.values, k=1)], method="average")
    labels = fcluster(Z, t=1 - rho_threshold, criterion="distance")
    clusters = [list(corr.columns[labels == lbl]) for lbl in set(labels)]
    return [c for c in clusters if len(c) > 1]

for cluster in find_clusters(X[numeric_cols], rho_threshold=0.85):
    keep, drop, reason = select_cluster_representative(X, y, cluster)
    if keep is None:
        print(f"⚠ AMBIGUOUS cluster — flag for review: {cluster}")
        continue
    print(f"cluster {cluster} → keep {keep} ({reason})")
    numeric_cols = [c for c in numeric_cols if c not in drop]

print(vif_table(X[numeric_cols]))                  # re-audit; expect VIF < 5 throughout
```

제거 후, 유지된 피처들의 VIF가 5 미만으로 돌아왔는지 **확인하기 위해
`vif_table`을 재실행하십시오**. 어떤 피처가 여전히 VIF > 5를 보인다면, 클러스터가
포착되지 않은 것입니다 — 클러스터를 확장(`rho_threshold`를 낮춤)하고 다시
시도하십시오.

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
