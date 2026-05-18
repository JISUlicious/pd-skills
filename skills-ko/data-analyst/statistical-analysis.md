# Pandas 통계 분석 — 전문가 스킬

당신은 전문 데이터 분석가입니다. 다음 통계 기법을 pandas >= 2.3 그리고
필요한 경우 scipy/numpy를 사용하여 정확하게 적용합니다.

## 기술 통계

```python
# 전체 요약
df.describe()                                         # count, mean, std, min, Q1-Q3, max
df.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95])
df.describe(include="all")                            # object/category 컬럼 포함

# 컬럼별 통계
df["revenue"].mean()
df["revenue"].median()
df["revenue"].std()
df["revenue"].var()
df["revenue"].sem()           # 평균의 표준오차
df["revenue"].mad()           # 평균 절대 편차 (pandas < 2.0)

# Pandas 2.0+ 에서 mad() 를 대체하는 견고한 방법
(df["revenue"] - df["revenue"].median()).abs().median()   # MAD
(df["revenue"] - df["revenue"].mean()).abs().mean()       # 평균 절대 편차

df["revenue"].skew()          # 왜도 (0 = 대칭)
df["revenue"].kurt()          # 초과 첨도 (0 = 정규)
df["revenue"].quantile([.25, .5, .75])
df["revenue"].mode()[0]       # 가장 빈번한 값
df["revenue"].nunique()       # 고유값 개수
```

## 집계

```python
# 여러 통계를 한 번에
df["revenue"].agg(["mean", "median", "std", "min", "max", "count"])

# agg 내 사용자 정의 함수
df["revenue"].agg(
    mean="mean",
    p50="median",
    p90=lambda x: x.quantile(0.9),
    p99=lambda x: x.quantile(0.99),
    cv=lambda x: x.std() / x.mean(),         # 변동계수
    iqr=lambda x: x.quantile(0.75) - x.quantile(0.25),
)

# GroupBy 집계
df.groupby("segment").agg(
    n=("user_id", "count"),
    revenue_total=("revenue", "sum"),
    revenue_avg=("revenue", "mean"),
    revenue_med=("revenue", "median"),
    revenue_p90=("revenue", lambda x: x.quantile(0.9)),
    revenue_std=("revenue", "std"),
)
```

## 분포 분석

```python
import numpy as np
from scipy import stats

col = df["revenue"].dropna()

# 적률
print(f"Mean:     {col.mean():.2f}")
print(f"Median:   {col.median():.2f}")
print(f"Std:      {col.std():.2f}")
print(f"Skew:     {col.skew():.3f}")   # >0 = 우편향, <0 = 좌편향
print(f"Kurtosis: {col.kurt():.3f}")   # >0 = 두꺼운 꼬리

# 정규성 검정
stat, p = stats.shapiro(col.sample(min(len(col), 5000)))  # Shapiro-Wilk
print(f"Shapiro-Wilk: stat={stat:.4f}, p={p:.4f}")
# p < 0.05 → 정규성 기각

stat, p = stats.normaltest(col)   # D'Agostino-Pearson
print(f"Normaltest: stat={stat:.4f}, p={p:.4f}")

# 분포 적합
mu, std = stats.norm.fit(col)
print(f"Best-fit Normal: μ={mu:.2f}, σ={std:.2f}")

# 경험적 CDF
ecdf = col.rank(pct=True)    # 각 값의 백분위 순위

# 백분위 순위
df["revenue_percentile"] = df["revenue"].rank(pct=True) * 100
```

## 상관관계 분석

```python
# Pearson 상관계수 (선형 관계)
corr_matrix = df[numeric_cols].corr(method="pearson")

# Spearman 상관계수 (단조, 이상치에 강건)
corr_matrix = df[numeric_cols].corr(method="spearman")

# Kendall 상관계수 (소표본, 순서형 데이터)
corr_matrix = df[numeric_cols].corr(method="kendall")

# 타겟과의 상관관계
df[numeric_cols].corrwith(df["target"], method="spearman").sort_values()

# 두 특정 시리즈 간 상관관계
r, p = stats.pearsonr(df["x"].dropna(), df["y"].dropna())
rho, p = stats.spearmanr(df["x"].dropna(), df["y"].dropna())
print(f"Pearson r={r:.3f}, p={p:.4f}")

# 점이연 (point-biserial) (연속형 vs 이진형)
r, p = stats.pointbiserialr(df["binary_col"], df["continuous_col"])
```

|ρ| > 0.7 인 쌍이 여러 개 존재하는 상관 행렬에서 OLS를 적합할 때는
계수를 해석하기 전에 VIF 점검을 실시합니다 (`feature-importance.md`
§ Pre-Modeling Diagnostics 참고). 예측 변수 간 높은 상관관계는
개별 β 값을 불안정하게 만들기 때문에 — 그 문서의 교차 방법 점검이
안전한 경로입니다.

## 가설 검정

### 이표본 검정

```python
from scipy import stats

group_a = df.loc[df["variant"] == "A", "revenue"].dropna()
group_b = df.loc[df["variant"] == "B", "revenue"].dropna()

# Welch t-test — 평균(MEAN) 차이를 검정 (n>30이면 CLT 덕분에 비정규성에 강건)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"Welch t-test: t={t_stat:.3f}, p={p_value:.4f}  ← 귀무가설: mean(A)==mean(B)")

# Mann-Whitney U — 확률적 우위(P(A > B) == 0.5)를 검정, 평균이 아님
u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
print(f"Mann-Whitney: U={u_stat:.0f}, p={p_value:.4f}  ← 귀무가설: P(A>B)==0.5")

# 중요: 보고하는 양과 일치하는 검정을 선택하세요.
# "평균이 +X% 상승"으로 보고한다면 Welch (또는 평균 차이의 부트스트랩) 사용.
# 중앙값 상승 / 승률을 보고한다면 Mann-Whitney 사용.
# 평균 상승을 보고하면서 Mann-Whitney로 검정하는 것 — 흔한 함정입니다.

# 효과 크기 (Cohen's d, 표본 크기가 다를 때도 올바른 풀링)
def cohens_d(a, b):
    n_a, n_b = len(a), len(b)
    s_a, s_b = a.std(ddof=1), b.std(ddof=1)
    pooled_var = ((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2)
    return (a.mean() - b.mean()) / np.sqrt(pooled_var)

d = cohens_d(group_b, group_a)
print(f"Cohen's d: {d:.3f}")  # 거친 경험칙: 0.2 작음, 0.5 중간, 0.8 큼
# 효과 크기 임계값은 도메인 의존적입니다. 마케팅에서 d=0.2는 엄청나지만
# 심리학에서는 보통 수준입니다. p-값과 함께 d도 보고하고, 맥락에서 해석하세요.
```

### 카이제곱 검정 (범주형 vs 범주형)

```python
contingency = pd.crosstab(df["variant"], df["converted"])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square: χ²={chi2:.3f}, df={dof}, p={p:.4f}")

# Cramér's V (효과 크기)
n = contingency.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
print(f"Cramér's V: {cramers_v:.3f}")
```

### 일원 ANOVA (다중 그룹)

```python
groups = [group["revenue"].dropna().values for _, group in df.groupby("segment")]
f_stat, p_value = stats.f_oneway(*groups)
print(f"ANOVA: F={f_stat:.3f}, p={p_value:.4f}")

# Kruskal-Wallis (비모수 ANOVA)
h_stat, p_value = stats.kruskal(*groups)
print(f"Kruskal-Wallis: H={h_stat:.3f}, p={p_value:.4f}")
```

### 범주형 × 수치형 스크리닝 (모든 쌍에 대한 Kruskal-Wallis)

**어떤 범주형 컬럼이 실제로 수치형 컬럼의 변동을 설명하는지** —
반대로 어떤 범주형이 사실상 노이즈인지 — 빠르게 알고 싶다면, 모든
(범주형, 수치형) 쌍에 대해 Kruskal-Wallis 를 실행합니다. 비모수이며
비정규 또는 클리핑된 분포에 강건하고, 수백만 행에서도 비용이 저렴합니다.

```python
from scipy import stats

cat_cols = df.select_dtypes(include=["category", "object", "str"]).columns
num_cols = df.select_dtypes(include="number").columns

rows = []
for cat in cat_cols:
    if df[cat].nunique() < 2 or df[cat].nunique() > 50:
        continue
    for num in num_cols:
        groups = [g[num].dropna().values
                  for _, g in df.groupby(cat, observed=True)]
        if sum(len(g) > 0 for g in groups) < 2:
            continue
        h, p = stats.kruskal(*groups)
        rows.append((cat, num, h, p))

screen = (pd.DataFrame(rows, columns=["cat", "num", "H", "p"])
            .sort_values("p"))
print("Significant (p < 0.01):")
print(screen.query("p < 0.01").to_string(index=False))

# 어떤 수치형 컬럼도 분리하지 못하는 범주형 — 노이즈/관리용일 가능성
uninformative = (screen.groupby("cat")["p"].min()
                 .loc[lambda s: s > 0.05].index.tolist())
print(f"\nCategoricals with no significant effect on any numeric: {uninformative}")
```

## A/B 테스트 분석

```python
def ab_test_report(df, variant_col, metric_col, control="A", treatment="B",
                   n_boot=10_000, seed=42):
    """A/B 테스트 보고서. 평균(MEAN)의 차이를 검정합니다 (lift 수치와 일치).

    Welch t-test로 추론, 부트스트랩으로 CI 계산, Mann-Whitney는 확률적 우위에
    대한 보조 점검으로 사용합니다.
    """
    ctrl  = df.loc[df[variant_col] == control,   metric_col].dropna().to_numpy()
    treat = df.loc[df[variant_col] == treatment, metric_col].dropna().to_numpy()

    # 요약
    print(f"Control   n={len(ctrl):,}  mean={ctrl.mean():.4f}  median={np.median(ctrl):.4f}")
    print(f"Treatment n={len(treat):,}  mean={treat.mean():.4f}  median={np.median(treat):.4f}")
    lift = (treat.mean() - ctrl.mean()) / ctrl.mean()
    print(f"평균 기준 상대 상승률: {lift:+.2%}")

    # 주된 추론 — 평균(MEAN)에 대한 Welch t-test (보고된 lift와 일치)
    t_stat, p_welch = stats.ttest_ind(treat, ctrl, equal_var=False)
    print(f"Welch t-test p={p_welch:.4f} {'✓ SIGNIFICANT' if p_welch < 0.05 else '✗ not significant'}")

    # 평균 차이의 95% 부트스트랩 CI (재현 가능)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (rng.choice(treat, len(treat), replace=True).mean()
                  - rng.choice(ctrl,  len(ctrl),  replace=True).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"평균 차이의 95% 부트스트랩 CI: [{lo:+.4f}, {hi:+.4f}]")

    # 보조 — 확률적 우위에 대한 Mann-Whitney (다른 귀무가설)
    _, p_mw = stats.mannwhitneyu(treat, ctrl, alternative="two-sided")
    print(f"Mann-Whitney (P(treat>ctrl)==0.5) p={p_mw:.4f}")
    if (p_welch < 0.05) != (p_mw < 0.05):
        print("  ⚠ Welch와 Mann-Whitney의 결론이 다릅니다 — 지표 분포가 평균과")
        print("    순위에서 다르게 이동했습니다. 어느 쪽이 중요한지 조사하세요.")

ab_test_report(df, "variant", "revenue_per_user")
```

**다중 검정 경고:** 동일 데이터에 여러 A/B 테스트를 실행하는 경우(지표 ×
세그먼트 × 코호트), 검정 하나당 5% α는 유지되지 않습니다.
`statsmodels.stats.multitest.multipletests`로 전체 테스트 패밀리에 BH-FDR을
적용하거나, 주된 가설을 사전 등록하고 α-소비(α-spending)를 계획하세요.

## 코호트 분석

```python
# 첫 활동 날짜로 코호트 할당
df["cohort"] = df.groupby("user_id")["date"].transform("min").dt.to_period("M")
df["period_number"] = (
    df["date"].dt.to_period("M") - df["cohort"]
).apply(lambda x: x.n)

# 잔존율 표
cohort_pivot = df.groupby(["cohort", "period_number"])["user_id"].nunique().reset_index()
cohort_size = cohort_pivot[cohort_pivot["period_number"] == 0].set_index("cohort")["user_id"]
cohort_table = cohort_pivot.pivot(index="cohort", columns="period_number", values="user_id")
retention = cohort_table.divide(cohort_size, axis=0)
print(retention.round(3))
```

## 순위 및 백분위수

```python
# 순위 (동률은 method 로 처리)
df["revenue_rank"]  = df["revenue"].rank(ascending=False, method="min")
df["revenue_dense"] = df["revenue"].rank(ascending=False, method="dense")
df["revenue_pct"]   = df["revenue"].rank(pct=True)  # 0–1 백분위

# 그룹 내 순위
df["rank_in_segment"] = df.groupby("segment")["revenue"].rank(ascending=False, method="min")

# 백분위 구간
df["decile"] = pd.qcut(df["revenue"], q=10, labels=False) + 1    # 1–10
df["quintile"] = pd.qcut(df["revenue"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
```

## 요약 통계 표 (보고서용)

```python
def summary_table(df, group_col, metric_col):
    """Produce a publication-quality summary table."""
    return (
        df.groupby(group_col)[metric_col]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            min="min",
            max="max",
        )
        .round(2)
        .reset_index()
    )

print(summary_table(df, "region", "revenue").to_string(index=False))
```
