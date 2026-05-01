# 근본 원인 분석 — 전문가 스킬

당신은 **제품 결함 조사, 수율 이상(yield excursion), 공정 드리프트, 사건
검토**를 지원하는 전문 데이터 분석가입니다. 여기의 방법론은 인과적이고
시간상 국소적입니다 — `feature-importance.md`의 연관적 순위와는 구별됩니다.

핵심 규칙: **SHAP 중요도는 인과적 증거가 아닙니다.** `chamber_pressure`의
SHAP 값 0.5는 "이 피처가 예측 오차를 줄인다"를 말하는 것이지 — *"챔버
압력을 고치면 수율이 고쳐진다"*는 의미가 *아닙니다*. 근본 원인 작업은
결과 변화 *이전에* 일어난 *변화*를 식별한 뒤, 대안적 설명을 배제해야
합니다.

## 이 파일을 사용할 때 vs `feature-importance.md`를 사용할 때

| 질문 | 파일 |
|---|---|
| *"어떤 X가 Y를 가장 잘 예측하는가?"* | `feature-importance.md` |
| *"시점 T에 Y가 변화한 원인이 무엇인가?"* | 여기 |
| *"어떤 장비 / 로트 / 작업자가 불량과 연관되어 있는가?"* | 둘 다 — 시간상 국소적 이상은 여기부터 시작, 함께 사용 |
| *"이 공정 변경이 수율을 개선했는가?"* | 여기 (DiD, § 6) |
| *"DOE를 했는데 — 무엇이 중요했는가?"* | 여기 (ANOVA, § 7) |

질문이 시간상 국소적("4월 2일 이후")이거나 인과적("X를 *변경한 것이* Y의
변화를 일으켰는가?")이라면 여기서 시작하세요. 순수하게 연관적("피처를
중요도 순으로 매기기")이라면 `feature-importance.md`로 충분합니다.

## 환경 설정 — 선택적 의존성

```python
try:
    import ruptures as rpt           # 변화점 탐지 (CPD)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_RCA = True
except Exception as e:
    HAS_RCA = False
    print(f"RCA stack unavailable ({type(e).__name__}: {e}).")
    print("Install: uv pip install ruptures statsmodels mlxtend")
    print("Optional (causal): uv pip install dowhy econml")

try:
    import dowhy                      # 인과 추론 (무거움, 선택적)
    HAS_CAUSAL = True
except Exception:
    HAS_CAUSAL = False
```

`scipy`와 `sklearn`(`feature-importance.md`에서 이미 필요)이 Fisher
정확검정, Mann-Whitney, 성향 점수 매칭, Tukey HSD를 모두 다룹니다.

## 1. 변화점 탐지 (CPD)

이상(excursion)의 대표 질문은 **언제 시작되었는가?**입니다. 이동 평균은
답을 흐리게 만들기 때문에, 공식적인 분기점 탐지기가 필요합니다.

### 기본값: PELT (오프라인, 빠름)

```python
import numpy as np
import pandas as pd
import ruptures as rpt

def detect_changepoints(series, model="rbf", min_size=30, penalty=None):
    """지표 분포가 변하는 지점의 인덱스를 반환.

    series   : 1-D numpy 배열 또는 pd.Series (시간 순서 필수)
    model    : "rbf" (기본, 강건), "l2" (가우시안 평균 이동),
               "l1" (중앙값 이동, 이상치에 강건)
    min_size : 변화점 사이의 최소 샘플 수
    penalty  : None이면 log(n) * 분산 사용 (BIC 스타일); 높이면 큰 변화만
               탐지, 낮추면 더 많이 탐지
    """
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    if penalty is None:
        penalty = np.log(n) * float(np.var(s))
    algo = rpt.Pelt(model=model, min_size=min_size).fit(s)
    bkps = algo.predict(pen=penalty)
    return bkps[:-1]                  # 끝점 제거

# 시각적 정상성 점검은 필수:
import matplotlib.pyplot as plt
def plot_cps(series, bkps, ax=None):
    ax = ax or plt.gca()
    ax.plot(series, lw=1)
    for b in bkps:
        ax.axvline(b, color="red", ls="--", lw=1)
    ax.set_title(f"{len(bkps)} change points detected")
```

### 작은 지속 드리프트에는 CUSUM

PELT는 종종 미묘한 드리프트(0.5σ 이동이 200 샘플 동안 지속)를 놓칩니다.
CUSUM은 편차를 누적하여 누적치가 임계값을 넘으면 알람을 발생시킵니다:

```python
def cusum(series, target=None, k=0.5, h=5):
    """양방향 CUSUM. +/- CUSUM이 h*sigma를 초과하는 인덱스를 반환.

    k : 표준편차 단위의 기준값 (0.5 = 1σ 이동에 민감)
    h : 표준편차 단위의 결정 임계값 (5가 정통적인 기본값)
    """
    s = np.asarray(series, dtype=np.float64)
    target = target if target is not None else s.mean()
    sigma = s.std()
    pos = np.zeros(len(s)); neg = np.zeros(len(s))
    for i in range(1, len(s)):
        pos[i] = max(0, pos[i-1] + (s[i] - target - k * sigma))
        neg[i] = min(0, neg[i-1] + (s[i] - target + k * sigma))
    alarms = np.where((pos > h * sigma) | (neg < -h * sigma))[0]
    return alarms
```

### 결정 표

| 데이터 형태 | 메서드 |
|---|---|
| 알려진 단일 분기점, 급격함 | PELT, model="l2" |
| 알려지지 않은 다중 분기점 | PELT, model="rbf", BIC 페널티 |
| 느린 드리프트, 명확한 단계 없음 | CUSUM (k=0.5) 또는 EWMA 차트 (§ 2) |
| 강한 이상치 | PELT, model="l1" (중앙값 기반) |
| 스트리밍 / 온라인 탐지 | 베이지안 온라인 CPD (`bayesian_changepoint_detection` 라이브러리) |

### 변화점을 믿기 전 정상성 점검

탐지된 분기점은 다음을 만족하지 않으면 **잡음**입니다:
1. 양쪽에 ≥30 샘플 (그렇지 않으면 통계적 검정력이 너무 낮음)
2. pre/post 평균이 국소 잡음의 ≥1σ 만큼 차이남
3. pre vs post의 Mann-Whitney가 p < 0.01 반환
4. 플롯에서 시각적으로 명백함

항상 네 가지 점검을 모두 출력하세요. 이를 거치지 않고 "변화점"을
보고하는 것이 가장 흔한 과탐지 실패입니다.

## 2. 통계적 공정 관리 (SPC)

지속 모니터링용 (vs 과거 데이터의 사후 CPD). SPC는 제조 엔지니어의
어휘입니다 — 보고할 때 자연스럽게 사용하세요.

### Western Electric 룰을 적용한 Shewhart X̄/R 차트

```python
def shewhart_chart(values, subgroup_size=5):
    """X̄와 R 차트의 한계 + Western Electric 룰 위반을 계산.
    중심선/UCL/LCL 그리고 부분군별 위반 플래그를 dict로 반환.
    """
    s = np.asarray(values, dtype=np.float64)
    n_sub = len(s) // subgroup_size
    s = s[:n_sub * subgroup_size].reshape(n_sub, subgroup_size)
    xbar = s.mean(axis=1)
    rng  = s.max(axis=1) - s.min(axis=1)
    # n=5 부분군에 대한 상수 (다른 크기는 A2, D3, D4 룩업)
    A2, D3, D4 = 0.577, 0.0, 2.114
    cl_x  = xbar.mean()
    rbar  = rng.mean()
    ucl_x = cl_x + A2 * rbar
    lcl_x = cl_x - A2 * rbar
    ucl_r = D4 * rbar
    lcl_r = D3 * rbar
    sigma = (ucl_x - cl_x) / 3                  # 차트 한계로부터 함의된 σ

    # X̄에 대한 Western Electric 룰
    flags = pd.DataFrame({"xbar": xbar})
    flags["rule_1"] = np.abs(xbar - cl_x) > 3 * sigma                       # 1점 > 3σ
    flags["rule_2"] = (np.abs(xbar - cl_x) > 2 * sigma).rolling(3).sum() >= 2  # 3개 중 2개 > 2σ
    flags["rule_3"] = (np.abs(xbar - cl_x) > sigma).rolling(5).sum() >= 4      # 5개 중 4개 > 1σ
    flags["rule_4"] = pd.Series((xbar > cl_x).astype(int)).rolling(8).sum() \
                        .isin([0, 8])                                          # 8연속 같은 쪽
    flags["any_violation"] = flags[["rule_1","rule_2","rule_3","rule_4"]].any(axis=1)
    return {"cl_x": cl_x, "ucl_x": ucl_x, "lcl_x": lcl_x,
            "cl_r": rbar, "ucl_r": ucl_r, "lcl_r": lcl_r,
            "sigma": sigma, "flags": flags}
```

### 느린 드리프트용 EWMA 차트

```python
def ewma_chart(values, lambda_=0.2, L=3):
    """EWMA 관리도 — 작고 지속적인 드리프트에 민감.
    lambda_ : 가중 인자 (0.05–0.3); 작을수록 더 매끄럽고 느림
    L       : 시그마 단위의 관리 한계 폭 (기본 3)
    """
    s = np.asarray(values, dtype=np.float64)
    target, sigma = s.mean(), s.std()
    z = np.zeros(len(s)); z[0] = target
    for i in range(1, len(s)):
        z[i] = lambda_ * s[i] + (1 - lambda_) * z[i-1]
    ucl = target + L * sigma * np.sqrt(lambda_ / (2 - lambda_))
    lcl = target - L * sigma * np.sqrt(lambda_ / (2 - lambda_))
    violations = np.where((z > ucl) | (z < lcl))[0]
    return {"ewma": z, "ucl": ucl, "lcl": lcl, "violations": violations}
```

### 공정 능력 지수

```python
def capability(values, lsl, usl, k=6):
    """규격한계 대비 측정값의 Cp, Cpk, Pp, Ppk.
    Cp/Cpk는 부분군 내 시그마(합리적 부분군) 사용; Pp/Ppk는 전체 시그마 사용.
    수치 값과 함께 실용적 해석을 보고합니다.
    """
    s = np.asarray(values, dtype=np.float64)
    sigma_overall = s.std(ddof=1)
    sigma_within  = sigma_overall                # 부분군 없으면 동일; 있으면 R-bar/d2로 계산
    mu = s.mean()
    cp  = (usl - lsl) / (k * sigma_within)
    cpk = min((usl - mu) / (3 * sigma_within), (mu - lsl) / (3 * sigma_within))
    pp  = (usl - lsl) / (k * sigma_overall)
    ppk = min((usl - mu) / (3 * sigma_overall), (mu - lsl) / (3 * sigma_overall))

    def interpret(v):
        if v < 1.0:  return "incapable (defects expected)"
        if v < 1.33: return "marginal"
        if v < 1.67: return "capable"
        return "highly capable"
    return {"Cp": cp, "Cpk": cpk, "Pp": pp, "Ppk": ppk,
            "Cpk_interp": interpret(cpk), "Ppk_interp": interpret(ppk)}
```

**Cp vs Cpk:** Cp는 중심을 무시; Cpk는 중심 이탈에 페널티. **Cp vs Pp:**
Cp는 부분군 내 변동(단기); Pp는 전체 변동(장기, 드리프트 포함). Pp ≪
Cp이면 공정이 부분군 사이에서 드리프트하고 있는 것 — 조사하세요.

### SPC vs CPD, 언제 어느 것을?

| 사용 사례 | 메서드 |
|---|---|
| 안정적인 공정의 온라인 모니터링 | SPC (Shewhart 또는 EWMA) |
| 알려진 이상에 대한 사후 포렌식 | CPD (PELT) |
| 현 상태의 결함률 추정 | Cp/Cpk |
| 수정이 효과 있었는지 확인 | 둘 다: 수정 시점에 CPD + 수정 이후 SPC |

**안티패턴:** 자기상관된 데이터에 SPC를 적용하면 거짓 알람이 발생합니다.
`series.autocorr() > 0.5`이면 먼저 AR(1) 잔차를 적합한 뒤 그 잔차를
차트화하세요.

## 3. 공통성 분석 (Commonality Analysis)

### 다중공선 센서 클러스터를 먼저 축소

제조 데이터에서 같은 물리 상태를 측정하는 인접한 센서들(챔버 온도 ↔ 벽
온도 ↔ 척 온도; 레시피 압력 ↔ 측정 압력)은 거의 완벽하게 상관됩니다.
**먼저 축소하지 않으면 공통성 표는 하나의 물리 클러스터의 8개
멤버를 8개의 별도 발견으로 보고합니다** — 보고서를 읽는 엔지니어에게는
무용합니다.

`data-exploration.md` § 6의 프레임워크(5-우선순위
`select_cluster_representative()` 헬퍼)를 사용하되, 한 가지 결정적 오버라이드를
적용합니다:

> **RCA용 우선순위 3 오버라이드:** 클러스터 멤버를 *전체 타겟 상관*이
> 아니라 **|변화점에서의 이동 크기|**(pre 기간의 σ 단위)로 순위 매김.
> 전체적으로 가장 예측력이 좋은 센서와 이상이 발생했을 때 가장 많이
> 움직인 센서는 종종 다릅니다. SECOM이 이를 경험적으로 검증했습니다:
> SHAP 상위 10개와 CPD 공통성 상위 10개는 10개 중 1개만 겹쳤습니다.

```python
def shift_score(pre_df, post_df, col):
    """RCA-context 우선순위 3: pre 기간의 σ 단위로 |Δ|."""
    pre = pre_df[col].dropna()
    post = post_df[col].dropna()
    if len(pre) < 30 or pre.std() == 0:
        return 0.0
    return abs(post.mean() - pre.mean()) / pre.std()

# 클러스터별 — pre_df / post_df는 변화점 윈도우로 정의
for cluster in find_clusters(X[sensor_cols], rho_threshold=0.85):
    keep, drop, reason = select_cluster_representative(
        X, y=None, cluster_cols=cluster,
        score_fn=lambda Xf, _y, c: shift_score(pre_df, post_df, c),
    )
    if keep is None:
        print(f"⚠ AMBIGUOUS cluster — flag for engineer review: {cluster}")
        continue
    print(f"cluster {cluster} → represented by {keep} ({reason})")
    sensor_cols = [c for c in sensor_cols if c not in drop]
```

RCA 서술에 클러스터 단위 발견을 보고:

> "**8개 챔버 상태 센서 클러스터** (s406, s540, s268, s405, s539,
> s267, s058, s007)가 변화점에서 이동했습니다. 클러스터 대표는
> `s406` (Δ = +5.15, 정규화 +4.5σ). 근본 원인 조사 시 단일 물리
> 현상으로 취급하세요."

— *"8개 개별 센서가 이동했습니다"*가 *아닙니다*. 후자는 같은 물리적
사건의 중복 측정값들 사이로 엔지니어의 주의를 분산시킵니다.

### 범주형 요인별 공통성

공선 센서 클러스터를 축소한 뒤, 남은 집합에 공통성을 실행합니다.
"*어떤 범주형 수준이 모든 불량 로트에는 나타나지만 양품 로트에는 거의
없는가?*" — 회귀가 아니라 집합 중첩 문제입니다.

```python
from scipy.stats import fisher_exact

def commonality(df, factor_col, defect_col, defect_value=1, alpha=0.01):
    """수준별 Fisher 정확검정 + 오즈비 + 리프트. Bonferroni 보정.

    불량 vs 비불량에서 그 존재가 유의미하게 상승된 수준을 반환합니다.
    """
    rows = []
    levels = df[factor_col].dropna().unique()
    n_tests = len(levels)
    for lvl in levels:
        is_lvl = (df[factor_col] == lvl)
        is_def = (df[defect_col] == defect_value)
        a = int((is_lvl & is_def).sum())             # 이 수준이고 불량
        b = int((is_lvl & ~is_def).sum())            # 이 수준이고 양품
        c = int((~is_lvl & is_def).sum())            # 다른 수준이고 불량
        d = int((~is_lvl & ~is_def).sum())           # 다른 수준이고 양품
        if a + b == 0:
            continue
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        # 리프트: P(level | defect) / P(level)
        lift = (a / (a + c)) / ((a + b) / (a + b + c + d)) if (a + c) else 0
        rows.append({
            "level": lvl, "n_with_defect": a, "n_without_defect": b,
            "defect_rate_in_level": a / (a + b) if (a + b) else 0.0,
            "odds_ratio": float(odds), "lift": round(lift, 2),
            "p_fisher": float(p),
            "p_bonferroni": min(1.0, float(p) * n_tests),
        })
    return (pd.DataFrame(rows)
              .sort_values("p_bonferroni")
              .query(f"p_bonferroni < {alpha}")
              .reset_index(drop=True))

# 모든 범주형 요인 스윕:
for factor in categorical_cols:
    sig = commonality(df, factor, "is_defective")
    if len(sig):
        print(f"\n{factor}: {len(sig)} significant levels")
        print(sig.to_string(index=False))
```

다중 요인 공통성(어떤 *조합* — 장비 + 레시피 + 로트 — 이 불량에
나타나는가)에는 빈발 항목집합 마이닝을 사용합니다:

```python
# pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# 각 factor=level을 불리언 칼럼으로 인코딩
encoded = pd.get_dummies(df[categorical_cols].astype(str), dtype=bool)
encoded["is_defective"] = (df["defect"] == 1)

freq = apriori(encoded, min_support=0.01, use_colnames=True)
rules = association_rules(freq, metric="lift", min_threshold=2.0)
defect_rules = rules[rules["consequents"].astype(str).str.contains("is_defective")]
print(defect_rules[["antecedents", "support", "confidence", "lift"]].head(10))
```

## 4. 인과 추론

핵심 규칙: **예측은 인과가 아닙니다.** X가 Y를 일으켰다고 주장하기
전에 다음을 배제하세요:

1. **역인과** — Y가 먼저 변하고 X가 따라온 것은 아닌가?
2. **공통 원인** — X와 Y를 모두 추동하는 상류의 Z가 있는가?
3. **선택 편향** — X가 위험 모집단에 과대 표집되었는가?
4. **시간 불일치** — X의 변화가 실제로 Y의 이동에 *선행*했는가?

### 최소 DAG (방향 비순환 그래프) 구축

SHAP / 공통성으로 부상한 모든 후보 원인 X에 대해, 손으로 최소 DAG를
그리세요:
```
   recipe_change ──→ chamber_pressure ──→ defect_rate
        │                  │
        ▼                  ▼
   tool_uptime ←──────→ wafer_temp
```
그런 다음 뒷문 경로(backdoor path)를 식별합니다: 차단되지 않은 공통
조상을 통해 X에서 Y로 가는 모든 경로는 교란변수입니다. 해결책은 (a)
교란변수를 조건화, (b) 도구 변수 찾기, 또는 (c) 보고서에 한계를
명시하는 것입니다.

### 이중차분법 (DiD) — pre/post 변화 이상의 핵심 도구

알려진 공정 변경이 시점 T에 일어났고 그것의 Y에 대한 인과 효과를
추정하고 싶다면, 다음을 비교합니다:
- 영향을 받은 그룹(처리)에 대한 Δ(post − pre)
- 영향을 받지 않은 그룹(대조)에 대한 Δ(post − pre)

```python
def diff_in_diff(df, time_col, group_col, outcome, change_time,
                 treated_value, control_value):
    """OLS 상호작용을 통한 닫힌 형태 DiD."""
    df = df.copy()
    df["post"]    = (df[time_col] >= change_time).astype(int)
    df["treated"] = (df[group_col] == treated_value).astype(int)
    model = smf.ols(f"{outcome} ~ post + treated + post:treated", data=df).fit()
    did_estimate = model.params["post:treated"]
    print(model.summary().tables[1])
    print(f"\nDiD estimate: {did_estimate:+.4f}  (95% CI: "
          f"[{model.conf_int().loc['post:treated',0]:.4f}, "
          f"{model.conf_int().loc['post:treated',1]:.4f}])")
    # 정상성: 평행추세 점검 (양 그룹의 pre 기간 시각화)
    return model
```

DiD는 pre 기간의 **평행추세(parallel trends)**를 가정합니다 — 변화가
없었다면 처리군과 대조군이 함께 움직였을 것. 항상 양 그룹의 pre 기간을
플롯하세요. T 이전부터 추세가 발산했다면 DiD는 무효입니다.

### 성향 점수 매칭 — 무작위화되지 않은 관측 비교

DiD를 사용할 수 없을 때(깨끗한 pre/post가 없을 때), 처리 단위를 관측된
공변량에 대해 비슷한 대조 단위에 매칭합니다:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_match(df, treatment_col, outcome_col, covariates, caliper=0.1):
    X = df[covariates].fillna(df[covariates].median())
    t = df[treatment_col].astype(int)
    # 성향 점수 추정
    ps_model = LogisticRegression(max_iter=1000).fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]
    df = df.assign(ps=ps)
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    # 1:1 PS 최근접 이웃 (나쁜 매치는 캘리퍼로 제거)
    nn = NearestNeighbors(n_neighbors=1).fit(control[["ps"]].values)
    dist, idx = nn.kneighbors(treated[["ps"]].values)
    keep = dist.ravel() < caliper
    matched = pd.concat([
        treated.iloc[keep],
        control.iloc[idx.ravel()[keep]],
    ])
    ate = (matched.loc[matched[treatment_col]==1, outcome_col].mean()
         - matched.loc[matched[treatment_col]==0, outcome_col].mean())
    print(f"Matched n: {keep.sum()} pairs (of {len(treated)} treated units)")
    print(f"ATE estimate: {ate:+.4f}")
    return matched, ate
```

### `dowhy` — 명시적 가정을 가진 원칙적 인과 추론

DAG가 비자명할 때(다중 교란변수, 도구, 매개체), `dowhy`를 사용하세요 —
가정을 명시적으로 적도록 강제하고, 각 가정 하에서 효과를 추정하며, 반박
검정을 실행합니다:

```python
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment="recipe_change",                  # 의심되는 원인
    outcome="defect_rate",                      # 지표
    common_causes=["tool_uptime", "wafer_lot_age", "operator_shift"],
    # 선택사항: instruments, effect_modifiers, ...
)
model.view_model()                              # DAG 렌더링

# 인과 추정량 식별 (닫힌 형태 식)
identified = model.identify_effect(proceed_when_unidentifiable=False)
print(identified)

# 뒷문 조정 + 성향 점수로 추정
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.propensity_score_matching",
    target_units="ate",
)
print(f"Causal effect estimate: {estimate.value:+.4f}")

# 반박: 추정치가 다음 도전을 견딜 수 있는가?
refute_random  = model.refute_estimate(identified, estimate, "random_common_cause")
refute_placebo = model.refute_estimate(identified, estimate, "placebo_treatment_refuter")
refute_subset  = model.refute_estimate(identified, estimate, "data_subset_refuter")
print("Refutation summary:")
print(f"  Random common cause:  Δ = {refute_random.new_effect - estimate.value:+.4f}")
print(f"  Placebo treatment:    Δ = {refute_placebo.new_effect - estimate.value:+.4f}")
print(f"  Data subset:          Δ = {refute_subset.new_effect - estimate.value:+.4f}")
```

강건한 인과 추정치는 세 가지 반박을 모두 작은 변화로 통과합니다.
플라시보 반박이 큰 효과를 보이면(구성상 그래서는 안 되는데), 원래
추정치는 의심됩니다.

## 5. DOE / ANOVA — 계획된 실험을 위해

팀이 실제 실험을 수행했을 때(2k 완전요인, 부분요인, Plackett-Burman),
`statsmodels`로 ANOVA를 사용하세요:

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 상호작용 포함 이원 ANOVA
model = smf.ols("yield ~ C(tool) * C(recipe)", data=df).fit()
print(sm.stats.anova_lm(model, typ=2))      # Type-II SS

# 효과 크기: η² (eta-squared) 와 ω² (omega-squared, 덜 편향)
ss = sm.stats.anova_lm(model, typ=2)
ss["eta_sq"]   = ss["sum_sq"] / ss["sum_sq"].sum()
ss["omega_sq"] = ((ss["sum_sq"] - ss["df"] * model.mse_resid) /
                  (ss["sum_sq"].sum() + model.mse_resid))
print(ss.round(4))

# Tukey HSD 사후 검정: 어떤 수준 쌍이 다른가?
tukey = pairwise_tukeyhsd(df["yield"], df["tool"], alpha=0.05)
print(tukey.summary())
```

**통계적 유의성 ≠ 실용적 유의성.** η² < 0.01은 "이 요인이 분산의 1%
미만을 설명한다"는 의미입니다 — p < 0.0001이라도 효과가 무관할 수
있습니다. 항상 효과 크기를 p-값과 함께 보고하세요.

## 6. RCA 의사결정 치트시트

```
START: 어떤 종류의 질문인가?
   │
   ├── "Y가 한동안 나빴다" (정상 상태)
   │       └→ SPC + Cp/Cpk + 공통성 (§§ 2, 3)
   │
   ├── "Y가 시점 T에 이동했다" (이상)
   │       └→ 시점 확인 위해 CPD (§ 1)
   │          └→ pre vs post 공통성 (§ 3)
   │             └→ T에 알려진 변경 있으면 DiD (§ 4)
   │                └→ 인과 반박 (§ 4: dowhy)
   │
   ├── "DOE를 했는데, 무엇이 중요했는가?"
   │       └→ ANOVA + Tukey HSD + 효과 크기 (§ 5)
   │
   └── "수정을 했다 — 효과가 있었는가?"
           └→ pre/post DiD + 변경 후 SPC 모니터링 (§§ 2, 4)

매 단계마다 묻기: "Z에 의해 교란될 수 있는가?" 그렇다면 Z를 조건화
(회귀 / 매칭)하거나 한계를 명시하세요.
```

## 7. RCA 보고 템플릿

SHAP 표는 RCA 보고서가 아닙니다. 산출물은 증거를 가진 **인과적
서술**입니다:

```
## 이상: <지표>가 <date>에 <old>에서 <new>로 이동

### 1. 타임라인
- 발견: <처음 인지된 시점>
- 추정 변화점: <date> (PELT, BIC 페널티, MW p < 0.01,
  pre n=200 / post n=180)
- 시간에 따른 지표 플롯, 변화점 표시.

### 2. 후보 원인 (증거 강도순)
| 후보 | 증거 (공통성 / DiD / SHAP) | 배제할 교란변수 |
|---|---|---|
| T-2의 레시피 v3.2 배포 | Lift=4.1, Fisher p=0.0003 | 같은 날 장비 정비 |
| 챔버 5 (장비 A) | 결함률 12% vs 베이스라인 2% | 로트 믹스 변화 |
| 신규 작업자 시프트 | 결함률 8% vs 베이스라인 2% | 야간 시프트 하드웨어와 교란 |

### 3. 인과 분석
- DiD 추정 (레시피 v3.2): +5.2pp 결함률, 95% CI [3.1, 7.3]
- 반박: 무작위 공통원인 Δ=+0.1, 플라시보 Δ=−0.05 → 강건
- 평행추세 점검: ✓ (부록 플롯)

### 4. 권장 조치
- 레시피 v3.2 롤백 (가장 강한 증거)
- 다음 5개 로트에 대해 챔버 5 샘플링 증가 (보조 신호)
- 야간 시프트 훈련 감사 (최저 우선순위 — 교란됨)

### 5. 배제할 수 없는 사항
- 같은 주 미관측 공급자 자재 드리프트
- 챔버 5 픽스처 마모 (계측 없음)
```

## 8. RCA 안티패턴

| 안티패턴 | 증상 | 해결 |
|---|---|---|
| SHAP 순위를 인과 순위로 취급 | "가장 영향력 있는 피처가 X이니 X를 고친다" | 조치 권장 전에 DiD 또는 반박 단계 추가 |
| 시간 인덱스 데이터에 CPD 건너뛰기 | 시작 시점 확인 없이 이상 보고 | 항상 PELT 실행, 정상성 점검과 함께 변화점 보고 |
| 케이스 믹스 통제 없이 장비/작업자 비교 | 장비 A가 "최악으로 보임"이지만 어려운 로트만 처리 | 로트 유형으로 층화하거나 성향 매칭 사용 |
| 자기상관 데이터에 SPC | 거의 매 부분군마다 거짓 알람 | `series.autocorr()` 점검; > 0.5이면 AR(1) 잔차 차트화 |
| "가장 영향받은" 엔티티만 골라쓰기 | 헤드라인 원인이 시스템 드라이버가 아니라 이상치 | 모든 엔티티 스윕; Bonferroni-유의한 lift 요구 |
| 표면 원인 vs 근본 원인 혼동 | "압력이 높았다" — 그런데 *왜* 압력이 높았는가? | 5-why 깊이 적용; 근본은 직접 변화의 상류 |
| 보정 없는 다중 검정 | "940 센서에 α=0.05로 47개 유의 요인 발견" | Bonferroni 또는 BH-FDR; SECOM 규모는 ~5% 거짓 양성 예상 |
| 효과 크기 없는 p-값 보고 | 5pp 계절성을 가진 지표의 0.1pp 이동에 "p < 0.0001" | 항상 p-값을 η² / Cohen's d / lift와 함께 보고 |

## 9. 성능 치트시트

| 이슈 | 해결 |
|---|---|
| > 100k 샘플에서 PELT가 느림 | `model="l2"` 사용(가장 빠름); `min_size`를 키워 후보 분기점 줄임 |
| `ruptures` 과세분 | `penalty` 올리기 (기본 BIC × 2); 시각적 정상성 점검 추가 |
| 큰 n에서 `dowhy` 느림 | 반박은 10k로 서브샘플; 점 추정에는 전체 데이터 |
| 많은 요인의 공통성 스윕이 느림 | `pd.crosstab`으로 벡터화; 요인별 루프를 `joblib`으로 병렬화 |
| 불균형 설계에서 ANOVA 충돌 | `anova_lm`에서 `typ=3`으로 Type-III SS 사용 |
| 성향 매칭이 너무 많은 단위를 떨어뜨림 | `caliper`를 성향 분포의 0.2σ로 완화 |

## 10. 종단 간 RCA 워크플로

```python
# 1. 변화 발생 확인: 지표에 CPD
bkps = detect_changepoints(metric_series, model="rbf", min_size=30)
# 2. 변화점 검증: pre vs post Mann-Whitney + 플롯
mw_p = stats.mannwhitneyu(pre, post).pvalue
assert mw_p < 0.01, "Change point not significant — investigate noise"

# 3. 공통성: 변경 후 어떤 요인이 과대 표집되었는가?
post_df = df.loc[df["timestamp"] >= change_time]
for factor in categorical_factors:
    sig = commonality(post_df, factor, "is_defective")
    # ... 후보 수집

# 4. 인과: 가장 강한 후보에 DiD
model = diff_in_diff(df, "timestamp", "treatment_group", "defect_rate",
                     change_time, treated_value=1, control_value=0)

# 5. 반박: 후보의 DAG로 dowhy
cm = CausalModel(data=df, treatment="candidate", outcome="defect_rate",
                 common_causes=identified_confounders)
estimate = cm.estimate_effect(cm.identify_effect(),
                              method_name="backdoor.propensity_score_matching")
refute = cm.refute_estimate(cm.identify_effect(), estimate, "random_common_cause")

# 6. 보고: 템플릿의 다섯 섹션을 모두 갖춘 구조화된 서술
```

이 파이프라인의 어느 단계라도 실패하면(CPD가 변화를 못 찾음, 공통성이
유의한 결과 없음, DiD가 효과 없음, 반박이 추정을 무너뜨림) — **데이터가
인과 주장을 뒷받침하지 못한다고 보고하고 추가 조사를 권장하세요**. RCA의
거짓 양성은 매우 비쌉니다: 진짜 이슈를 가리는 잘못된 수정을 트리거합니다.
