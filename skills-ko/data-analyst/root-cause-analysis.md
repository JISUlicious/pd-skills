# 근본 원인 분석 — 전문가 스킬

당신은 **제품 결함 조사, 수율 이상(yield excursion), 공정 드리프트, 사건
검토**를 지원하는 전문 데이터 분석가입니다. 여기의 방법론은 인과적이고
시간상 국소적입니다 — `feature-importance.md`의 연관적 순위와는 구별됩니다.
*범위:* 이것은 RCA의 분석적 D4(탐지 → 귀속 → 검증)를 다룹니다. 신뢰성
*수명* 모델링(Weibull / 중도절단 생존 — `lifelines` 사용)과 정식 결함
트리 분석(FTA)은 이 스킬이 구현하지 않는 인접 방법입니다.

핵심 규칙: **SHAP 중요도는 인과적 증거가 아닙니다.** `chamber_pressure`의
SHAP 값 0.5는 "이 피처가 예측 오차를 줄인다"를 말하는 것이지 — *"챔버
압력을 고치면 수율이 고쳐진다"*는 의미가 *아닙니다*. 근본 원인 작업은
결과 변화 *이전에* 일어난 *변화*를 식별한 뒤, 대안적 설명을 배제해야
합니다.

## 목차

- § 0: 8D 프레임워크 맥락 (D0-D8 + CAPA 참고)
- `feature-importance.md` 대비 이 파일을 사용할 때
- 환경 설정 — 선택적 의존성 + 시간 컷오프 분할
- § 1: 변화점 탐지 (PELT, CUSUM)
- § 2: 통계적 공정 관리 (Shewhart, EWMA, Cp/Cpk)
- § 6: RCA 의사결정 치트시트
- § 8: RCA 안티패턴
- § 9: 성능 치트시트
- § 10: 종단 간 워크플로

**확장 참조 (`SKILL.md`에서 레벨-1):**
- `rca-commonality.md`     — Fisher / BH-FDR / 클러스터 축소 (기존 § 3)
- `rca-causal-analysis.md` — DAG / DiD / dowhy / DOE (기존 § 4 + § 5)
- `rca-qualitative.md`     — 파레토 / 어골도 / 5-Why + 반증
- `rca-d5-verification.md` — 수정이 작동했는가?
- `rca-wafer-spatial.md`   — 웨이퍼 맵 / 공간적 결함 패턴
- `rca-reporting.md`       — Tier 2 템플릿 + 양식 가이드 + HTML (기존 § 7 + § 7a + § 7b)

## § 0. 8D 프레임워크 맥락

규제 제조 산업(자동차, 항공우주, 의료기기)에서 RCA는 **8D 프레임워크**
(포드의 원조 *Eight Disciplines*) 안에 자리 잡습니다. 이 스킬이
생성하는 Tier 2 보고서가 곧 **D4 증거 산출물**입니다 — 잘 수행된 D4는
D5-D8을 수월하게 만듭니다.

| 8D 단계 | 의미 | 이 스킬이 돕는 지점 |
|---|---|---|
| **D0** — 계획 | 비상 대응 기준 충족? | 범위 외 |
| **D1** — 팀 | 교차기능 팀 구성 | 범위 외 |
| **D2** — 문제 | 정량화된 문제 진술 | § 1 CPD가 변화 확인(지표, 시점, 크기) |
| **D3** — 봉쇄 | 고객 보호 임시 조치 | 범위 외, D3 상태는 Tier 2 헤더에 포함 |
| **D4** — 근본 원인 | *이것이 RCA.* | 전체 스킬 적용: §§ 1–2, `rca-commonality.md`, `rca-causal-analysis.md`, `rca-qualitative.md` |
| **D5** — 검증 | 수정이 작동함을 증명 | `rca-d5-verification.md` |
| **D6** — 실행 | 영구 조치 전개 | 범위 외 |
| **D7** — 예방 | SOP / 관리 계획 / FMEA 업데이트 | 유출 원인 보고 (`rca-causal-analysis.md` 참조) |
| **D8** — 축하 | 팀 인정 | 범위 외 |

**CAPA** (Corrective and Preventive Action)는 규제 산업에서 D5-D7을
지칭하는 이름입니다. FDA / ISO / IATF 감사가 요구하는 기록은 Tier 2
보고서의 엄격한 상위 집합입니다 — 같은 내용에 서명된 감사 기록이
추가됩니다. 분석을 CAPA 기록이 보고서에서 직접 추출될 수 있게
구조화하세요.

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

### 시간 인덱스 데이터 — 무작위 `train_test_split` 사용 금지

대부분의 RCA 데이터는 시간 인덱스 데이터입니다 (로트별 결함률, 센서별
측정값 시퀀스). 무작위 분할은 **시간적 누수**를 발생시킵니다: 훈련 행에
테스트 행 이후의 시점이 포함되어 모델이 암묵적으로 "미래를 봅니다." 이는
홀드아웃 R² / AUC를 조용히 부풀립니다.

```python
# 시간 인덱스 데이터에서는 잘못된 방법:
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# 올바른 방법 — 명시적 시간 컷오프:
cutoff = df["timestamp"].quantile(0.8)
train = df[df["timestamp"] < cutoff]
test  = df[df["timestamp"] >= cutoff]

# 또는 CV에는 sklearn의 TimeSeriesSplit:
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X.sort_index()):
    ...
```

무작위 분할은 행이 교환 가능할 때 — 일반적으로 분석 질문이 단면적(한
시점에서 모집단 비교)이고 시간적이지 않을 때 — 만 허용됩니다.

## 1. 변화점 탐지 (CPD)

이상(excursion)의 대표 질문은 **언제 시작되었는가?**입니다. 이동 평균은
답을 흐리게 만들기 때문에, 공식적인 분기점 탐지기가 필요합니다.

### 기본값: PELT (오프라인, 빠름)

```python
import numpy as np
import pandas as pd
import ruptures as rpt

def detect_changepoints(series, model="rbf", min_size=30, penalty=None,
                        ref_window=100):
    """지표 분포가 변하는 지점의 인덱스를 반환.

    series   : 1-D numpy 배열 또는 pd.Series (시간 순서 필수)
    model    : "rbf" (기본, 강건), "l2" (가우시안 평균 이동),
               "l1" (중앙값 이동, 이상치에 강건)
    min_size : 변화점 사이의 최소 샘플 수
    penalty  : None이면 모델별로 선택(아래 참조); 높이면 큰 변화만 탐지,
               낮추면 더 많이 탐지
    ref_window: model="l2"의 노이즈 스케일 추정에 쓰는 샘플 수
    """
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    if penalty is None:
        # 페널티는 선택한 모델의 비용 스케일과 맞아야 합니다.
        # - l2 비용은 잔차 제곱합 → 노이즈 σ²에 비례. var(s)가 아니라
        #   안정된 참조 윈도우에서 σ²를 추정하세요: 전체 분산은 탐지하려는
        #   바로 그 이동으로 부풀려져 페널티를 높이고 UNDER-분할합니다.
        # - rbf 비용은 정규화된 커널 Gram 행렬([0,1] 값) → O(1)이므로
        #   var(s)를 곱하는 것은 차원적으로 틀립니다. ~log(n)을 쓰고 아래
        #   시각적 점검으로 조정하세요.
        if model == "l2":
            sigma2 = float(np.var(s[:ref_window])) or float(np.var(s))
            penalty = np.log(n) * sigma2
        else:                         # rbf / l1
            penalty = np.log(n)
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
def cusum(series, target=None, sigma=None, k=0.5, h=5, ref_window=50):
    """양방향 CUSUM. +/- CUSUM이 h*sigma를 초과하는 인덱스를 반환.

    target : 기준 평균. None이면 처음 `ref_window` 샘플에서 추정 (전체 시리즈
             아님 — 그러면 탐지하려는 드리프트가 포함됨).
    sigma  : 기준 표준편차. None이면 같은 이유로 처음 `ref_window` 샘플에서 추정.
    k      : 표준편차 단위의 기준값 (0.5 = 1σ 이동에 민감)
    h      : 표준편차 단위의 결정 임계값 (5가 정통적인 기본값)

    target/sigma를 전체 시리즈에서 추정하는 것은 흔한 버그입니다: 드리프트된
    시리즈는 σ가 부풀려지고, 슬랙 k·σ가 너무 관대해져, 탐지하려던 드리프트를
    오히려 놓칩니다.
    """
    s = np.asarray(series, dtype=np.float64)
    ref = s[:min(ref_window, len(s) // 5)]               # 클린 기간 윈도우
    target = target if target is not None else ref.mean()
    sigma  = sigma  if sigma  is not None else ref.std()
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

    # X̄에 대한 Western Electric 룰. WE 존 규칙은 *한쪽 방향*입니다 —
    # "3개 중 2개가 2σ 초과"는 같은 쪽을 의미합니다. np.abs()를 쓰면
    # +2σ / −2σ 혼합에서도 발동하는데, 이는 WE 위반이 아닙니다. 또한
    # xbar는 NumPy 배열이므로 .rolling() 전에 pd.Series로 감싸야 합니다.
    dev    = pd.Series(xbar - cl_x)
    upper2 = (dev >  2 * sigma); lower2 = (dev < -2 * sigma)   # 2σ 초과, 쪽별
    upper1 = (dev >      sigma); lower1 = (dev <     -sigma)   # 1σ 초과, 쪽별
    flags = pd.DataFrame({"xbar": xbar})
    flags["rule_1"] = (dev.abs() > 3 * sigma)                                   # 1점 > 3σ
    flags["rule_2"] = ((upper2.rolling(3).sum() >= 2) |
                       (lower2.rolling(3).sum() >= 2))                          # 3개 중 2개 > 2σ, 같은 쪽
    flags["rule_3"] = ((upper1.rolling(5).sum() >= 4) |
                       (lower1.rolling(5).sum() >= 4))                          # 5개 중 4개 > 1σ, 같은 쪽
    flags["rule_4"] = (pd.Series((xbar > cl_x).astype(int)).rolling(8).sum()
                         .isin([0, 8]))                                         # 8연속 같은 쪽
    flags["any_violation"] = flags[["rule_1","rule_2","rule_3","rule_4"]].any(axis=1)
    return {"cl_x": cl_x, "ucl_x": ucl_x, "lcl_x": lcl_x,
            "cl_r": rbar, "ucl_r": ucl_r, "lcl_r": lcl_r,
            "sigma": sigma, "flags": flags}
```

### 느린 드리프트용 EWMA 차트

```python
def ewma_chart(values, lambda_=0.2, L=3, ref_window=50):
    """EWMA 관리도 — 작고 지속적인 드리프트에 민감.
    lambda_    : 가중 인자 (0.05–0.3); 작을수록 더 매끄럽고 느림
    L          : 시그마 단위의 관리 한계 폭 (기본 3)
    ref_window : target/sigma 설정에 쓰는 샘플 수. 전체 시리즈가 아니라
                 안정된 참조 윈도우에서 추정하세요 — 드리프트된 시리즈는
                 평균을 이상 쪽으로 끌어당겨 이상을 가립니다(CUSUM 절이
                 경고하는 바로 그 버그).
    """
    s = np.asarray(values, dtype=np.float64)
    ref = s[:ref_window]
    target, sigma = ref.mean(), ref.std()
    z = np.zeros(len(s)); z[0] = target
    for i in range(1, len(s)):
        z[i] = lambda_ * s[i] + (1 - lambda_) * z[i-1]
    # 정확한 시변 한계: 분산이 점근값을 향해 커지므로, 초기 점은
    # 정상상태 공식보다 더 TIGHT한 한계를 가집니다.
    i = np.arange(1, len(s) + 1)
    width = L * sigma * np.sqrt(lambda_ / (2 - lambda_)
                                * (1 - (1 - lambda_) ** (2 * i)))
    ucl, lcl = target + width, target - width
    violations = np.where((z > ucl) | (z < lcl))[0]
    return {"ewma": z, "ucl": ucl, "lcl": lcl, "violations": violations}
```

### 공정 능력 지수

```python
# R-bar 방법용 Hartley's d2 상수 (부분군 크기 2..10)
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
       7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

def capability(values, lsl, usl, subgroup_size=None, k=6):
    """규격한계 대비 Cp, Cpk (부분군 내 σ) 및 Pp, Ppk (전체 σ).

    subgroup_size : 주어지면 부분군 범위로부터 R-bar/d2 방식 (합리적 부분군 방법)
                    으로 σ_within을 추정합니다. 그렇지 않으면 σ_within = σ_overall이
                    되어 Cp == Pp가 되고, within/long-term 비교는 의미를 잃습니다.
    """
    s = np.asarray(values, dtype=np.float64)
    sigma_overall = s.std(ddof=1)
    mu = s.mean()

    if subgroup_size is not None and subgroup_size in _D2:
        n_sub = len(s) // subgroup_size
        sg = s[:n_sub * subgroup_size].reshape(n_sub, subgroup_size)
        rbar = (sg.max(axis=1) - sg.min(axis=1)).mean()
        sigma_within = rbar / _D2[subgroup_size]
    else:
        sigma_within = sigma_overall                  # 폴백; Cp == Pp

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
            "sigma_within": sigma_within, "sigma_overall": sigma_overall,
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

## 3. 공통성 분석

클러스터 축소 (RCA 전용 이동-점수 오버라이드 포함), 범주형 요인별
Fisher 정확검정 + BH-FDR, 다중 요인 조합의 빈발 항목집합 마이닝, 그리고
다중성 범위 경고는 `rca-commonality.md`를 참조하세요.

## 4-5. 인과 추론과 DOE

DAG + 뒷문 경로 프레임워크, 이중차분법(DiD), 성향 점수 매칭, 반박이
포함된 `dowhy`, 효과 크기 해석이 있는 DOE / ANOVA / Tukey HSD는
`rca-causal-analysis.md`를 참조하세요.

핵심 규칙은 변함없이 툴킷과 함께 있습니다: **예측은 인과가 아닙니다.**
X가 Y를 일으켰다고 주장하기 전에 역인과, 공통 원인, 선택 편향, 시간
불일치를 배제하세요.

## 6. RCA 의사결정 치트시트

```
START: 어떤 종류의 질문인가?
   │
   ├── "실패 모드 백로그가 큰데 어디서 시작?"
   │       └→ 파레토 (rca-qualitative.md) — 상위 20% 모드 = 손실의 80%
   │
   ├── "이 실패 모드의 가능한 원인은?"
   │       └→ 어골도 → 5-Why → 통계적 검증 (rca-qualitative.md)
   │
   ├── "Y가 오랜 기간 나쁘다" (정상 상태)
   │       └→ SPC + Cp/Cpk + 공통성 (§ 2 + rca-commonality.md)
   │
   ├── "Y가 시점 T에 이동했다" (이상)
   │       └→ CPD로 시점 확인 (§ 1)
   │          └→ pre-vs-post 공통성 (rca-commonality.md)
   │             └→ T에 알려진 변화가 있다면 DiD (rca-causal-analysis.md)
   │                └→ 인과 반박 (rca-causal-analysis.md: dowhy)
   │
   ├── "DOE를 수행했는데 무엇이 중요했는가?"
   │       └→ ANOVA + Tukey HSD + 효과 크기 (rca-causal-analysis.md)
   │
   ├── "결함에 (x, y) 좌표가 있다 — 공간 패턴?"
   │       └→ KDE + Ripley의 K + 패턴 분류 (rca-wafer-spatial.md)
   │
   └── "수정을 적용했는데 효과가 있었는가?"
           └→ 검정력 / pre-post / 규칙 안정성 (rca-d5-verification.md)

각 단계에서 물으세요: "Z에 의해 교란될 수 있는가?" 만약 그렇다면 Z를
조건화(회귀 / 매칭)하거나 한계를 명시하세요.
```

## 7. 보고

Tier 2 템플릿(§ 7), 양식 가이드와 작성 규율(§ 7a), 그리고 자체 완결적
HTML 출력 사양(§ 7b, 그대로 사용할 CSS 블록 포함)은
`rca-reporting.md`를 참조하세요.

## 8. RCA 안티패턴

| 안티패턴 | 증상 | 해결 |
|---|---|---|
| SHAP 순위를 인과 순위로 취급 | "가장 영향력 있는 피처가 X이니 X를 고친다" | 조치 권장 전에 DiD 또는 반박 단계 추가 |
| 시간 인덱스 데이터에 CPD 건너뛰기 | 시작 시점 확인 없이 이상 보고 | 항상 PELT 실행, 정상성 점검과 함께 변화점 보고 |
| 케이스 믹스 통제 없이 장비/작업자 비교 | 장비 A가 "최악으로 보임"이지만 어려운 로트만 처리 | 로트 유형으로 층화하거나 성향 매칭 사용 |
| 자기상관 데이터에 SPC | 거의 매 부분군마다 거짓 알람 | `series.autocorr()` 점검; > 0.5이면 AR(1) 잔차 차트화 |
| "가장 영향받은" 엔티티만 골라쓰기 | 헤드라인 원인이 시스템 드라이버가 아니라 이상치 | 모든 엔티티 스윕; Bonferroni-유의한 lift 요구 |
| 표면 원인 vs 근본 원인 혼동 | "압력이 높았다" — 그런데 *왜* 압력이 높았는가? | 5-why 깊이 적용; 근본은 직접 변화의 상류 |
| 보정 없는 다중 검정 | "940 센서에 α=0.05로 47개 유의 요인 발견" | Bonferroni는 family-wise 오류를 통제(*어떤* 거짓 양성이라도 나올 확률 ≤5% — 보수적); BH-FDR은 거짓 발견율을 통제(*선언된* 발견의 ~5%가 거짓). SECOM 규모에서는 보통 BH-FDR이 옳은 절충. |
| 효과 크기 없는 p-값 보고 | 5pp 계절성을 가진 지표의 0.1pp 이동에 "p < 0.0001" | 항상 p-값을 η² / Cohen's d / lift와 함께 보고 |
| 발생 원인만 보고 | "레시피 변경이 결함을 유발함 — 종료" (조용히: 모니터가 잡지 못했고, 재발할 것) | **발생(occurrence)** 과 **유출(escape)** 을 모두 보고. `rca-causal-analysis.md` § 4.5 참조. D7이 재발 방지를 위해 유출 원인이 필요합니다. |
| 검증 기간 없이 D5 성공 선언 | "지난주 결함률이 좋아 보였음, CAPA 종료" | `rca-d5-verification.md` 사용: ≥20 관리 상태 부분군 + 변동 점검 + 검정력 계획된 n |
| 어골도를 답으로 발표 | 덱에 6가지 화이트보드 사진, 검증 없음 | 어골도는 후보 생성기. 최소 상위 가지에 통계 검정을 붙이세요 (`rca-qualitative.md`). |
| 반증 없는 5-Why 추측 | 5개의 주장 사슬, 어느 링크에도 검정 없음 | 각 '왜'는 그것을 반박할 통계 검정을 지명해야 합니다. `rca-qualitative.md` 참조. |

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
