# RCA 인과 추론과 계획된 실험

RCA에서 상관과 인과의 구분. 관측·소급 툴킷(DAG, DiD, 성향점수 매칭,
dowhy)과 계획된 실험 툴킷(ANOVA / Tukey / 효과 크기)을 모두 포함합니다.

## 목차

- 배제: 역인과 / 공통 원인 / 선택 편향 / 시간 불일치
- 최소 DAG 구축
- 이중차분법 (DiD)
- 성향 점수 매칭
- `dowhy` — 명시적 가정을 가진 원칙적 인과 추론
- **§ 4.5: 발생 원인 vs 유출 원인** (두 질문 프로토콜)
- DOE / ANOVA — 계획된 실험
- 효과 크기: 통계적 유의성 vs 실용적 유의성

이 파일을 읽는 경우: "X가 Y의 이동을 *일으켰는가*", "반사실 추정",
"레시피 변경을 롤백해야 하는가", "DOE를 수행했는데 무엇이 중요했는가"
관련 작업. `root-cause-analysis.md` § 1 (CPD가 pre/post 윈도우 제공)과
`rca-commonality.md` (DiD / dowhy에 넣을 후보 원인 발굴)와 함께
사용하세요.

## 배제부터

핵심 규칙: **예측은 인과가 아닙니다.** X가 Y를 일으켰다고 주장하기
전에 다음을 배제하세요:

1. **역인과** — Y가 먼저 변하고 X가 따라온 것은 아닌가?
2. **공통 원인** — X와 Y를 모두 변하게 만드는 상류의 Z가 있는가?
3. **선택 편향** — X가 위험 모집단에 과대 표집되었는가?
4. **시간 불일치** — X의 변화가 실제로 Y의 이동에 *선행*했는가?

## 최소 DAG (방향 비순환 그래프) 구축

SHAP / 공통성으로 떠오른 모든 후보 원인 X에 대해, 직접 최소 DAG를
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

## 이중차분법 (DiD) — pre/post 변화 이상의 핵심 도구

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
없었다면 처리군과 대조군이 함께 움직였을 것이라는 가정입니다. 항상 양
그룹의 pre 기간을 플롯하세요. T 이전부터 추세가 발산했다면 DiD는
무효입니다.

## 성향 점수 매칭 — 무작위화되지 않은 관측 비교

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

## `dowhy` — 명시적 가정을 가진 원칙적 인과 추론

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

강건한 인과 추정치는 세 반박을 거치고도 큰 변화 없이 유지됩니다.
플라시보 반박이 큰 효과를 보인다면(구조상 그래서는 안 되므로), 원래
추정치를 의심해야 합니다.

## § 4.5: 발생 원인 vs 유출 원인

모든 결함에는 **두 개의 원인**이 있고, D7 예방 조치 단계는 둘 다
필요합니다.

- **발생 원인 (occurrence cause)** — *왜 발생했는가?* 결함을 만들어낸
  물리 / 공정 사슬. 이를 되돌리는 것이 D6 시정 조치입니다.
- **유출 원인 (escape cause)** — *왜 감지가 임팩트 전에 잡지 못했는가?*
  결함이 빠져나가게 한 모니터링 / SPC / 감사 공백. 이를 닫는 것이 D7
  예방 조치입니다.

**발생 원인만 보고하는 것** — 가장 흔한 D4 실패 — 은 유출을 열어두어
발생 메커니즘이 다시 촉발되는 다음 번에 같은 결함이 재발하도록
만듭니다. "해결됨" 이후에도 같은 이상이 분기마다 재발하는 이유입니다.

**두 질문 프로토콜:** 통계적으로 검증하는 모든 발생 후보에 대해 두 번째
질문을 하세요:

> "그리고 어떤 모니터 / SPC 규칙 / 감사가 임팩트 전에 이것을 잡았어야
> *하는가* — 그리고 왜 잡지 못했는가?"

답이 유출 원인입니다. 다음 중 하나일 수 있습니다:
- 이 신호에 모니터가 존재하지 않았음 (이동된 센서에 SPC 추가)
- 모니터는 있었지만 민감도가 낮았음 (Shewhart X-bar가 sub-3σ 드리프트를
  놓침 → EWMA로 전환 — `root-cause-analysis.md` § 2 참조)
- 모니터는 있었지만 알람 임계값이 잘못됨 (재보정)
- 수동 감사를 건너뜀 (자동화)

**작동 예시 (SECOM 스타일):**
- 발생: 2008-08-21의 레시피 v3.2 배포가 s406 클러스터를 +4.5σ 이동
  → 결함률 4.8% → 14.0%. D6 조치: 레시피 롤백.
- 유출: s406에 기존 SPC는 Shewhart X-bar였는데, 레시피 v3.2가 유도한
  종류의 sub-3σ 지속 드리프트에 둔감. D7 조치: s406 차트를 EWMA
  (λ=0.2)로 전환, h=3.5σ에서 알람.

**보고 통합.** Tier 2 보고서 (`rca-reporting.md` § 7)는 후보 원인 표에
**발생 증거**와 **유출 증거** 별도 열을 가져야 합니다. 발생 증거는
있는데 유출 열이 없는 후보는 D7 공백이 분석되지 않았다는 신호입니다 —
D4로 되돌리세요.

## DOE / ANOVA — 계획된 실험

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
