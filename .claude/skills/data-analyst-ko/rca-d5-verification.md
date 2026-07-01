# D5 — 시정 조치의 검증

수정이 통계적 엄격성을 갖고 작동했음을 증명합니다. 검증 기간 없이
성공을 선언하는 것이 가장 흔한 D5 실패이고, 많은 "해결됨" RCA가 한
분기 안에 재발하는 이유입니다.

## 목차

- 표본 크기 계획 (검정력 분석)
- Pre/post 실험 설계
- 세 가지 검증 검정 (중심 경향 / 변동 / 규칙 안정성)
- 성공 기준 (모두 통과 필수)
- 안티패턴

이 파일을 읽는 경우: "수정을 적용했는데 효과가 있었는가?", "시정
조치를 종료하기 전에 얼마나 오래 모니터링?", "이번 주 지표가 좋아
보이는데 완료로 부를 수 있을까?". `root-cause-analysis.md` § 2 (검증
윈도우의 SPC 모니터링)와 `rca-causal-analysis.md` (병렬 대조 라인이
존재할 때 DiD)와 함께 사용하세요.

## 표본 크기 계획 (검정력 분석)

검증 기간을 시작하기 전, 선택된 검정력으로 목표 개선을 감지하는 데
실제로 얼마나 많은 사후 데이터가 필요한지 계산하세요. 작은 n으로
검증하고 완료로 부르면 거짓 통과율이 부풀려집니다.

```python
from statsmodels.stats.power import NormalIndPower

# D3(봉쇄)과 D4(근본 원인)의 입력값:
pre_std       = 0.030               # pre 기간 결함률 표준편차
target_delta  = 0.020               # 봐야 할 개선(2 pp)
effect_size   = target_delta / pre_std

analysis = NormalIndPower()
n_per_group = analysis.solve_power(
    effect_size=effect_size,
    alpha=0.05, power=0.80,
    ratio=1.0, alternative="larger",  # "개선했다"는 단측
)
print(f"검정력=0.80에서 그룹당 n≥{int(n_per_group)+1} 필요")
```

이항(결함 / 비결함) 지표의 경우, arcsin 변환된 비율에
`statsmodels.stats.power.NormalIndPower`를 사용하거나
`proportions_ztest`의 직접 검정력 계산을 사용하세요. 모든 검증 결과와
함께 `n_needed`를 보고하세요 — "2주 동안 실행함"은 설계가 아니고,
"380 웨이퍼가 필요했고, 415를 관측했으며, 검정력 0.84 달성"이
설계입니다.

## Pre/post 실험 설계

인프라에 따라 두 가지 설계 중 하나를 선택하세요:

- **병렬 라인 A/B** — 동일 레시피의 여러 동일 라인, 하나에만 수정 적용.
  가장 깨끗한 설계. 처리 효과 추정에 DiD(`rca-causal-analysis.md`) 사용.
- **시간 내 비교 + SPC** — 병렬 라인 없음. 사후 수정 윈도우를 매칭된
  사전 수정 윈도우(동일 길이, 동일 제품 믹스)와 비교. 모니터링 중 회귀
  감지에 SPC 사용.

두 설계 모두 실행 불가능하면(예: 대조 없는 일회성 장비), D5 결과를
"검증됨"이 아니라 "수정과 일치"로 라벨링하세요 — 인과 주장이 더 약합니다.

## 세 가지 검증 검정

세 가지 모두 통과해야 합니다. "평균이 내려갔다"는 단일 검정만으로는
충분하지 않습니다 — 변동과 규칙 안정성은 독립적으로 실패할 수 있습니다.

### 1. 중심 경향

사후 평균 vs 사전 평균 — 연속 지표는 Welch t-검정, 결함률은 비율
z-검정. 병렬 대조 라인이 존재할 때는 계절이나 드리프트 효과가 상쇄되는
DiD를 대신 사용.

```python
from scipy import stats
t_stat, p = stats.ttest_ind(post_samples, pre_samples, equal_var=False)
delta = post_samples.mean() - pre_samples.mean()
print(f"Δmean = {delta:+.4f}  Welch p = {p:.4f}")
```

### 2. 변동

사후 분산 vs 사전 분산 — Levene 검정(비정규성에 강건) 또는 고전적
F-검정. **평균을 낮추면서 분산을 두 배로 만드는 수정은 헤드라인 결함률을
줄이면서 새로운 실패 모드를 도입할 수 있습니다.** 이 검정이 그것을
잡습니다.

```python
lev = stats.levene(pre_samples, post_samples, center="median")
print(f"Levene p = {lev.pvalue:.4f}  ratio σ_post/σ_pre = "
      f"{post_samples.std()/pre_samples.std():.3f}")
```

### 3. 규칙 안정성

**Western Electric 규칙 하에서 N = 20개 연속 관리 상태 부분군.**
가장 강한 단일 검정입니다 — "공정이 변화가 순간적 노이즈가 아니라고
결론 내리기에 충분히 긴 윈도우 동안 안정적으로 유지되었다"고 말합니다.
어떤 WE 규칙 위반도 카운터를 리셋합니다.

```python
# root-cause-analysis.md § 2의 SPC 헬퍼 사용
subgroup_stats = compute_subgroups(post_samples, subgroup_size=5)
we_violations = check_western_electric(subgroup_stats)
consecutive_ok = count_consecutive_in_control(we_violations)
print(f"연속 관리 상태 부분군: {consecutive_ok}/20")
assert consecutive_ok >= 20, "D5 검증 미완 — 모니터링 연장"
```

## 성공 기준 (모두 통과 필수)

D5를 사인 오프하기 전에 네 기준 모두 성립해야 합니다:

- **Δmean ≥ 목표 개선** (D5 킥오프에서 정의됨, 데이터를 본 후 조정 금지)
- **사후 σ ≤ 사전 σ** (또는 사전 선언된 허용치 내의 Δσ)
- **≥ 20 관리 상태 부분군** Western Electric 규칙 하 사후 수정
- **검증 기간 중 WE 규칙 위반 없음**

어느 하나라도 실패하면 팀은 D4로 돌아가 근본 원인이 올바르게 식별되었는지
재고려하거나, D6로 가서 시정 조치를 강화해야 합니다.

## 안티패턴

| 안티패턴 | 증상 | 해결 |
|---|---|---|
| 좋은 일주일에 성공 선언 | "지난주 결함률이 떨어짐, D5 종료" | 검증 기간은 공정의 자연 변동 주기의 ≥ 2배여야 합니다. 단일 윈도우가 아니라 SPC 규칙 안정성을 사용하세요. |
| 사후 목표 조정 | "목표는 2 pp 개선; 1.4를 얻음 — 충분" | D5 킥오프의 목표는 구속력. 목표 미달은 D4 또는 D6 재검토 필요. |
| 변동 점검 없음 | "Δmean이 유의, 완료" | Levene 검정 실행. 같은 평균 + 두 배 분산 = 더 나쁜 공정. |
| 검정력 계산 없음 | "2주 모니터링했음" | `n_needed`와 `power_achieved`를 명시. 검정력 부족 검증은 우연으로 통과. |
| 사용 가능한데 대조 비교 없음 | "사후 수정률이 사전 수정률보다 낮음, 완료" | 병렬 미영향 라인이 존재하면 DiD 사용. 계절이나 공정 전반의 드리프트가 거짓 pre/post 차이를 만들 수 있습니다. |
