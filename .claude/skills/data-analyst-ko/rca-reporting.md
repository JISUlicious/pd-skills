# RCA 보고 — Tier 2 템플릿 + HTML 출력

산출물은 SHAP 플롯이 아니라 증거를 가진 **인과적 서술**입니다. 이 파일은
보고 산출물을 다룹니다: 템플릿, 작성 규율, 자체 완결적 HTML 렌더링.

## 목차

- § 7: Tier 2 RCA 보고 템플릿 (보정용 참조)
- § 7a: Tier 2 보고서 작성법 — 양식 가이드 + 규율
  - 사전 조건 체크리스트
  - 섹션별 규율
  - 도메인별 슬롯 매핑
  - 분량 예산
  - 품질 체크리스트 (배포 전)
  - 세 가지 실패 모드
- § 7b: HTML 보고서 생성 — 자체 완결적 파일 사양
  - HTML을 언제 쓰는가 vs 마크다운
  - 출력 사양
  - CSS 블록 (그대로 사용)
  - 섹션별 HTML 요소
  - 생성 방식 — 작업당 스크립트
  - 렌더링 예시
  - 품질 체크리스트 (HTML 전용)
  - 세 가지 실패 모드 (HTML 전용)

이 파일을 읽는 경우: Tier 2 RCA 보고서 작성, 이해관계자용 HTML 산출물
생성, 또는 "발견 메모(Findings memo)"로 격하될 규율 공백을 감사할 때.
`SKILL.md`에서 레벨-1 참조: `root-cause-analysis.md` §§ 1–2,
`rca-commonality.md`, `rca-causal-analysis.md`.

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
| 후보 | 발생 증거 | 유출 증거 (왜 감지가 잡지 못했는가?) | 배제할 교란변수 |
|---|---|---|---|
| T-2의 레시피 v3.2 배포 | Lift=4.1, Fisher p=0.0003 | s406의 Shewhart X-bar가 sub-3σ 드리프트에 둔감 | 같은 날 장비 정비 |
| 챔버 5 (장비 A) | 결함률 12% vs 베이스라인 2% | 이 지표에 챔버별 알람 없음 | 로트 믹스 변화 |
| 신규 작업자 시프트 | 결함률 8% vs 베이스라인 2% | 시프트 단위 SPC 없음 | 야간 시프트 하드웨어와 교란 |

### 3. 인과 분석
- DiD 추정 (레시피 v3.2): +5.2pp 결함률, 95% CI [3.1, 7.3]
- 반박: 무작위 공통원인 Δ=+0.1, 플라시보 Δ=−0.05 → 강건
- 평행추세 점검: ✓ (부록 플롯)

### 4. 권장 조치 (D6 시정 + D7 예방)
- **D6 시정:** 레시피 v3.2 롤백 (가장 강한 증거)
- **D7 예방:** s406 SPC 차트를 EWMA λ=0.2로 전환 (유출 원인 해소)
- 다음 5개 로트에 대해 챔버 5 샘플링 증가 (보조 신호)
- 야간 시프트 훈련 감사 (최저 우선순위 — 교란됨)

### 5. 검증 계획 (D5)
- 검정력 계획된 n = 380 웨이퍼 사후 수정 (목표 Δ = 2pp, α=0.05, 검정력=0.80)
- 성공 기준: Δmean ≥ 2pp, 사후 σ ≤ 사전 σ, 20 관리 상태 부분군,
  WE 규칙 위반 없음. `rca-d5-verification.md` 참조.

### 6. 배제할 수 없는 사항
- 같은 주 미관측 공급자 자재 드리프트
- 챔버 5 픽스처 마모 (계측 없음)
```

## 7a. Tier 2 RCA 보고서 작성법

위의 § 7 템플릿은 산출물이고, 이 절은 임의의 분석 결과에서 그 산출물을
만들어내는 규율입니다.

### 사전 조건 체크리스트

메모장에 복사해 체크하며 확인하세요. 어느 항목이라도 통과하지 못하면
"발견 메모(Findings memo)"이지 RCA 보고서가 아닙니다. 라벨을 바꾸고
§ 인과분석 + § 권장 조치는 삭제하세요.

```
RCA 사전 조건:
- [ ] 확정된 변화점: p<0.01, 양쪽 각 30 샘플 이상, 시각적으로 명확
- [ ] 전처리 로그: 칼럼별 결측 처리, 공선 클러스터 축소
- [ ] 상위 3개 후보 원인 순위 + 각 항목의 가장 강한 증거 수치
- [ ] #1 드라이버의 사분위 (또는 수준별) 분포와 타겟 관계
- [ ] #1에 대한 인과 추정치(DiD / 매칭) + 최소 1개의 반박(refutation)
- [ ] 대안 원인이 될 수 있는 미관측 변수의 구체적 목록
```

### 섹션별 규율

다음 순서대로 작성하세요. TL;DR은 마지막에 씁니다.

| 섹션 | 강제되는 규율 |
|---|---|
| **TL;DR** (1–2문장) | 확언. 하나의 원인, 하나의 조치. 애매한 표현, "또는", 날짜 없는 표현 금지. |
| **변화(The change)** | 이동이 노이즈가 아니라 실제임을 증명. 방법 + p-값 + 양쪽 n + 시각적 참조 필수. |
| **데이터 & 방법** (표 2개) | 감사 가능한 전처리. 전처리 표 열: 단계 / 조치 / 이유. 방법 표 열: 질문 / 방법 / 이 방법을 쓴 이유. |
| **신호가 어디에 있는가** | 실행 가능한 임계값. #1 드라이버의 사분위 표 + 임계 문장: "피처 > 값이면 기저 대비 N배". |
| **상위 후보 원인** (최대 3행) | 순위 + 무결성. 모든 행에 교란변수 열 필수 — 비어 있으면 비판적 사고가 없었다는 신호. |
| **인과분석** (#1만) | 상관 vs 인과. 추정치 + CI + 반박 1회 이상. 세 후보 모두에 적용하면 보고서가 희석됨. |
| **권장 조치** (번호 리스트) | 운영자가 실행 가능한 무언가. #1은 되돌릴 수 있고 증거가 강해야 함. "추가 조사"가 #1이면 분석이 미완. |
| **배제할 수 없는 사항** (≥3개) | 무결성. 이 절이 비어 있으면 과신, 일반적 면책 문구("데이터가 불완전할 수 있음")는 무의미. |

### 도메인별 슬롯 매핑

템플릿은 동일하고 명사만 바뀝니다.

| 슬롯 | 제조 | 웹 프로덕트 | 헬스케어 | 금융 | SRE / 운영 |
|---|---|---|---|---|---|
| 지표 | 결함률 | 전환율, 이탈률 | 재입원율, 합병증률 | 부도율, 사기율 | 오류율, p99 지연 |
| 변경 주체 | 레시피, 장비 정비 | 피처 플래그, 배포, A/B | 프로토콜, 처방집 | 모델 버전, 정책 | 배포, 설정 푸시 |
| 코호트 | 로트, 웨이퍼 | 유저, 세션 | 환자, 병동 | 계좌, 거래 | 요청, 호스트 그룹 |
| 드라이버 형태 | 센서 판독 | 세션 지속 시간 | 랩 값, 스코어 | 리스크 스코어, 속도 | 큐 깊이, GC 지연 |
| 인과 레버 | 레시피 롤백 | 플래그 롤백 | 프로토콜 되돌리기 | 모델 롤백 | 배포 되돌리기 |

### 분량 예산

- **적정선:** 60–75줄 / 450–550 단어.
- **상한:** 90줄 / 약 700 단어. 초과하면 부록으로 이동.
- **하한:** 40줄. 미만이면 누락된 섹션이 있음.

### 품질 체크리스트 (배포 전)

- [ ] TL;DR의 원인이 § 권장 조치 #1과 일치 (같은 명사, 같은 수치)
- [ ] 보고서의 모든 수치가 분석 로그에 존재 — 보고서에서 새로 만들지 않았음
- [ ] § 신호가 어디에 있는가 → 구체적 임계 문장을 만들어냈음
- [ ] § 상위 후보가 3행 이하, 모든 행에 교란변수가 명시됨
- [ ] § 인과분석은 #1만 다루고 반박 검정을 최소 1회 포함
- [ ] § 권장 조치 #1이 되돌릴 수 있고 증거가 강함
- [ ] § 배제할 수 없는 사항이 구체적 공백을 나열 (일반적 면책 문구 아님)
- [ ] TL;DR을 마지막에 작성

### 세 가지 실패 모드와 해결

1. **SHAP 플롯을 산문으로 감싼 보고서** — 사분위 표도, 임계값도 없어 독자가
   행동할 수 없음. → § 신호가 어디에 있는가에 사분위 표 + 임계 문장 추가.
2. **세 후보 모두에 인과분석을 적용** — 주의가 희석되고 남획으로 읽힘. →
   § 인과는 #1만 다루고 나머지는 부록으로 강등.
3. **§ 권장 조치 #1이 "추가 조사"** — 분석이 끝나지 않음. → 되돌릴 수 있는
   첫 단계(롤백, 임계 강화, 샘플링 증가)를 찾거나, 문서 라벨을 "발견 메모"로
   바꾸고 § 인과 + § 조치를 삭제.

## 7b. HTML 보고서 생성하기

이해관계자 전달, PDF 인쇄, 이메일 첨부 용도로는 Tier 2 보고서를 자체
완결적인 HTML 파일로 생성합니다. § 7a의 모든 규율은 유지되고, 이 절은
렌더링을 다룹니다.

### HTML을 언제 쓰는가 vs 마크다운

| 기능 | 마크다운 | HTML |
|---|:---:|:---:|
| Plotly 인터랙티브 플롯 인라인 | ✗ | ✓ |
| 접히는 details / 확장 가능한 섹션 | ✗ | ✓ |
| 페이지 나눔이 있는 PDF 인쇄 | ~ | ✓ |
| 브라우저에서 오프라인 실행 | ~ | ✓ |
| 이메일에 서식 보존하며 붙여넣기 | ✗ | ✓ |

내부 노트북 아티팩트와 스크래치 분석용은 마크다운, 이해관계자에게 보이는
것은 HTML.

### 출력 사양

**한 파일. `.html`. 자체 완결적.**

- 단일 `<html>` 문서. 외부 CSS 없음, Plotly CDN 외 외부 JS 없음.
- 인라인 `<style>` 블록에 아래 CSS를 그대로 삽입(팔레트는 브랜드 지침이
  있을 때만 조정).
- Plotly 그림은 `fig.to_html(include_plotlyjs="cdn", full_html=False)`로
  임베드 — 첫 그림에서만 CDN 로드, 이후 그림은 `include_plotlyjs=False`.
- Matplotlib 폴백 플롯: base64 `<img src="data:image/png;base64,…">`.
- Plotly 외 JavaScript 사용 금지. Plotly가 실행되지 않아도 수치 + 표는
  그대로 보이도록 우아하게 저하됨.
- 파일 크기 목표: 500 KB 미만. 초과 시 무거운 플롯은 `_appendix.html`로
  분리하고 § 부록에서 링크.

### CSS 블록 — 그대로 사용

```css
:root {
  --fg: #1a1a1a; --muted: #6b6b6b; --bg: #ffffff;
  --border: #e5e5e5; --code-bg: #f7f7f7;
  --sev: #c62828; --ok: #2e7d32; --accent: #1565c0;
}
* { box-sizing: border-box; }
body {
  font: 14px/1.55 ui-sans-serif, -apple-system, system-ui, sans-serif;
  color: var(--fg); background: var(--bg);
  max-width: 880px; margin: 32px auto; padding: 0 24px;
}
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 8px; padding-bottom: 4px;
     border-bottom: 1px solid var(--border); }
h3 { font-size: 14px; margin: 20px 0 6px; color: var(--muted); }
.meta { color: var(--muted); font-size: 13px; margin: 0 0 20px; }
.tldr { padding: 12px 16px; border-left: 3px solid var(--sev);
        background: #fff5f5; margin: 16px 0; }
.tldr strong { color: var(--sev); }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px;
        font-size: 13px; }
th, td { border-bottom: 1px solid var(--border); padding: 6px 10px;
         text-align: left; vertical-align: top; }
th { background: #fafafa; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code, pre { background: var(--code-bg); border-radius: 3px;
            font: 12.5px/1.4 ui-monospace, Menlo, monospace; }
code { padding: 1px 5px; }
pre { padding: 10px 12px; overflow-x: auto; }
.threshold { padding: 10px 14px; background: #fffde7;
             border-left: 3px solid #f9a825; margin: 8px 0; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
         font-size: 11px; font-weight: 600; text-transform: uppercase; }
.sev-high { background: var(--sev); color: white; }
.sev-med  { background: #ef6c00; color: white; }
.sev-low  { background: var(--muted); color: white; }
details { margin: 8px 0; }
summary { cursor: pointer; color: var(--accent); font-weight: 500; }
aside { padding: 10px 14px; background: #f5f9ff;
        border-left: 3px solid var(--accent); margin: 8px 0; }
@media print {
  body { max-width: none; margin: 0; padding: 12mm; }
  h2 { page-break-before: auto; page-break-inside: avoid; }
  details { page-break-inside: avoid; }
  details:not([open]) summary::after { content: " (expand in appendix)"; }
}
```

### 섹션별 HTML 요소

§ 7a와 동일한 8개 섹션. 섹션별 요소 선택:

| 섹션 | HTML 요소 |
|---|---|
| 헤더 | `<header>` + `<h1>` + 심각도 `<span class="badge sev-…">` + `<p class="meta">` (작성자, 날짜, 소스) |
| TL;DR | `<section class="tldr">` — 색상 좌측 보더, 큰 글꼴 |
| 변화 | `<section>` + `<dl>` (지표 / 방향 / 변화점) + 변화점 수직선을 그은 Plotly 라인 차트 |
| 데이터 & 방법 | `<table>` 2개를 `<details><summary>` 접기 안에 배치 (기본 접힘) |
| 신호가 어디에 있는가 | 사분위 표 `<table>` + `<p class="threshold">` 임계 문장 + Plotly 막대 차트(옵션) |
| 상위 후보 원인 | `<table>` — 최대 3행, 수치는 `<code>` |
| 인과분석 | `<section>` + 추정치·CI용 `<aside>` 박스 + 반박은 `<ul>` |
| 권장 조치 | `<ol>` + 동사에 `<strong>` |
| 배제할 수 없는 사항 | `<ul>` |
| 부록 | `<details>` 접기, 링크·썸네일 리스트 |

Details 접기는 데이터 & 방법 + 부록에만 사용. 상위 후보, 임계 문장, 권장
조치는 절대 접지 마세요 — 첫 화면에서 보여야 합니다.

### 생성 방식 — 작업당 스크립트

파이썬 헬퍼는 번들하지 않습니다. 분석 아티팩트로부터 페이로드를 계산하고,
Plotly 프래그먼트를 렌더링하고, 단일 HTML 파일로 포맷팅하는 스크립트를
작업마다 생성합니다. 스켈레톤:

```python
# 1. 분석 아티팩트에서 페이로드 계산 (섹션별 dict)
payload = {
    "metric": "defect_rate",
    "date": "2026-04-12",
    "severity": "high",              # → 배지의 CSS 클래스
    "tldr": "defect_rate가 recipe v3.2 배포에서 2.1% → 10.5%로 상승. 롤백 권장.",
    "before": 0.021, "after": 0.105, "delta_pp": 8.4,
    "changepoint": {"ts": "2026-04-12 14:00", "method": "PELT + BIC",
                    "test": "MW", "p": 0.001, "n_pre": 200, "n_post": 180},
    "prep_rows": [...],              # (step, action, why) 리스트
    "methods_rows": [...],           # (question, method, why) 리스트
    "quintile_rows": [...],          # (label, range, n, rate, delta) 리스트
    "threshold_sentence": "s406 > 6.8 → 기저 대비 4배",
    "candidates": [...],             # (rank, name, evidence, confounders) 리스트
    "causal": {"est": +0.052, "ci_lo": +0.031, "ci_hi": +0.073,
               "refute_random": +0.001, "refute_placebo": -0.005,
               "refute_subset": +0.003, "parallel_ok": True},
    "actions": ["레시피 v3.2 롤백 (~2h, 되돌릴 수 있음)", ...],
    "cannot_rule_out": ["공급자 자재 드리프트 (로트별 데이터 없음)", ...],
}

# 2. Plotly 프래그먼트 렌더링 (CDN은 첫 프래그먼트에만 로드)
fig1_html = fig_metric_over_time.to_html(include_plotlyjs="cdn",  full_html=False)
fig2_html = fig_quintile_bar.to_html(include_plotlyjs=False, full_html=False)

# 3. f-string 템플릿으로 파일 조립. 구조는 위 요소 매핑 표를 그대로 따르세요.
#    조건 분기는 한 단계 이상 깊어지면 가독성이 급락합니다.
html = f"""<!doctype html><html lang="en"><head>
  <meta charset="utf-8"><title>RCA: {payload['metric']} on {payload['date']}</title>
  <style>{CSS}</style></head><body>
  <header>...</header>
  <section class="tldr">...</section>
  ...
</body></html>"""

# 4. 디스크에 기록
from pathlib import Path
out = Path(f"reports/rca_{payload['metric']}_{payload['date']}.html")
out.parent.mkdir(exist_ok=True); out.write_text(html)
print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
```

**페이로드 분리 패턴:** 여러 청중(임원 / 감사 / 엔지니어링)용으로 재생성이
필요한 보고서는 페이로드를 `payload.json` 사이드카에 쓰고 템플릿 스크립트에서
로드하세요. 분석 1회 + 페이로드 1개 + 템플릿 1개 → 다양한 렌더링 산출물.

### 렌더링 예시 — 헤더 프래그먼트

보정용 참조. 렌더링된 보고서 상단의 HTML 소스:

```html
<header>
  <h1>RCA: defect_rate shifted on 2026-04-12
      <span class="badge sev-high">High</span></h1>
  <p class="meta">Author: J. Kim &middot; Generated 2026-05-19 &middot;
     Source: fab_metrics.parquet + mes_events.csv</p>
</header>

<section class="tldr">
  <strong>TL;DR.</strong> defect_rate가 4월 12일 14:00에 2.1% → 10.5%로
  상승, recipe v3.2 배포 시각과 일치. DiD 추정치 +5.2pp
  (95% CI: +3.1 ~ +7.3), 반박 통과. <strong>롤백 권장.</strong>
</section>

<section>
  <h2>변화</h2>
  <dl>
    <dt>지표</dt> <dd>defect_rate (50-로트 롤링 윈도우)</dd>
    <dt>방향</dt> <dd>2.1% → 10.5% (Δ = +8.4pp, 기저 대비 5배)</dd>
    <dt>변화점</dt>
    <dd>2026-04-12 14:00 — PELT (BIC); MW p=0.001;
        pre n=200 / post n=180 </dd>
  </dl>
  <!-- 변화점 수직선이 포함된 metric-over-time Plotly 프래그먼트 -->
  {fig_metric_over_time_html}
</section>
```

### 품질 체크리스트 (HTML 전용, § 7a 위에 추가)

- [ ] Chrome, Safari, Outlook 프리뷰에서 정상 표시
- [ ] 2–3페이지로 인쇄되고 페이지 나눔이 자연스러움
- [ ] `<title>`이 `<h1>`과 일치 (브라우저 탭 / 이메일 제목 프리뷰)
- [ ] 헤더 심각도 배지가 TL;DR의 어조와 일치
- [ ] TL;DR의 수치가 본문의 수치와 정확히 일치
- [ ] 모든 표가 실제 `<table>` — 표를 이미지로 렌더한 플롯 금지
- [ ] `<details>` 접기는 데이터 & 방법 + 부록에만 적용
- [ ] Plotly `include_plotlyjs="cdn"`은 첫 그림만
- [ ] `<img src>` 나 `<a href>`에 절대 경로 사용 금지 (base64 또는 상대경로)
- [ ] 파일 크기 500 KB 미만

### 세 가지 실패 모드 (HTML 전용)

1. **수치 하나 바꾸려 스크립트를 재편집해야 함.** → `payload.json`을 템플릿
   스크립트와 분리.
2. **Outlook / Slack 프리뷰에서 Plotly가 렌더링 안 됨.** → matplotlib PNG를
   base64로 임베드한 `_static.html` 동반 파일 배포.
3. **플롯이 임베드되면서 파일이 500 KB 초과.** → 무거운 플롯은
   `_appendix.html`로 옮기고 § 부록에서 링크.
