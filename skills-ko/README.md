# skills-ko — 한국어 번역본

`.claude/skills/data-analyst/`(영문 원본)의 한국어 번역본을 보관하는
디렉터리입니다. 디렉터리 구조와 파일명은 원본과 1:1 대응됩니다.

## 디렉터리 구조

```
skills-ko/
├── README.md                          ← 이 파일
└── data-analyst/
    ├── SKILL.md                       ← .claude/skills/data-analyst/SKILL.md
    ├── data-loading.md
    ├── data-exploration.md
    ├── data-cleaning.md
    ├── data-transformation.md
    ├── indexing-selection.md
    ├── merging-joining.md
    ├── time-series.md
    ├── statistical-analysis.md
    ├── performance-optimization.md
    ├── visualization.md
    ├── advanced-pandas.md
    ├── feature-importance.md
    └── root-cause-analysis.md
```

## 번역 원칙

1. **코드 블록은 그대로 유지** — 함수명, 변수명, API 호출, 라이브러리
   명(pandas, numpy, scikit-learn 등)은 번역하지 않습니다. 코드 안의
   한글 주석은 자연스러운 한국어로 작성하되 영문 코드와 명확히
   구분되도록 합니다.

2. **기술 용어는 한국어 + 괄호 영문 병기 (최초 등장 시)** — 검색 가능성과
   원본 문서 참조 편의를 위해 첫 등장에는 `다중공선성(multicollinearity)`
   처럼 영문을 병기하고, 이후에는 한국어 용어만 사용합니다.

3. **문서 구조는 영문 원본과 동일하게 유지** — 섹션 번호, 헤더 레벨,
   순서, 표의 행 수까지 동일하게 번역하여 영문 원본 업데이트 시
   diff 기반으로 동기화할 수 있도록 합니다.

4. **고유명사/제품명은 원어 유지** — Anthropic, Plotly, XGBoost, SHAP,
   PyArrow, Ridge, Lasso 등은 영문 그대로 사용합니다.

## 유지보수 흐름

영문 원본이 업데이트되면 다음 절차로 한국어 본을 동기화합니다:

```bash
# 1. 영문 원본의 변경 diff 확인
git log -p -- .claude/skills/data-analyst/data-exploration.md

# 2. 변경된 섹션을 한국어 본에서 찾아 번역
$EDITOR skills-ko/data-analyst/data-exploration.md

# 3. 한국어 본 변경사항을 같은 commit에 포함하지 말고
#    "Sync Korean translation: <원본 commit hash>" 메시지로 별도 커밋
git add skills-ko/data-analyst/data-exploration.md
git commit -m "Sync Korean translation: <영문 원본 commit hash>"
```

## 활성 스킬로 사용하려면

이 디렉터리는 **참고용 번역**이며, Claude Code 스킬로 자동 발견되지
않습니다(`.claude/skills/` 하위에만 자동 발견됨). 한국어 본을 활성
스킬로 사용하려면:

```bash
# 옵션 1: 영문 본을 한국어 본으로 교체 (한 번에 한 언어만 활성)
cp -r skills-ko/data-analyst/* .claude/skills/data-analyst/

# 옵션 2: 별도 스킬로 등록 (양쪽 동시 활성, name 충돌 방지를 위해
# SKILL.md frontmatter의 name을 변경 필요)
mkdir -p .claude/skills/data-analyst-ko
cp -r skills-ko/data-analyst/* .claude/skills/data-analyst-ko/
# SKILL.md의 name을 'data-analyst-ko'로 수정
```

## 용어 일관성 사전(요약)

| 영문 | 한국어 (병기 표기) |
|---|---|
| feature | 피처 (feature) |
| target | 타겟 (target) / 종속변수 |
| column | 칼럼 (column) / 열 |
| row | 행 |
| missing value, null | 결측값 (null) |
| duplicate | 중복 |
| outlier | 이상치 (outlier) |
| correlation | 상관계수 |
| multicollinearity | 다중공선성 (multicollinearity) |
| skewness | 왜도 (skewness) |
| kurtosis | 첨도 (kurtosis) |
| regression | 회귀 |
| standardized coefficient (β) | 표준화 계수 (β) |
| Variance Inflation Factor (VIF) | 분산팽창인수 (VIF) |
| imputation | 대체 / 결측 대체 |
| attenuation | 감쇄 |
| change-point detection (CPD) | 변화점 탐지 (CPD) |
| root cause analysis (RCA) | 근본 원인 분석 (RCA) |
| categorical | 범주형 |
| numerical | 수치형 |
| ordinal | 순서형 |
| holdout | 홀드아웃 (holdout) |
| permutation importance | 순열 중요도 (permutation importance) |
| informative missingness | 정보 있는 결측 (informative missingness) |
| tautology | 동어반복 (tautology) |
