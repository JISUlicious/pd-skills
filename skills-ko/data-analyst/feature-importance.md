# 피처 중요도 & SHAP — 전문가 스킬

당신은 전문 데이터 분석가입니다. 선형 분석(Spearman ρ, 표준화 OLS β +
ΔR²)이 부족하고 타겟 변수에 대해 **비선형 중요도**, **상호작용 탐지**,
또는 **행 단위 설명**이 필요할 때 다음 방법론을 적용합니다.

이 파일은 `statistical-analysis.md`의 **보완**이지 대체가 아닙니다. 항상
선형 분석을 먼저 수행하세요 — 더 빠르고, 더 해석 가능하며, 종종
충분합니다. 아래 조건이 충족되었을 때만 ML 중요도로 에스컬레이트합니다.

## 목차

- 에스컬레이트 시점 (선형 부족 → ML 중요도)
- 환경 설정 — 선택적 의존성
- 모델링 전 진단: VIF (§ 1) · 대표 선택 (§ 1a) · 타겟 왜도 (§ 1b) ·
  결측 처리 (§ 2)
- **계산 메서드 → `importance-methods.md`** (XGBoost 레시피,
  gain / 순열 / SHAP, 메서드 간 점검, 상호작용 탐지)
- 안티패턴 / 누수
- 보고 템플릿
- 성능 + 의사결정 치트시트

## 에스컬레이트 시점

> **질문이 *무엇이 변화를 일으켰는가?*(*수준을 무엇이 예측하는가?*가
> 아니라)라면 먼저 `root-cause-analysis.md`를 보세요.** 피처 중요도는
> 연관적입니다. 근본 원인 분석에는 변화점 탐지(CPD), 공통성 분석, 그리고
> 인과 추론 도구가 필요하며 이 파일은 그것들을 다루지 않습니다.

다음 중 **하나라도** 해당하면 선형 → 트리 기반 중요도로 에스컬레이트합니다:

| 트리거 | 선형 분석이 부적절한 이유 |
|---|---|
| 모든 가능한 요인을 포함했는데 선형 모델 R² < 0.4 | 비선형 또는 상호작용 효과가 지배적 |
| 잔차 플롯에서 곡률, U자형, 불연속이 보임 | 선형 β는 비선형 신호를 단일 기울기로 압축 |
| 도메인 근거로 상호작용 의심 (예: "Y가 높을 때 X가 더 중요") | 가법 선형 모델은 상호작용을 표현할 수 없음 |
| 이해관계자가 "*이 개인*이 왜 높/낮은가?"를 물음 | β는 모집단 수준이며, SHAP은 행 단위 |
| 예측 변수가 강하게 왜곡되거나, 영-팽창되었거나, 클리핑됨 | OLS 계수 추론은 편향됨; 트리는 강건함 |
| Spearman ρ 순위와 표준화 β 순위가 일치하지 않음 | 두 관점을 중재할 세 번째 시각이 필요 |

이 중 어느 것도 해당하지 않으면 **선형 결과를 보고하고 멈추세요** — ML을
더하면 의존성과 해석 부담만 늘 뿐 인사이트는 얻지 못합니다.

## 환경 설정 — 선택적 의존성

`xgboost`, `shap`, `scikit-learn`은 기본 데이터 분석 환경의 일부가
아닙니다. 항상 import를 보호하고 폴백 메시지를 제공하세요. **`ImportError`만
잡지 말고 `Exception`을 잡으세요** — macOS에서 `libomp`가 없으면 xgboost는
import 시점에 `XGBoostError`를 발생시키며 이는 `ImportError`가 아닙니다:

```python
try:
    import xgboost as xgb
    import shap
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import r2_score, mean_absolute_error
    HAS_ML = True
except Exception as e:
    HAS_ML = False
    print(f"ML stack unavailable ({type(e).__name__}: {e}).")
    print("Install with: uv pip install xgboost shap scikit-learn")
    print("On macOS, xgboost also needs OpenMP: brew install libomp")
```

`HAS_ML`이 False라면 `statistical-analysis.md`의 선형 분석으로 폴백하고,
ML 교차 검증은 건너뛰었음을 사용자에게 명시적으로 알립니다.

## 모델링 전 진단(Pre-Modeling Diagnostics)

중요도 모델을 적합시키기 전 두 가지 점검이 필수입니다. 몇 초면 끝나지만
가장 흔한 두 가지 무음 실패 — 공선 예측 변수가 좌우하는 순위, 결측 제거
이후 남은 행이 좌우하는 순위 — 를 막아줍니다.

### 1. 다중공선성(multicollinearity) 감사 — VIF

칼럼 j에 대한 VIF(분산팽창인수, Variance Inflation Factor)는
`1 / (1 − R²_j)`로 정의됩니다. `R²_j`는 칼럼 j를 다른 모든 예측 변수에
대해 회귀했을 때의 결정계수입니다. 해석:

| VIF | 의미 | 조치 |
|---:|---|---|
| 1 | 독립 | 없음 |
| 1–5 | 약함 | 없음 |
| 5–10 | 중간 | OLS β 불안정; SHAP/순열 교차 검증에 의존 |
| > 10 | 심각 | 중복 쌍 중 하나를 제거하거나, Ridge/Lasso로 전환 |

```python
import numpy as np
import pandas as pd

def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """VIF_j = 1 / (1 - R²_j), R²_j는 칼럼 j를 나머지에 회귀한 결과."""
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
            vif = float("inf")
        else:
            r2 = 1 - ((y - yhat) ** 2).sum() / ss_tot
            vif = float("inf") if r2 >= 0.9999 else 1 / (1 - r2)
        rows.append({"feature": col, "R²_on_others": round(float(r2), 4),
                     "VIF": round(float(vif), 2)})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)

vif = vif_table(X)
print(vif.to_string(index=False))

severe   = vif.query("VIF > 10")["feature"].tolist()
moderate = vif.query("5 < VIF <= 10")["feature"].tolist()
if severe:
    print(f"⚠ SEVERE multicollinearity (VIF > 10): {severe}")
    print("  → drop one of each redundant pair OR switch to RidgeCV / LassoCV")
elif moderate:
    print(f"⚠ Moderate multicollinearity (VIF 5-10): {moderate}")
    print("  → linear β is unstable; trust SHAP & permutation rankings over β")
```

VIF > 10일 때, 선형 베이스라인으로는 정규화 회귀를 선호합니다:

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xz = scaler.fit_transform(X)
yz = (y - y.mean()) / y.std()
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xz, yz)
ridge_coefs = pd.Series(ridge.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print(f"Best α: {ridge.alpha_}, R²: {ridge.score(Xz, yz):.4f}")
print(ridge_coefs)
```

항상 **VIF를 보고에 포함**하세요 — OLS β 값을 독립 효과로 해석할 수 있는지를
독자에게 가장 간결하게 알리는 방법입니다.

### 1a. 제거할 때 대표(representative) 선택하기

`collinearity-diagnostics.md`의 프레임워크(6-우선순위 선택자 +
`select_cluster_representative()` 헬퍼)를 사용하세요.

피처 중요도 작업의 경우, **우선순위 3은 기본값인 `|ρ(target)|`
점수**입니다. 제거/유지 결정이 이후 교차 검증된 R²에 전파되어 발생할 수
있는 소프트 타겟 누수(soft target leakage)를 피하려면 **훈련 폴드에서만**
계산하세요:

```python
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
fold_mask = X.index.isin(X_tr.index)

for cluster in find_clusters(X[numeric_cols]):
    keep, drop, reason = select_cluster_representative(
        X, y, cluster, fold_mask=fold_mask
    )
    # 양쪽 폴드에 제거 적용한 뒤 재적합
```

제거 후, 재적합하고 다음 세 가지를 검증합니다:
1. R²가 눈에 띄게 떨어지지 않음 (남긴 피처가 제거된 신호를 흡수)
2. 남긴 피처의 표준화 β / SHAP 크기가 빈 자리를 메우면서 증가
3. 남긴 피처의 VIF가 < 5로 복귀 (클러스터가 완전히 포착됨)

홀드아웃에서 R²가 0.05 초과로 떨어지면, 클러스터에 정말로 구별되는
신호가 있던 것 — 제거하지 말고 RidgeCV로 전환하세요.

### 1b. 타겟 왜도 점검 (회귀 전용)

회귀를 적합시키기 전, 타겟의 분포를 확인합니다. 강한 우편향(가격, 횟수,
지속 시간, 매출에서 흔히 나타남)은 OLS β에 대한 꼬리 영역 관측치의
영향력을 부풀리고, 이분산 잔차를 만듭니다.

**타겟 유형별 결정 규칙:**
- 양수만, 우편향(`skew > 1`, `y > 0`) → `log1p` 또는 **Box-Cox**
  (Box-Cox는 최적 λ를 선택; `log`는 Box-Cox의 λ=0 경우)
- 카운트(Poisson 형태) → Poisson 회귀 / `XGBRegressor(objective="count:poisson")`
- [0, 1] 사이의 비율 → logit 변환
- 좌편향 → Yeo-Johnson (음수도 처리) 또는 제곱 변환
- 다봉 / 두꺼운 꼬리 분포 → **분위수 회귀** 고려
  (`XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5)`로 중앙값)

```python
from scipy.stats import skew, boxcox
print(f"Target skew: {skew(y):+.2f}")

if skew(y) > 1 and (y > 0).all():
    # 기본: log1p. 더 유연하게 하려면 Box-Cox:
    #   y_model, lam = boxcox(y + 1e-9); print(f"Box-Cox λ = {lam:.3f}")
    print("  → 강한 우편향; 모델링에 log1p(y) 사용")
    y_model = np.log1p(y)
elif skew(y) < -1:
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method="yeo-johnson").fit(y.values.reshape(-1, 1))
    y_model = pt.transform(y.values.reshape(-1, 1)).ravel()
    print(f"  → 강한 좌편향; Yeo-Johnson 적용, 새 왜도 {skew(y_model):+.2f}")
else:
    y_model = y

print(f"Skew after transform: {skew(y_model):+.2f}")
```

원본 타겟과 변환된 타겟의 R²를 **둘 다** 보고하여 독자가 예측력 향상을
확인할 수 있게 하세요. 로그 변환을 적용했다면 모델 오차도 이해관계자를
위해 **행마다** 원본 단위로 변환합니다 — 단일 곱셈 근사를 쓰지 마세요:

```python
# 잘못된 방법 — mae_log가 작을 때만 유효 (Taylor 전개):
# mae_dollars = y_raw.median() * (np.exp(mae_log) - 1)

# 올바른 방법 — 행별로 예측을 역변환한 뒤 원본 단위로 MAE 계산:
pred_orig = np.expm1(pred)              # log1p를 사용했다면
y_orig    = np.expm1(y_test)
mae_dollars = mean_absolute_error(y_orig, pred_orig)
print(f"MAE in $: ${mae_dollars:,.0f} (median-priced item: ${y_orig.median():,.0f})")
```

### 2. 모델링을 위한 결측 처리 정책

데이터 정제 스킬(`data-cleaning.md`)은 fillna / dropna / 보간을 일반적으로
다룹니다. **모델링에서만** 적용되는 추가 규칙이 세 가지 있는데, 어기기
쉽습니다:

```python
def null_audit(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """피처별 결측 비율 + 결측과 타겟의 연관성.
    |miss-target assoc|이 0이 아니면 결측 자체가 신호를 가짐 —
    정보 있는 결측 — 이며 결측 지시자 칼럼이 필요합니다.
    타겟 dtype에 따라 적절한 연관성 측도를 선택합니다."""
    from scipy import stats
    is_num   = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    is_bin   = pd.api.types.is_bool_dtype(y) or (is_num and y.nunique() == 2)
    rows = []
    for col in X.columns:
        miss_pct = float(X[col].isna().mean())
        if miss_pct == 0:
            rows.append({"feature": col, "null_pct": 0.0, "miss_target_assoc": 0.0})
            continue
        is_miss = X[col].isna().astype(int)
        if is_bin:                                  # 이진 타겟 → 점이연 r
            r, _ = stats.pointbiserialr(is_miss, pd.Series(y).astype(int))
            assoc = float(r)
        elif is_num:                                # 연속 수치 타겟 → Spearman ρ
            assoc = float(is_miss.corr(y, method="spearman"))
        else:                                       # 다클래스 범주형 → Cramér's V
            ct = pd.crosstab(is_miss, y)
            chi2, *_ = stats.chi2_contingency(ct)
            n = ct.sum().sum()
            denom = n * max(min(ct.shape) - 1, 1)
            assoc = float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0
        rows.append({"feature": col, "null_pct": round(miss_pct, 4),
                     "miss_target_assoc": round(assoc, 4)})
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False)

audit = null_audit(X, y)
print(audit.to_string(index=False))
```

`miss_target_assoc`는 수치형 타겟에는 Spearman ρ, 이진형에는 점이연 r,
다클래스에는 Cramér's V입니다. 세 가지 모두 [-1, 1] / [0, 1] 비교 가능
척도에 있으므로 아래의 동일한 0.05 임계값이 적용됩니다.

**임계값 주의:** 0.05는 일반적인 n ≈ 1k–100k 환경을 가정한 경험적
값입니다. n = 1M에서 |assoc| 0.05는 통계적으로 매우 유의하지만 실용적
효과는 작습니다 — 매우 큰 n에서는 임계값을 ~0.10로 상향 조정하세요.
n < 200에서는 ~0.20로 (추정치가 노이즈에 민감함). 임계값은 통계적
유의성이 아닌 **실용적** 유의성을 의도한 것입니다.

**피처별 결정 규칙 (DataFrame 전체가 아니라 칼럼별 적용):**

| null_pct | \|miss_target_assoc\| | 권장 조치 |
|---:|---:|---|
| 0% | — | 없음 |
| < 1% | 무관 | 해당 행 제거 |
| 1–10% | < 0.05 | XGBoost: NaN 그대로 통과 (네이티브 처리); 선형: 중앙값 대체, **R² 감쇄에 대해 경고** |
| 1–10% | ≥ 0.05 | **정보 있는 결측** — `{col}_is_missing` 지시자 칼럼 추가 후 원본을 중앙값 대체 |
| 10–50% | < 0.05 | 지시자 + 중앙값 대체 (위와 동일; MAR이라도 결측 비율이 커서 지시자가 도움) |
| 10–50% | ≥ 0.05 | 지시자 + 중앙값 대체, 그리고 강한 결측 신호를 보고에 명시 |
| > 50% | 무관 | 칼럼 자체를 제거; "모델링에 너무 희소함"으로 보고 |

**대체 감쇄 편향 (선형 회귀에서만):** 평균/중앙값 대체는
`null_pct × σ_imputed / σ_observed` 정도로 β를 0 쪽으로 축소시킵니다.
10% 결측이 있는 칼럼이라면 β의 약 5–10% 감쇄를 예상할 수 있습니다. 항상
대체 전후의 R²를 보고하세요:

```python
# 대체 전 (NaN 행 제거)
clean_mask = X.notna().all(axis=1) & y.notna()
r2_clean = fit_linear(X[clean_mask], y[clean_mask])

# 대체 후 (모든 행 유지)
X_imp = X.fillna(X.median(numeric_only=True))
r2_imp = fit_linear(X_imp, y)

print(f"R² before imputation: {r2_clean:.4f}  (rows: {clean_mask.sum():,})")
print(f"R² after  imputation: {r2_imp:.4f}    (rows: {len(X):,})")
print(f"Attenuation:          {(r2_clean - r2_imp):.4f}")
```

감쇄가 0.05를 초과하고 **동시에** 결측에 정보가 있으면
(`miss_target_assoc` ≥ 0.05), 지시자+대체 패턴을 사용하세요 — 지시자가
대체로 인해 사라지는 결측 신호를 포착합니다.

순수하게 무작위인 결측(`miss_target_assoc` ≈ 0)에는 단순한 중앙값
대체로 충분합니다. 이 경우 지시자+대체는 지시자가 분산을 흡수하기
때문에 원본 칼럼의 β를 오히려 약간 *악화*시킬 수 있습니다(다만 R²
보존은 더 낫습니다). 실제 데이터에서 어떤 패턴이 정보를 가지는지
파악하고 그에 맞는 방식을 선택하세요 — 반사적으로 지시자를 추가하지
마세요.

**XGBoost의 네이티브 NaN 처리:** XGBoost (>= 1.6)는 각 분기에서 NaN의
"기본 방향"을 학습합니다. NaN을 그대로 통과시키세요. 결측 지시자 칼럼을
추가하지 않는다면 적합 전 대체를 **하지 마세요**. 이것은 선형과 트리
경로가 정당하게 갈라지는 몇 안 되는 지점입니다 — 선형 경로는 대체가
필요하고, 트리 경로는 그렇지 않습니다.

```python
# XGBoost 경로: NaN 그대로 통과
model = xgb.XGBRegressor(...).fit(X_train_with_nans, y_train)

# 선형 경로: 지시자 + 대체
for col in cols_with_nulls:
    X[f"{col}_is_missing"] = X[col].isna().astype("int8")
X = X.fillna(X.median(numeric_only=True))
```

## 계산 메서드 → `importance-methods.md`

XGBoost 베이스라인 레시피, 세 가지 중요도 렌즈(gain / 순열 / SHAP),
메서드 간 일치도 점검, 상호작용 탐지는 `importance-methods.md`로
이동했습니다. 위의 진단을 먼저 실행한 뒤 거기서 중요도를 계산하세요.

## 안티패턴 / 누수

이런 실패는 트리 기반 "중요도" 결과를 무가치하게 만듭니다. 보고 전에 각
항목을 점검하세요.

| 안티패턴 | 증상 | 해결 |
|---|---|---|
| 결과 이후 변수를 예측 변수로 사용 | 타겟의 파생/합성 변수 하나가 중요도를 지배 (예: `mental_health_index`로 `stress_level`을 예측) | 타겟에서 계산되거나 타겟 이후의 변수는 제거 |
| 훈련/테스트 분할 없음 | "R² = 0.99"인데 새 데이터는 0.30 | 적합 전 항상 20% 홀드아웃 분리 |
| SHAP을 훈련 데이터에서 적합 | SHAP 플롯이 "너무 깔끔" — 모든 게 완벽하게 중요해 보임 | SHAP은 홀드아웃 / 보류된 샘플에서 계산해야 함 |
| 명목 범주형을 라벨 인코딩 | 트리 분기가 거짓 순서 관계를 만듦 | 명목형은 `get_dummies`; 진정한 순서형만 순서 인코딩 |
| Gain 단독 보고 | 연속 피처가 지배; 이진 피처가 사소해 보임 | 항상 순열 또는 SHAP 교차 검증 포함 |
| `random_state` 없음 | 재실행마다 순위가 바뀜 | split, 모델, SHAP 샘플 모두에 `random_state=` 설정 |
| 수렴하지 않은 모델의 중요도 | 실행 간 큰 변동 | 훈련 오차가 안정화되었는지 확인; `n_estimators`를 올리거나 클래스 불균형 점검 |
| 경고 없이 선형 회귀에 평균/중앙값 대체 | β가 0 쪽으로 감쇄 (대체된 칼럼당 ~5–15%) | 대체 전에 `{col}_is_missing` 지시자 추가, 또는 대체 전후 R² 보고 |
| 결측이 타겟과 상관될 때 행 제거 | 표본이 편향됨; β는 응답한 사람만 반영 | `null_audit()` 실행; `miss_target_rho >= 0.05`이면 지시자+대체 사용, 절대 제거하지 말 것 |
| VIF > 10일 때 OLS β 보고 | β 값이 재실행 시 부호가 바뀌거나 크기가 변동 | 중복 쌍 중 하나 제거, 또는 RidgeCV로 전환 |
| "가장 중요한 피처"가 그 자체로 등급 요약 | 상위 드라이버가 `OverallQual`/`Score`/`Rating`이고 구성 요소 등급(`KitchenQual`, `ExterQual`, …)이 바로 뒤에 있음 | 중복을 표면화: 등급 클러스터를 하나의 합성으로 보고, 또는 요약 피처를 제외하고 중요도를 다시 계산해 무엇이 그 자리를 채우는지 확인 |
| 범주형이 존재하는데 수치형만 사용한 VIF 보고 | VIF ≈ 1인 수치형 칼럼이 범주형과 드러나지 않게 결합되어 있음 (예: Ames `Garage Yr Blt` ↔ `Garage Finish` η² = 0.998) — β가 의미 있어 보이지만 해석 불가 | `collinearity-diagnostics.md`의 `mixed_type_vif()`와 `cross_type_binding()` 실행; 설계 행렬 max VIF > 10 또는 쌍별 η² > 0.7인 모든 소스를 공선으로 처리 |

**첫 번째** 안티패턴은 EDA에서 가장 흔합니다: 타겟이 설문 척도 또는
합성치(스트레스, 만족도, NPS, 정신 건강 지수)일 때, 같은 도구의 다른
설문 척도들은 종종 수학적으로 관련됩니다. 적합 전에 제거하세요.

## 보고 템플릿

교차 검증을 마쳤다면, 단일 순위 비교 표와 한 단락의 해석을 제공합니다.
표가 핵심 산출물입니다:

```
| Feature             | Spearman ρ | std β | Perm. Imp. | SHAP \|mean\| | Rank consensus |
|---------------------|-----------:|------:|-----------:|--------------:|---------------:|
| financial_stress    | +0.453     | +0.47 | 0.214      | 0.79          | 1 (all 4)      |
| exam_pressure       | +0.444     | +0.46 | 0.087      | 0.71          | 2 (all 4)      |
| family_expectation  | +0.336     | +0.35 | 0.118      | 0.52          | 3 (all 4)      |
| sleep_hours         | -0.254     | -0.26 | 0.066      | 0.31          | 4 (all 4)      |
| physical_activity   | -0.167     | -0.17 | 0.029      | 0.18          | 5 (all 4)      |
| study_hours_per_day | +0.340     |  0.00 | 0.001      | 0.04          | 9 → ρ misled by collinearity with exam_pressure |
```

필수 산문 요소:
1. **헤드라인 발견:** 합의된 #1 드라이버를 표준화 β + SHAP 크기와 함께.
2. **메서드 일치도 요약:** 네 렌즈 사이의 순위 상관.
3. **이름으로 호명한 불일치:** 선형-순위와 SHAP-순위가 ≥ 3 차이 나는 피처 —
   이유 설명 (다중공선성, 비선형성, 또는 상호작용).
4. **발견된 상호작용:** 있다면 상위 SHAP 의존성 쌍.
5. **홀드아웃 R² (또는 ROC-AUC):** 미관측 데이터에 대한 모델의 예측력.
6. **유의사항:** SHAP 샘플 크기, 누수로 제거된 변수, 선택적 라이브러리
   버전.

## 성능 치트시트

| 이슈 | 해결 |
|---|---|
| 수백만 행에서 XGBoost 훈련이 느림 | `tree_method="hist"`, `subsample=0.5`, 더 작은 `n_estimators` |
| SHAP 중 OOM | 20k–50k 행으로 샘플링; `KernelExplainer`가 아니라 `TreeExplainer` 사용 |
| SHAP 플롯이 빽빽함 | `shap.summary_plot`에 `max_display=20` 설정 |
| `permutation_importance`가 느림 | `n_repeats`를 3으로 낮춤; 테스트 세트를 20k로 서브샘플링 |
| 실행 간 다른 중요도 순위 | 모든 곳에 `random_state=` 설정; 모델이 수렴했는지 점검 |

## 의사결정 치트시트

```
PRE-FIT — 항상 먼저 실행:
   1. vif_table(X)        — VIF > 10인가? 중복 피처 제거하거나 RidgeCV 사용.
   2. null_audit(X, y)    — 칼럼별 규칙 적용 (drop / impute / indicator+impute / remove).
   3. 결과 이후 누수 변수 제거 (타겟의 합성치).

선형 분석 완료 (Spearman ρ + std β + ΔR²)?
   ↓ 아니오  → 먼저 실행 (statistical-analysis.md)
   ↓ 예
R² ≥ 0.4이고 순위가 안정적인가?
   ↓ 예 → 멈추세요. 선형 결과를 보고. ML이 추가 가치를 주지 않음.
   ↓ 아니오
XGBoost + 순열 + SHAP 교차 검증 실행.
   ↓
메서드가 일치 (Spearman 순위 상관 ≥ 0.8)?
   ↓ 예 → 합의 순위 보고, ML이 선형 분석을 확인했다고 명시.
   ↓ 아니오
불일치 조사:
   - 다중공선성?  → 중복 피처 제거 후 재실행.
   - 비선형성? → SHAP 의존성 플롯, 형태 보고.
   - 상호작용?   → 서브샘플에 SHAP 상호작용 값, 쌍 보고.
세 순위 모두와 그것들을 화해시키는 해석을 보고.
```
