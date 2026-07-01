# 특성 중요도 — 계산 메서드

드라이버 분석의 계산 계층: XGBoost 베이스라인을 적합한 뒤 세 가지 보완
렌즈로 중요도를 읽고 서로 대조합니다. 먼저 실행해야 하는 *진단*(VIF,
결측 처리, 왜도, 누수)과 보고 규율은 `feature-importance.md`에 남습니다.

## 목차

- XGBoost 베이스라인 레시피 (회귀 / 이진 / 범주형 인코딩)
- 세 가지 중요도 렌즈 (Gain, 순열, SHAP)
- 메서드 간 일치도 점검 (렌즈 간 순위 상관)
- 상호작용 탐지 (SHAP 의존도 + 상호작용 값)

이 파일을 읽는 경우: XGBoost 모델 적합, SHAP / 순열 / gain 중요도 계산,
불일치하는 순위 조정, 또는 특성 상호작용 탐지. 선행 진단과 보고
템플릿은 `feature-importance.md`에 있습니다.

## XGBoost 베이스라인 레시피

합리적인 기본값을 사용하세요 — 명시적으로 요청받지 않았다면 하이퍼파라미터
튜닝은 하지 않습니다. 목표는 피처 중요도이지 프로덕션 모델이 아닙니다.

### 회귀 타겟

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# X: 수치형 피처만 (범주형은 먼저 인코딩 — 아래 참고)
# y: 수치형 타겟
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",      # 큰 데이터에서 빠름
    n_jobs=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

pred = model.predict(X_test)
print(f"R²  on holdout: {r2_score(y_test, pred):.4f}")
print(f"MAE on holdout: {mean_absolute_error(y_test, pred):.4f}")
```

### 이진 분류 타겟

```python
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

from sklearn.metrics import roc_auc_score, average_precision_score
proba = model.predict_proba(X_test)[:, 1]
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
print(f"PR-AUC : {average_precision_score(y_test, proba):.4f}")
```

### XGBoost 전 범주형 인코딩

XGBoost (>= 1.6)는 `enable_categorical=True`로 범주형을 네이티브 처리하지만,
이식성을 위해서는 명시적 인코딩을 사용합니다:

```python
# 카디널리티가 낮음 (< 20 고유값) → 원-핫
X = pd.get_dummies(df[features], columns=cat_cols, drop_first=False, dtype="int8")

# 카디널리티가 높음 → 순서형 (타겟 인코딩은 CV 폴드 안에서만)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[high_card_cols] = oe.fit_transform(X[high_card_cols])
```

*순서가 없는* 범주형을 절대 라벨 인코딩하지 마세요 — XGBoost는 코드를
수치로 다루어 거짓 순서 관계를 추론합니다.

## 세 가지 중요도 렌즈

항상 **최소 두 가지를** 계산하세요. 하나만 보고하는 것은 직무 유기입니다.

### 1. Gain (XGBoost 분기 중요도)

빠르고, 무료이고, 편향됨. 앙상블 내 모든 분기에서 각 피처에 귀속된 총
손실 감소를 보고합니다.

```python
gain = pd.Series(model.feature_importances_, index=X.columns) \
         .sort_values(ascending=False)
print(gain.head(15))
```

**알아야 할 편향:** Gain은 카디널리티가 높거나 연속형인 피처에 대해
중요도를 부풀립니다 (분기 후보가 더 많을수록 분기로 채택될 기회가
더 많음). 이진 피처가 진정한 드라이버여도 10-수준 범주형이 이를
압도할 수 있습니다. **Gain을 단독으로 보고하지 마세요.**

### 2. 순열 중요도 (sklearn — 모델 무관)

모델 무관(model-agnostic) 베이스라인입니다. 단일 피처의 값을 무작위로
섞었을 때 홀드아웃 점수가 얼마나 떨어지는지를 측정합니다.

```python
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,                            # 10–30 권장; 5는 너무 노이즈가 큼
    random_state=42,
    n_jobs=-1,
    scoring="r2",                            # 아래 참고 — 타겟 형태에 따라 선택
)
perm_df = (pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True))
print(perm_df.head(15).round(4))
```

**타겟 형태별 `scoring=`:**
- 대칭적 수치형 타겟 → `"r2"` (기본)
- 두꺼운 꼬리 / 로그 변환된 타겟 → `"neg_mean_absolute_error"`
  (R²는 긴 꼬리에 지배됨; MAE는 일반적 행의 영향을 반영)
- 이진 분류, 균형 잡힘 → `"roc_auc"`
- 이진 분류, 불균형 (양성 < 10%) → `"average_precision"` (PR-AUC)

훈련 데이터가 아니라 **홀드아웃 세트**에서 계산하세요. 훈련 데이터에서는
순열 중요도가 무의미합니다 (모델이 이미 학습 데이터를 외운 상태이기
때문).

**상관된 피처에 대한 알려진 편향 (Strobl 2007, Hooker–Mentch 2021):**
X₁과 X₂가 상관되어 있을 때, X₁을 섞어도 모델이 X₂에서 신호를 회복할 수
있어서 *둘 다* 약해 보입니다 — 중요도가 클러스터 전체에 분산됩니다. 항상
§ 1의 VIF / 클러스터 축소 단계와 함께 사용하세요. 그렇지 않으면 상관된
피처들이 실제 효과와 무관하게 독립 피처보다 일관되게 낮은 순위를
받습니다.

매우 큰 홀드아웃(> 100k 행)에서는 50k로 서브샘플링해 속도를 확보하세요 —
순열은 모델을 `n_repeats × n_features`회 실행합니다.

### 3. SHAP (TreeExplainer)

트리 모델을 위한 황금 표준. 각 피처의 각 개별 예측에 대한 기여도를,
이론적 보장(협력 게임 이론의 Shapley 값)과 함께 제공합니다. 전역(평균
|SHAP|) 및 행 단위 설명 모두에 사용합니다.

**`feature_perturbation=` 선택이 상관된 피처에서 중요합니다:**
- `"tree_path_dependent"` (기본) — 관측적 SHAP, 빠름, 배경 데이터 불필요.
  상관된 피처에서는 클러스터 내 각 피처에 파트너의 효과 일부가 귀속됩니다.
- `"interventional"` — 인과적/개입적 SHAP, 배경 데이터셋 필요, 더 느림.
  인과적 해석에 더 깔끔. SHAP을 예측 설명이 아닌 피처 *효과*에 대한
  논증으로 사용할 때 필수입니다.

```python
import shap

# 중요: 큰 데이터에서는 SHAP 전에 샘플링 — TreeSHAP은 O(n × trees × leaves²)
X_shap = X_test.sample(min(50_000, len(X_test)), random_state=42)

# 관측적 SHAP (기본 — 빠름, 예측 설명에 적합)
explainer = shap.TreeExplainer(model)

# 또는 개입적 SHAP (느림, 효과 해석에 더 깔끔):
# X_bg = X_train.sample(min(1000, len(X_train)), random_state=42)
# explainer = shap.TreeExplainer(model, X_bg,
#                                feature_perturbation="interventional")

shap_values = explainer.shap_values(X_shap)
# shap_values shape: 회귀/이진의 경우 (n_rows, n_features)

# 전역 중요도: 평균 |SHAP|
shap_global = (pd.Series(np.abs(shap_values).mean(axis=0), index=X_shap.columns)
                 .sort_values(ascending=False))
print(shap_global.head(15).round(4))

# 시각적 요약 (노트북에서; 스크립트에서는 파일로 저장)
import matplotlib.pyplot as plt
shap.summary_plot(shap_values, X_shap, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
```

**샘플링 경험칙:**

| 데이터 행 수 | SHAP 샘플 크기 | 예상 실행 시간 |
|---:|---:|---:|
| ≤ 10,000 | 전체 | < 30초 |
| 10k–100k | 20,000 | 1–3분 |
| 100k–1M | 50,000 | 3–10분 |
| > 1M | 50,000 (≤ 5%) | 5–15분 |

샘플이 더 커도 순위는 거의 바뀌지 않습니다 — `summary_plot`의 시각적
밀도만 더 단단해질 뿐입니다.

## 메서드 간 일치도 점검

중요한 신호는 **세 가지 메서드가 모두 일치**하는 것입니다. 불일치는 그
자체로 하나의 발견입니다 — 보고하세요. 승자를 조용히 고르지 마세요.

```python
def rank_compare(*, gain, perm_mean, shap_mean, spearman_rho):
    """나란히 놓는 순위 표 작성. 입력은 피처가 인덱스인 pd.Series."""
    df = pd.DataFrame({
        "spearman_|rho|": spearman_rho.abs(),
        "xgb_gain":       gain,
        "perm_importance": perm_mean,
        "shap_|mean|":     shap_mean,
    })
    ranks = df.rank(ascending=False, method="min").astype(int)
    ranks.columns = [f"rank_{c}" for c in df.columns]
    out = pd.concat([df.round(4), ranks], axis=1).sort_values("rank_shap_|mean|")
    return out

table = rank_compare(
    gain=gain,
    perm_mean=perm_df.set_index("feature")["importance_mean"],
    shap_mean=shap_global,
    spearman_rho=df[features].corrwith(df[target], method="spearman"),
)
print(table.head(15).to_string())

# 일치도 정량화: 네 순위 칼럼에 대한 Spearman 상관
agreement = (table.filter(like="rank_").corr(method="spearman")
                .round(3))
print("\nRank-method agreement (Spearman):")
print(agreement)
```

해석:
- **비대각 > 0.8:** 메서드 일치 → 순위가 메서드 선택에 강건.
- **0.5–0.8:** 대부분 일치 → 합의된 상위 K를 보고하고, 불일치 표시.
- **< 0.5:** 메서드 간 불일치 → 비선형성, 피처 상호작용, 또는 어느 한
  메서드가 잘못된 신호를 잡았다고 의심해야 합니다. 순위를 발표하기
  전에 원인을 조사하세요.

**합의가 알려주는 것과 알려주지 않는 것:** 네 렌즈 사이의 강한 일치는
**순위**가 메서드 선택에 강건하다는 의미입니다 — 근본적인 관계가
인과적이라는 의미는 아닙니다. 네 메서드 모두 같은 데이터로 적합한 같은
모델의 다른 시각이며, 상관된 증인이지 독립적 증인이 아닙니다. 인과를
주장하려면 `root-cause-analysis.md` § 4 (DAG, DiD, 반박)을 보세요.

## 상호작용 탐지

SHAP과 선형 β가 어떤 피처의 중요도에 대해 일치하지 않을 때, 가장 흔한
원인은 상호작용 효과입니다. SHAP 의존성 플롯으로 점검합니다:

```python
# 상위 피처 의존성 — SHAP 값이 다른 피처에 의존하는가?
top_feature = shap_global.index[0]
shap.dependence_plot(top_feature, shap_values, X_shap, show=False)
plt.tight_layout(); plt.savefig(f"shap_dep_{top_feature}.png", dpi=120); plt.close()
```

명시적인 상호작용 강도 행렬 (작은 데이터에서만):

```python
# 경고: O(n × features²) — features ≤ 30, rows ≤ 10,000일 때만 사용
if X_shap.shape[1] <= 30 and len(X_shap) <= 10_000:
    shap_inter = explainer.shap_interaction_values(X_shap)
    # 비대각 크기 = 쌍별 상호작용 강도
    inter_mat = np.abs(shap_inter).mean(axis=0)
    np.fill_diagonal(inter_mat, 0)
    pairs = (pd.DataFrame(inter_mat, index=X_shap.columns, columns=X_shap.columns)
             .stack().reset_index()
             .rename(columns={"level_0":"a","level_1":"b",0:"interaction"})
             .query("a < b").sort_values("interaction", ascending=False))
    print(pairs.head(10).round(4))
```

큰 데이터에서는 `dependence_plot`의 색상 분산으로 상호작용을 정성적으로
추론한 뒤, 10k 서브샘플에서 검증합니다.
