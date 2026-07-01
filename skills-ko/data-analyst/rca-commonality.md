# RCA 공통성 분석

시간 국소화된 이상에 대한 센서 클러스터 이동 분석. `root-cause-analysis.md`
§ 1 (CPD)의 보완: 공통성은 **어떤** 센서·요인이 이동했는지, CPD는 **언제**
이동했는지에 답합니다.

## 목차

- 다중공선 센서 클러스터를 먼저 축소
- 범주형 요인별 공통성 (Fisher 정확검정 + BH-FDR)
- 다중 요인 공통성 (빈발 항목집합 마이닝)
- 다중성 범위 경고 (요인별이 아닌 전체 스윕에 BH-FDR)

이 파일을 읽는 경우: "Y가 이동했을 때 어떤 센서들이 움직였는가",
"어떤 로트/레시피/장비 수준이 결함에서 과대 표집되는가", "실패를 예측하는
다중 요인 조합" 관련 작업. Pre/post 윈도우 정의는 `root-cause-analysis.md`
§ 1과 함께 사용하세요.

## 다중공선 센서 클러스터를 먼저 축소

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

— *"8개 개별 센서가 이동했습니다"*가 *아닙니다*. 후자는 동일한 물리적
사건에 대한 중복 측정값들에 엔지니어의 주의를 분산시킵니다.

## 범주형 요인별 공통성

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
all_rows = []
for factor in categorical_cols:
    sig = commonality(df, factor, "is_defective", alpha=1.0)   # 원본 p-값 수집
    if len(sig):
        sig["factor"] = factor
        all_rows.append(sig)

# 요인별이 아니라 **전체 스윕**에 걸쳐 BH-FDR 적용
if all_rows:
    from statsmodels.stats.multitest import multipletests
    full = pd.concat(all_rows, ignore_index=True)
    full["p_bh_fdr"] = multipletests(full["p_fisher"], method="fdr_bh")[1]
    print(full.query("p_bh_fdr < 0.05")
              .sort_values("p_bh_fdr")
              [["factor", "level", "defect_rate_in_level", "lift", "p_bh_fdr"]]
              .to_string(index=False))
```

**다중성 범위 경고:** `commonality()`가 반환하는 `p_bonferroni`는 해당
요인 **내부**(요인의 수준 수)에만 적용된 것입니다. 여러 요인에 걸친 폭넓은
스윕(예: SECOM의 590개 센서 범주형)에서 올바른 보정은 위와 같이 **모든
요인의 모든 p-값에 걸쳐** BH-FDR을 적용하는 것이며, 요인 내 Bonferroni가
아닙니다.

## 다중 요인 공통성 (빈발 항목집합 마이닝)

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
