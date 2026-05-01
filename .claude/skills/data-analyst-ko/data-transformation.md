# Pandas 데이터 변환 — 전문가 스킬

당신은 전문 데이터 분석가입니다. pandas >= 2.3 모범 사례를 적용하여 다음
변환 패턴을 사용합니다. 루프보다 벡터화된 연산을 우선합니다.

## 칼럼 생성 및 수정

```python
# assign() — 권장: 새 DataFrame을 반환하며 체이닝 가능
df = df.assign(
    revenue=lambda x: x["price"] * x["quantity"],
    revenue_k=lambda x: x["revenue"] / 1000,
    is_high_value=lambda x: x["revenue"] > 10_000,
    log_revenue=lambda x: np.log1p(x["revenue"]),
)

# 직접 할당 (제자리에서 변경)
df["margin"] = (df["revenue"] - df["cost"]) / df["revenue"]
df["bucket"] = pd.cut(df["score"], bins=[0, 60, 80, 100], labels=["C", "B", "A"])
```

## 벡터화된 연산 — 항상 루프보다 우선

```python
import numpy as np

# 산술 연산
df["total"] = df["qty"] * df["unit_price"] * (1 - df["discount"])

# .str 접근자를 통한 문자열 연산
df["first_name"] = df["full_name"].str.split().str[0]
df["domain"] = df["email"].str.extract(r"@(.+)$")
df["upper"] = df["status"].str.upper()
df["trimmed"] = df["notes"].str.strip()

# .dt 접근자를 통한 datetime 연산
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["weekday"] = df["date"].dt.day_name()
df["quarter"] = df["date"].dt.quarter
df["is_weekend"] = df["date"].dt.dayofweek >= 5

# 조건부 칼럼을 위한 np.where
df["tier"] = np.where(df["revenue"] > 100_000, "enterprise",
             np.where(df["revenue"] > 10_000,  "mid-market", "smb"))

# 다중 조건을 위한 np.select
conditions = [
    df["score"] >= 90,
    df["score"] >= 75,
    df["score"] >= 60,
]
choices = ["A", "B", "C"]
df["grade"] = np.select(conditions, choices, default="F")
```

## map()과 replace() — 원소 단위

```python
# map: Series에 함수 또는 dict 매핑 적용
df["status_code"] = df["status"].map({"active": 1, "inactive": 0, "pending": 2})
df["label"] = df["id"].map(id_to_label_dict)   # dict 조회
df["doubled"] = df["value"].map(lambda x: x * 2)

# replace: 값 치환 (map보다 유연하며 매핑되지 않은 값 보존)
df["region"] = df["region"].replace({"USA": "US", "U.S.A": "US", "United States": "US"})
df = df.replace({"status": {"Active": "active", "Inactive": "inactive"}})
```

## apply() — 행 또는 열 단위

```python
# 칼럼 단위 (axis=0, 기본값) — 각 칼럼 Series에 함수 적용
col_means = df[["a", "b", "c"]].apply("mean")

# 행 단위 (axis=1) — 각 행 Series에 함수 적용 (대용량 데이터에서는 느림)
df["max_feature"] = df[["f1", "f2", "f3"]].apply("max", axis=1)

# 벡터화된 대안이 없을 때만 사용
def complex_logic(row):
    if row["type"] == "A":
        return row["x"] * 2
    return row["y"] + 10

df["result"] = df.apply(complex_logic, axis=1)  # 느림 — 가능하면 np.select 사용
```

**성능 참고:** `apply(axis=1)`은 벡터화된 연산보다 100~1000배 느립니다.
`np.where`, `np.select`, `.str` 접근자, 산술 연산자를 우선합니다.

## GroupBy — 분할-적용-결합

```python
# 기본 집계
df.groupby("category")["revenue"].sum()
df.groupby(["region", "category"])["revenue"].agg(["sum", "mean", "count"])

# 명명된 집계 (pandas 0.25+)
agg = df.groupby("category").agg(
    total_revenue=("revenue", "sum"),
    avg_revenue=("revenue", "mean"),
    n_orders=("order_id", "count"),
    max_order=("revenue", "max"),
    p90_revenue=("revenue", lambda x: x.quantile(0.9)),
)

# 다중 칼럼, 다중 집계
df.groupby("region").agg({
    "revenue": ["sum", "mean"],
    "quantity": "sum",
    "discount": "mean",
})

# transform() — 그룹 결과를 원본 형태로 다시 추가
df["group_mean"] = df.groupby("category")["revenue"].transform("mean")
df["pct_of_group"] = df["revenue"] / df.groupby("category")["revenue"].transform("sum")
df["rank_in_group"] = df.groupby("category")["revenue"].rank(ascending=False)

# filter() — 조건을 만족하는 그룹 유지
large_groups = df.groupby("category").filter(lambda g: len(g) >= 100)
active_regions = df.groupby("region").filter(lambda g: g["revenue"].sum() > 1_000_000)

# 그룹에 대한 apply() — 임의의 그룹 단위 변환
def normalize_group(g):
    g["norm_score"] = (g["score"] - g["score"].mean()) / g["score"].std()
    return g

df = df.groupby("category", group_keys=False).apply(normalize_group)
```

## 피벗과 형태 변경

```python
# pivot_table — Excel 피벗과 유사하며 집계 포함
pivot = df.pivot_table(
    index="region",
    columns="category",
    values="revenue",
    aggfunc="sum",
    fill_value=0,
    margins=True,       # 행/열 합계
    margins_name="Total",
)

# pivot — 집계 없음 (고유한 index+column 조합 필요)
wide = df.pivot(index="date", columns="metric", values="value")

# melt — 와이드에서 롱으로 (언피벗)
long = pd.melt(
    df,
    id_vars=["id", "date"],
    value_vars=["jan", "feb", "mar"],
    var_name="month",
    value_name="amount",
)

# stack / unstack — MultiIndex와 함께 작동
stacked = df.stack()           # 칼럼 → 가장 안쪽 인덱스 레벨
unstacked = df.unstack()       # 가장 안쪽 인덱스 레벨 → 칼럼

# crosstab — 빈도표
ct = pd.crosstab(df["region"], df["status"], values=df["revenue"], aggfunc="sum")
```

## 비닝 및 이산화

```python
# 동일 너비 빈
df["age_band"] = pd.cut(
    df["age"],
    bins=[0, 18, 35, 50, 65, 120],
    labels=["<18", "18-34", "35-49", "50-64", "65+"],
    right=False,
)

# 동일 빈도 (분위수) 빈
df["revenue_quartile"] = pd.qcut(df["revenue"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
df["revenue_decile"] = pd.qcut(df["revenue"], q=10, labels=False)  # 0-9 정수

# np.digitize를 사용한 사용자 정의 빈
bins = [0, 100, 500, 1000, np.inf]
df["tier"] = pd.cut(df["revenue"], bins=bins, labels=["Low", "Medium", "High", "VIP"])
```

## 정렬

```python
# 단일 칼럼 정렬
df = df.sort_values("revenue", ascending=False)

# 다중 칼럼 정렬 (다중 키)
df = df.sort_values(["region", "revenue"], ascending=[True, False])

# 인덱스 정렬
df = df.sort_index()

# 안정 정렬 (동률에 대해 원본 순서 보존)
df.sort_values("score", kind="mergesort")

# nlargest / nsmallest (sort + head보다 빠름)
df.nlargest(10, "revenue")
df.nsmallest(5, "error_rate")
```

## 문자열 변환 (`.str` 접근자)

```python
# 분리
df[["first", "last"]] = df["full_name"].str.split(" ", n=1, expand=True)
df["words"] = df["sentence"].str.split()           # 단어 리스트

# 패턴 추출
df[["area_code", "number"]] = df["phone"].str.extract(r"(\d{3})-(\d{7})")

# 포함 / 매치
df["has_promo"] = df["notes"].str.contains("promo|discount", case=False, na=False)

# 치환
df["clean"] = df["text"].str.replace(r"<[^>]+>", "", regex=True)  # HTML 제거

# 패딩 / 정렬
df["padded_id"] = df["id"].astype(str).str.zfill(8)
```

## 날짜/시간 변환 (`.dt` 접근자)

```python
df["date"] = pd.to_datetime(df["date"])

# 구성 요소
df["year"]     = df["date"].dt.year
df["month"]    = df["date"].dt.month
df["day"]      = df["date"].dt.day
df["hour"]     = df["date"].dt.hour
df["week"]     = df["date"].dt.isocalendar().week
df["quarter"]  = df["date"].dt.quarter
df["weekday"]  = df["date"].dt.day_of_week          # 0=월요일
df["is_month_end"] = df["date"].dt.is_month_end

# 산술 연산
df["age_days"] = (pd.Timestamp.now() - df["birth_date"]).dt.days
df["next_week"] = df["date"] + pd.Timedelta(weeks=1)

# 반올림
df["hour_bucket"] = df["timestamp"].dt.floor("h")
df["day_bucket"]  = df["timestamp"].dt.normalize()
```

## 메서드 체이닝 — 권장 스타일

```python
result = (
    df
    .query("status == 'active' and amount > 0")
    .assign(
        revenue=lambda x: x["price"] * x["qty"],
        margin=lambda x: (x["revenue"] - x["cost"]) / x["revenue"],
    )
    .groupby(["region", "category"])
    .agg(total_revenue=("revenue", "sum"), avg_margin=("margin", "mean"))
    .sort_values("total_revenue", ascending=False)
    .reset_index()
    .head(20)
)
```
