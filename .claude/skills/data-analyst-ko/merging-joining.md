# Pandas 병합(merge), 조인(join) 및 연결(concat) — 전문가 스킬

당신은 전문 데이터 분석가입니다. DataFrame을 결합할 때 다음 패턴을 정확하게
적용하십시오. 잘못된 조인(join) 유형을 선택하는 것은 데이터 품질 오류의
흔한 원인이므로, 어떤 조인이든 수행한 후에는 항상 행 수를 검증해야 합니다.

## pd.merge() — SQL 스타일 조인

```python
# 문법
result = pd.merge(left, right, how="inner", on="key")

# how 옵션:
# "inner" — 일치하는 키만 (기본값, 가장 안전)
# "left"  — 좌측의 모든 행, 일치하지 않는 우측은 NaN
# "right" — 우측의 모든 행, 일치하지 않는 좌측은 NaN
# "outer" — 양쪽의 모든 행, 일치하지 않으면 NaN
# "cross" — 카르테시안 곱(모든 조합)
```

### 일반적인 조인 패턴

```python
# 단일 키
orders = pd.merge(orders, customers, on="customer_id", how="left")

# 다중 키 (복합 키)
merged = pd.merge(
    df_a, df_b,
    on=["user_id", "date"],
    how="inner"
)

# 각 DataFrame에서 칼럼 이름이 다른 경우
merged = pd.merge(
    orders, products,
    left_on="product_id",
    right_on="id",
    how="left"
)

# 인덱스로 조인
merged = pd.merge(df_a, df_b, left_index=True, right_index=True, how="inner")
merged = pd.merge(df_a, df_b, left_on="id", right_index=True, how="left")

# 접미사(suffix)로 중복 칼럼 이름 처리
merged = pd.merge(
    orders, returns,
    on="order_id",
    how="left",
    suffixes=("_order", "_return"),
)
```

### 조인 검증(validate)

```python
# 조인 유형을 검증하여 데이터 무결성 문제를 잡아냅니다
pd.merge(df_a, df_b, on="id", how="inner", validate="one_to_one")
pd.merge(orders, customers, on="customer_id", how="left", validate="many_to_one")
pd.merge(products, tags, on="product_id", how="left", validate="one_to_many")
# 가정이 위반되면 MergeError를 발생시킵니다
```

**항상 행 수를 확인하십시오:**
```python
n_before = len(orders)
result = pd.merge(orders, customers, on="customer_id", how="left")
n_after = len(result)
print(f"Rows before: {n_before}, after: {n_after}, delta: {n_after - n_before}")
assert n_after == n_before, "Left join should not change row count!"
```

### 일치하지 않는 키 탐지

```python
# 일치하는 고객이 없는 주문은 무엇인가?
merged = pd.merge(orders, customers, on="customer_id", how="left", indicator=True)
unmatched = merged[merged["_merge"] == "left_only"]
print(f"Orders with no customer: {len(unmatched)}")

# anti-join (역조인): 좌측에는 있지만 우측에는 없는 행
anti = merged.loc[merged["_merge"] == "left_only"].drop(columns="_merge")
```

## DataFrame.join() — 인덱스 기반

```python
# join은 기본적으로 인덱스를 사용합니다
result = df_a.join(df_b, how="left")                    # 인덱스 대 인덱스
result = df_a.join(df_b.set_index("id"), on="id")       # 칼럼 대 인덱스
result = df_a.join([df_b, df_c])                         # 여러 개를 한 번에 조인
```

## pd.concat() — DataFrame 쌓기(stack)

```python
# 수직(행 방향) 쌓기 — 동일한 칼럼
df_all = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)

# 출처 라벨 포함
df_all = pd.concat(
    {"2022": df_2022, "2023": df_2023},
    names=["year", "row"],
)

# 수평(칼럼 방향) 쌓기 — 동일한 행
df_wide = pd.concat([df_features, df_targets], axis=1)

# 일치하지 않는 칼럼 처리
df_all = pd.concat([df_a, df_b], ignore_index=True, sort=False)
# 누락된 칼럼은 NaN으로 채워집니다
```

### concat과 merge 비교

| 작업 | 사용 |
|---|---|
| 동일한 스키마, 행 쌓기 | `pd.concat(axis=0)` |
| 나란히 칼럼 배치(동일 인덱스) | `pd.concat(axis=1)` |
| 키로 조인하기 | `pd.merge()` |
| 인덱스로 조인하기 | `df.join()` |

## 효율적으로 행 추가하기

**더 이상 사용되지 않는 `.append()`를 사용하지 마십시오** (pandas 2.0에서 제거됨).

```python
# DataFrame을 리스트에 모은 뒤, 마지막에 한 번 concat
chunks = []
for batch in data_batches:
    processed = process(batch)
    chunks.append(processed)
df = pd.concat(chunks, ignore_index=True)   # 마지막에 한 번 concat
```

루프 내에서 연결(concat)하면 O(n²)입니다. 항상 모은 뒤 한 번에 연결하십시오.

## 시간 순서 데이터 병합

### Merge Asof (가장 가까운 이전 키)

```python
# merge_asof: 좌측의 각 행을 우측에서 가장 가까운 이전 행과 매칭
# 두 DataFrame 모두 키 칼럼으로 정렬되어 있어야 합니다
quotes = quotes.sort_values("timestamp")
trades = trades.sort_values("timestamp")

result = pd.merge_asof(
    trades,
    quotes,
    on="timestamp",
    by="ticker",              # 이 칼럼은 정확히 일치해야 함
    direction="backward",     # 가장 가까운 이전 시세
    tolerance=pd.Timedelta("1min"),  # 1분 이내에서만 매칭
)
```

활용 사례: 거래를 마지막으로 가능한 시세에 매칭, 센서 측정값을 가장 가까운
구성 변경에 조인.

### Merge Ordered

```python
# merge_ordered: 선택적 채우기를 동반한 외부 조인
result = pd.merge_ordered(
    df_a, df_b,
    on="date",
    fill_method="ffill",    # 병합 후 forward-fill
)
```

## update()와 combine_first()로 결합하기

```python
# update: 다른 DataFrame에서 NaN 값을 제자리에서 덮어쓰기 (인덱스로 정렬)
df.update(df_updates)              # df를 제자리에서 수정, NaN만

# combine_first: 다른 DataFrame에서 NaN 값을 채우기
df_combined = df.combine_first(df_fallback)
# df가 non-null이면 df 사용, 아니면 df_fallback로 대체
```

## 흔한 실수와 해결책

| 문제 | 증상 | 해결책 |
|---|---|---|
| 다대다(many-to-many) 조인 | 행 폭증 | `validate=` 파라미터 추가; 키 고유성 확인 |
| 내부 조인 후 행 누락 | 예상보다 적은 행 수 | `how="left"`와 `indicator=True`로 진단 |
| 중복 칼럼 | `_x`, `_y` 접미사 | `suffixes=`를 명시적으로 설정; 병합 후 삭제/이름 변경 |
| 키의 잘못된 dtype | 값이 일치해도 매칭되지 않음 | dtype 정렬: `df["id"] = df["id"].astype(int)` |
| 대용량 데이터에서 느린 병합 | 타임아웃 | 먼저 인덱스 설정: 조인 전에 `df.set_index("key")` |

## 전체 예시: 주문 보강 파이프라인

```python
# 시작: orders DataFrame
# customers (다대일), products (다대일), promotions (좌측) 조인
result = (
    orders
    .merge(customers[["customer_id", "name", "region"]],
           on="customer_id", how="left", validate="many_to_one")
    .merge(products[["product_id", "name", "category", "cost"]],
           on="product_id", how="left", validate="many_to_one",
           suffixes=("", "_product"))
    .merge(promotions[["order_id", "discount_pct"]],
           on="order_id", how="left", validate="one_to_one")
)

# 좌측 조인으로 인한 행 수 변화가 없는지 확인
assert len(result) == len(orders), "Row count changed unexpectedly"

# 일치하지 않는 키 확인
print("Unmatched customers:", result["name"].isna().sum())
print("Unmatched products:", result["category"].isna().sum())
```
