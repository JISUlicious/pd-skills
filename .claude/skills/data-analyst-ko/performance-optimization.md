# Pandas 성능 및 메모리 최적화 — 전문가 스킬

여러분은 전문 데이터 분석가입니다. 다음 기법들을 적용하여 pandas >= 2.3 환경에서 pandas 코드를 빠르고 메모리 효율적으로 만듭니다.

## 1단계 — 최적화보다 프로파일링이 먼저

```python
import time
import tracemalloc

# 코드 블록 시간 측정
start = time.perf_counter()
result = df.groupby("category")["revenue"].sum()
elapsed = time.perf_counter() - start
print(f"Time: {elapsed:.3f}s")

# 메모리 사용량
tracemalloc.start()
result = expensive_operation(df)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"Peak memory: {peak / 1e6:.1f} MB")

# DataFrame 메모리
print(df.memory_usage(deep=True).sum() / 1e6, "MB")
print(df.memory_usage(deep=True).sort_values(ascending=False).head(10))

# Jupyter 환경에서의 %timeit
# %timeit df.groupby("category")["revenue"].sum()
```

## 2단계 — dtype 최적화 (가장 영향력이 큰 변경)

```python
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 dtype을 자동으로 다운캐스트합니다."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category")
    return df

# dtype별 메모리 영향
# int64  → int32:  50% 감소
# float64 → float32: 50% 감소
# object → category: 최대 95% 감소 (낮은 카디널리티)
# object → string (arrow): 약 30% 감소 + 더 빠른 연산
```

**Dtype 크기 참조표:**

| Dtype | 바이트 | 사용 시점 |
|---|---|---|
| `bool` | 1 | 이진 플래그 |
| `boolean` | 1+null | 결측 허용(nullable) 이진 플래그 |
| `int8` | 1 | [-128, 127] |
| `int16` | 2 | [-32768, 32767] |
| `int32` | 4 | [-2B, 2B] |
| `int64` | 8 | 더 큰 정수 (기본값) |
| `float32` | 4 | 대부분의 지표에서 허용 가능한 정밀도 |
| `float64` | 8 | 높은 정밀도가 필요할 때 |
| `category` | 가변 | 고유값이 50% 미만일 때 |
| `string` (Arrow) | 압축됨 | 텍스트 컬럼 |

## 3단계 — PyArrow 백엔드 (pandas 2.0+)

```python
# PyArrow 엔진으로 로드 (CSV에서 가장 빠름)
df = pd.read_csv("data.csv", engine="pyarrow", dtype_backend="pyarrow")

# 기존 DataFrame 변환
df = df.convert_dtypes(dtype_backend="pyarrow")

# 왜 PyArrow인가?
# - 문자열이 메모리에 연속적으로 저장됨 (Python 객체가 아님)
# - SIMD 가속 연산
# - Arrow <> pandas 간 제로카피 교환
# - 모든 타입에 대해 nullable dtype 지원 (nullable int를 위한 float64 트릭 불필요)
```

## 4단계 — 벡터화된 연산 (절대 루프를 돌리지 마세요)

```python
# 느림 — Python 레벨 루프
for i, row in df.iterrows():                   # 벡터화 대비 약 1000배 느림
    df.at[i, "revenue"] = row["price"] * row["qty"]

# 느림 — 행 단위 apply
df["revenue"] = df.apply(lambda r: r["price"] * r["qty"], axis=1)

# 빠름 — 벡터화 산술 연산
df["revenue"] = df["price"] * df["qty"]

# 빠름 — numpy 연산
import numpy as np
df["log_revenue"] = np.log1p(df["revenue"])
df["clipped"] = np.clip(df["value"], 0, 100)

# 빠름 — .str 접근자 (벡터화된 문자열 연산)
df["domain"] = df["email"].str.extract(r"@(.+)$")

# 빠름 — .dt 접근자 (벡터화된 datetime 연산)
df["month"] = df["date"].dt.month
```

### apply()를 피할 수 없을 때

```python
# 수치 연산 apply는 numba로 벡터화
from numba import njit

@njit
def custom_calc(x, y, z):
    return x * y + z ** 2

df["result"] = custom_calc(df["a"].values, df["b"].values, df["c"].values)
```

## 5단계 — CoW (Copy-on-Write)

CoW는 pandas >= 3.0에서 항상 활성화되어 있습니다 (2.x에서는 옵트인). pandas 3.0에서는 경고가 발생하므로 `pd.options.mode.copy_on_write = True`를 **설정하지 마세요**.

```python
# CoW는 뷰가 변경 시점에 복사됨을 의미하며, 동작이 항상 예측 가능합니다
# 메서드 체이닝은 완전히 안전하고 메모리 효율적입니다
result = (df
    .query("status == 'active'")
    .assign(revenue=lambda x: x["price"] * x["qty"])
    .groupby("region")["revenue"]
    .sum()
)

# inplace=True는 절대 사용하지 마세요 — pandas 3.0에서 deprecated 처리됨
# 대신 할당을 사용하세요: df = df.dropna()
```

## 6단계 — 대용량 파일 처리

```python
# 1. 청크 단위로 읽기
results = []
for chunk in pd.read_csv("huge.csv", chunksize=200_000, dtype={"id": "int32"}):
    processed = chunk.query("status == 'active'").groupby("category")["revenue"].sum()
    results.append(processed)
final = pd.concat(results).groupby(level=0).sum()

# 2. 컬럼 프루닝 (필요한 컬럼만 로드)
df = pd.read_csv("huge.csv", usecols=["id", "revenue", "date"])

# 3. 반복 접근에는 Parquet 사용
df.to_parquet("data.parquet", engine="pyarrow", compression="snappy")
df = pd.read_parquet("data.parquet", columns=["id", "revenue"])  # 컬럼 프루닝

# 4. 진정한 대규모 데이터셋(> RAM)에는 Dask 사용
import dask.dataframe as dd
ddf = dd.read_csv("huge_*.csv")
result = ddf.groupby("category")["revenue"].sum().compute()
```

## 7단계 — groupby (그룹화) 성능

```python
# 빠름: 내장 집계 함수 (C 구현)
df.groupby("category")["revenue"].sum()       # 빠름
df.groupby("category")["revenue"].mean()      # 빠름
df.groupby("category")["revenue"].std()       # 빠름

# 빠름: 내장 함수를 사용한 named agg
df.groupby("category").agg(total=("revenue", "sum"), n=("id", "count"))

# 더 느림: lambda 또는 사용자 정의 함수 (Python 레벨)
df.groupby("category")["revenue"].agg(lambda x: x.quantile(0.9))

# 카테고리형에서 observed=False 회피 (모든 조합을 계산함)
df.groupby("category", observed=True)["revenue"].sum()  # 존재하는 카테고리만
```

## 8단계 — 병합 성능

```python
# merge_asof 전에 정렬
left = left.sort_values("key")
right = right.sort_values("key")

# 해시 조인 사용 (동등 조인의 기본값 — 이미 빠름)
pd.merge(left, right, on="id")

# 같은 키로 반복 조인할 경우: 먼저 인덱스로 설정
df_b = df_b.set_index("customer_id")
result = df_a.join(df_b, on="customer_id")  # 인덱스 조인이 더 빠름
```

## 9단계 — 흔한 안티패턴 회피

```python
# ❌ 루프 안에서 DataFrame 키우기
df = pd.DataFrame()
for item in items:
    df = pd.concat([df, pd.DataFrame([item])])   # O(n²) — 절대 금지

# ✅ 모은 뒤 한 번에 concat
rows = []
for item in items:
    rows.append(item)
df = pd.DataFrame(rows)                          # O(n)

# ❌ 연쇄 인덱싱 (느리고 버그 발생 가능)
df[df["a"] > 0]["b"] = 1

# ✅ loc 사용
df.loc[df["a"] > 0, "b"] = 1

# ❌ 단순 산술에 apply(axis=1) 사용
df.apply(lambda r: r["x"] + r["y"], axis=1)

# ✅ 벡터화
df["x"] + df["y"]

# ❌ 루프 안에서 df.shape / df.columns 반복 호출
for _ in range(10_000):
    n = len(df)  # len()은 O(1)

# ❌ 컬럼에 한 번 적용해도 될 astype을 매 행마다 적용하기
```

## 10단계 — PyArrow ADBC를 통한 SQL (pandas 2.2+)

```python
# 큰 결과 집합에서 SQLAlchemy보다 훨씬 빠름
import adbc_driver_postgresql.dbapi as adbc

with adbc.connect("postgresql://user:pw@host/db") as conn:
    df = pd.read_sql("SELECT * FROM large_table", conn)
    df.to_sql("output", conn, if_exists="replace", index=False)
```

## 메모리 절감 체크리스트

- [ ] `df.info(memory_usage="deep")` — 큰 컬럼 식별
- [ ] 범위가 허용하는 경우 `int64` → `int32`/`int16` 다운캐스트
- [ ] 정밀도가 결정적이지 않은 경우 `float64` → `float32` 다운캐스트
- [ ] 고유값 < 50%인 `object` 문자열 컬럼을 `category`로 변환
- [ ] `usecols=`로 필요한 컬럼만 로드
- [ ] `read_csv()`에 `dtype=`을 지정하여 object 폴백 방지
- [ ] 메서드 체이닝을 사용해 복사 회피 (pandas 3.0에서는 CoW가 항상 활성화됨)
- [ ] 반복 분석에는 CSV 대신 Parquet 사용
- [ ] 대용량 파일은 청크 단위로 처리
