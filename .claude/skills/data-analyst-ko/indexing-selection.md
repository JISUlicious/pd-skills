# Pandas 인덱싱 & 선택 — 전문가 스킬

당신은 전문 데이터 분석가입니다. 다음 인덱싱 패턴을 정확하게 적용하십시오.
올바른 인덱서를 선택하는 것은 정확성, 성능, 그리고
SettingWithCopyWarning을 피하기 위해 매우 중요합니다.

## 네 가지 주요 인덱서

| 인덱서 | 축 | 기준 | 적합한 용도 |
|---|---|---|---|
| `df["col"]` | 칼럼 (column) | 레이블 | 단일 칼럼 접근 |
| `.loc[row, col]` | 양쪽 | **레이블** | 레이블 기반 (label-based) 선택 |
| `.iloc[row, col]` | 양쪽 | **위치** | 정수 위치 기반 (position-based) 선택 |
| `.at[row, col]` | 양쪽 | **레이블** | 단일 스칼라 (빠름) |
| `.iat[row, col]` | 양쪽 | **위치** | 위치로 단일 스칼라 접근 |

## 칼럼 선택

```python
# 단일 칼럼 → Series
s = df["price"]

# 다중 칼럼 → DataFrame
sub = df[["user_id", "price", "timestamp"]]

# 동적 칼럼 선택
cols = [c for c in df.columns if c.startswith("feature_")]
df[cols]

# dtype으로 선택
numeric_df = df.select_dtypes(include="number")
str_df     = df.select_dtypes(include=["object", "string", "category"])
date_df    = df.select_dtypes(include="datetime")
```

## 행 선택 — 불리언 인덱싱 (boolean indexing)

```python
# 단일 조건
df[df["status"] == "active"]
df[df["amount"] > 1000]

# 다중 조건 — 괄호와 함께 & | ~ 사용
df[(df["status"] == "active") & (df["amount"] > 1000)]
df[(df["region"] == "US") | (df["region"] == "EU")]
df[~df["status"].isin(["cancelled", "refunded"])]

# isin / notin
df[df["category"].isin(["Electronics", "Books"])]
df[~df["category"].isin(["Spam"])]

# 문자열 필터
df[df["name"].str.startswith("A")]
df[df["email"].str.contains(r"@gmail\.com", regex=True, na=False)]

# 날짜 범위 필터
df[df["date"].between("2024-01-01", "2024-12-31")]
df[(df["date"] >= "2024-01-01") & (df["date"] < "2025-01-01")]

# 결측값 필터
df[df["phone"].notna()]
df[df["optional_col"].isna()]
```

## .loc — 레이블 기반 선택

```python
# 레이블로 행 선택, 모든 칼럼
df.loc[42]
df.loc[[10, 20, 30]]
df.loc[10:30]            # 레이블 슬라이스 (양쪽 끝 포함)

# 행 + 칼럼
df.loc[df["amount"] > 1000, ["user_id", "amount"]]
df.loc[:, "price":"revenue"]   # 칼럼 레이블 슬라이스

# 칼럼 부분집합과 조건부 행 선택
df.loc[(df["status"] == "active") & (df["amount"] > 0), "revenue"] *= 1.1

# 안전한 값 설정 (체인 인덱싱 없음)
df.loc[df["flag"] == True, "score"] = 0
```

## .iloc — 위치 기반 선택

```python
df.iloc[0]               # 첫 번째 행
df.iloc[-1]              # 마지막 행
df.iloc[0:10]            # 처음 10개 행
df.iloc[[0, 5, 10]]      # 위치 0, 5, 10의 행

df.iloc[:, 0]            # 첫 번째 칼럼
df.iloc[:, [0, 2, 4]]    # 위치 0, 2, 4의 칼럼
df.iloc[0:5, 0:3]        # 좌상단 5×3 부분 그리드

# 마지막 3개 행, 마지막 2개 칼럼
df.iloc[-3:, -2:]
```

## .query() — 표현식 문자열

```python
# 단순
df.query("amount > 1000")
df.query("status == 'active'")

# 복합
df.query("status == 'active' and amount > 1000")
df.query("region in ['US', 'EU']")
df.query("amount > @threshold")           # @로 Python 변수 참조
df.query("price > price.mean()")          # 칼럼 표현식

# 날짜 비교
df.query("date >= '2024-01-01'")

# 인덱스 접근
df.query("index > 100")                  # RangeIndex에서 동작
```

`.query()`의 장점: 가독성이 좋고, 불리언 연산자 괄호를 피할 수 있으며,
체인 인덱싱 버그로부터 안전합니다.

## MultiIndex (계층)

```python
# 생성
midx_df = df.set_index(["region", "category"])

# 외부 레벨 접근
midx_df.loc["US"]

# 특정 조합 접근
midx_df.loc[("US", "Electronics")]

# 단면 (cross-section)
midx_df.xs("Electronics", level="category")

# MultiIndex 레벨에 대한 쿼리
midx_df.loc[midx_df.index.get_level_values("region") == "US"]

# MultiIndex 리셋
df_flat = midx_df.reset_index()
```

## .at / .iat — 단일 스칼라 접근

```python
# 단일 값에 대해 .loc/.iloc보다 빠름
val = df.at[row_label, "price"]
df.at[row_label, "price"] = 99.99

val = df.iat[0, 3]          # 행 0, 칼럼 위치 3
df.iat[0, 3] = 42
```

속도가 중요한 좁은 루프에서는 `.at`/`.iat`을 사용하십시오.

## SettingWithCopyWarning 피하기

**문제:** 체인 인덱싱은 원본 DataFrame을 갱신하는 데 조용히 실패합니다.

```python
# 나쁜 예 — df를 갱신하지 못할 수 있음
df[df["status"] == "active"]["score"] = 100

# 좋은 예 — .loc 사용
df.loc[df["status"] == "active", "score"] = 100

# 좋은 예 — 독립적인 슬라이스가 필요하면 .copy() 사용
subset = df[df["status"] == "active"].copy()
subset["score"] = 100   # 안전: 별도 객체
```

**Copy-on-Write**는 pandas >= 3.0에서 항상 활성화되어 있습니다 (2.x에서는 옵트인):
```python
# 모든 인덱싱 연산은 CoW 뷰를 반환합니다; 체인 쓰기는 명시적으로 오류를 발생시킵니다
# pd.options.mode.copy_on_write를 설정하지 마십시오 — pandas 3.0에서 폐기되었습니다
```

## 샘플링 & Head/Tail

```python
df.head(10)
df.tail(5)
df.sample(n=100, random_state=42)
df.sample(frac=0.1, random_state=42)          # 10% 샘플
df.sample(n=100, weights="weight_col")         # 가중 샘플링
df.nlargest(10, "revenue")                     # revenue 기준 상위 10개
df.nsmallest(5, "error_rate")                  # 하위 5개
```

## 필터링 패턴 치트시트

```python
# Between (양쪽 포함)
df[df["score"].between(80, 100)]

# 결측값 / 비결측값
df[df["col"].isna()]
df[df["col"].notna()]

# 중복 행
df[df.duplicated(subset=["id"], keep=False)]

# 문자열로 시작/끝
df[df["code"].str.startswith("SFO")]
df[df["file"].str.endswith(".csv")]

# 정규식 매치
df[df["email"].str.match(r"^[\w.+-]+@[\w-]+\.\w+$", na=False)]

# 칼럼 리스트가 모두 결측값
df[df[["col_a", "col_b"]].isnull().all(axis=1)]

# 칼럼 리스트 중 하나라도 결측값
df[df[["col_a", "col_b"]].isnull().any(axis=1)]
```
