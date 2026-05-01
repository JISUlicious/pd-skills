# 고급 Pandas v2.3 — 전문가 스킬

당신은 전문 데이터 분석가입니다. pandas >= 2.3 으로 정교한 데이터 조작을
수행하기 위해 다음의 고급 기법들을 적용합니다.

## pipe() 를 활용한 메서드 체이닝

```python
# pipe() inserts the DataFrame as the first argument to any function
# Enables fully chainable custom transformations

def add_revenue(df, price_col="price", qty_col="qty"):
    return df.assign(revenue=df[price_col] * df[qty_col])

def normalize(df, col, group_col):
    df = df.copy()
    df[f"{col}_norm"] = df.groupby(group_col)[col].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    return df

def tag_outliers(df, col, n_std=3):
    mu, sigma = df[col].mean(), df[col].std()
    return df.assign(
        is_outlier=(df[col] - mu).abs() > n_std * sigma
    )

result = (
    df
    .pipe(add_revenue)
    .pipe(normalize, col="revenue", group_col="region")
    .pipe(tag_outliers, col="revenue_norm")
    .query("not is_outlier")
    .groupby("region")
    .agg(total=("revenue", "sum"), avg=("revenue", "mean"))
)
```

## Nullable 확장 타입 (결측 허용, pandas 1.0+)

레거시 NumPy dtype 과 달리, nullable 타입은 정수를 float64 로 강제 변환하지
않고도 `NA` (결측값) 를 지원합니다.

```python
# Nullable integer
s = pd.array([1, 2, None, 4], dtype="Int64")   # 대문자 I
df["count"] = df["count"].astype("Int32")

# Nullable float
df["rate"] = df["rate"].astype("Float32")

# Nullable boolean
df["flag"] = df["flag"].astype("boolean")

# Arrow-backed string (pandas 2.0+)
df["name"] = df["name"].astype("string")       # 기본 StringDtype
# 또는 명시적으로 Arrow 백엔드 사용:
df["name"] = df["name"].astype(pd.StringDtype(storage="pyarrow"))

# 전체 DataFrame 을 최적의 nullable 타입으로 변환
df = df.convert_dtypes()                        # 최적의 nullable 타입 추론
df = df.convert_dtypes(dtype_backend="pyarrow") # PyArrow 백엔드 강제
```

**핵심 차이점:**

```python
# 레거시: NaN 때문에 int → float64 로 강제 변환됨
pd.Series([1, 2, None]).dtype       # float64
# Nullable: NA 와 함께 int 를 그대로 보존
pd.array([1, 2, None], dtype="Int64").dtype  # Int64
```

## 카테고리형 dtype — 고급 활용

```python
# 명시적 카테고리 순서 정의 (정렬/비교에 사용)
size_type = pd.CategoricalDtype(
    categories=["XS", "S", "M", "L", "XL", "XXL"],
    ordered=True
)
df["size"] = df["size"].astype(size_type)

# 순서 비교
df[df["size"] >= "L"]                         # ordered=True 이므로 동작함
df.sort_values("size")                        # 정의된 카테고리 순서로 정렬

# 모든 카테고리(빈 카테고리 포함)에 대한 GroupBy
df.groupby("size", observed=False)["revenue"].sum()

# 카테고리 이름 변경
df["size"] = df["size"].cat.rename_categories({"XS": "Extra Small"})

# 카테고리 추가 / 제거
df["size"] = df["size"].cat.add_categories(["XXXL"])
df["size"] = df["size"].cat.remove_unused_categories()

# 카테고리 코드 (정수 표현 — ML 에 유용)
df["size_code"] = df["size"].cat.codes
```

## MultiIndex — 계층적 인덱싱

```python
# MultiIndex 생성
df = df.set_index(["year", "quarter", "region"])

# 레벨 접근
df.loc["2024"]                                # 외부 레벨
df.loc[("2024", "Q1")]                        # 외부 + 중간
df.loc[("2024", "Q1", "US")]                  # 정확히 지정
df.loc[("2024", slice(None), "US")]           # 2024년의 모든 분기, US

# 단면(xs)
df.xs("US", level="region")
df.xs(("2024", "Q1"), level=["year", "quarter"])

# 더 깔끔한 슬라이싱을 위한 IndexSlice
idx = pd.IndexSlice
df.loc[idx["2024":"2025", "Q1":"Q2", :], "revenue"]

# 레벨 순서 교체
df = df.swaplevel("quarter", "region")
df = df.sort_index()

# MultiIndex 컬럼 평탄화 (pivot_table 이후)
df.columns = ["_".join(col).strip("_") for col in df.columns]
```

## eval() 과 query() — 표현식 엔진

```python
# eval: 빠른 벡터화 표현식 평가
# 대규모 DataFrame 의 복잡한 산술 연산에 최적 (임시 배열 회피)
df = df.eval("margin = (revenue - cost) / revenue")
df = df.eval("""
    gross = price * qty
    discount_amt = gross * discount_pct
    net = gross - discount_amt
""")

# query: 표현식 문자열로 필터링
df.query("region == 'US' and revenue > @min_rev and status in ['active', 'trial']")

# 로컬 변수 참조에는 @ 사용
threshold = df["revenue"].quantile(0.9)
df.query("revenue > @threshold")

# pandas eval 은 속도를 위해 numexpr 를 사용 (numexpr 설치 시 2~3배 향상)
```

## 윈도우 함수 — 고급

```python
# 사용자 지정 오프셋의 롤링 (DatetimeIndex 필요)
df = df.set_index("date")
df["7d_mean"]  = df["revenue"].rolling("7D").mean()
df["30d_mean"] = df["revenue"].rolling("30D").mean()
df["90d_mean"] = df["revenue"].rolling("90D").mean()

# 한 번의 롤링 패스로 여러 통계량 계산
rolled = df["revenue"].rolling(30).agg(
    roll_mean="mean",
    roll_std="std",
    roll_min="min",
    roll_max="max",
)

# 볼린저 밴드 (Bollinger Bands)
df["bb_mid"]   = df["price"].rolling(20).mean()
df["bb_upper"] = df["bb_mid"] + 2 * df["price"].rolling(20).std()
df["bb_lower"] = df["bb_mid"] - 2 * df["price"].rolling(20).std()

# 확장 윈도우 (누적 통계량)
df["running_avg"]  = df["revenue"].expanding().mean()
df["running_best"] = df["revenue"].expanding().max()

# halflife 를 timedelta 로 지정한 EWM (pandas 1.1+)
df["ewm_halflife"] = df["price"].ewm(
    halflife=pd.Timedelta("7 days"),
    times=df.index
).mean()
```

## DataFrame 스타일링 (출력 포맷팅)

```python
# Jupyter 노트북 / HTML 리포트용 pandas Styler
styled = (
    df.style
    .format({
        "revenue": "${:,.0f}",
        "margin": "{:.1%}",
        "growth": "{:+.1%}",
        "date": "{:%Y-%m-%d}",
    })
    .background_gradient(subset=["revenue"], cmap="Blues")
    .background_gradient(subset=["margin"], cmap="RdYlGn", vmin=0, vmax=0.5)
    .bar(subset=["growth"], align="zero", color=["#d65f5f", "#5fba7d"])
    .highlight_max(subset=["revenue"], color="lightgreen")
    .highlight_min(subset=["revenue"], color="salmon")
    .set_caption("Revenue Summary by Region")
    .set_table_styles([
        {"selector": "th", "props": [("font-size", "11px"), ("text-align", "center")]},
    ])
)

# 내보내기
styled.to_excel("report.xlsx", engine="openpyxl")
styled.to_html("report.html")

# Jupyter 에서 표시
styled  # 변수만 출력하면 됨
```

## 사용자 정의 접근자 확장

```python
# DataFrame 에 사용자 정의 네임스페이스 등록
@pd.api.extensions.register_dataframe_accessor("biz")
class BizAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def revenue_summary(self, group_col):
        return self._obj.groupby(group_col).agg(
            total_revenue=("revenue", "sum"),
            avg_revenue=("revenue", "mean"),
            n_orders=("order_id", "count"),
        )

    def top_n(self, col, n=10):
        return self._obj.nlargest(n, col)

# 사용
df.biz.revenue_summary("region")
df.biz.top_n("revenue")
```

## 효율적인 반복 패턴

```python
# 반복이 정말로 필요할 때, 다음 순서로 사용 (가장 빠른 것부터):

# 1. itertuples — namedtuple 사용, iterrows 대비 약 10배 빠름
for row in df.itertuples(index=False):
    process(row.revenue, row.category)

# 2. to_dict("records") — 딕셔너리의 리스트
for record in df.to_dict("records"):
    process(record["revenue"])

# 3. numpy/numba 로 벡터화 (수치 연산에 가장 빠름)
import numba
@numba.njit
def loop_calc(revenues, costs):
    results = np.empty(len(revenues))
    for i in range(len(revenues)):
        results[i] = revenues[i] - costs[i]
    return results
df["profit"] = loop_calc(df["revenue"].values, df["cost"].values)

# 4. iterrows — 가장 느림, 대규모 DataFrame 에서는 피할 것
for idx, row in df.iterrows():   # 벡터화 대비 1000배 느림
    ...
```

## Pandas 옵션

```python
# 표시 설정
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 120)

# 성능 설정
pd.set_option("compute.use_numexpr", True)          # eval/query 에서 numexpr 활성화
# 참고: pandas 3.0 에서 CoW 는 항상 켜져 있음; mode.copy_on_write 를 설정하지 말 것
# 참고: pandas 3.0 에서 Arrow 문자열이 기본값임; future.infer_string 을 설정하지 말 것

# 임시 설정용 컨텍스트 매니저
with pd.option_context("display.max_rows", 200, "display.float_format", "{:.4f}".format):
    display(df)

# 모든 옵션 초기화
pd.reset_option("all")
```
