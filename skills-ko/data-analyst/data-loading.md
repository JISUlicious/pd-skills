# Pandas 데이터 로딩 & IO — 전문가 스킬

당신은 pandas >= 2.3의 IO 도구에 깊이 있는 지식을 갖춘 데이터 분석 전문가입니다.
데이터를 로드하거나 저장할 때 다음 지식을 정확하게 적용합니다.

## 핵심 리더 함수

### CSV / 평문 파일
```python
# 최소 사용 – 빠른 C 엔진 (기본값)
df = pd.read_csv("file.csv")

# 프로덕션 수준: dtype 지정, 날짜 파싱, NA 처리, 청크 단위 처리
df = pd.read_csv(
    "file.csv",
    dtype={"id": "int32", "category": "category"},
    parse_dates=["timestamp"],
    na_values=["NA", "N/A", "-", ""],
    usecols=["id", "name", "value", "timestamp"],  # 필요한 칼럼만 로드
    chunksize=None,          # 큰 파일의 경우 예: 100_000 으로 설정
    engine="c",              # "c" (빠름) | "python" (기능 풍부) | "pyarrow"
    low_memory=False,        # C 엔진에서 혼합 타입 칼럼 회피
)

# 큰 파일: 청크 반복 처리
chunks = []
for chunk in pd.read_csv("big.csv", chunksize=100_000):
    chunks.append(chunk.query("value > 0"))
df = pd.concat(chunks, ignore_index=True)
```

주요 파라미터:
- `sep` / `delimiter` — 구분자 문자 (기본값 `,`)
- `header` — 칼럼 이름으로 사용할 행 번호 (0 = 첫 번째 행)
- `index_col` — 행 인덱스로 사용할 칼럼
- `skiprows` / `nrows` — 시작 행 건너뛰기; 읽을 행 수 제한
- `thousands` / `decimal` — 로케일 인식 숫자 파싱
- `encoding` — "utf-8", "latin-1", "utf-8-sig" (BOM) 등
- `compression` — "gzip", "bz2", "zip", "xz", "zstd" (자동 감지)

### Excel
```python
# 단일 시트
df = pd.read_excel("data.xlsx", sheet_name="Sales", engine="openpyxl")

# 여러 시트 → DataFrame 딕셔너리
sheets = pd.read_excel("data.xlsx", sheet_name=None)  # 모든 시트
first_two = pd.read_excel("data.xlsx", sheet_name=[0, 1])

# 쓰기
with pd.ExcelWriter("out.xlsx", engine="openpyxl", mode="w") as w:
    df1.to_excel(w, sheet_name="Summary", index=False)
    df2.to_excel(w, sheet_name="Detail", index=False)
```

### JSON
```python
df = pd.read_json("data.json", orient="records", dtype={"id": int})
# orient 옵션: "records", "split", "index", "columns", "values", "table"

df.to_json("out.json", orient="records", lines=True, date_format="iso")
```

### Parquet (칼럼 기반 저장에 권장)
```python
# pyarrow 또는 fastparquet 필요
df = pd.read_parquet("data.parquet", engine="pyarrow")
df = pd.read_parquet("data.parquet", columns=["id", "value"])  # 칼럼 프루닝

df.to_parquet("out.parquet", engine="pyarrow", compression="snappy", index=False)
```

Parquet은 다음 용도에 가장 적합한 포맷입니다:
- 칼럼 기반 접근 패턴 (필요한 칼럼만 읽기)
- 타입 충실도 (dtype을 정확하게 보존)
- 압축 효율성

### SQL
```python
import sqlalchemy as sa

engine = sa.create_engine("postgresql+psycopg2://user:pw@host/db")

# 전체 테이블
df = pd.read_sql_table("orders", con=engine)

# 커스텀 쿼리
df = pd.read_sql_query(
    "SELECT id, amount FROM orders WHERE status = 'complete'",
    con=engine,
    parse_dates=["created_at"],
    chunksize=50_000,
)

# 쓰기
df.to_sql("results", con=engine, if_exists="replace", index=False, method="multi")

# pandas 2.2+: ADBC 드라이버 (Arrow 네이티브 DB에서 훨씬 빠름)
# pip install adbc-driver-postgresql
import adbc_driver_postgresql.dbapi as adbc
conn = adbc.connect("postgresql://user:pw@host/db")
df = pd.read_sql("SELECT * FROM orders", conn)
```

### HDF5
```python
df.to_hdf("store.h5", key="df", mode="w", complevel=9)
df = pd.read_hdf("store.h5", key="df")

# 여러 데이터셋을 위한 HDFStore
with pd.HDFStore("store.h5") as store:
    store["sales"] = sales_df
    store["returns"] = returns_df
```

### 기타 포맷
```python
pd.read_clipboard()          # 클립보드에서 붙여넣기
pd.read_html("url")[0]       # HTML 페이지의 첫 번째 테이블
pd.read_feather("file.feather")
pd.read_orc("file.orc")
pd.read_xml("file.xml", xpath=".//record")
pd.read_stata("file.dta")
pd.read_spss("file.sav")
```

## 로드 시 dtype 최적화

object 칼럼으로의 암묵적 폴백을 피하기 위해 항상 dtype을 명시적으로 지정합니다:

```python
dtype_map = {
    "id": "int32",
    "user_id": "int32",
    "amount": "float32",
    "status": "category",
    "flag": "boolean",      # nullable boolean (pandas 1.0+)
    "name": "string",       # Arrow 기반 string (pandas 2.0+)
}
df = pd.read_csv("data.csv", dtype=dtype_map)
```

카디널리티가 낮은 문자열 칼럼에는 `category` dtype을 사용합니다 — 메모리를 10~100배 절약합니다.

## PyArrow 기반 dtype (pandas 2.0+)

```python
# Arrow 네이티브 타입을 위해 PyArrow 엔진으로 전체 CSV 로드
df = pd.read_csv("data.csv", engine="pyarrow", dtype_backend="pyarrow")

# 또는 로드 후 변환
df = df.convert_dtypes(dtype_backend="pyarrow")
```

## 로드 후 검증

로드 직후 항상 검증을 수행합니다:
```python
print(df.shape)           # 행 × 칼럼
print(df.dtypes)          # 모든 dtype이 예상과 일치하는지 확인
print(df.isnull().sum())  # 칼럼별 결측값 (null) 개수
print(df.duplicated().sum())  # 중복 행
print(df.head())
print(df.describe())
```

## 흔한 함정 및 해결책

| 문제 | 해결책 |
|---|---|
| 혼합 타입 칼럼이 `object`로 로드됨 | `dtype=`을 명시적으로 설정하거나 `low_memory=False` 사용 |
| 날짜가 문자열로 로드됨 | `parse_dates=["col"]` 사용 |
| 큰 파일에서 MemoryError 발생 | `chunksize=` 사용하여 반복 처리 |
| 인코딩 오류 | `encoding="latin-1"` 또는 `encoding_errors="replace"` 시도 |
| 느린 SQL 읽기 | ADBC 드라이버로 전환하거나 `chunksize=` 사용 |
| NaN 때문에 정수가 float이 됨 | nullable int 사용: `dtype="Int64"` (대문자 I) |
