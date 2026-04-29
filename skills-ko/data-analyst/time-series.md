# Pandas 시계열 분석 — 전문가 스킬

당신은 pandas >= 2.3 환경에서 시계열을 전문으로 다루는 데이터 분석가입니다.
날짜/시간 기반 분석을 수행할 때 다음 기법들을 적용합니다.

## 날짜·시간 파싱과 초기 설정

```python
# 문자열에서 파싱
df["date"] = pd.to_datetime(df["date"])
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
df["date"] = pd.to_datetime(df["date"], utc=True)           # UTC로 파싱

# Unix 타임스탬프에서 변환
df["date"] = pd.to_datetime(df["unix_ts"], unit="s")        # 초 단위
df["date"] = pd.to_datetime(df["unix_ts_ms"], unit="ms")    # 밀리초 단위

# 날짜 범위 생성
idx = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
idx = pd.date_range(start="2024-01-01", periods=12, freq="ME")  # 월말
idx = pd.bdate_range(start="2024-01-01", end="2024-12-31")      # 영업일

# Period 범위 (회계/달력 기간에 더 적합)
periods = pd.period_range(start="2024-01", periods=12, freq="M")
```

## DatetimeIndex — 시계열의 기반

```python
# 시간 기반 연산을 위해 DatetimeIndex 설정
df = df.set_index("date").sort_index()

# 부분 문자열 인덱싱 (매우 강력함)
df["2024"]                          # 2024년 전체
df["2024-Q1"]                       # 2024년 1분기
df["2024-06"]                       # 2024년 6월
df["2024-01-01":"2024-03-31"]       # 날짜 범위 슬라이스

# 인덱스 구성 요소 접근
df.index.year
df.index.month
df.index.day
df.index.dayofweek                  # 0=월요일
df.index.is_month_end
df.index.quarter
```

## Series에서의 .dt 접근자

```python
# 구성 요소 추출
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["day"]     = df["date"].dt.day
df["hour"]    = df["date"].dt.hour
df["minute"]  = df["date"].dt.minute
df["weekday"] = df["date"].dt.day_name()      # "Monday", "Tuesday", ...
df["week"]    = df["date"].dt.isocalendar().week
df["quarter"] = df["date"].dt.quarter

# 불리언 플래그
df["is_weekend"]   = df["date"].dt.dayofweek >= 5
df["is_month_end"] = df["date"].dt.is_month_end
df["is_leap"]      = df["date"].dt.is_leap_year

# 반올림 / 절단
df["hour_block"] = df["date"].dt.floor("h")
df["day"]        = df["date"].dt.normalize()        # 자정으로 정규화
df["week_start"] = df["date"].dt.to_period("W").dt.start_time
```

## 리샘플링 — 시간 기반 GroupBy

```python
# DatetimeIndex가 설정되어 있어야 함
daily = df.set_index("date")

# 다운샘플: 더 낮은 빈도로 집계
monthly = daily["revenue"].resample("ME").sum()     # 월말
weekly  = daily["revenue"].resample("W-MON").sum()  # 월요일로 끝나는 주
hourly  = daily["price"].resample("h").ohlc()       # OHLC (금융)

# 다중 집계
monthly = daily["revenue"].resample("ME").agg(
    total=("sum"),
    average=("mean"),
    peak=("max"),
    transactions=("count"),
)

# 업샘플: 더 높은 빈도로 채우기
upsampled = monthly.resample("D").ffill()           # 전방 채우기
upsampled = monthly.resample("D").interpolate()     # 보간

# 자주 쓰는 빈도 별칭
# "D"   = 달력 기준 일
# "B"   = 영업일
# "W"   = 주 (일요일 기준)
# "W-MON" = 월요일로 끝나는 주
# "ME"  = 월말 (pandas 2.2+; 이전에는 "M")
# "MS"  = 월초
# "QE"  = 분기말
# "QS"  = 분기초
# "YE"  = 연말
# "h"   = 시 (pandas 2.2+; 이전에는 "H")
# "min" = 분 (pandas 2.2+; 이전에는 "T")
# "s"   = 초
```

**참고:** Pandas 2.2+에서는 대문자 별칭("M", "H", "T", "S")이 폐기 예정(deprecated)입니다.
소문자 형태("ME", "h", "min", "s") 또는 접미사가 붙은 형태를 사용하세요.

## 롤링 윈도우

```python
# 고정 윈도우
df["revenue_7d_avg"] = df["revenue"].rolling(window=7).mean()
df["revenue_30d_sum"] = df["revenue"].rolling(window=30).sum()
df["revenue_7d_std"] = df["revenue"].rolling(window=7).std()

# 최소 관측치 수 지정 (시작 부분의 NaN 회피)
df["ma7"] = df["revenue"].rolling(window=7, min_periods=3).mean()

# 윈도우 가운데 정렬 (스무딩에 적합)
df["centered_ma"] = df["revenue"].rolling(window=7, center=True).mean()

# 시간 기반 롤링 (DatetimeIndex 필요)
df["7d_avg"] = df["revenue"].rolling("7D").mean()
df["30d_avg"] = df["revenue"].rolling("30D").mean()

# 사용자 정의 롤링 집계
df["roll_iqr"] = df["value"].rolling(30).apply(
    lambda x: x.quantile(0.75) - x.quantile(0.25), raw=False
)
```

## 확장 윈도우

```python
# 시작부터 현재 행까지의 모든 데이터
df["cumulative_sum"]  = df["revenue"].expanding().sum()
df["cumulative_mean"] = df["revenue"].expanding().mean()
df["cumulative_max"]  = df["revenue"].expanding().max()
df["running_total"]   = df["revenue"].cumsum()
df["running_max"]     = df["revenue"].cummax()
```

## 지수가중 (EWM)

```python
# 지수가중 이동평균 (최근 데이터일수록 가중치 큼)
df["ewma_span10"] = df["price"].ewm(span=10, adjust=False).mean()
df["ewma_com5"]   = df["price"].ewm(com=5).mean()
df["ewma_alpha"]  = df["price"].ewm(alpha=0.3).mean()

# 매개변수 (하나만 사용):
# span    — N-기간 등가
# com     — 질량 중심 (com = (span-1)/2)
# halflife — 기간 또는 시간 오프셋 단위 반감기
# alpha   — 평활화 계수 [0,1] (높을수록 최근 데이터 가중치 큼)
```

## 래그와 리드 피처

```python
# 래그 (이전 값) — 시계열 모델링에 핵심
df["revenue_lag1"]  = df["revenue"].shift(1)     # 직전 기간
df["revenue_lag7"]  = df["revenue"].shift(7)     # 1주 전
df["revenue_lag30"] = df["revenue"].shift(30)    # 1개월 전

# 리드 (미래 값) — 사후(retrospective) 분석에서만 사용
df["revenue_lead1"] = df["revenue"].shift(-1)

# 기간 대비 변화
df["mom_change"] = df["revenue"] - df["revenue"].shift(1)      # 절대값
df["mom_pct"]    = df["revenue"].pct_change(1)                 # 백분율
df["yoy_pct"]    = df["revenue"].pct_change(12)                # 전년 동월 대비 (월별 데이터)

# 로그 수익률 (금융)
import numpy as np
df["log_return"] = np.log(df["price"] / df["price"].shift(1))
```

## 타임존 처리

```python
# 타임존 정보가 없는 타임스탬프를 로컬라이즈
df["ts"] = df["ts"].dt.tz_localize("UTC")
df["ts"] = df["ts"].dt.tz_localize("America/New_York")

# 타임존 간 변환
df["ts_eastern"] = df["ts"].dt.tz_convert("America/New_York")
df["ts_london"]  = df["ts"].dt.tz_convert("Europe/London")

# 타임존 정보 제거 (먼저 UTC로 정규화 후 제거)
df["ts_naive"] = df["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
```

## 누락된 타임스탬프와 갭

```python
# 일별 시계열에서 누락된 날짜 찾기
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
missing = full_range.difference(df.index)
print(f"Missing dates: {len(missing)}")

# 갭을 채우기 위해 reindex
df = df.reindex(full_range)
df["value"] = df["value"].ffill()   # 갭을 전방 채움

# 예상보다 큰 갭 탐지
gaps = df.index.to_series().diff().dropna()
large_gaps = gaps[gaps > pd.Timedelta("2D")]
print(large_gaps)
```

## 계절성 및 추세 분해 (decomposition)

```python
# statsmodels 사용 (별도 패키지)
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df["revenue"], model="additive", period=12)
# 접근: result.trend, result.seasonal, result.resid

# STL 분해 (decomposition) (강건함)
from statsmodels.tsa.seasonal import STL
stl = STL(df["revenue"], period=12, robust=True).fit()
df["trend"]    = stl.trend
df["seasonal"] = stl.seasonal
df["residual"] = stl.resid
```

## 시계열 집계 패턴

```python
# 일별 → 월별 매출과 영업일 수 집계
daily = df.set_index("date")
monthly = daily.resample("ME").agg(
    revenue=("revenue", "sum"),
    avg_daily=("revenue", "mean"),
    trading_days=("revenue", "count"),
    peak_day=("revenue", "max"),
)

# 주간 대비 성장률
monthly["wow_growth"] = monthly["revenue"].pct_change(1)

# 전년 동기 대비 비교
monthly["revenue_ly"] = monthly["revenue"].shift(12)   # 전년 (월별)
monthly["yoy"] = (monthly["revenue"] / monthly["revenue_ly"] - 1)
```

## 실전 시계열 체크리스트

- [ ] `pd.to_datetime()`으로 날짜를 파싱하고 변환 후 NaT가 없는지 확인
- [ ] 리샘플링이나 롤링 연산 전에 datetime 인덱스 기준으로 정렬
- [ ] 타임존을 명시적으로 처리 (항상 UTC로 저장하고 표시할 때 로컬 시간으로 변환)
- [ ] `date_range.difference(df.index)`로 갭 / 누락된 기간 점검
- [ ] 롤링에서 `min_periods=`를 사용해 앞부분 NaN 누적 방지
- [ ] 리샘플링 빈도 별칭이 pandas 2.2+ 호환(소문자)인지 확인
- [ ] 래그 피처는 반드시 `.shift()` 사용 — 학습 시 shift(-n)으로 미래 정보를 참조하지 말 것
