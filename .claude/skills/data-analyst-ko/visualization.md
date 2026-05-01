# Pandas 데이터 시각화 — 전문가 스킬

당신은 전문 데이터 분석가입니다. **Plotly가 기본 시각화 라이브러리입니다.**
인터랙티브 차트(호버, 줌, 클릭 필터, 자체 완결형 HTML로 내보내기 가능)는
정적 PNG보다 이해관계자에게 훨씬 더 매력적이며, 코드 복잡성 측면에서도
추가 비용이 들지 않습니다.

**matplotlib + seaborn은 다음과 같은 경우에만** 사용합니다:
- 출판물이나 논문을 위한 인쇄용 정적 그림이 필요한 경우
- HTML 임베딩이 불가능한 슬라이드 자료
- Plotly 문법으로 표현할 수 없는 커스텀 레이아웃을 정밀하게 제어해야 하는 경우
- 브라우저가 없는 환경에서의 서버 측 이미지 생성

그 외 모든 경우 — 탐색적 분석, 대시보드, 내부 보고서, 노트북, 웹 앱 —
에서는 Plotly가 올바른 선택입니다.

## Plotly — 기본 레시피

```python
import plotly.express as px
import plotly.graph_objects as go
```

### 선 차트 / 시계열

```python
# 단일 시리즈
fig = px.line(df, x="date", y="revenue", title="Daily Revenue")
fig.show()

# 색상으로 구분되는 다중 시리즈
fig = px.line(df, x="date", y="revenue", color="region",
              title="Revenue by Region", markers=False)
fig.update_layout(hovermode="x unified")           # 호버 시 모든 시리즈 표시
fig.show()

# 평활화된 추세선 포함 (이동 평균 오버레이)
df_long = df.melt(id_vars="date", value_vars=["revenue", "revenue_7d"],
                  var_name="series", value_name="value")
fig = px.line(df_long, x="date", y="value", color="series",
              title="Daily Revenue + 7-day Rolling Mean")
fig.show()
```

### 막대 차트

```python
# 세로 막대
fig = px.bar(
    df.groupby("category", as_index=False)["revenue"].sum(),
    x="category", y="revenue",
    title="Revenue by Category",
    text_auto=".2s",                               # 자동 포맷된 레이블
)
fig.show()

# 그룹 막대 (카테고리 × 시간)
pivot = (df.groupby(["region", "quarter"])["revenue"].sum()
           .reset_index())
fig = px.bar(pivot, x="region", y="revenue", color="quarter",
             barmode="group", text_auto=".2s",
             title="Revenue by Region and Quarter")
fig.show()

# 누적 막대
fig = px.bar(pivot, x="region", y="revenue", color="quarter",
             barmode="stack", title="Revenue Composition by Region")
fig.show()

# 가로 막대 — 카테고리가 많을 때 가장 적합
top_products = (df.groupby("product", as_index=False)["revenue"].sum()
                  .nlargest(15, "revenue").sort_values("revenue"))
fig = px.bar(top_products, x="revenue", y="product",
             orientation="h", text_auto=".2s",
             title="Top 15 Products by Revenue")
fig.show()
```

### 산점도

```python
# 호버, 색상, 크기 인코딩이 있는 산점도
fig = px.scatter(
    df, x="ad_spend", y="revenue",
    color="segment", size="margin",
    hover_data=["product", "region", "date"],
    opacity=0.6,
    title="Ad Spend vs Revenue (size = margin)",
    trendline="ols",                               # 내장 회귀선
)
fig.show()

# 패싯 산점도 (스몰 멀티플)
fig = px.scatter(df, x="ad_spend", y="revenue", color="segment",
                 facet_col="region", facet_col_wrap=3,
                 title="Spend vs Revenue by Region")
fig.show()
```

### 분포

```python
# 히스토그램
fig = px.histogram(df, x="revenue", nbins=50,
                   title="Revenue Distribution")
fig.show()

# 그룹별 히스토그램 (오버레이 또는 누적)
fig = px.histogram(df, x="revenue", color="segment",
                   nbins=40, barmode="overlay", opacity=0.6,
                   title="Revenue by Segment")
fig.show()

# 그룹별 박스 플롯
fig = px.box(df, x="segment", y="revenue", color="segment",
             points="outliers", title="Revenue Distribution by Segment")
fig.show()

# 바이올린 플롯 — 전체 분포 형태
fig = px.violin(df, x="region", y="revenue", color="region",
                box=True, points="all",
                title="Revenue Distribution by Region")
fig.show()
```

### 상관관계 히트맵 (sns.heatmap 대체)

```python
corr = df[numeric_cols].corr().round(2)
fig = px.imshow(
    corr, text_auto=True, aspect="auto",
    color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
    title="Feature Correlation Matrix",
)
fig.show()
```

### 지리 / 지도

```python
# 단계 구분도 (choropleth) — 국가 단위
country_rev = df.groupby("country", as_index=False)["revenue"].sum()
fig = px.choropleth(country_rev, locations="country",
                    locationmode="country names", color="revenue",
                    color_continuous_scale="Viridis",
                    title="Revenue by Country")
fig.show()

# 지도 위 산점도 (위도/경도 점)
fig = px.scatter_geo(df, lat="lat", lon="lon", color="region", size="revenue",
                     hover_name="city", projection="natural earth")
fig.show()
```

### 트리맵 / 선버스트 (계층형)

```python
fig = px.treemap(df, path=["region", "category", "product"],
                 values="revenue", color="margin",
                 color_continuous_scale="RdYlGn",
                 title="Revenue Hierarchy")
fig.show()

fig = px.sunburst(df, path=["region", "category"], values="revenue",
                  title="Revenue by Region → Category")
fig.show()
```

### 깔때기 (funnel) / 전환

```python
funnel_df = pd.DataFrame({
    "stage": ["Visit", "Sign-up", "Trial", "Paid"],
    "count": [100_000, 12_000, 4_000, 800],
})
fig = px.funnel(funnel_df, x="count", y="stage",
                title="Conversion Funnel")
fig.show()
```

### 주석이 있는 시계열

```python
fig = px.line(df, x="date", y="metric", title="Daily Metric")
# 알려진 이벤트 표시
fig.add_vline(x="2024-06-15", line_dash="dash", line_color="red",
              annotation_text="Recipe v3 deploy", annotation_position="top")
# 구간 강조
fig.add_vrect(x0="2024-06-15", x1="2024-06-22",
              fillcolor="red", opacity=0.15, line_width=0)
fig.show()
```

### 저장 / 공유

```python
# 자체 완결형 HTML — 모든 브라우저에서 작동, 데이터 포함
fig.write_html("/tmp/chart.html", include_plotlyjs="cdn")

# 정적 내보내기 (`kaleido` 필요 — `uv pip install kaleido`)
fig.write_image("/tmp/chart.png", width=1200, height=700, scale=2)
fig.write_image("/tmp/chart.svg")                  # 슬라이드용 벡터
```

### Plotly 테마 기본값

```python
import plotly.io as pio

pio.templates.default = "plotly_white"             # 더 깔끔한 배경
# 기타 내장 테마: "plotly_dark", "ggplot2", "seaborn", "simple_white"

# 그림별 오버라이드 (제한적으로 사용)
fig.update_layout(
    font_family="Inter, sans-serif",
    title_font_size=16,
    legend=dict(orientation="h", y=-0.15),
    margin=dict(l=40, r=20, t=60, b=40),
    hovermode="closest",
)
```

## Plotly 대시보드 (서브플롯)

여러 패널로 구성된 보고서 형식의 그림이 필요할 때는 Plotly의 `make_subplots`를
사용합니다 — 여전히 인터랙티브하고, 단일 HTML로 유지되며, matplotlib도 필요 없습니다:

```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=("Weekly Revenue", "Revenue by Region",
                    "Margin Distribution", "Top 10 Products",
                    "Cohort Retention", "Spend vs Revenue"),
    specs=[[{"type": "scatter"}, {"type": "bar"},     {"type": "histogram"}],
           [{"type": "bar"},     {"type": "heatmap"}, {"type": "scatter"}]],
)

# 좌상단: 주간 매출
weekly = df.set_index("date")["revenue"].resample("W").sum().reset_index()
fig.add_trace(go.Scatter(x=weekly["date"], y=weekly["revenue"], mode="lines"),
              row=1, col=1)

# 상단 중앙: 지역별 매출
region_rev = df.groupby("region", as_index=False)["revenue"].sum()
fig.add_trace(go.Bar(x=region_rev["region"], y=region_rev["revenue"]),
              row=1, col=2)

# 우상단: 마진 히스토그램
fig.add_trace(go.Histogram(x=df["margin"], nbinsx=40), row=1, col=3)

# 좌하단: 상위 10개 제품
top10 = (df.groupby("product")["revenue"].sum()
           .nlargest(10).sort_values().reset_index())
fig.add_trace(go.Bar(x=top10["revenue"], y=top10["product"], orientation="h"),
              row=2, col=1)

# 하단 중앙: 코호트 리텐션 히트맵
fig.add_trace(go.Heatmap(z=retention.values, x=retention.columns,
                         y=retention.index.astype(str),
                         colorscale="YlGn"), row=2, col=2)

# 우하단: 산점도
fig.add_trace(go.Scatter(x=df["ad_spend"], y=df["revenue"], mode="markers",
                         marker=dict(size=4, opacity=0.3)),
              row=2, col=3)

fig.update_layout(height=900, width=1600, showlegend=False,
                  title_text="Executive Dashboard — Q1 2024", title_x=0.5)
fig.write_html("/tmp/dashboard.html", include_plotlyjs="cdn")
fig.show()
```

## 정적 그림 — Matplotlib 및 Seaborn

산출물이 정적 이미지인 경우에만 사용합니다. 위의 Plotly 레시피가 대부분의
요구를 충족합니다.

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# 전문적인 스타일 기본값 (스크립트 상단에서 한 번 적용)
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.labelsize": 12, "axes.titlesize": 14,
    "legend.fontsize": 10,
})
sns.set_theme(style="whitegrid")
```

### Matplotlib 빠른 패턴

```python
# 선 차트
df.set_index("date")["revenue"].plot(title="Daily Revenue", figsize=(12, 4))
plt.tight_layout(); plt.savefig("revenue.png", dpi=300, bbox_inches="tight")

# 막대 차트
(df.groupby("category")["revenue"].sum().sort_values()
   .plot(kind="barh", figsize=(8, 6)))
plt.title("Revenue by Category")
plt.tight_layout(); plt.savefig("by_category.png", dpi=300, bbox_inches="tight")

# 히스토그램 + KDE
fig, ax = plt.subplots(figsize=(10, 5))
df["revenue"].plot(kind="hist", bins=50, density=True, alpha=0.5, ax=ax)
df["revenue"].plot(kind="kde", ax=ax, color="red", linewidth=2)
plt.tight_layout(); plt.savefig("dist.png", dpi=300, bbox_inches="tight")

# 그룹별 박스 플롯
df.boxplot(column="revenue", by="segment", figsize=(10, 6))
plt.suptitle("")  # 자동 추가된 제목 제거
plt.tight_layout(); plt.savefig("box.png", dpi=300, bbox_inches="tight")
```

### 통계 그림을 위한 Seaborn

```python
# 상관관계 히트맵 (출판 스타일)
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax)
plt.title("Feature Correlation Matrix")
plt.tight_layout(); plt.savefig("corr.png", dpi=300, bbox_inches="tight")

# 페어플롯
sns.pairplot(df[["revenue", "cost", "margin", "segment"]],
             hue="segment", plot_kws={"alpha": 0.4})
plt.savefig("pairplot.png", dpi=300, bbox_inches="tight")

# 신뢰 구간이 있는 회귀
sns.regplot(data=df, x="ad_spend", y="revenue", scatter_kws={"alpha": 0.3})

# 패싯 스몰 멀티플
g = sns.FacetGrid(df, col="region", row="quarter", height=4)
g.map(sns.histplot, "revenue", bins=20)
g.savefig("facets.png", dpi=300, bbox_inches="tight")
```

### 정적 대시보드 (matplotlib 서브플롯)

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Executive Dashboard — Q1 2024", fontsize=16, y=1.02)
df.set_index("date")["revenue"].resample("W").sum().plot(ax=axes[0, 0])
df.groupby("region")["revenue"].sum().sort_values().plot(kind="barh", ax=axes[0, 1])
df["margin"].plot(kind="hist", bins=40, ax=axes[0, 2])
df.groupby("product")["revenue"].sum().nlargest(10).plot(kind="barh", ax=axes[1, 0])
sns.heatmap(retention.head(6), annot=True, fmt=".0%", ax=axes[1, 1], cmap="YlGn")
axes[1, 2].scatter(df["ad_spend"], df["revenue"], alpha=0.2, s=10)
plt.tight_layout()
plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
```

### Y축 포매터

```python
def format_yaxis_millions(ax):
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

def format_yaxis_pct(ax):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
```

## 시각화 모범 사례

- **Plotly를 기본으로 사용합니다.** 이해관계자는 호버와 줌으로 상호작용하며,
  정적 PNG는 비교적 무미건조하게 느껴집니다.
- **올바른 차트 선택:** 추세에는 선 차트, 비교에는 막대 차트, 상관관계에는 산점도,
  분포에는 히스토그램, 그룹 간 분포 비교에는 박스 플롯/바이올린 플롯, 행렬 데이터에는
  히트맵, 계층적 비율에는 트리맵을 사용합니다.
- **모든 것에 레이블을 붙입니다:** 제목, 축 레이블, 단위, 데이터 출처. Plotly의
  호버 레이블은 그림 내 주석의 필요성을 줄여주지만 축 레이블을 대체하지는 않습니다.
- **의도적으로 색상을 사용합니다:** 양에는 순차(blues / Viridis), 중간값을 중심으로
  한 편차에는 발산(RdBu), 카테고리에는 정성적(Tab10) 컬러스케일을 사용합니다.
  데이터가 양수와 음수에 걸쳐 있을 때 발산 스케일에 Plotly의 `color_continuous_midpoint=0`
  은 필수입니다.
- **차트정크 (chartjunk) 회피:** 3D 효과, 불필요한 그라데이션은 사용하지 않습니다.
  Plotly의 `plotly_white` 템플릿은 대부분의 기본 장식을 제거합니다.
- **접근성:** 색맹 친화적 팔레트(`px.colors.qualitative.Safe`, Plotly의 "Viridis",
  또는 seaborn "colorblind")를 사용합니다. 빨강/녹색만 사용하는 것은 피합니다.
- **공유용으로 HTML 저장:** `fig.write_html(..., include_plotlyjs="cdn")`은
  모든 브라우저에서 작동하는 50 KB 미만의 자체 완결형 파일을 생성합니다.
- **수십 개의 `fig.show()`를 자동으로 열지 마십시오** — 대신 파일로 저장한 후
  보고서에 경로를 나열합니다.
