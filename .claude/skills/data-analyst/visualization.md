# Pandas Data Visualization — Expert Skill

You are an expert data analyst. Apply the following visualization patterns
using pandas >= 2.3 built-in plotting and complementary libraries.

## Pandas Built-in Plotting

Pandas wraps Matplotlib and provides a `.plot` accessor on DataFrames and Series.
Always import matplotlib for fine-tuned control.

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
```

### Line Charts (Time Series)

```python
# Simple line
df.set_index("date")["revenue"].plot(title="Daily Revenue", figsize=(12, 4))
plt.tight_layout(); plt.show()

# Multiple lines
df.set_index("date")[["revenue", "cost", "margin"]].plot(
    title="Financial Trends", figsize=(12, 5), linewidth=1.5
)
plt.ylabel("USD")
plt.legend(loc="upper left")
plt.tight_layout(); plt.show()
```

### Bar Charts

```python
# Grouped bar (category × metric)
pivot = df.groupby(["region", "quarter"])["revenue"].sum().unstack("quarter")
pivot.plot(kind="bar", figsize=(12, 5), width=0.8)
plt.title("Revenue by Region and Quarter")
plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.show()

# Horizontal bar (good for many categories)
df.groupby("product")["revenue"].sum().nlargest(15).plot(kind="barh", figsize=(8, 8))
plt.title("Top 15 Products by Revenue")
plt.xlabel("Revenue ($)")
plt.tight_layout(); plt.show()

# Stacked bar
pivot.plot(kind="bar", stacked=True, figsize=(12, 5))
```

### Histograms & KDE

```python
# Histogram
df["revenue"].plot(kind="hist", bins=50, edgecolor="white", figsize=(10, 5))
plt.title("Revenue Distribution")
plt.xlabel("Revenue")
plt.tight_layout(); plt.show()

# KDE (smooth density estimate)
df["revenue"].plot(kind="kde", figsize=(10, 5))

# Both together
fig, ax = plt.subplots(figsize=(10, 5))
df["revenue"].plot(kind="hist", bins=50, density=True, alpha=0.5, ax=ax)
df["revenue"].plot(kind="kde", ax=ax, color="red", linewidth=2)
plt.title("Revenue Distribution")
plt.tight_layout(); plt.show()

# Multiple columns
df[["revenue", "cost"]].plot(kind="hist", bins=40, alpha=0.5, figsize=(10, 5))
```

### Box Plots & Violin Plots

```python
# Box plot by group
df.boxplot(column="revenue", by="segment", figsize=(10, 6))
plt.suptitle("")  # remove auto title
plt.title("Revenue by Customer Segment")
plt.tight_layout(); plt.show()

# Pandas box plot (all numeric)
df[["revenue", "cost", "margin"]].plot(kind="box", figsize=(8, 6))
```

### Scatter Plots

```python
# Basic scatter
df.plot(kind="scatter", x="ad_spend", y="revenue", alpha=0.3, figsize=(8, 6))
plt.title("Ad Spend vs Revenue")
plt.tight_layout(); plt.show()

# With color encoding
df.plot(
    kind="scatter", x="ad_spend", y="revenue",
    c="margin", colormap="RdYlGn", alpha=0.5,
    figsize=(10, 6)
)
plt.colorbar(label="Margin")
plt.tight_layout(); plt.show()
```

### Area Charts

```python
df.set_index("date")[["product_a", "product_b", "product_c"]].plot(
    kind="area", stacked=True, alpha=0.7, figsize=(12, 5)
)
plt.title("Revenue by Product (Stacked)")
plt.tight_layout(); plt.show()
```

## Seaborn — Statistical Visualization

```python
import seaborn as sns
sns.set_theme(style="whitegrid")

# Correlation heatmap
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, square=True, linewidths=0.5, ax=ax
)
plt.title("Feature Correlation Matrix")
plt.tight_layout(); plt.show()

# Pair plot (relationships between all numeric columns)
sns.pairplot(df[["revenue", "cost", "margin", "segment"]], hue="segment", plot_kws={"alpha": 0.4})
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

# Distribution by group
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=df, x="revenue", hue="segment", kde=True, ax=axes[0])
sns.boxplot(data=df, x="segment", y="revenue", hue="segment", legend=False, ax=axes[1])
plt.tight_layout(); plt.show()

# Violin plot (shows full distribution)
sns.violinplot(data=df, x="region", y="revenue", hue="region", legend=False, inner="quartile")

# Bar plot with confidence intervals
sns.barplot(data=df, x="segment", y="revenue", estimator="mean", errorbar="ci")

# Regression plot
sns.regplot(data=df, x="ad_spend", y="revenue", scatter_kws={"alpha": 0.3})

# Facet grid (small multiples)
g = sns.FacetGrid(df, col="region", row="quarter", height=4)
g.map(sns.histplot, "revenue", bins=20)
```

## Plotly — Interactive Charts

```python
import plotly.express as px
import plotly.graph_objects as go

# Interactive time series
fig = px.line(df, x="date", y="revenue", color="region", title="Revenue Trends")
fig.show()

# Interactive bar
fig = px.bar(
    df.groupby("category")["revenue"].sum().reset_index(),
    x="category", y="revenue", title="Revenue by Category",
    text_auto=".2s"  # auto labels
)
fig.show()

# Scatter with hover
fig = px.scatter(
    df, x="ad_spend", y="revenue", color="segment", size="margin",
    hover_data=["product", "region"], title="Spend vs Revenue"
)
fig.show()

# Choropleth map
fig = px.choropleth(
    df.groupby("country")["revenue"].sum().reset_index(),
    locations="country", locationmode="country names",
    color="revenue", title="Revenue by Country"
)
fig.show()

# Treemap
fig = px.treemap(df, path=["region", "category", "product"], values="revenue")
fig.show()

# Funnel chart
fig = px.funnel(funnel_df, x="count", y="stage", title="Conversion Funnel")
fig.show()
```

## Report-Quality Figure Templates

```python
# Professional style defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
})

def format_yaxis_millions(ax):
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

def format_yaxis_pct(ax):
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

# Save high-resolution figure
plt.savefig("chart.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("chart.svg", bbox_inches="tight")   # vector for publications
```

## Dashboard Layout

```python
# Multi-panel dashboard with matplotlib
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Executive Dashboard — Q1 2024", fontsize=16, y=1.02)

# Top-left: revenue trend
df.set_index("date")["revenue"].resample("W").sum().plot(ax=axes[0, 0])
axes[0, 0].set_title("Weekly Revenue")

# Top-middle: revenue by region
df.groupby("region")["revenue"].sum().sort_values().plot(kind="barh", ax=axes[0, 1])
axes[0, 1].set_title("Revenue by Region")

# Top-right: margin distribution
df["margin"].plot(kind="hist", bins=40, ax=axes[0, 2])
axes[0, 2].set_title("Margin Distribution")

# Bottom-left: top products
df.groupby("product")["revenue"].sum().nlargest(10).plot(kind="barh", ax=axes[1, 0])
axes[1, 0].set_title("Top 10 Products")

# Bottom-middle: monthly cohort retention
sns.heatmap(retention.head(6), annot=True, fmt=".0%", ax=axes[1, 1], cmap="YlGn")
axes[1, 1].set_title("Cohort Retention")

# Bottom-right: scatter
axes[1, 2].scatter(df["ad_spend"], df["revenue"], alpha=0.2, s=10)
axes[1, 2].set_title("Ad Spend vs Revenue")

plt.tight_layout()
plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Visualization Best Practices

- **Choose the right chart:** line for trends, bar for comparison, scatter for correlation, histogram for distribution
- **Label everything:** title, axis labels, units, data source
- **Color intentionally:** use sequential (blues) for quantity, diverging (RdBu) for deviation, qualitative for categories
- **Avoid chartjunk:** remove top/right spines, use light gridlines, no 3D effects
- **Accessibility:** use colorblind-safe palettes (e.g., seaborn "colorblind" or "tab10")
- **Interactivity:** use Plotly for exploratory analysis, matplotlib for publication figures
- **Consistent scale:** don't truncate y-axis to exaggerate differences unless intentional
