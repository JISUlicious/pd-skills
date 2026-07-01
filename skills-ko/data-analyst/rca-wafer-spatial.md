# 공간적 결함 패턴 분석 (웨이퍼 맵)

(x, y) 위치가 있는 제조 결함 데이터. **공간 패턴은 종종 어떤 공통성
표보다 물리적 원인을 더 빠르게 드러냅니다** — 엣지 링은 핸들링 /
edge-bead 제거를 시사하고, 반복 스텝 패턴은 스테퍼를 시사하며, 챔버
중심 핫스팟은 증착 단계 비균일성을 시사합니다.

## 목차

- 적용 가능한 경우 (x, y 좌표 필요)
- 2D 커널 밀도 추정
- Ripley의 K 함수 (군집 vs 무작위)
- 패턴 분류 (엣지 / 중심 / 방사형 / 스테퍼 / 무작위 / 스트라이프)
- 해석 가이드 (패턴 → 유력한 물리 원인)
- 합성 웨이퍼 예시

이 파일을 읽는 경우: 웨이퍼 맵 결함 분석, 공간 검사 데이터, "왜 우리
불량은 모두 웨이퍼 한쪽에 모여 있는가?", 또는 (x, y) 좌표가 기록되는
반도체 제조 결함 조사. 이 방법론은 PCB 결함 맵, 스크린 코팅 결함, 그
외 모든 2D 좌표 결함 데이터에도 적용됩니다.

## 적용 가능한 경우

각 결함이 기판상의 (x, y) 좌표를 가져야 합니다. 결함 로그가 로트별 또는
웨이퍼별 집계만 있다면(SECOM 스타일), 이 파일의 방법은 적용되지
않습니다 — `rca-commonality.md`를 대신 사용하세요.

전형적 스키마:
```
lot_id | wafer_id | defect_id | x_mm | y_mm | defect_class | detected_at
```

## 2D 커널 밀도 추정

결함이 군집되는지 보는 가장 빠른 방법은 웨이퍼 다이어그램에 KDE를
오버레이하는 것입니다.

```python
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

pts = df.loc[df["defect_class"] == "particle", ["x_mm", "y_mm"]].values.T
kde = gaussian_kde(pts, bw_method="scott")

# 200x200 그리드에서 평가, 웨이퍼 원반 내부로 제한
r_wafer_mm = 150
gx, gy = np.mgrid[-r_wafer_mm:r_wafer_mm:200j, -r_wafer_mm:r_wafer_mm:200j]
inside = (gx**2 + gy**2) < r_wafer_mm**2
density = np.where(inside, kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape), np.nan)

fig = go.Figure(go.Heatmap(x=gx[:, 0], y=gy[0], z=density.T, colorscale="Viridis"))
fig.add_scatter(x=pts[0], y=pts[1], mode="markers",
                marker=dict(color="white", size=3, opacity=0.5))
fig.update_layout(title="결함 KDE — 150 mm 웨이퍼",
                  xaxis_title="x (mm)", yaxis_title="y (mm)",
                  yaxis_scaleanchor="x", width=600, height=600)
```

편향된 패턴에는 로그 밀도 스케일링을 사용하고 웨이퍼 원반 바깥은
마스킹(NaN 설정)하세요 — 그렇지 않으면 KDE가 존재하지 않는 영역으로
번집니다.

## Ripley의 K 함수 — 군집 vs 무작위 검정

KDE는 결함이 *어디에* 집중되는지 보여줍니다. Ripley의 K는 전체 패턴이
완전 공간 무작위성(CSR)과 다른지 검정합니다. 관측된 K가 CSR 봉투를
초과하면 그 반경에서 패턴은 군집되어 있고, 아래이면 규칙적입니다.

```python
# pointpats 또는 astropy가 제대로 된 구현을 제공; 스케치:
from pointpats import ripley
k_obs, radii = ripley.k_estimate(pts.T, support=np.linspace(0, 50, 50))
env_lo, env_hi = ripley.k_envelope(pts.T, n_permutations=99)
```

r=5-15 mm에서 군집을 보이는 웨이퍼는 국소화된 공정 사건(픽스처 스크래치,
노즐 막힘)을 시사합니다. r > 50 mm에서의 군집은 전역 기울기(온도,
필름 두께)를 시사합니다.

## 패턴 분류

웨이퍼 패턴은 작은 어휘로 떨어집니다. 형태를 인식하면 검정을 실행하기
전에 물리적 원인 후보를 얻습니다.

| 패턴 | 유력한 물리 원인 |
|---|---|
| **엣지 링** — r ≈ r_wafer에 밀집 | 핸들링 손상 / edge-bead 제거 / 린스 디스펜스 / 엣지 배제 오설정 |
| **중심 밀집** — (0, 0)에 핫스팟 | 챔버 중심 비균일성; 증착 또는 식각 속도가 중심에서 최고 |
| **방사형 기울기** — 밀도가 r에 따라 부드럽게 변화 | 회전 관련 공정(스핀 코트, CMP); 플래튼 또는 척 동심성 점검 |
| **반복 스캔 패턴** — 사각 격자상 주기적 핫스팟 | 스테퍼 / 리소그래피 노광 결함; 샷 단위 결함 |
| **무작위 / 균일** — CSR 봉투가 K̂(r) 포함 | 수율 플로어 / 무작위 파티클 배경; 국소화된 원인 없음 |
| **군집 / 스트라이프** — 하나의 밀집 군집 또는 선형 스트릭 | 픽스처 손상, 스크래치, 카세트 이송 접촉 |

**진단 시퀀스:**
1. KDE 플롯 후 패턴 눈으로 확인.
2. 위 분류에 매칭.
3. Ripley의 K 실행하여 군집을 통계적으로 확인.
4. 공정 이력과 패턴을 교차 참조: 흐름의 어느 단계가 패턴과 일치하는
   기하학을 가지는가?

## 합성 웨이퍼 예시

현재 조사에 (x, y) 데이터가 없을 때, 방법 보정을 위해 합성 웨이퍼를
생성하세요:

```python
# 중심 핫 패턴 180개 + 균일 무작위 배경 20개, 총 200개 결함
r_wafer = 150
rng = np.random.default_rng(0)

center = rng.normal(loc=(0, 0), scale=25, size=(180, 2))
uniform_r = r_wafer * np.sqrt(rng.random(20))
uniform_th = 2 * np.pi * rng.random(20)
uniform = np.column_stack([uniform_r * np.cos(uniform_th),
                           uniform_r * np.sin(uniform_th)])
pts = np.vstack([center, uniform])
pts = pts[(pts**2).sum(axis=1) < r_wafer**2].T   # 웨이퍼 원반으로 클립

# ... 위의 KDE + Ripley 파이프라인 실행
```

KDE는 중심 밀집 시그니처를 복원해야 하고, Ripley의 K는 r < 30 mm에서
CSR 봉투를 초과해야 합니다. 새로운 공간 데이터 소스를 통합할 때마다
이를 스모크 테스트로 사용하세요.
