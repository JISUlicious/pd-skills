# Root-Cause Analysis — Expert Skill

You are an expert data analyst supporting **product-defect investigation,
yield excursions, process drift, and incident reviews**. The methodology
here is causal and time-localized — distinct from the associational
ranking in `feature-importance.md`.

The headline rule: **SHAP-importance is not causal evidence.** A SHAP
value of 0.5 for `chamber_pressure` says "this feature reduces prediction
error" — *not* "fixing chamber pressure will fix yield." Root-cause work
needs to identify the *change* that preceded the outcome shift, then rule
out alternative explanations.

## Contents

- When to use vs. `feature-importance.md`
- Setup — optional dependencies + time-cutoff-split
- § 1: Change-point detection (PELT, CUSUM)
- § 2: Statistical Process Control (Shewhart, EWMA, Cp/Cpk)
- § 6: RCA decision cheatsheet
- § 8: RCA anti-patterns
- § 9: Performance cheatsheet
- § 10: End-to-end workflow

**Extended references (level-1 from `SKILL.md`):**
- `rca-commonality.md`      — Fisher / BH-FDR / cluster collapse (was § 3)
- `rca-causal-analysis.md`  — DAG / DiD / dowhy / DOE (was § 4 + § 5)
- `rca-reporting.md`        — Tier 2 template + form guide + HTML (was § 7 + § 7a + § 7b)

## When to Use This File vs. `feature-importance.md`

| Question | File |
|---|---|
| *"Which X predicts Y best?"* | `feature-importance.md` |
| *"What caused Y to shift at time T?"* | here |
| *"Which tool / lot / operator is associated with bad parts?"* | both — start here for time-localized excursions, use both together |
| *"Did this process change improve yield?"* | here (DiD, § 6) |
| *"We ran a DOE — what mattered?"* | here (ANOVA, § 7) |

If the question is time-localized ("since April 2"), or causal ("did
*changing* X produce the shift in Y?"), start here. If it's purely
associational ("rank features by importance"), `feature-importance.md`
is sufficient.

## Setup — Optional Dependencies

```python
try:
    import ruptures as rpt           # change-point detection
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_RCA = True
except Exception as e:
    HAS_RCA = False
    print(f"RCA stack unavailable ({type(e).__name__}: {e}).")
    print("Install: uv pip install ruptures statsmodels mlxtend")
    print("Optional (causal): uv pip install dowhy econml")

try:
    import dowhy                      # causal inference (heavyweight, optional)
    HAS_CAUSAL = True
except Exception:
    HAS_CAUSAL = False
```

`scipy` and `sklearn` (already required by `feature-importance.md`) cover
Fisher's exact, Mann-Whitney, propensity matching, and Tukey HSD.

### Time-indexed data — do not use random `train_test_split`

Most RCA data is time-indexed (per-lot defect rate, per-sensor reading
sequence). A random split causes **temporal leakage**: training rows
include moments after test rows, so the model implicitly "sees the
future." This silently inflates holdout R² / AUC.

```python
# WRONG for time-indexed data:
# X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# RIGHT — explicit time cutoff:
cutoff = df["timestamp"].quantile(0.8)
train = df[df["timestamp"] < cutoff]
test  = df[df["timestamp"] >= cutoff]

# OR sklearn's TimeSeriesSplit for CV:
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X.sort_index()):
    ...
```

Random split is acceptable only when the rows are exchangeable — typically
when the analysis question is cross-sectional (compare populations at one
point in time), not temporal.

## 1. Change-Point Detection (CPD)

The signature question of an excursion is **when did it start?** Rolling
means smear the answer; you need a formal break detector.

### Default: PELT (offline, fast)

```python
import numpy as np
import pandas as pd
import ruptures as rpt

def detect_changepoints(series, model="rbf", min_size=30, penalty=None):
    """Return indices where the metric distribution changes.

    series   : 1-D numpy array or pd.Series (chronological order required)
    model    : "rbf" (default, robust), "l2" (gaussian mean shift),
               "l1" (median shift, robust to outliers)
    min_size : minimum samples between change points
    penalty  : if None, uses log(n) * variance (BIC-style); raise to detect
               fewer (only large) change points; lower to detect more
    """
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    if penalty is None:
        penalty = np.log(n) * float(np.var(s))
    algo = rpt.Pelt(model=model, min_size=min_size).fit(s)
    bkps = algo.predict(pen=penalty)
    return bkps[:-1]                  # drop the trailing endpoint

# Visual sanity check is mandatory:
import matplotlib.pyplot as plt
def plot_cps(series, bkps, ax=None):
    ax = ax or plt.gca()
    ax.plot(series, lw=1)
    for b in bkps:
        ax.axvline(b, color="red", ls="--", lw=1)
    ax.set_title(f"{len(bkps)} change points detected")
```

### CUSUM for small persistent drifts

PELT often misses subtle drifts (a 0.5σ shift sustained for 200 samples).
CUSUM accumulates deviation and raises an alarm when accumulation crosses
a threshold:

```python
def cusum(series, target=None, sigma=None, k=0.5, h=5, ref_window=50):
    """Two-sided CUSUM. Returns indices where +/- CUSUM exceeds h*sigma.

    target : reference mean. If None, estimated from the first `ref_window` samples
             (NOT the full series — that would include any drift you're trying
             to detect).
    sigma  : reference standard deviation. If None, estimated from the first
             `ref_window` samples for the same reason.
    k      : reference value in standard deviations (0.5 = sensitive to 1σ shifts)
    h      : decision threshold in standard deviations (5 is canonical)

    Estimating target/sigma from the full series is a common bug: a drifted
    series has inflated σ, the slack k·σ becomes too generous, and the test
    under-detects the drift it was supposed to find.
    """
    s = np.asarray(series, dtype=np.float64)
    ref = s[:min(ref_window, len(s) // 5)]               # clean-period window
    target = target if target is not None else ref.mean()
    sigma  = sigma  if sigma  is not None else ref.std()
    pos = np.zeros(len(s)); neg = np.zeros(len(s))
    for i in range(1, len(s)):
        pos[i] = max(0, pos[i-1] + (s[i] - target - k * sigma))
        neg[i] = min(0, neg[i-1] + (s[i] - target + k * sigma))
    alarms = np.where((pos > h * sigma) | (neg < -h * sigma))[0]
    return alarms
```

### Decision table

| Data shape | Method |
|---|---|
| One known break, abrupt | PELT, model="l2" |
| Multiple unknown breaks | PELT, model="rbf", BIC penalty |
| Slow drift, no clear step | CUSUM (k=0.5) or EWMA chart (§ 2) |
| Heavy outliers | PELT, model="l1" (median-based) |
| Streaming / online detection | Bayesian Online CPD (`bayesian_changepoint_detection` lib) |

### Sanity check before believing a change point

A detected break is **noise** unless:
1. It has ≥30 samples on either side (else statistical power is too low)
2. The pre/post means differ by ≥1σ of the local noise
3. A Mann-Whitney pre vs. post returns p < 0.01
4. It is visually obvious in the plot

Always print all four checks. Reporting a "change point" without them is
the most common over-detection failure.

## 2. Statistical Process Control (SPC)

For ongoing monitoring (vs. forensic CPD on historical data). SPC is the
manufacturing-engineer vocabulary — speak it natively when reporting.

### Shewhart X̄/R Chart with Western Electric Runs Rules

```python
def shewhart_chart(values, subgroup_size=5):
    """Compute X̄ and R chart limits + Western Electric rule violations.
    Returns dict with center/UCL/LCL and per-subgroup violation flags.
    """
    s = np.asarray(values, dtype=np.float64)
    n_sub = len(s) // subgroup_size
    s = s[:n_sub * subgroup_size].reshape(n_sub, subgroup_size)
    xbar = s.mean(axis=1)
    rng  = s.max(axis=1) - s.min(axis=1)
    # Constants for n=5 subgroups (look up A2, D3, D4 for other sizes)
    A2, D3, D4 = 0.577, 0.0, 2.114
    cl_x  = xbar.mean()
    rbar  = rng.mean()
    ucl_x = cl_x + A2 * rbar
    lcl_x = cl_x - A2 * rbar
    ucl_r = D4 * rbar
    lcl_r = D3 * rbar
    sigma = (ucl_x - cl_x) / 3                  # implied σ from chart limits

    # Western Electric runs rules on X̄
    flags = pd.DataFrame({"xbar": xbar})
    flags["rule_1"] = np.abs(xbar - cl_x) > 3 * sigma                       # 1 point > 3σ
    flags["rule_2"] = (np.abs(xbar - cl_x) > 2 * sigma).rolling(3).sum() >= 2  # 2 of 3 > 2σ
    flags["rule_3"] = (np.abs(xbar - cl_x) > sigma).rolling(5).sum() >= 4      # 4 of 5 > 1σ
    flags["rule_4"] = pd.Series((xbar > cl_x).astype(int)).rolling(8).sum() \
                        .isin([0, 8])                                          # 8 in a row same side
    flags["any_violation"] = flags[["rule_1","rule_2","rule_3","rule_4"]].any(axis=1)
    return {"cl_x": cl_x, "ucl_x": ucl_x, "lcl_x": lcl_x,
            "cl_r": rbar, "ucl_r": ucl_r, "lcl_r": lcl_r,
            "sigma": sigma, "flags": flags}
```

### EWMA chart for slow drifts

```python
def ewma_chart(values, lambda_=0.2, L=3):
    """EWMA control chart — sensitive to small persistent drifts.
    lambda_ : weighting factor (0.05–0.3); smaller = smoother, slower
    L       : control limit width in sigma (3 default)
    """
    s = np.asarray(values, dtype=np.float64)
    target, sigma = s.mean(), s.std()
    z = np.zeros(len(s)); z[0] = target
    for i in range(1, len(s)):
        z[i] = lambda_ * s[i] + (1 - lambda_) * z[i-1]
    ucl = target + L * sigma * np.sqrt(lambda_ / (2 - lambda_))
    lcl = target - L * sigma * np.sqrt(lambda_ / (2 - lambda_))
    violations = np.where((z > ucl) | (z < lcl))[0]
    return {"ewma": z, "ucl": ucl, "lcl": lcl, "violations": violations}
```

### Process Capability Indices

```python
# Hartley's d2 constants for the R-bar method (subgroup sizes 2..10)
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
       7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

def capability(values, lsl, usl, subgroup_size=None, k=6):
    """Cp, Cpk (within-subgroup σ) and Pp, Ppk (overall σ) vs spec limits.

    subgroup_size : if given, estimate σ_within from subgroup ranges via R-bar/d2
                    (the rational-subgroup method). Otherwise σ_within = σ_overall,
                    in which case Cp == Pp and the within/long-term comparison
                    is uninformative.
    """
    s = np.asarray(values, dtype=np.float64)
    sigma_overall = s.std(ddof=1)
    mu = s.mean()

    if subgroup_size is not None and subgroup_size in _D2:
        n_sub = len(s) // subgroup_size
        sg = s[:n_sub * subgroup_size].reshape(n_sub, subgroup_size)
        rbar = (sg.max(axis=1) - sg.min(axis=1)).mean()
        sigma_within = rbar / _D2[subgroup_size]
    else:
        sigma_within = sigma_overall                  # falls back; Cp == Pp

    cp  = (usl - lsl) / (k * sigma_within)
    cpk = min((usl - mu) / (3 * sigma_within), (mu - lsl) / (3 * sigma_within))
    pp  = (usl - lsl) / (k * sigma_overall)
    ppk = min((usl - mu) / (3 * sigma_overall), (mu - lsl) / (3 * sigma_overall))

    def interpret(v):
        if v < 1.0:  return "incapable (defects expected)"
        if v < 1.33: return "marginal"
        if v < 1.67: return "capable"
        return "highly capable"
    return {"Cp": cp, "Cpk": cpk, "Pp": pp, "Ppk": ppk,
            "sigma_within": sigma_within, "sigma_overall": sigma_overall,
            "Cpk_interp": interpret(cpk), "Ppk_interp": interpret(ppk)}
```

**Cp vs. Cpk:** Cp ignores centering; Cpk penalizes off-center processes.
**Cp vs. Pp:** Cp uses within-subgroup variation (short-term); Pp uses
overall variation (long-term, includes drift). When Pp << Cp, the process
is drifting between subgroups — investigate.

### When SPC vs. CPD?

| Use case | Method |
|---|---|
| Online monitoring of a stable process | SPC (Shewhart or EWMA) |
| Retrospective forensic on a known excursion | CPD (PELT) |
| Estimating defect-rate at current state | Cp/Cpk |
| Confirming a fix took effect | Both: CPD on the fix timestamp + post-fix SPC |

**Anti-pattern:** SPC on autocorrelated data raises false alarms. If
`series.autocorr() > 0.5`, fit AR(1) residuals first and chart those.

## 3. Commonality Analysis

See `rca-commonality.md` for cluster collapse (with the RCA-specific
shift-score override), per-categorical-factor Fisher exact + BH-FDR,
frequent-itemset mining for multi-factor combinations, and the
multiplicity-scope warning.

## 4-5. Causal Inference and DOE

See `rca-causal-analysis.md` for the DAG + backdoor-path framework,
Diff-in-Differences, propensity score matching, `dowhy` with refutation,
and DOE / ANOVA / Tukey HSD with effect-size interpretation.

The headline rule is unchanged and lives with the toolkit: **prediction
is not causation**. Rule out reverse causation, common cause, selection
bias, and temporal mismatch before claiming X caused Y.

## 6. RCA Decision Cheatsheet

```
START: What kind of question?
   │
   ├── "Y has been bad for a while" (steady state)
   │       └→ SPC + Cp/Cpk + commonality (§§ 2, 3)
   │
   ├── "Y shifted at time T" (excursion)
   │       └→ CPD to confirm timing (§ 1)
   │          └→ Commonality on pre-vs-post (§ 3)
   │             └→ DiD if a known change at T (§ 4)
   │                └→ Causal refutation (§ 4: dowhy)
   │
   ├── "We ran a DOE, what mattered?"
   │       └→ ANOVA + Tukey HSD + effect size (§ 5)
   │
   └── "We made a fix — did it work?"
           └→ DiD pre/post + post-change SPC monitoring (§§ 2, 4)

At every step, ask: "Could this be confounded by Z?" If yes, condition on
Z (regression / matching) or acknowledge the limitation.
```

## 7. Reporting

See `rca-reporting.md` for the Tier 2 template (§ 7), form guide and
writing discipline (§ 7a), and self-contained HTML output spec with the
verbatim CSS block (§ 7b).

## 8. RCA Anti-Patterns

| Anti-pattern | Symptom | Fix |
|---|---|---|
| Treating SHAP-rank as causal rank | "The most influential feature is X, so fix X" | Add a DiD or refutation step before recommending action |
| Skipping CPD on time-indexed data | Reporting an excursion without a confirmed start time | Always run PELT and report the change point with sanity checks |
| Comparing tools/operators without case-mix control | Tool A "looks worst" but only runs the hard lots | Stratify by lot type, or use propensity matching |
| SPC on autocorrelated data | False alarms every other subgroup | Check `series.autocorr()`; if > 0.5, chart AR(1) residuals |
| Cherry-picking the "most affected" entity | Headline cause is the outlier, not the systemic driver | Sweep all entities; require Bonferroni-significant lift |
| Surface-cause vs. root-cause confusion | "Pressure was high" — but *why* was pressure high? | Apply 5-why depth; the root is upstream of the proximate change |
| Multiple testing without correction | "Found 47 significant factors" with α=0.05 across 940 sensors | Bonferroni or BH-FDR; for SECOM-scale, expect ~5% false positives |
| Reporting p-value without effect size | "p < 0.0001" on a 0.1pp shift in a metric with 5pp seasonality | Always pair p-values with η² / Cohen's d / lift |

## 9. Performance Cheatsheet

| Issue | Fix |
|---|---|
| PELT slow on > 100k samples | Use `model="l2"` (fastest); set `min_size` larger to reduce candidate breaks |
| `ruptures` over-segments | Raise `penalty` (default BIC × 2); add visual sanity check |
| `dowhy` slow for large n | Subsample to 10k for refutation; full data for point estimate |
| Commonality sweep slow on many factors | Vectorize with `pd.crosstab`; parallelize per-factor loop with `joblib` |
| ANOVA crashes on unbalanced design | Use Type-III SS via `typ=3` in `anova_lm` |
| Propensity matching drops too many units | Loosen `caliper` to 0.2σ of the propensity distribution |

## 10. End-to-End RCA Workflow

```python
# 1. Confirm a change happened: CPD on the metric
bkps = detect_changepoints(metric_series, model="rbf", min_size=30)
# 2. Validate the change point: pre vs post Mann-Whitney + plot
mw_p = stats.mannwhitneyu(pre, post).pvalue
assert mw_p < 0.01, "Change point not significant — investigate noise"

# 3. Commonality: which factors are over-represented post-change?
post_df = df.loc[df["timestamp"] >= change_time]
for factor in categorical_factors:
    sig = commonality(post_df, factor, "is_defective")
    # ... collect candidates

# 4. Causal: DiD on the strongest candidate
model = diff_in_diff(df, "timestamp", "treatment_group", "defect_rate",
                     change_time, treated_value=1, control_value=0)

# 5. Refute: dowhy with the candidate's DAG
cm = CausalModel(data=df, treatment="candidate", outcome="defect_rate",
                 common_causes=identified_confounders)
estimate = cm.estimate_effect(cm.identify_effect(),
                              method_name="backdoor.propensity_score_matching")
refute = cm.refute_estimate(cm.identify_effect(), estimate, "random_common_cause")

# 6. Report: structured narrative with all five sections of the template
```

If any step in this pipeline fails (CPD finds no change, commonality
returns nothing significant, DiD shows no effect, refutation knocks out
the estimate) — **report that the data does not support a causal claim
and recommend further investigation**. False positives in RCA are very
costly: they trigger wrong fixes that mask the real issue.
