# RCA Causal Inference and Designed Experiments

The distinction between correlation and causation for RCA. Contains both
the observational / retrospective toolkit (DAG, DiD, propensity matching,
dowhy) and the designed-experiment toolkit (ANOVA / Tukey / effect size).

## Contents

- Reject: reverse causation / common cause / selection / temporal mismatch
- Build a minimal DAG
- Diff-in-Differences (DiD)
- Propensity score matching
- `dowhy` — principled causal inference with explicit assumptions
- DOE / ANOVA — designed experiments
- Effect size: statistical vs. practical significance

Read this file when working on: "did X *cause* Y to shift", "estimating
the counterfactual", "should we roll back the recipe change", or "we
ran a DOE, what mattered". Pairs with `root-cause-analysis.md` § 1
(CPD provides the pre/post window) and `rca-commonality.md` (surfaces
the candidate cause to feed into DiD / dowhy).

## Reject before believing

The headline rule: **prediction is not causation.** Before claiming X
caused Y, rule out:

1. **Reverse causation** — did Y change first and X follow?
2. **Common cause** — is there an upstream Z that drives both X and Y?
3. **Selection bias** — is X over-represented in the population at risk?
4. **Temporal mismatch** — did X's change actually precede Y's shift?

## Build a minimal DAG (Directed Acyclic Graph)

For every candidate cause X surfaced by SHAP/commonality, draw the
minimal DAG by hand:
```
   recipe_change ──→ chamber_pressure ──→ defect_rate
        │                  │
        ▼                  ▼
   tool_uptime ←──────→ wafer_temp
```
Then identify backdoor paths: any path from X to Y through an unblocked
common ancestor is a confounder. The fix is to either (a) condition on
the confounder, (b) find an instrument, or (c) acknowledge the limitation
in the report.

## Diff-in-Differences (DiD) — workhorse for pre/post-change excursions

When a known process change happened at time T and you want to estimate
its causal effect on Y, compare:
- Δ(post − pre) for the affected group (treated)
- Δ(post − pre) for an unaffected group (control)

```python
def diff_in_diff(df, time_col, group_col, outcome, change_time,
                 treated_value, control_value):
    """Closed-form DiD via OLS interaction."""
    df = df.copy()
    df["post"]    = (df[time_col] >= change_time).astype(int)
    df["treated"] = (df[group_col] == treated_value).astype(int)
    model = smf.ols(f"{outcome} ~ post + treated + post:treated", data=df).fit()
    did_estimate = model.params["post:treated"]
    print(model.summary().tables[1])
    print(f"\nDiD estimate: {did_estimate:+.4f}  (95% CI: "
          f"[{model.conf_int().loc['post:treated',0]:.4f}, "
          f"{model.conf_int().loc['post:treated',1]:.4f}])")
    # Sanity: parallel-trends check (visualize pre-period for both groups)
    return model
```

DiD assumes **parallel trends** in the pre-period — the treated and
control groups would have moved together absent the change. Always plot
the pre-period for both groups; if their trends diverge before T, DiD
is invalid.

## Propensity Score Matching — for non-randomized observational comparisons

When you can't use DiD (no clean pre/post), match treated units to
similar control units on observed covariates:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_match(df, treatment_col, outcome_col, covariates, caliper=0.1):
    X = df[covariates].fillna(df[covariates].median())
    t = df[treatment_col].astype(int)
    # Estimate propensity score
    ps_model = LogisticRegression(max_iter=1000).fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]
    df = df.assign(ps=ps)
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    # 1:1 nearest neighbor on PS (with caliper to drop bad matches)
    nn = NearestNeighbors(n_neighbors=1).fit(control[["ps"]].values)
    dist, idx = nn.kneighbors(treated[["ps"]].values)
    keep = dist.ravel() < caliper
    matched = pd.concat([
        treated.iloc[keep],
        control.iloc[idx.ravel()[keep]],
    ])
    ate = (matched.loc[matched[treatment_col]==1, outcome_col].mean()
         - matched.loc[matched[treatment_col]==0, outcome_col].mean())
    print(f"Matched n: {keep.sum()} pairs (of {len(treated)} treated units)")
    print(f"ATE estimate: {ate:+.4f}")
    return matched, ate
```

## `dowhy` — for principled causal inference with explicit assumptions

When the DAG is non-trivial (multiple confounders, instruments,
mediators), use `dowhy` — it forces you to write the assumptions down,
estimates the effect under each, and runs refutation tests:

```python
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment="recipe_change",                  # the suspected cause
    outcome="defect_rate",                      # the metric
    common_causes=["tool_uptime", "wafer_lot_age", "operator_shift"],
    # Optional: instruments, effect_modifiers, ...
)
model.view_model()                              # renders the DAG

# Identify the causal estimand (closed-form expression)
identified = model.identify_effect(proceed_when_unidentifiable=False)
print(identified)

# Estimate via backdoor adjustment + propensity score
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.propensity_score_matching",
    target_units="ate",
)
print(f"Causal effect estimate: {estimate.value:+.4f}")

# Refutation: would the estimate survive these challenges?
refute_random  = model.refute_estimate(identified, estimate, "random_common_cause")
refute_placebo = model.refute_estimate(identified, estimate, "placebo_treatment_refuter")
refute_subset  = model.refute_estimate(identified, estimate, "data_subset_refuter")
print("Refutation summary:")
print(f"  Random common cause:  Δ = {refute_random.new_effect - estimate.value:+.4f}")
print(f"  Placebo treatment:    Δ = {refute_placebo.new_effect - estimate.value:+.4f}")
print(f"  Data subset:          Δ = {refute_subset.new_effect - estimate.value:+.4f}")
```

A robust causal estimate survives all three refutations with little
change. If the placebo refuter shows a large effect (it shouldn't, by
construction), the original estimate is suspect.

## DOE / ANOVA — for designed experiments

When the team ran a real experiment (2k full-factorial, fractional
factorial, Plackett-Burman), use ANOVA with `statsmodels`:

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Two-way ANOVA with interaction
model = smf.ols("yield ~ C(tool) * C(recipe)", data=df).fit()
print(sm.stats.anova_lm(model, typ=2))      # Type-II SS

# Effect size: η² (eta-squared) and ω² (omega-squared, less biased)
ss = sm.stats.anova_lm(model, typ=2)
ss["eta_sq"]   = ss["sum_sq"] / ss["sum_sq"].sum()
ss["omega_sq"] = ((ss["sum_sq"] - ss["df"] * model.mse_resid) /
                  (ss["sum_sq"].sum() + model.mse_resid))
print(ss.round(4))

# Tukey HSD post-hoc: which level pairs differ?
tukey = pairwise_tukeyhsd(df["yield"], df["tool"], alpha=0.05)
print(tukey.summary())
```

**Statistical significance ≠ practical significance.** η² < 0.01 means
"this factor explains < 1% of the variance" — even if p < 0.0001, the
effect may be irrelevant. Always report effect size alongside p-values.
