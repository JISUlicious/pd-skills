# pd-skills — Expert data-analyst skill for AI agents

A pandas >= 2.3 expert data-analyst skill for Claude Code (and any agent
that consumes the [Anthropic Agent Skills format](https://docs.anthropic.com/en/docs/claude-code/skills)).
Available in **English** and **Korean**.

## Install

Via the [vercel-labs/skills CLI](https://github.com/vercel-labs/skills):

```bash
# Default — installs both variants
npx skills add JISUlicious/pd-skills

# Or pick a specific variant
npx skills add JISUlicious/pd-skills -s data-analyst       # English only
npx skills add JISUlicious/pd-skills -s data-analyst-ko    # Korean only
```

For Claude Code users without the CLI:

```bash
git clone https://github.com/JISUlicious/pd-skills.git
cp -r pd-skills/.claude/skills/data-analyst    ~/.claude/skills/   # or project-local
cp -r pd-skills/.claude/skills/data-analyst-ko ~/.claude/skills/
```

## What's inside

A 14-file methodology covering the full data-analysis workflow, with two
complementary entry points: feature importance (associational) and
root-cause analysis (causal). Highlights:

- **EDA workflow** — structured 9-step exploration with point-mass guard,
  Spearman correlations, derived-column / target-leakage check, and
  mandatory **VIF + mixed-type collinearity audit** (numeric, design-matrix
  with one-hot dummies, plus η² / Cramér's V for cross-type bindings).
- **Pre-modeling diagnostics** — null-handling decision tree (with dtype-
  aware association measure), target-skew check, cluster-representative
  selector with priority-ordered rules, log-transform recipe.
- **Feature-importance pipeline** — XGBoost + permutation importance + SHAP
  with cross-method agreement check, three-lens rank table, anti-pattern
  catalog including the rating-tautology trap.
- **Root-cause analysis** — change-point detection (PELT, CUSUM), SPC
  (Shewhart, EWMA, Cp/Cpk), commonality with Fisher exact + lift +
  Bonferroni, causal-inference primer (DAG, DiD, propensity matching,
  dowhy refutation), DOE/ANOVA with effect sizes.
- **Plotly default** — interactive HTML by default; matplotlib/seaborn as
  fallback for static publication figures.
- **pandas 3.0 compliant** — no `inplace=True`, CoW-safe, lowercase
  frequency aliases, dtype-aware nullable types.

## Skills shipped

| Skill name | Path | Language |
|---|---|---|
| `data-analyst` | `.claude/skills/data-analyst/` | English (source of truth) |
| `data-analyst-ko` | `.claude/skills/data-analyst-ko/` | Korean (translation, synced from `skills-ko/data-analyst/`) |

The Korean translation is structurally 1:1 with the English source. Code
blocks are byte-identical (only inline `# comments` translated). When the
English version updates, the corresponding Korean file is synced via diff
translation — see [`skills-ko/README.md`](skills-ko/README.md) for the
maintenance workflow.

## Verification artifacts

The `analysis/` directory contains scripts that exercised the skill
end-to-end on real public datasets and were used to surface and verify
each diagnostic capability:

- `stress_factors.py` / `stress_factors_ml.py` — synthetic student-stress
  dataset, linear vs. ML cross-check
- `null_collinearity_probe.py` / `skill_diagnostics_check.py` — VIF +
  null-audit verification
- `ames_drivers.py` — Ames Housing end-to-end (rating-tautology check,
  log-transform, fair linear vs. ML comparison)
- `secom_rca.py` — SECOM semiconductor manufacturing RCA (CPD +
  commonality + DiD; demonstrates the SHAP-vs-RCA divergence)
- `vif_mixed_type_probe.py` / `mixed_type_vif_test.py` — mixed-type VIF
  and cross-type binding (η² / Cramér's V)
- `cluster_selector_test.py` — cluster-representative selector (5
  priority rules)

These also serve as worked examples of how to apply the skill to a fresh
dataset.

## Layout

```
.
├── .claude/
│   └── skills/
│       ├── data-analyst/         ← English skill (Anthropic Agent Skills spec)
│       └── data-analyst-ko/      ← Korean skill (auto-discovered)
├── .claude-plugin/
│   └── marketplace.json          ← explicit manifest for the skills CLI
├── skills-ko/
│   ├── README.md                 ← Korean translation maintenance guide
│   └── data-analyst/             ← Korean source-of-truth (kept in sync)
└── analysis/                     ← worked examples / verification scripts
```

## Compatibility

- **Claude Code** — auto-discovered from `.claude/skills/`
- **vercel-labs/skills CLI** — auto-discovered + explicit manifest
- **Cursor / other agents** — copy `.claude/skills/data-analyst*/` to the
  agent-specific path (e.g. `.cursor/skills/`); the skill format is
  agent-agnostic

## License & attribution

Skill methodology authored by jisu (@JISUlicious). Empirical validation
on public datasets:
[Ames Housing](https://jse.amstat.org/v19n3/decock.pdf) by Dean De Cock,
[SECOM](https://archive.ics.uci.edu/dataset/179/secom) from UCI.
