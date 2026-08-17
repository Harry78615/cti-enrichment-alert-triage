# Evaluating Cyber Threat Intelligence Enrichment for Reducing False Positives in Machine Learning Alert Triage

MSc Cyber Security dissertation, Sheffield Hallam University.
Author: Syed Muhammad Haris Shah · Supervisor: Dr John Haggerty · September 2026

## What this study asks

Threat intelligence enrichment is widely credited with reducing false positives in SOC alert
triage, but the claim is asserted far more often than it is measured. This study builds two
closely matched machine-learning models that differ **only** in whether MITRE ATT&CK technique
context is attached, and measures what that context adds — and what it costs.

**RQ1** Which threat-intelligence features can be produced reliably for public alert data?
**RQ2** How much do they improve effectiveness (false positive rate, recall, F1) under standard and harder evaluation?
**RQ3** What efficiency cost does enrichment add, and is any gain worth it?
**RQ4** Which features matter most, shown through explainability analysis?

## Data

**Microsoft GUIDE** (Freitas et al., 2025) — ~1M real security incidents with analyst-assigned
triage grades (TruePositive / BenignPositive / FalsePositive), 6.1k organisations,
441 ATT&CK techniques. Released under CDLA-2.0 on Kaggle:
`kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction`

**MITRE ATT&CK Enterprise** STIX bundle: `github.com/mitre-attack/attack-stix-data`

Neither dataset is committed here (size and licence). Download both, place `GUIDE_Train.csv`,
`GUIDE_Test.csv` and `enterprise-attack.json` in `Dataset/`, then run the pipeline below.

## Pipeline

Run in order from the repository root. Each step writes a JSON manifest recording inputs,
row counts, seed and outputs, so any result can be traced back or regenerated.

| Step | Script | What it does |
|---|---|---|
| 1 | `01a_convert_slim.py train\|test` | Raw CSV → slim Parquet (the 6 columns sampling needs) |
| 2 | `01b_build_incident_index.py train\|test` | Evidence rows → one row per incident (grade, category, first timestamp) |
| 3 | `02_draw_samples.py` | Draws the six protocol samples, seed 42 |
| 4 | `03_extract_ti_context.py stix Dataset/enterprise-attack.json` | ATT&CK technique context + revoked-ID alias map (also supports `opencti` mode via pycti) |
| 5 | `04a_aggregate.py` | Streams raw rows for sampled incidents → cached incident-level aggregates |
| 6 | `04b_build_features.py` | Feature set A (73 features) and B (102 = A + 29 TI features) |
| 7 | `05_run_experiments.py standard\|crossorg\|acrosstime` then `significance` | Trains RF + GB on A and B, evaluates, McNemar tests |
| 8 | `06_subgroup_analysis.py` | Re-scores predictions on incidents that carry ATT&CK techniques |
| 9 | `07_explainability.py <condition>` | TreeSHAP, three-run stability check, permutation cross-check |
| 10 | `08_make_figures.py` | Figures for the dissertation |
| 11 | `09_make_orange_files.py` | Reduced, Orange-ready files for the visual workflow |
| 12 | `10_pipeline_diagram.py` | The pipeline diagram (Figures/fig0_pipeline.png) |

Requirements: Python 3.10+, `pandas pyarrow scikit-learn scipy shap matplotlib`
(`pycti` only for the OpenCTI route). Orange 3 with the Explain add-on for
`Orange/triage_workflow.ows`.

## Design decisions worth knowing

- **Unit of analysis: the incident.** Evidence rows are aggregated to alert then incident level,
  mirroring the Freitas et al. baseline. An incident never straddles a train/test split.
- **Binary primary task.** TruePositive = "needs investigation" versus BenignPositive +
  FalsePositive = "can be safely deprioritised"; three-class results are reported as secondary.
- **Three evaluation conditions.** Standard (provided split), cross-organisation (a seeded 20%
  of organisations held out entirely — the provided split is *not* organisation-disjoint), and a
  short-horizon across-time split (GUIDE's incidents concentrate in a two-week June window).
- **Leakage controls.** `IncidentGrade`, `ActionGrouped`, `ActionGranular`, `LastVerdict`,
  `OrgId` and raw high-cardinality identifiers are never features. Feature set B draws only on
  `MitreTechniques`, available at alert time.
- **Enrichment is technique-level, not indicator-level**, because GUIDE pseudo-anonymises entity
  values (SHA1) — a property of privacy-preserving data, not a limitation of the method.

## Repository layout

```
Pipeline/   numbered scripts, run in order
Results/    experiment results, significance tests, SHAP rankings (CSV/JSON)
Figures/    dissertation figures; Figures/orange/ holds Orange workflow screenshots
Orange/     the Orange workflow file (.ows)
Dataset/    empty in git — place downloaded data here (see Data above)
```

## Reproducibility

Seed 42 throughout. Hyperparameters fixed a priori, not tuned on evaluation data. Every
sampling and feature step writes a JSON manifest (counts at each filter step, class
proportions, content hashes). Manifests are committed even though the data is not.

## Licence and attribution

Code released for academic assessment. The GUIDE dataset remains under CDLA-2.0 (Microsoft);
MITRE ATT&CK is © The MITRE Corporation.
