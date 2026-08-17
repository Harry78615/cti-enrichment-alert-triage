"""
Step 07 - Explainability analysis (RQ4).

Answers "which features matter most, and can that answer be trusted?" using:
  1. TreeSHAP on the feature-set-B Gradient Boosting model (exact for tree ensembles -
     unlike the perturbation-based KernelSHAP criticised by Warnecke et al. (2020), so the
     instability they report does not automatically apply here).
     Note on model choice: TreeSHAP on the 300-tree Random Forest proved computationally
     prohibitive (about 27s per 200 incidents, versus 1.9s per 1500 for Gradient Boosting,
     whose trees are depth-limited). This is Warnecke et al.'s "efficiency" criterion met in
     practice and is reported as a finding rather than hidden; Gradient Boosting is therefore
     the explained model, and permutation importance covers the Random Forest if needed.
  2. A stability check in the spirit of Warnecke et al.: SHAP is recomputed on three
     independent evaluation subsamples and the overlap of the top-10 features reported.
  3. Permutation importance as an independent, model-agnostic cross-check
     (decisions log, 15 July): agreement between two methods makes any claim about
     which features matter much harder to dispute.
  4. An artefact check (Arp et al. P4; Lanvin et al.; D'hooge et al.): the ranking is
     inspected for features that could be organisational or detector shortcuts rather
     than security signal.

Usage: python3 Pipeline/07_explainability.py <condition> [n_sample]
Outputs: Results/shap_<condition>.csv, Results/shap_stability_<condition>.json,
         Results/permutation_<condition>.csv
"""
import sys, json, time
import numpy as np, pandas as pd, shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance

cond = sys.argv[1] if len(sys.argv) > 1 else "standard"
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SEED = 42

tr = pd.read_csv(f"Dataset/derived/features/{cond}_train_B.csv")
ev = pd.read_csv(f"Dataset/derived/features/{cond}_eval_B.csv")
feat = [c for c in tr.columns if c != "label"]
sw = compute_sample_weight("balanced", tr.label.values)
model = HistGradientBoostingClassifier(max_iter=300, early_stopping=False, random_state=SEED
        ).fit(tr[feat].values, tr.label.values, sample_weight=sw)

# ---- 1 & 2: TreeSHAP with a three-run stability check ----
expl = shap.TreeExplainer(model)
tops, means = [], []
for run, seed in enumerate([SEED, SEED+1, SEED+2]):
    sub = ev.sample(n=min(NS, len(ev)), random_state=seed)
    t0 = time.perf_counter()
    sv = expl.shap_values(sub[feat].values, check_additivity=False)
    sv = np.asarray(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    m = np.abs(sv).mean(axis=0)
    means.append(m)
    tops.append([feat[i] for i in np.argsort(-m)[:10]])
    print(f"  run {run+1}: {time.perf_counter()-t0:.1f}s")

pairs = [(0,1),(0,2),(1,2)]
overlaps = [len(set(tops[a]) & set(tops[b]))/10 for a,b in pairs]
mean_abs = np.mean(means, axis=0)
order = np.argsort(-mean_abs)
shap_df = pd.DataFrame({"feature":[feat[i] for i in order],
                        "mean_abs_shap":np.round(mean_abs[order],6),
                        "rank":range(1,len(order)+1)})
shap_df.to_csv(f"Results/shap_{cond}.csv", index=False)
json.dump({"condition":cond,"n_per_run":NS,"runs":3,
           "top10_per_run":tops,"pairwise_top10_overlap":overlaps,
           "mean_top10_overlap":round(float(np.mean(overlaps)),3),
           "interpretation":"1.0 means the top-10 features are identical across runs "
                            "(Warnecke et al. stability criterion)"},
          open(f"Results/shap_stability_{cond}.json","w"), indent=2)

# ---- 3: permutation importance cross-check ----
sub = ev.sample(n=min(NS, len(ev)), random_state=SEED)
pi = permutation_importance(model, sub[feat].values, sub.label.values,
                            n_repeats=5, random_state=SEED, n_jobs=-1, scoring="f1")
pdf = pd.DataFrame({"feature":feat,"perm_importance":np.round(pi.importances_mean,6)}
                   ).sort_values("perm_importance", ascending=False).reset_index(drop=True)
pdf["rank"] = pdf.index+1
pdf.to_csv(f"Results/permutation_{cond}.csv", index=False)

agree = len(set(shap_df.feature[:10]) & set(pdf.feature[:10]))/10
print(f"\nSHAP top-10 stability across runs: {np.mean(overlaps):.2f}")
print(f"SHAP vs permutation top-10 agreement: {agree:.2f}")
print("\nTop 12 by SHAP:")
print(shap_df.head(12).to_string(index=False))
ti_cols = [c for c in feat if c.startswith(("ta_","n_tech","n_subtech","share_sub","n_tactics",
           "kill_chain","group_use","software_use","mitigations","technique_age","days_since","has_ti","ti_alert"))]
in_top20 = [f for f in shap_df.feature[:20] if f in ti_cols]
print(f"\nTI features in SHAP top-20: {in_top20}")
json.dump({"shap_vs_permutation_top10_agreement":agree,
           "ti_features_in_top20":in_top20,
           "best_ti_rank":int(shap_df[shap_df.feature.isin(ti_cols)]["rank"].min())},
          open(f"Results/explain_summary_{cond}.json","w"), indent=2)
