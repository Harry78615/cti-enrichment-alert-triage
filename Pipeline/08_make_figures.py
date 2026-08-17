"""
Step 08 - Generate the figures for the Findings chapter.
Every figure is produced directly from the saved result files, so each can be regenerated
and audited. Titles are deliberately plain; captions belong in the dissertation text.
Usage: python3 Pipeline/08_make_figures.py
Outputs: Figures/*.png (300 dpi)
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi":300, "font.size":9, "font.family":"DejaVu Sans",
                     "axes.spines.top":False, "axes.spines.right":False})
CA, CB = "#4C72B0", "#DD8452"      # feature set A, feature set B
COND = ["standard","crossorg","acrosstime"]
LBL = {"standard":"Standard","crossorg":"Cross-organisation","acrosstime":"Across-time"}

res = pd.read_csv("Results/experiment_results.csv")
sub = pd.read_csv("Results/subgroup_results.csv")

def grouped(ax, df, metric, title, ylabel):
    models = ["RandomForest","GradientBoosting"]
    x = np.arange(len(COND)*len(models)); w = 0.38; ticks=[]
    for i,(c,m) in enumerate([(c,m) for c in COND for m in models]):
        a = df[(df.condition==c)&(df.model==m)&(df.feature_set=="A")][metric].values[0]
        b = df[(df.condition==c)&(df.model==m)&(df.feature_set=="B")][metric].values[0]
        ax.bar(i-w/2, a, w, color=CA, label="A (alert features)" if i==0 else "")
        ax.bar(i+w/2, b, w, color=CB, label="B (+ threat intelligence)" if i==0 else "")
        ax.text(i-w/2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=6)
        ax.text(i+w/2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=6)
        ticks.append(f"{LBL[c]}\n{'RF' if m=='RandomForest' else 'GB'}")
    ax.set_xticks(x); ax.set_xticklabels(ticks, fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, frameon=False)

# Fig 1 - F1 and FPR, all incidents
fig, axes = plt.subplots(2,1, figsize=(7,6.5))
grouped(axes[0], res, "f1", "F1 score by condition and model (all evaluation incidents)", "F1")
grouped(axes[1], res, "fpr", "False positive rate by condition and model (all evaluation incidents)", "FPR")
axes[1].invert_yaxis()
plt.tight_layout(); plt.savefig("Figures/fig1_overall_f1_fpr.png", bbox_inches="tight"); plt.close()

# Fig 2 - subgroup: incidents that carry techniques
fig, axes = plt.subplots(2,1, figsize=(7,6.5))
grouped(axes[0], sub, "f1", "F1 score - incidents carrying ATT&CK techniques only", "F1")
grouped(axes[1], sub, "fpr", "False positive rate - incidents carrying ATT&CK techniques only", "FPR")
axes[1].invert_yaxis()
plt.tight_layout(); plt.savefig("Figures/fig2_subgroup_has_technique.png", bbox_inches="tight"); plt.close()

# Fig 3 - FPR change delta (B - A), both populations
fig, ax = plt.subplots(figsize=(7,3.4))
labels, alls, subs = [], [], []
for c in COND:
    for m in ["RandomForest","GradientBoosting"]:
        f=lambda d,fs: d[(d.condition==c)&(d.model==m)&(d.feature_set==fs)]["fpr"].values[0]
        labels.append(f"{LBL[c]}\n{'RF' if m=='RandomForest' else 'GB'}")
        alls.append(f(res,"B")-f(res,"A")); subs.append(f(sub,"B")-f(sub,"A"))
x=np.arange(len(labels)); w=0.38
ax.bar(x-w/2, alls, w, color="#999999", label="All incidents")
ax.bar(x+w/2, subs, w, color="#55A868", label="Incidents with techniques")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Change in FPR (B - A)\nnegative = enrichment helps")
ax.set_title("Effect of threat-intelligence enrichment on false positive rate", fontsize=10)
ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig("Figures/fig3_fpr_delta.png", bbox_inches="tight"); plt.close()

# Fig 4 - SHAP top-20 with TI features highlighted
TI_PREF=("ta_","n_tech","n_subtech","share_sub","n_tactics","kill_chain","group_use",
         "software_use","mitigations","technique_age","days_since","has_ti","ti_alert")
sh = pd.read_csv("Results/shap_standard.csv").head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(6.5,5))
cols = [CB if f.startswith(TI_PREF) else CA for f in sh.feature]
ax.barh(sh.feature, sh.mean_abs_shap, color=cols)
ax.set_xlabel("Mean |SHAP value|"); ax.set_title("Top 20 features by SHAP (standard condition, feature set B)", fontsize=10)
h=[plt.Rectangle((0,0),1,1,color=CA), plt.Rectangle((0,0),1,1,color=CB)]
ax.legend(h,["Alert features","Threat-intelligence features"], fontsize=7, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig("Figures/fig4_shap_top20.png", bbox_inches="tight"); plt.close()

# Fig 5 - SHAP stability across runs
st = json.load(open("Results/shap_stability_standard.json"))
fig, ax = plt.subplots(figsize=(5,3))
ax.bar(["Runs 1-2","Runs 1-3","Runs 2-3"], st["pairwise_top10_overlap"], color="#55A868")
ax.set_ylim(0,1.05); ax.set_ylabel("Top-10 feature overlap")
ax.set_title("SHAP stability across independent subsamples", fontsize=10)
for i,v in enumerate(st["pairwise_top10_overlap"]): ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout(); plt.savefig("Figures/fig5_shap_stability.png", bbox_inches="tight"); plt.close()

# Fig 6 - efficiency cost of enrichment
eff = res.groupby(["model","feature_set"])["train_seconds"].mean().unstack()
fig, ax = plt.subplots(figsize=(5,3.2))
x=np.arange(len(eff)); w=0.38
ax.bar(x-w/2, eff["A"], w, color=CA, label="A"); ax.bar(x+w/2, eff["B"], w, color=CB, label="B")
for i,(a,b) in enumerate(zip(eff["A"],eff["B"])):
    ax.text(i-w/2,a,f"{a:.1f}s",ha="center",va="bottom",fontsize=7)
    ax.text(i+w/2,b,f"{b:.1f}s ({(b/a-1)*100:+.0f}%)",ha="center",va="bottom",fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(eff.index, fontsize=8); ax.set_ylabel("Mean training time (s)")
ax.set_title("Efficiency cost of enrichment (RQ3)", fontsize=10); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig("Figures/fig6_efficiency.png", bbox_inches="tight"); plt.close()

# Fig 7 - technique coverage (RQ1)
import glob, os
cov=[]
for p in sorted(glob.glob("Dataset/derived/features/*_manifest.json")):
    m=json.load(open(p)); cov.append((m["sample"], m["technique_coverage_incident_level"], m["mean_ti_alert_coverage"]))
cv=pd.DataFrame(cov, columns=["sample","incident_coverage","alert_coverage"]).sort_values("sample")
fig, ax = plt.subplots(figsize=(7,3.2))
x=np.arange(len(cv)); w=0.38
ax.bar(x-w/2, cv.incident_coverage, w, color=CA, label="Incidents with >=1 technique")
ax.bar(x+w/2, cv.alert_coverage, w, color=CB, label="Mean share of alerts carrying techniques")
ax.set_xticks(x); ax.set_xticklabels(cv["sample"], rotation=20, ha="right", fontsize=7)
ax.set_ylabel("Proportion"); ax.set_ylim(0,0.6)
ax.set_title("ATT&CK technique coverage in the sampled incidents (RQ1)", fontsize=10)
ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig("Figures/fig7_ti_coverage.png", bbox_inches="tight"); plt.close()
print("figures written:", len(glob.glob("Figures/*.png")))
