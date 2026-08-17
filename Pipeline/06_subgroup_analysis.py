"""
Step 06 - Subgroup analysis: does enrichment help where intelligence actually exists?

Only about a third of GUIDE incidents carry MITRE ATT&CK techniques, so the overall
A-vs-B comparison dilutes any enrichment effect across incidents that have no
intelligence to add. This step re-scores the SAME trained predictions on the subset of
evaluation incidents where has_ti = 1, which is the fair test of the enrichment claim.

No retraining and no new fitting: predictions from step 05 are reused, so this is a
partition of existing results, not a second experiment.

Usage: python3 Pipeline/06_subgroup_analysis.py
Output: Results/subgroup_results.csv
"""
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import binomtest

rows, sig = [], []
for cond in ["standard","crossorg","acrosstime"]:
    ev = pd.read_csv(f"Dataset/derived/features/{cond}_eval_B.csv")
    y = ev.label.values
    mask = (ev.has_ti == 1).values
    for model in ["RandomForest","GradientBoosting"]:
        p = {fs: np.load(f"Results/predictions/{cond}_{fs}_{model}.npy") for fs in ["A","B"]}
        for fs in ["A","B"]:
            yy, pp = y[mask], p[fs][mask]
            tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0,1]).ravel()
            rows.append({"condition":cond,"model":model,"feature_set":fs,
                         "subgroup":"has_technique","n":int(mask.sum()),
                         "precision":round(precision_score(yy,pp,zero_division=0),4),
                         "recall":round(recall_score(yy,pp,zero_division=0),4),
                         "f1":round(f1_score(yy,pp,zero_division=0),4),
                         "fpr":round(fp/(fp+tn) if (fp+tn) else 0,4)})
        a_ok = (p["A"][mask] == y[mask]); b_ok = (p["B"][mask] == y[mask])
        n01, n10 = int((~a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())
        pv = binomtest(n01, n01+n10, 0.5).pvalue if (n01+n10) else 1.0
        sig.append({"condition":cond,"model":model,"subgroup":"has_technique",
                    "B_right_A_wrong":n01,"A_right_B_wrong":n10,
                    "net_gain_for_B":n01-n10,"mcnemar_exact_p":round(float(pv),6),
                    "significant_at_0.05": bool(pv<0.05)})
df = pd.DataFrame(rows); df.to_csv("Results/subgroup_results.csv", index=False)
sg = pd.DataFrame(sig); sg.to_csv("Results/subgroup_significance.csv", index=False)
print(df.pivot_table(index=["condition","model"], columns="feature_set",
                     values=["f1","fpr","recall","precision"]).round(4).to_string())
print("\n", sg.to_string(index=False))
