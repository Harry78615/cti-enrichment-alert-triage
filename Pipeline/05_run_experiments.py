"""
Step 05 - Run the matched-model experiments.

For each condition (standard, crossorg, acrosstime) and each learner
(Random Forest, Gradient Boosting), trains on feature set A and feature set B and
evaluates on the matching evaluation sample. Records effectiveness (RQ2),
efficiency (RQ3) and a paired significance test of B against A.

Design notes:
  - Binary task: TruePositive = 1 ("needs investigation"), BenignPositive/FalsePositive = 0
    (supervisor-approved framing, 17 July).
  - Class imbalance handled with class_weight='balanced' (RF); Gradient Boosting has no
    class_weight, so sample weights are passed instead - equivalent effect, recorded here.
  - No test data touches training: hyperparameters are fixed a priori (documented below),
    not tuned on evaluation data.
  - Seed 42 everywhere.
  - McNemar's exact test on paired predictions answers "is B different from A, significantly?"

Usage:
    python3 Pipeline/05_run_experiments.py standard     # one condition at a time
    python3 Pipeline/05_run_experiments.py crossorg
    python3 Pipeline/05_run_experiments.py acrosstime
    python3 Pipeline/05_run_experiments.py significance # after all three
Outputs: Results/experiment_results.csv, Results/significance_tests.csv,
         Results/predictions/*.npy, Results/run_manifest.json
"""
import os, sys, json, time, platform
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import binomtest

SEED = 42
FEAT = "Dataset/derived/features"
OUT = "Results"; os.makedirs(f"{OUT}/predictions", exist_ok=True)
ALL_CONDITIONS = ["standard", "crossorg", "acrosstime"]
SETS = ["A", "B"]
arg = sys.argv[1] if len(sys.argv) > 1 else "all"
CONDITIONS = ALL_CONDITIONS if arg in ("all", "significance") else [arg]

def make_models():
    """Fixed hyperparameters, chosen a priori for tabular data of this size."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=SEED),
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=False, random_state=SEED),
    }

rows, preds = [], {}
for cond in ([] if arg == "significance" else CONDITIONS):
    for fs in SETS:
        tr = pd.read_csv(f"{FEAT}/{cond}_train_{fs}.csv")
        ev = pd.read_csv(f"{FEAT}/{cond}_eval_{fs}.csv")
        assert list(tr.columns) == list(ev.columns), f"column mismatch {cond} {fs}"
        Xtr, ytr = tr.drop(columns=["label"]).values, tr.label.values
        Xev, yev = ev.drop(columns=["label"]).values, ev.label.values

        for mname, model in make_models().items():
            t0 = time.perf_counter()
            if mname == "GradientBoosting":
                sw = compute_sample_weight("balanced", ytr)
                model.fit(Xtr, ytr, sample_weight=sw)
            else:
                model.fit(Xtr, ytr)
            train_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            yp = model.predict(Xev)
            score_s = time.perf_counter() - t1
            proba = model.predict_proba(Xev)[:, 1]

            tn, fp, fn, tp = confusion_matrix(yev, yp, labels=[0, 1]).ravel()
            rows.append({
                "condition": cond, "feature_set": fs, "model": mname,
                "n_train": len(ytr), "n_eval": len(yev), "n_features": Xtr.shape[1],
                "precision": round(precision_score(yev, yp, zero_division=0), 4),
                "recall": round(recall_score(yev, yp, zero_division=0), 4),
                "f1": round(f1_score(yev, yp, zero_division=0), 4),
                "fpr": round(fp / (fp + tn) if (fp + tn) else 0, 4),
                "roc_auc": round(roc_auc_score(yev, proba), 4),
                "pr_auc": round(average_precision_score(yev, proba), 4),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "train_seconds": round(train_s, 2),
                "score_seconds_per_1k": round(score_s / len(yev) * 1000, 4),
            })
            preds[f"{cond}|{fs}|{mname}"] = yp
            np.save(f"{OUT}/predictions/{cond}_{fs}_{mname}.npy", yp)
            print(f"{cond:11s} {fs} {mname:17s} F1={rows[-1]['f1']:.3f} "
                  f"FPR={rows[-1]['fpr']:.3f} recall={rows[-1]['recall']:.3f} ({train_s:.0f}s train)")

if rows:   # append so conditions can be run in separate sessions
    res = pd.DataFrame(rows)
    p = f"{OUT}/experiment_results.csv"
    if os.path.exists(p):
        old = pd.read_csv(p)
        old = old[~old.condition.isin(res.condition.unique())]
        res = pd.concat([old, res], ignore_index=True)
    res.sort_values(["condition","model","feature_set"]).to_csv(p, index=False)

# ---- paired significance: B vs A on identical evaluation incidents ----
sig = []
for cond in (ALL_CONDITIONS if arg in ("all","significance") else []):
    yev = pd.read_csv(f"{FEAT}/{cond}_eval_A.csv").label.values
    for mname in make_models():
        a = preds.get(f"{cond}|A|{mname}", None)
        if a is None: a = np.load(f"{OUT}/predictions/{cond}_A_{mname}.npy")
        b = preds.get(f"{cond}|B|{mname}", None)
        if b is None: b = np.load(f"{OUT}/predictions/{cond}_B_{mname}.npy")
        a_ok, b_ok = (a == yev), (b == yev)
        n01 = int((~a_ok & b_ok).sum())   # B right, A wrong
        n10 = int((a_ok & ~b_ok).sum())   # A right, B wrong
        p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
        sig.append({"condition": cond, "model": mname, "B_right_A_wrong": n01,
                    "A_right_B_wrong": n10, "mcnemar_exact_p": round(float(p), 6),
                    "significant_at_0.05": bool(p < 0.05),
                    "net_gain_for_B": n01 - n10})
if sig: pd.DataFrame(sig).to_csv(f"{OUT}/significance_tests.csv", index=False)

json.dump({"seed": SEED, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "python": platform.python_version(),
           "sklearn": __import__("sklearn").__version__,
           "models": {k: str(v) for k, v in make_models().items()},
           "note": "Binary task; class imbalance via balanced class/sample weights; "
                   "hyperparameters fixed a priori, not tuned on evaluation data."},
          open(f"{OUT}/run_manifest.json", "w"), indent=2)
if sig:
    print("\n--- significance (B vs A) ---")
    print(pd.DataFrame(sig).to_string(index=False))
