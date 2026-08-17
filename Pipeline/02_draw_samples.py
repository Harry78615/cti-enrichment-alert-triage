"""
Step 02 - Draw the protocol samples (Sampling_Protocol_Haris.docx v1).
Seed 42 everywhere. Incident-level. Stratified by (IncidentGrade, Category).
Per-organisation cap: max 5% of a sample from any single OrgId.
Conditions:
  standard   : train sample from train index; eval sample from test index (provided split)
  crossorg   : 20% of train orgs held out (seeded); train sample from the 80%,
               eval sample from held-out orgs only
  acrosstime : train sample from incidents first_ts < 2024-06-10 (v1.1 revision);
               eval sample from incidents first_ts >= 2024-06-10
Budgets: TRAIN_N=100_000, EVAL_N=50_000 (protocol s4; reduced only if a pool is smaller).
Labels: binary primary - TruePositive=1 ('needs investigation'), BenignPositive/FalsePositive=0;
        three-class kept in 'grade' column for the secondary task.
Outputs: Dataset/derived/samples/{name}.csv + manifest JSON each.
Usage: python3 Pipeline/02_draw_samples.py
"""
import json, os, time, hashlib
import pandas as pd
import numpy as np

SEED = 42
TRAIN_N, EVAL_N = 100_000, 50_000
ORG_CAP = 0.05
os.makedirs("Dataset/derived/samples", exist_ok=True)
rng = np.random.RandomState(SEED)

def load(which):
    df = pd.read_parquet(f"Dataset/derived/{which}_incident_index.parquet")
    steps = [("loaded", len(df))]
    df = df[df.IncidentGrade.fillna("") != ""].copy()
    steps.append(("dropped_missing_grade", len(df)))
    df["label"] = (df.IncidentGrade == "TruePositive").astype(int)
    df["Category"] = df.Category.fillna("Unknown").replace("", "Unknown")
    return df, steps

def stratified_capped(pool, n, name):
    """Proportional stratified sample by (grade, category), then enforce org cap."""
    n = min(n, len(pool))
    # proportional allocation per stratum
    samp = (pool.groupby(["IncidentGrade","Category"], group_keys=False)
                .apply(lambda g: g.sample(n=max(1, round(len(g)/len(pool)*n)), random_state=SEED) if len(g)>0 else g))
    # trim/expand to n exactly
    if len(samp) > n: samp = samp.sample(n=n, random_state=SEED)
    # org cap
    cap = int(ORG_CAP * n)
    for _ in range(6):
        over = samp.OrgId.value_counts()
        over = over[over > cap]
        if over.empty: break
        drop_idx = []
        for org, cnt in over.items():
            rows = samp[samp.OrgId == org]
            drop_idx.extend(rows.sample(n=cnt-cap, random_state=SEED).index)
        samp = samp.drop(index=drop_idx)
        pool_rest = pool[~pool.index.isin(samp.index)]
        pool_rest = pool_rest[~pool_rest.OrgId.isin(over.index)]
        need = n - len(samp)
        if need > 0 and len(pool_rest) > 0:
            samp = pd.concat([samp, pool_rest.sample(n=min(need,len(pool_rest)), random_state=SEED)])
    return samp

def save(df, name, steps, extra):
    out = f"Dataset/derived/samples/{name}.csv"
    cols = ["OrgId","IncidentId","IncidentGrade","label","Category","first_ts","n_evidence"]
    df[cols].to_csv(out, index=False)
    h = hashlib.sha256(open(out,"rb").read()).hexdigest()[:16]
    m = {"name": name, "rows": len(df), "seed": SEED, "steps": steps,
         "grade_counts": df.IncidentGrade.value_counts().to_dict(),
         "label_counts": df.label.value_counts().to_dict(),
         "n_orgs": int(df.OrgId.nunique()),
         "max_org_share": round(df.OrgId.value_counts().max()/len(df), 4),
         "sha256_16": h, **extra}
    json.dump(m, open(f"Dataset/derived/samples/{name}_manifest.json","w"), indent=2)
    print(name, "->", len(df), "rows |", m["grade_counts"], "| max org share", m["max_org_share"])

t0=time.time()
train, tsteps = load("train")
test, esteps = load("test")

# --- Condition 1: standard ---
s_tr = stratified_capped(train, TRAIN_N, "std")
save(s_tr, "standard_train", tsteps, {"condition":"standard","role":"train"})
s_te = stratified_capped(test, EVAL_N, "std")
save(s_te, "standard_eval", esteps, {"condition":"standard","role":"eval"})

# --- Condition 2: cross-organisation ---
orgs = np.array(sorted(train.OrgId.unique()))
rng.shuffle(orgs)
n_hold = int(0.2*len(orgs))
hold, keep = set(orgs[:n_hold].tolist()), set(orgs[n_hold:].tolist())
pool_tr = train[train.OrgId.isin(keep)]
pool_ev = train[train.OrgId.isin(hold)]
x_tr = stratified_capped(pool_tr, TRAIN_N, "xorg")
save(x_tr, "crossorg_train", tsteps+[("orgs_kept", len(keep)), ("pool", len(pool_tr))], {"condition":"crossorg","role":"train","held_out_orgs": n_hold})
x_ev = stratified_capped(pool_ev, EVAL_N, "xorg")
save(x_ev, "crossorg_eval", tsteps+[("orgs_held_out", n_hold), ("pool", len(pool_ev))], {"condition":"crossorg","role":"eval"})
assert set(x_tr.OrgId) & set(x_ev.OrgId) == set(), "org leakage!"

# --- Condition 3: across-time ---
CUT = "2024-06-10"  # protocol v1.1: 95% of incidents fall 3-17 Jun 2024, so the split is within the bulk (short-horizon); see decisions log 28 Jul
pool_tr = train[train.first_ts < CUT]
pool_ev = train[train.first_ts >= CUT]
t_tr = stratified_capped(pool_tr, TRAIN_N, "time")
save(t_tr, "acrosstime_train", tsteps+[("pool_before_cut", len(pool_tr))], {"condition":"acrosstime","role":"train","cutoff":CUT})
t_ev = stratified_capped(pool_ev, EVAL_N, "time")
save(t_ev, "acrosstime_eval", tsteps+[("pool_after_cut", len(pool_ev))], {"condition":"acrosstime","role":"eval","cutoff":CUT})
assert t_tr.first_ts.max() < CUT <= t_ev.first_ts.min(), "temporal leakage!"

print("total elapsed", round(time.time()-t0,1), "s")
