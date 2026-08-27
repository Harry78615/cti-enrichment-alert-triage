"""
Step 04b - Build model-ready feature tables from the cached aggregates (04a).

For each of the six protocol samples writes two CSVs:
    <sample>_A.csv - plain alert features only
    <sample>_B.csv - the same features plus the threat-intelligence block
A and B differ ONLY in the TI columns.

Leakage controls: IncidentGrade, ActionGrouped, ActionGranular, LastVerdict, OrgId and raw
high-cardinality identifiers are never features. TI context is joined from ti_context.csv
using only MitreTechniques, which is available at alert time. Revoked technique IDs are
resolved through ti_alias.csv before being counted as unmatched.

Usage: python3 Pipeline/04b_build_features.py
Outputs: Dataset/derived/features/*.csv 
"""
import os, json, time, glob
import pandas as pd, numpy as np

SAMP, OUTD = "Dataset/derived/samples", "Dataset/derived/features"
os.makedirs(OUTD, exist_ok=True)
REF_DATE = pd.Timestamp("2024-06-17")   # dataset end date - technique age measured against this
TACTICS = ["reconnaissance","resource-development","initial-access","execution","persistence",
           "privilege-escalation","defense-evasion","credential-access","discovery",
           "lateral-movement","collection","command-and-control","exfiltration","impact"]

samples = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in sorted(glob.glob(f"{SAMP}/*.csv"))}
src_of = {n: ("test" if n == "standard_eval" else "train") for n in samples}
agg = {w: pd.read_parquet(f"Dataset/derived/agg_{w}.parquet").set_index("k") for w in ["train","test"]}

ti = pd.read_csv("Dataset/derived/ti_context.csv")
cre = pd.to_datetime(ti.created, format="mixed", utc=True).dt.tz_localize(None)
mod = pd.to_datetime(ti.modified, format="mixed", utc=True).dt.tz_localize(None)
# plain-dict lookup: far faster than pandas indexing inside a per-incident loop
META = {r.technique_id: (int(r.is_subtechnique),
                         tuple(x for x in str(r.tactics).split(";") if x and x != "nan"),
                         int(r.n_groups), int(r.n_softwares), int(r.n_mitigations),
                         float((REF_DATE - c).days), float((REF_DATE - m).days))
        for r, c, m in zip(ti.itertuples(index=False), cre, mod)}
ALIAS = {}
try:
    _a = pd.read_csv("Dataset/derived/ti_alias.csv")
    ALIAS = dict(zip(_a.old_id, _a.new_id))
except FileNotFoundError:
    pass
TA_COLS = [f"ta_{t.replace('-','_')}" for t in TACTICS]
ZERO = dict({"has_ti":0,"n_techniques":0,"n_subtechniques":0,"share_subtechniques":0.0,"n_tactics":0,
             "kill_chain_breadth":0.0,"group_use_max":0,"group_use_mean":0.0,"software_use_max":0,
             "software_use_mean":0.0,"mitigations_mean":0.0,"technique_age_min_days":0.0,
             "technique_age_mean_days":0.0,"days_since_modified_min":0.0}, **{c:0 for c in TA_COLS})

def key(df): return df.OrgId.astype(str) + "|" + df.IncidentId.astype(str)

def ti_row(cell, unknown):
    """Threat-intelligence block for one incident (structural zeros when no techniques)."""
    if not cell or not isinstance(cell, str): return ZERO
    ids = sorted({x.strip() for x in cell.split(";") if x.strip()})
    if not ids: return ZERO
    d = dict(ZERO); d["has_ti"] = 1; d["n_techniques"] = len(ids)
    known = []
    for i in ids:
        mt = META.get(i)
        if mt is None:                      # try the revoked-by replacement ID
            mt = META.get(ALIAS.get(i, ""))
        if mt is None: unknown[i] = unknown.get(i,0)+1
        else: known.append(mt)
    if not known: return d
    subs = [k[0] for k in known]; tset = set()
    for k in known: tset.update(k[1])
    grp = [k[2] for k in known]; sof = [k[3] for k in known]
    mit = [k[4] for k in known]; age = [k[5] for k in known]; dmod = [k[6] for k in known]
    d["n_subtechniques"] = sum(subs); d["share_subtechniques"] = round(sum(subs)/len(subs),4)
    d["n_tactics"] = len(tset); d["kill_chain_breadth"] = round(len(tset)/len(TACTICS),4)
    for t in tset:
        c = f"ta_{t.replace('-','_')}"
        if c in d: d[c] = 1
    d["group_use_max"] = max(grp); d["group_use_mean"] = round(sum(grp)/len(grp),3)
    d["software_use_max"] = max(sof); d["software_use_mean"] = round(sum(sof)/len(sof),3)
    d["mitigations_mean"] = round(sum(mit)/len(mit),3)
    d["technique_age_min_days"] = min(age); d["technique_age_mean_days"] = round(sum(age)/len(age),1)
    d["days_since_modified_min"] = min(dmod)
    return d

for name, s in samples.items():
    t0 = time.time()
    s = s.copy(); s["k"] = key(s)
    m = agg[src_of[name]].reindex(s["k"]).reset_index(drop=True)
    df = pd.concat([s[["label"]].reset_index(drop=True), m], axis=1)

    ts_min = pd.to_datetime(df.ts_min, format="mixed", utc=True, errors="coerce").dt.tz_localize(None)
    ts_max = pd.to_datetime(df.ts_max, format="mixed", utc=True, errors="coerce").dt.tz_localize(None)
    df["duration_hours"] = ((ts_max - ts_min).dt.total_seconds()/3600).fillna(0).round(3)
    df["first_alert_hour"] = ts_min.dt.hour.fillna(0).astype(int)
    df["first_alert_dayofweek"] = ts_min.dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df.first_alert_dayofweek >= 5).astype(int)
    df["evidence_per_alert"] = (df.n_evidence / df.n_alerts.replace(0, np.nan)).fillna(0).round(3)
    ti_alert_cov = (df.alerts_with_ti / df.n_alerts.replace(0, np.nan)).fillna(0).round(4)

    unknown = {}
    tif = pd.DataFrame([ti_row(c, unknown) for c in df.pop("tech_lists")])
    tif["ti_alert_coverage"] = ti_alert_cov.values
    df = df.drop(columns=["ts_min","ts_max","alerts_with_ti"]).fillna(0)

    A = df
    B = pd.concat([A, tif], axis=1)
    A.to_csv(f"{OUTD}/{name}_A.csv", index=False)
    B.to_csv(f"{OUTD}/{name}_B.csv", index=False)

    man = {"sample": name, "rows": int(len(df)), "features_A": int(A.shape[1]-1), "features_B": int(B.shape[1]-1),
           "label_counts": {int(k): int(v) for k,v in df.label.value_counts().items()},
           "technique_coverage_incident_level": round(float(tif.has_ti.mean()),4),
           "mean_ti_alert_coverage": round(float(tif.ti_alert_coverage.mean()),4),
           "mean_techniques_when_present": round(float(tif.loc[tif.has_ti==1,"n_techniques"].mean() if tif.has_ti.any() else 0),2),
           "unmatched_technique_ids_top20": dict(sorted(unknown.items(), key=lambda x:-x[1])[:20]),
           "n_unmatched_distinct": len(unknown),
           "alias_map_size": len(ALIAS),
           "elapsed_s": round(time.time()-t0,1)}
    json.dump(man, open(f"{OUTD}/{name}_manifest.json","w"), indent=2)
    print(f"{name}: A={man['features_A']} B={man['features_B']} feats | TI coverage {man['technique_coverage_incident_level']:.1%} | {man['elapsed_s']}s")

# ---- align evaluation tables to their training vocabulary (design doc s4) ----
# Composition-count columns are fixed from the training sample of each condition; levels seen
# only in evaluation data are dropped, levels missing there are added as zeros. This keeps the
# feature space identical across a condition without letting evaluation data shape the model.
for cond in ["standard","crossorg","acrosstime"]:
    for fs in ["A","B"]:
        tr_p, ev_p = f"{OUTD}/{cond}_train_{fs}.csv", f"{OUTD}/{cond}_eval_{fs}.csv"
        tr, ev = pd.read_csv(tr_p), pd.read_csv(ev_p)
        added = [c for c in tr.columns if c not in ev.columns]
        dropped = [c for c in ev.columns if c not in tr.columns]
        for c in added: ev[c] = 0
        ev = ev[tr.columns]
        ev.to_csv(ev_p, index=False)
        if added or dropped:
            print(f"aligned {cond}_eval_{fs}: added {added} | dropped {dropped}")
        json.dump({"condition":cond,"feature_set":fs,"columns":list(tr.columns),
                   "added_zero_columns":added,"dropped_unseen_columns":dropped},
                  open(f"{OUTD}/{cond}_{fs}_columns.json","w"), indent=2)
print("done")
