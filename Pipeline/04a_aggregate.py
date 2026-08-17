"""
Step 04a - Aggregate evidence rows for sampled incidents (cached) (Feature_Set_Design_Haris.docx v1).

For each of the six protocol samples, produces two CSVs:
    <sample>_A.csv  - plain alert features only
    <sample>_B.csv  - the same features plus the threat-intelligence block
so that A and B differ ONLY in the TI columns (matched-model design).

Method: one streaming pass per source file (train/test). Evidence rows are filtered
to the union of sampled incidents, aggregated to incident level with pyarrow, then
composition counts are pivoted and joined. TI features come from Dataset/derived/ti_context.csv.

Leakage controls applied here: IncidentGrade, ActionGrouped, ActionGranular, LastVerdict,
OrgId, and raw high-cardinality identifiers are never used as features (see design doc s3).

Usage: python3 Pipeline/04a_aggregate.py
Outputs: Dataset/derived/agg_{train,test}.parquet (cache for step 04b)
"""
import os, json, time, glob
import pyarrow as pa, pyarrow.csv as pv, pyarrow.compute as pc
import pandas as pd, numpy as np

SAMP = "Dataset/derived/samples"
OUTD = "Dataset/derived/features"
os.makedirs(OUTD, exist_ok=True)
REF_DATE = pd.Timestamp("2024-06-17")     # dataset end - used for technique age (no "today" leakage)
TACTICS = ["reconnaissance","resource-development","initial-access","execution","persistence",
           "privilege-escalation","defense-evasion","credential-access","discovery",
           "lateral-movement","collection","command-and-control","exfiltration","impact"]

USE_COLS = ["OrgId","IncidentId","AlertId","Timestamp","DetectorId","Category","MitreTechniques",
            "EntityType","EvidenceRole","SuspicionLevel","AntispamDirection","DeviceId","Sha256",
            "IpAddress","Url","AccountSid","AccountName","ApplicationId","FileName","CountryCode",
            "OSFamily","Roles"]
STR_COLS = ["Timestamp","Category","MitreTechniques","EntityType","EvidenceRole","SuspicionLevel",
            "AntispamDirection","CountryCode","OSFamily","Roles"]

samples = {os.path.basename(p)[:-4]: pd.read_csv(p) for p in sorted(glob.glob(f"{SAMP}/*.csv"))}
src_of = {n: ("test" if n == "standard_eval" else "train") for n in samples}
print("samples:", {n: len(d) for n,d in samples.items()})

ti = pd.read_csv("Dataset/derived/ti_context.csv")
ti["created_dt"] = pd.to_datetime(ti["created"], format="mixed", utc=True).dt.tz_localize(None)
ti["modified_dt"] = pd.to_datetime(ti["modified"], format="mixed", utc=True).dt.tz_localize(None)
ti_by_id = ti.set_index("technique_id")

def key(df): return df.OrgId.astype(str) + "|" + df.IncidentId.astype(str)

def aggregate_source(which, wanted):
    """Stream one raw CSV, keep only wanted incident keys, return joined incident-level frame."""
    SRC = {"train":"Dataset/GUIDE_Train.csv","test":"Dataset/GUIDE_Test.csv"}[which]
    opts = pv.ConvertOptions(include_columns=USE_COLS, column_types={c:"string" for c in STR_COLS})
    main, comp = [], {"Category":[], "EntityType":[], "EvidenceRole":[], "SuspicionLevel":[]}
    tech, cover = [], []
    t0=time.time(); seen=0
    with pv.open_csv(SRC, convert_options=opts, read_options=pv.ReadOptions(block_size=64*1024*1024)) as reader:
        for batch in reader:
            t = pa.Table.from_batches([batch])
            k = pc.binary_join_element_wise(pc.cast(t["OrgId"],pa.string()), pc.cast(t["IncidentId"],pa.string()), "|")
            t = t.append_column("k", k)
            t = t.filter(pc.is_in(t["k"], value_set=pa.array(wanted)))
            if t.num_rows == 0: continue
            seen += t.num_rows
            main.append(t.group_by("k").aggregate([
                ("AlertId","count_distinct"), ("AlertId","count"),
                ("Timestamp","min"), ("Timestamp","max"),
                ("DetectorId","count_distinct"), ("DeviceId","count_distinct"),
                ("AccountSid","count_distinct"), ("AccountName","count_distinct"),
                ("IpAddress","count_distinct"), ("Url","count_distinct"),
                ("Sha256","count_distinct"), ("ApplicationId","count_distinct"),
                ("FileName","count_distinct"), ("CountryCode","count_distinct"),
                ("OSFamily","count_distinct"), ("Roles","count_distinct"),
                ("EntityType","count_distinct"), ("Category","count_distinct")]))
            for c in comp:
                comp[c].append(t.group_by(["k", c]).aggregate([("AlertId","count")]))
            mt = t.filter(pc.and_(pc.is_valid(t["MitreTechniques"]), pc.not_equal(t["MitreTechniques"], "")))
            if mt.num_rows:
                # collapse each batch's distinct technique strings to one ';'-joined string per
                # incident (pyarrow cannot aggregate lists of lists across batches)
                tb = mt.group_by("k").aggregate([("MitreTechniques","distinct")]).to_pandas()
                tb.columns = ["k","tl"]
                tb["tl"] = tb["tl"].apply(lambda a: ";".join(sorted({s for s in (a if a is not None else []) if s})))
                tech.append(tb)
                cover.append(mt.group_by("k").aggregate([("AlertId","count_distinct")]))
    print(f"  {which}: {seen:,} evidence rows kept in {round(time.time()-t0,1)}s")

    m = pa.concat_tables(main).group_by("k").aggregate([
        ("AlertId_count_distinct","sum"), ("AlertId_count","sum"),
        ("Timestamp_min","min"), ("Timestamp_max","max"),
        ("DetectorId_count_distinct","max"), ("DeviceId_count_distinct","max"),
        ("AccountSid_count_distinct","max"), ("AccountName_count_distinct","max"),
        ("IpAddress_count_distinct","max"), ("Url_count_distinct","max"),
        ("Sha256_count_distinct","max"), ("ApplicationId_count_distinct","max"),
        ("FileName_count_distinct","max"), ("CountryCode_count_distinct","max"),
        ("OSFamily_count_distinct","max"), ("Roles_count_distinct","max"),
        ("EntityType_count_distinct","max"), ("Category_count_distinct","max")]).to_pandas()
    m.columns = ["k","n_alerts","n_evidence","ts_min","ts_max","n_detectors","n_devices",
                 "n_accountsids","n_accountnames","n_ips","n_urls","n_sha256","n_apps","n_files",
                 "n_countries","n_os_families","n_roles","n_entity_types","n_categories"]

    for c, parts in comp.items():
        cdf = pa.concat_tables(parts).group_by(["k", c]).aggregate([("AlertId_count","sum")]).to_pandas()
        cdf.columns = ["k","level","cnt"]
        cdf["level"] = c[:4].lower() + "_" + cdf["level"].fillna("na").replace("", "na").astype(str)
        piv = cdf.pivot_table(index="k", columns="level", values="cnt", aggfunc="sum", fill_value=0).reset_index()
        m = m.merge(piv, on="k", how="left")

    if tech:
        td = (pd.concat(tech, ignore_index=True).groupby("k")["tl"]
                .apply(lambda s: ";".join(sorted({x for v in s for x in str(v).split(";") if x})))
                .reset_index())
        td.columns = ["k","tech_lists"]
        cd = pa.concat_tables(cover).group_by("k").aggregate([("AlertId_count_distinct","sum")]).to_pandas()
        cd.columns = ["k","alerts_with_ti"]
        m = m.merge(td, on="k", how="left").merge(cd, on="k", how="left")
    else:
        m["tech_lists"] = None; m["alerts_with_ti"] = 0
    return m

def flatten_techs(cell):
    """';'-joined technique string -> sorted list of distinct technique IDs."""
    if cell is None or isinstance(cell, float) or cell == 0 or not str(cell).strip(): return []
    return sorted({x.strip() for x in str(cell).split(";") if x.strip()})

def ti_features(tech_ids):
    """Threat-intelligence block for one incident (zeros when no techniques present)."""
    base = {"has_ti":0,"n_techniques":0,"n_subtechniques":0,"share_subtechniques":0.0,
            "n_tactics":0,"kill_chain_breadth":0.0,"group_use_max":0,"group_use_mean":0.0,
            "software_use_max":0,"software_use_mean":0.0,"mitigations_mean":0.0,
            "technique_age_min_days":0.0,"technique_age_mean_days":0.0,"days_since_modified_min":0.0}
    base.update({f"ta_{t.replace('-','_')}":0 for t in TACTICS})
    if not tech_ids: return base, []
    known = [t for t in tech_ids if t in ti_by_id.index]
    unknown = [t for t in tech_ids if t not in ti_by_id.index]
    base["has_ti"] = 1; base["n_techniques"] = len(tech_ids)
    if not known: return base, unknown
    r = ti_by_id.loc[known]
    base["n_subtechniques"] = int(r.is_subtechnique.sum())
    base["share_subtechniques"] = round(float(r.is_subtechnique.mean()), 4)
    tset = set()
    for s in r.tactics.fillna(""):
        tset.update(x for x in str(s).split(";") if x)
    base["n_tactics"] = len(tset)
    base["kill_chain_breadth"] = round(len(tset)/len(TACTICS), 4)
    for t in tset:
        kk = f"ta_{t.replace('-','_')}"
        if kk in base: base[kk] = 1
    base["group_use_max"] = int(r.n_groups.max()); base["group_use_mean"] = round(float(r.n_groups.mean()),3)
    base["software_use_max"] = int(r.n_softwares.max()); base["software_use_mean"] = round(float(r.n_softwares.mean()),3)
    base["mitigations_mean"] = round(float(r.n_mitigations.mean()),3)
    age = (REF_DATE - r.created_dt).dt.days
    base["technique_age_min_days"] = float(age.min()); base["technique_age_mean_days"] = round(float(age.mean()),1)
    base["days_since_modified_min"] = float((REF_DATE - r.modified_dt).dt.days.min())
    return base, unknown


# ---- one streaming pass per source over the union of that source's sampled incidents ----
if __name__ == "__main__":
    for which in ["train","test"]:
        names = [n for n,s in src_of.items() if s == which]
        wanted = sorted({k for n in names for k in key(samples[n])})
        print(f"{which}: {len(wanted):,} unique incidents needed across {names}")
        out = aggregate_source(which, wanted)
        out.to_parquet(f"Dataset/derived/agg_{which}.parquet", index=False)
        print("  cached ->", f"Dataset/derived/agg_{which}.parquet", out.shape)
