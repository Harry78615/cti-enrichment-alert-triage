"""
Step 01b - Build the incident index from the slim Parquet (pyarrow-native for speed).
One row per incident (OrgId, IncidentId):
  IncidentGrade - analyst label ("max": any non-empty value wins over empty)
  Category      - deterministic representative category (lexicographic minimum of the
                  incident's evidence categories; used for stratification only)
  first_ts      - earliest alert Timestamp (drives the across-time split)
  n_evidence    - number of evidence rows
Distinct-alert counts / grade-conflict checks are deferred to step 03 (sampled incidents only).
Usage: python3 Pipeline/01b_build_incident_index.py train|test
"""
import sys, time, json
import pyarrow as pa, pyarrow.parquet as pq

which = sys.argv[1]
SRC = f"Dataset/derived/{which}_slim.parquet"
OUT = f"Dataset/derived/{which}_incident_index.parquet"

t0 = time.time()
pf = pq.ParquetFile(SRC)
partials = []
for batch in pf.iter_batches(batch_size=4_000_000, columns=["OrgId","IncidentId","Timestamp","Category","IncidentGrade"]):
    t = pa.Table.from_batches([batch])
    g = t.group_by(["OrgId","IncidentId"]).aggregate([
        ("IncidentGrade","max"), ("Category","min"), ("Timestamp","min"), ("Timestamp","count")])
    partials.append(g)
allp = pa.concat_tables(partials)
idx = allp.group_by(["OrgId","IncidentId"]).aggregate([
    ("IncidentGrade_max","max"), ("Category_min","min"), ("Timestamp_min","min"), ("Timestamp_count","sum")])
idx = idx.rename_columns(["OrgId","IncidentId","IncidentGrade","Category","first_ts","n_evidence"])
pq.write_table(idx, OUT, compression="zstd")

import pyarrow.compute as pc
grades = pc.fill_null(pc.replace_substring_regex(idx["IncidentGrade"], "^$", "MISSING"), "MISSING")
vc = {x["values"]: x["counts"] for x in pc.value_counts(grades).to_pylist()}
m = {"source": SRC, "out": OUT, "incidents": idx.num_rows, "grade_counts": vc,
     "ts_min": pc.min(idx["first_ts"]).as_py(), "ts_max": pc.min_max(idx["first_ts"]).as_py()["max"],
     "elapsed_s": round(time.time()-t0,1)}
json.dump(m, open(OUT.replace(".parquet","_manifest.json"),"w"), indent=2)
print(json.dumps(m, indent=2))
