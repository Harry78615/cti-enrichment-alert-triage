"""
Step 03 - Extract the technique-context table from OpenCTI (or the MITRE ATT&CK
STIX bundle as offline-equivalent fallback).
Output: Dataset/derived/ti_context.csv - one row per ATT&CK technique ID with:
  technique_id, name, is_subtechnique, tactics (semicolon list),
  n_groups (threat groups using it), n_software (malware/tools implementing it),
  n_mitigations, created, modified
Provenance of the run (opencti or stix_bundle) is recorded in the manifest - report in RQ1.

Usage (preferred - your OpenCTI must be running with the MITRE connector synced):
    OPENCTI_URL=http://localhost:8080 OPENCTI_TOKEN=<your token> \
    python3 Pipeline/03_extract_ti_context.py opencti

Usage (fallback - download enterprise-attack.json first from
       https://github.com/mitre-attack/attack-stix-data , folder enterprise-attack):
    python3 Pipeline/03_extract_ti_context.py stix Dataset/enterprise-attack.json
"""
import sys, json, time, csv, os
from collections import defaultdict

mode = sys.argv[1] if len(sys.argv) > 1 else "stix"
t0 = time.time()
rows = {}

if mode == "opencti":
    from pycti import OpenCTIApiClient  # pip install pycti
    api = OpenCTIApiClient(os.environ["OPENCTI_URL"], os.environ["OPENCTI_TOKEN"])
    aps = api.attack_pattern.list(getAll=True, withPagination=False)
    for ap in aps:
        tid = ap.get("x_mitre_id")
        if not tid: continue
        rows[tid] = {
            "technique_id": tid, "name": ap.get("name",""),
            "is_subtechnique": int("." in tid),
            "tactics": ";".join(sorted({kc.get("phase_name","") for kc in ap.get("killChainPhases",[]) or []})),
            "n_groups": 0, "n_softwares": 0, "n_mitigations": 0,
            "created": ap.get("created",""), "modified": ap.get("modified",""),
            "_internal_id": ap.get("id"),
        }
    # relationship counts per technique
    for tid, r in rows.items():
        rels = api.stix_core_relationship.list(toId=r["_internal_id"], relationship_type="uses", getAll=True)
        for rel in rels:
            ft = (rel.get("from") or {}).get("entity_type","")
            if ft == "Intrusion-Set": r["n_groups"] += 1
            elif ft in ("Malware","Tool"): r["n_softwares"] += 1
        mits = api.stix_core_relationship.list(toId=r["_internal_id"], relationship_type="mitigates", getAll=True)
        r["n_mitigations"] = len(mits)
        del_key = r.pop("_internal_id", None)
else:
    bundle_path = sys.argv[2]
    objs = json.load(open(bundle_path))["objects"]
    id2tid = {}
    for o in objs:
        if o.get("type") == "attack-pattern" and not o.get("revoked") and not o.get("x_mitre_deprecated"):
            tid = next((ref["external_id"] for ref in o.get("external_references",[]) if ref.get("source_name")=="mitre-attack"), None)
            if not tid: continue
            id2tid[o["id"]] = tid
            rows[tid] = {"technique_id": tid, "name": o.get("name",""),
                "is_subtechnique": int(o.get("x_mitre_is_subtechnique", False)),
                "tactics": ";".join(sorted({kc["phase_name"] for kc in o.get("kill_chain_phases",[]) if kc.get("kill_chain_name")=="mitre-attack"})),
                "n_groups": 0, "n_softwares": 0, "n_mitigations": 0,
                "created": o.get("created",""), "modified": o.get("modified","")}
    src_type = {}
    for o in objs:
        if o.get("type") in ("intrusion-set","malware","tool","course-of-action"):
            src_type[o["id"]] = o["type"]
    for o in objs:
        if o.get("type") != "relationship" or o.get("revoked"): continue
        tgt = o.get("target_ref",""); tid = id2tid.get(tgt)
        if not tid: continue
        st = src_type.get(o.get("source_ref",""))
        if o.get("relationship_type") == "uses":
            if st == "intrusion-set": rows[tid]["n_groups"] += 1
            elif st in ("malware","tool"): rows[tid]["n_softwares"] += 1
        elif o.get("relationship_type") == "mitigates" and st == "course-of-action":
            rows[tid]["n_mitigations"] += 1

# Revoked/superseded technique IDs -> their replacement, so that historical IDs appearing in
# alert data (GUIDE spans 2023-24) can still be matched against the current knowledge base.
alias = {}
if mode != "opencti":
    id2ext = {}
    for o in objs:
        if o.get("type") == "attack-pattern":
            e = next((r["external_id"] for r in o.get("external_references",[]) if r.get("source_name")=="mitre-attack"), None)
            if e: id2ext[o["id"]] = e
    for o in objs:
        if o.get("type") == "relationship" and o.get("relationship_type") == "revoked-by":
            a, b = id2ext.get(o.get("source_ref")), id2ext.get(o.get("target_ref"))
            if a and b and a != b: alias[a] = b
    with open("Dataset/derived/ti_alias.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["old_id","new_id"])
        for a in sorted(alias): w.writerow([a, alias[a]])

out = "Dataset/derived/ti_context.csv"
with open(out,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["technique_id","name","is_subtechnique","tactics","n_groups","n_softwares","n_mitigations","created","modified"])
    w.writeheader()
    for tid in sorted(rows): w.writerow(rows[tid])
m = {"mode": mode, "techniques": len(rows), "revoked_aliases": len(alias), "out": out, "elapsed_s": round(time.time()-t0,1),
     "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
json.dump(m, open("Dataset/derived/ti_context_manifest.json","w"), indent=2)
print(json.dumps(m, indent=2))
