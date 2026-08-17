"""
Step 09 - Orange-ready demo files.

Orange is used for the visual workflow demonstration of the deliverable (the scripted
runs in step 05 remain the primary results). Two adjustments are made here:
  1. Size: a seeded subset (20k train / 10k evaluation) so the workflow stays responsive
     on a 16 GB machine. Class proportions are preserved by stratified sampling.
  2. Label: written as text ("NeedsInvestigation" / "Deprioritise") so Orange infers a
     categorical target automatically instead of a numeric one.
Nothing else changes: same features, same seed, same protocol.
Usage: python3 Pipeline/09_make_orange_files.py
Outputs: Dataset/derived/orange/*.csv
"""
import os, json
import pandas as pd
SEED, NTR, NEV = 42, 20000, 10000
OUT = "Dataset/derived/orange"; os.makedirs(OUT, exist_ok=True)
man = {}
for cond in ["standard"]:
    for fs in ["A", "B"]:
        for role, n in [("train", NTR), ("eval", NEV)]:
            df = pd.read_csv(f"Dataset/derived/features/{cond}_{role}_{fs}.csv")
            s = (df.groupby("label", group_keys=False)
                   .apply(lambda g: g.sample(n=round(len(g)/len(df)*n), random_state=SEED)))
            s = s.sample(frac=1, random_state=SEED)           # shuffle
            s.insert(0, "class", s.pop("label").map({1: "NeedsInvestigation", 0: "Deprioritise"}))
            p = f"{OUT}/{cond}_{role}_{fs}_orange.csv"
            s.to_csv(p, index=False)
            man[os.path.basename(p)] = {"rows": len(s), "features": s.shape[1]-1,
                                        "class_counts": s["class"].value_counts().to_dict()}
            print(f"{os.path.basename(p)}: {len(s)} rows, {s.shape[1]-1} features")
json.dump({"seed": SEED, "note": "Stratified demo subsets for the Orange workflow; "
           "primary results come from Pipeline/05.", "files": man},
          open(f"{OUT}/manifest.json","w"), indent=2)
