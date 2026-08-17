"""
Step 10 - Pipeline diagram for the methodology chapter.
Usage: python3 Pipeline/10_pipeline_diagram.py -> Figures/fig0_pipeline.png
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"figure.dpi":300, "font.family":"DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0,100); ax.set_ylim(0,60); ax.axis("off")

C = {"data":"#DCE6F1", "proc":"#FCE4D6", "model":"#E2EFDA", "out":"#E4DFEC"}
E = {"data":"#2E5C8A", "proc":"#C55A11", "model":"#548235", "out":"#7030A0"}

def box(x, y, w, h, text, kind, fs=7.5):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle="round,pad=0.4",
                 fc=C[kind], ec=E[kind], lw=1.1))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)

def arrow(x1,y1,x2,y2, label=None, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle=style, mutation_scale=11,
                 lw=0.9, color="#666666", connectionstyle="arc3,rad=0"))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+1.1, label, ha="center", fontsize=6, color="#444444", style="italic")

# Row 1 - sources
box(2, 48, 20, 9, "GUIDE dataset\n9.5M evidence rows\n731k incidents, 5,769 orgs", "data")
box(2, 33, 20, 9, "MITRE ATT&CK\nEnterprise STIX bundle\n697 techniques", "data")

# Row 2 - preparation
box(28, 48, 19, 9, "01a/01b  Incident index\naggregate evidence rows\nto one row per incident", "proc")
box(28, 33, 19, 9, "03  Technique context\ntactics, groups, software,\nmitigations, age, aliases", "proc")

# Row 3 - sampling and features
box(53, 48, 19, 9, "02  Protocol samples\nseed 42, stratified,\n5% org cap", "proc")
box(53, 33, 19, 9, "04a/04b  Feature build\nA = 73 alert features\nB = A + 29 TI features", "proc")

# Conditions band
box(53, 20, 19, 8, "Three conditions\nstandard · cross-org\n· across-time", "proc", fs=7)

# Models
box(78, 44, 19, 8, "05  Random Forest\n+ Gradient Boosting\nA vs B, matched", "model")
box(78, 32, 19, 8, "06  Subgroup\nincidents carrying\ntechniques", "model")
box(78, 20, 19, 8, "07  TreeSHAP\n+ stability\n+ permutation", "model")

# Outputs
box(78, 6, 19, 9, "Results\nRQ1 coverage · RQ2 effectiveness\nRQ3 cost · RQ4 features", "out")
box(53, 6, 19, 9, "Orange workflow\nvisual demonstration\nof the deliverable", "out")

arrow(22,52.5, 28,52.5); arrow(22,37.5, 28,37.5)
arrow(47,52.5, 53,52.5); arrow(47,37.5, 53,37.5)
arrow(62.5,48, 62.5,42, "sampled incidents")
arrow(62.5,33, 62.5,28)
arrow(72,52.5, 78,48)
arrow(72,37.5, 78,36)
arrow(72,24, 78,24)
arrow(87.5,44, 87.5,40)
arrow(87.5,32, 87.5,28)
arrow(87.5,20, 87.5,15)
arrow(62.5,33, 62.5,15, style="-|>")

ax.text(50, 58.5, "Figure: Study pipeline from raw data to research answers",
        ha="center", fontsize=9.5, weight="bold")
ax.text(2, 1.5, "Every step writes a JSON manifest (inputs, row counts, seed, class proportions, hashes). "
                "Seed 42 throughout; hyperparameters fixed a priori.",
        fontsize=6.5, color="#555555", style="italic")
plt.tight_layout(); plt.savefig("Figures/fig0_pipeline.png", bbox_inches="tight")
print("written Figures/fig0_pipeline.png")
