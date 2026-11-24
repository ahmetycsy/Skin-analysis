import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT = "/Users/ahmet/Desktop/vlaude code"; sys.path.insert(0, ROOT)
from predictor import CLASSES
OUT = os.path.join(os.path.dirname(__file__), "figs")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
TEAL, AMB, BLU, GRY, RED = "#0F766E", "#B0701A", "#5B6EAE", "#8A9BA3", "#A63D57"

# 1) architecture
fig, ax = plt.subplots(figsize=(10, 6.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
def box(x, y, w, h, title, lines, fc="#FFFFFF", ec="#C9D8D6", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + 1.2, y + h - 1.4, title, fontsize=9, weight="bold", color="#1F3A44", va="top")
    for i, l in enumerate(lines):
        ax.text(x + 1.2, y + h - 4.6 - i * 2.6, l, fontsize=7.5, color="#4A5F67", va="top")
def arrow(x1, y1, x2, y2, ls="-"):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#5F747C", lw=1.2, ls=ls))
box(1, 30, 15, 9, "User", ["Physician or patient", "browser / phone"])
ax.add_patch(FancyBboxPatch((21, 15), 50, 49, boxstyle="round,pad=0.4,rounding_size=1.5", fc="none", ec=TEAL, lw=1.3, ls="--"))
ax.text(22.5, 62.3, "STREAMLIT APPLICATION (Streamlit Community Cloud)", fontsize=8, color=TEAL, weight="bold")
steps = [("1 Input & pre-checks", ["Clinical + dermoscopic image (clinical only in patient mode)", "Crop, rotate; sharpness, lighting, skin & scope checks"]),
         ("2 Prediction engine", ["4 EfficientNets (B3 + V2-S; clinical & dermoscopic), flip TTA", "Temperature scaling; Grad-CAM heat map"]),
         ("3 Decision layer", ["Referral thresholds (priority 30%, review 20%), melanoma rule", "Uncertainty flag; 90% conformal prediction set"]),
         ("4 Result, report & follow-up", ["Routing, PDF + QR, case tracking, physician review", "Calm patient-mode view, reminders, SUS survey"])]
for i, (t, l) in enumerate(steps):
    y = 49 - i * 11
    box(23, y, 46, 9, t, l)
    if i < 3: arrow(46, y - 0.4, 46, y - 1.6)
box(76, 42, 23, 11, "Hugging Face Hub", ["Private model repository", "Weights + model_config.json", "Downloaded at start-up"], fc="#E7F2F0", ec="#C5DFDA")
box(76, 20, 23, 14, "Supabase", ["PostgreSQL database", "Anonymous results (consent)", "Audit log, SUS survey", "No images / personal data"], fc="#E7F2F0", ec="#C5DFDA")
arrow(16, 35, 22.5, 52); arrow(75.5, 47, 69.8, 45); arrow(69.8, 22, 75.5, 26)
box(1, 1, 28, 10, "Data (offline)", ["MILK10k (5,240 lesions, primary)", "+ PAD-UFES-20, + ISIC 2019"], fc="#F6F8F8", ls="--")
box(34, 1, 29, 10, "Training (Kaggle T4 GPU)", ["PyTorch + timm, 12 epochs", "Split seed 42: 4,192 / 1,048"], fc="#F6F8F8", ls="--")
box(68, 1, 31, 10, "Evaluation (offline)", ["1,048-lesion validation, ablation,", "equity, calibration, ISIC hidden test"], fc="#F6F8F8", ls="--")
arrow(29.5, 6, 33.5, 6); arrow(63.5, 6, 67.5, 6); arrow(60, 11.6, 84, 41.5, ls="--")
fig.savefig(f"{OUT}/fig_architecture.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 2) model versions
vers = ["v1\nMILK10k only", "v2\n+ PAD-UFES-20", "v3\n+ ISIC 2019"]
ser = {"Physician mode (validation)": [0.546, 0.535, 0.594], "Patient mode (validation)": [0.484, 0.496, 0.496],
       "Phone-photo test (PAD-UFES-20)": [0.399, 0.668, 0.668], "ISIC hidden test (official)": [0.485, None, 0.483]}
fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = np.arange(3)
for (k, v), c, m in zip(ser.items(), [TEAL, AMB, BLU, RED], ["o", "s", "^", "D"]):
    xs = [i for i, a in enumerate(v) if a is not None]; ys = [a for a in v if a is not None]
    ax.plot(xs, ys, marker=m, color=c, lw=2, label=k)
    off = (0, -13) if "ISIC" in k else (0, 6)
    for xx, yy in zip(xs, ys): ax.annotate(f"{yy:.3f}", (xx, yy), textcoords="offset points", xytext=off, ha="center", fontsize=7.5, color=c)
ax.axhline(0.45, color=GRY, ls=":", lw=1); ax.text(1.0, 0.443, "proposal target 0.45", fontsize=7.5, color=GRY, va="top", ha="center")
ax.set_xticks(x); ax.set_xticklabels(vers); ax.set_ylabel("Macro F1"); ax.set_ylim(0.35, 0.72); ax.set_xlim(-0.3, 2.3)
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
fig.savefig(f"{OUT}/fig_versions.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 3) confusion matrix (physician, current)
d = json.load(open(f"{ROOT}/assets/eval_summary.json")); C = d["classes"]
cm = np.array(d["modes"]["h"]["cm"], float); cmn = cm / cm.sum(1, keepdims=True).clip(1)
fig, ax = plt.subplots(figsize=(6.2, 5.2)); im = ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=1)
for i in range(len(C)):
    for j in range(len(C)):
        if cm[i, j] > 0: ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if cmn[i, j] > .55 else "#1F3A44")
ax.set_xticks(range(len(C))); ax.set_xticklabels(C, rotation=60, fontsize=7.5); ax.set_yticks(range(len(C))); ax.set_yticklabels(C, fontsize=7.5)
ax.set_xlabel("Predicted (most likely type)"); ax.set_ylabel("True type"); fig.colorbar(im, ax=ax, fraction=0.046, label="Row-normalised share")
fig.savefig(f"{OUT}/fig_confusion.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 4) skin tone equity
tones = ["0–1\nvery dark/dark", "2\nmedium-dark", "3\nmedium", "4\nlight", "5\nvery light"]
n = [19, 113, 614, 223, 79]; f1 = [None, 0.594, 0.623, 0.507, 0.371]; acc = [None, 0.823, 0.813, 0.762, 0.785]; malref = [None, 0.953, 0.988, 0.987, 1.0]
fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = np.arange(5); w = 0.26
for k, (vals, c, lab) in enumerate(((f1, TEAL, "Macro F1"), (acc, AMB, "Accuracy"), (malref, BLU, "Malignant referred (sensitivity)"))):
    xs = [i for i, v in enumerate(vals) if v is not None]
    ax.bar(np.array(xs) + (k - 1) * w, [vals[i] for i in xs], w, color=c, label=lab)
ax.text(0, 0.3, "n = 19\ntoo few to\nmeasure", ha="center", fontsize=7.5, color=GRY)
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n(n={m})" for t, m in zip(tones, n)], fontsize=7.5); ax.set_ylim(0, 1.08)
ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13)); ax.set_ylabel("Score")
fig.savefig(f"{OUT}/fig_equity.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# 5) Gantt (status colours)
rows = [("WP1 Project management", "P", "Y"), ("WP2 Data acquisition & EDA", "D", "G"), ("WP2 Preprocessing & split", "D", "G"),
        ("WP3 Single-modality baselines", "D", "G"), ("WP4 Image ensemble + calibration", "D", "G"), ("WP4 Metadata fusion / ablation", "D", "G"),
        ("WP4 Extra data (PAD-UFES-20, ISIC 2019)", "D", "G"), ("WP5 Decision-support interface", "D", "G"), ("WP5 Usability testing (SUS)", "P", "Y"),
        ("WP6 Evaluation, equity, ISIC submission", "D", "G"), ("WP6 Reproducibility test", "P", "Y"), ("WP6 External validation (planned)", "P", "B"),
        ("WP6 Final report, poster, presentation", "P", "B")]
span = {"WP1 Project management": (1, 15), "WP2 Data acquisition & EDA": (1, 3), "WP2 Preprocessing & split": (2, 4), "WP3 Single-modality baselines": (4, 6),
        "WP4 Image ensemble + calibration": (6, 9), "WP4 Metadata fusion / ablation": (7, 10), "WP4 Extra data (PAD-UFES-20, ISIC 2019)": (9, 11),
        "WP5 Decision-support interface": (8, 12), "WP5 Usability testing (SUS)": (11, 14), "WP6 Evaluation, equity, ISIC submission": (10, 12),
        "WP6 Reproducibility test": (11, 12), "WP6 External validation (planned)": (12, 14), "WP6 Final report, poster, presentation": (13, 15)}
col = {"G": "#3C9D5D", "Y": "#E3B341", "R": "#C0392B", "B": "#4A78C2"}
fig, ax = plt.subplots(figsize=(9, 4.6))
for i, (name, _, st) in enumerate(rows):
    a, b = span[name]; ax.barh(i, b - a + 1, left=a - 0.5, color=col[st], height=0.6)
ax.axvline(11.5, color="#1F3A44", ls="--", lw=1); ax.text(11.6, -0.9, "now (progress report)", fontsize=7.5)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=8); ax.invert_yaxis()
ax.set_xticks(range(1, 16)); ax.set_xticklabels([f"W{i}" for i in range(1, 16)], fontsize=7.5); ax.set_xlim(0.5, 15.5)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col[k], label=l) for k, l in (("G", "Completed"), ("Y", "In progress"), ("R", "Delayed"), ("B", "Planned"))],
          fontsize=7.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.2))
fig.savefig(f"{OUT}/fig_gantt.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok", os.listdir(OUT))
