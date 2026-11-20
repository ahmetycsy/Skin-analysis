"""
Kaggle'dan inen bir calistirmanin (derm / clin) sonuclarini analiz eder.

Calistirma:
  python analyze_run.py --run-dir <zip'in acildigi klasor> --run derm_efficientnet_b3 \
      [--meta MILK10k_Training_Metadata.csv] [--out .]

Uretir:
  figures/cm_<run>.png             satir-normalize karisiklik matrisi (hucrede adet + satir %)
  results/<run>_per_class.csv      sinif bazinda precision / recall / F1 / destek
  results/<run>_summary.json       Macro F1, %95 bootstrap araligi, macro AUC, (varsa) ten gruplarina gore sonuc
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
TONE_GROUPS = {"0-2 (darker)": (0, 2), "3": (3, 3), "4-5 (lighter)": (4, 5)}
INK, INK2 = "#0b0b0b", "#52514e"
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.titlelocation": "left", "axes.titleweight": "bold",
                     "axes.titlepad": 20})


def plot_cm(y, yh, run, macro, out):
    cm = confusion_matrix(y, yh, labels=range(len(CLASSES)))
    rown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8.4))
    ax.imshow(rown, cmap=CMAP, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{cm[i, j]}\n{rown[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="white" if rown[i, j] > 0.55 else INK)
    n = cm.sum(axis=1)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([f"{c} (n={k})" for c, k in zip(CLASSES, n)])
    ax.set_xlabel("Predicted class", color=INK2)
    ax.set_ylabel("True class", color=INK2)
    ax.set_xticks(np.arange(-.5, len(CLASSES)), minor=True)
    ax.set_yticks(np.arange(-.5, len(CLASSES)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Confusion matrix: {run} (validation, n = {len(y):,})")
    ax.text(0, 1.01, f"Cell = lesions and row % (recall on the diagonal). Macro F1 = {macro:.3f}",
            transform=ax.transAxes, color=INK2, fontsize=9)
    fig.savefig(out / f"cm_{run}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    rd, out = Path(a.run_dir), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    probs = np.load(rd / f"val_probs_{a.run}.npy")
    val = pd.read_csv(rd / f"val_ids_{a.run}.csv")
    y, yh = val.label.values, probs.argmax(1)
    labels = list(range(len(CLASSES)))

    macro = f1_score(y, yh, average="macro", labels=labels, zero_division=0)
    pr, rc, f1, sup = precision_recall_fscore_support(y, yh, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": CLASSES, "precision": pr.round(4), "recall": rc.round(4),
                              "f1": f1.round(4), "support": sup})
    per_class.to_csv(out / "results" / f"{a.run}_per_class.csv", index=False)

    rng = np.random.default_rng(42)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        boot.append(f1_score(y[i], yh[i], average="macro", labels=labels, zero_division=0))

    summary = {
        "run": a.run,
        "n_val": int(len(y)),
        "macro_f1": round(float(macro), 4),
        "macro_f1_ci95_bootstrap": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
        "macro_f1_without_MAL_OTH": round(float(np.delete(f1, CLASSES.index("MAL_OTH")).mean()), 4),
        "accuracy": round(float((y == yh).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, yh)), 4),
        "macro_auc_ovr": round(float(roc_auc_score(np.eye(len(CLASSES))[y], probs, average="macro")), 4),
        "mel_predicted_as_nv": int(((y == CLASSES.index("MEL")) & (yh == CLASSES.index("NV"))).sum()),
        "mel_support": int((y == CLASSES.index("MEL")).sum()),
    }

    if a.meta:
        tone = pd.read_csv(a.meta).groupby("lesion_id")["skin_tone_class"].first()
        val["tone"] = val.lesion_id.map(tone)
        val["pred"] = yh
        rows = {}
        for g, (lo, hi) in TONE_GROUPS.items():
            d = val[val.tone.between(lo, hi)]
            rows[g] = {"n": int(len(d)), "accuracy": round(float((d.label == d.pred).mean()), 4),
                       "balanced_accuracy": round(float(balanced_accuracy_score(d.label, d.pred)), 4)}
        summary["by_skin_tone_group"] = rows

    with open(out / "results" / f"{a.run}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    plot_cm(y, yh, a.run, macro, out / "figures")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(per_class.to_string(index=False))


if __name__ == "__main__":
    main()
