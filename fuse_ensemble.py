"""
Onceden sabitlenmis birlestirme kurali (kontrol setine bakip agirlik SECILMEZ):
  1) Her goruntu turunun modelleri kendi icinde ortalanir   (derm = ort(derm modelleri), clin = ort(clin modelleri))
  2) Sonra derm ve clin esit agirlikla ortalanir            (p = (derm + clin) / 2)

Calistirma:
  python fuse_ensemble.py --derm <klasor>:<run> [<klasor>:<run> ...] --clin <klasor>:<run> [...] \
      [--name ensemble_b3_v2] [--out .] [--meta MILK10k_Training_Metadata.csv] [--archive milk10k.csv]

Uretir:
  results/<name>.json             Macro F1, dogruluk, AUC, melanom ve malign olcutleri, bootstrap farklar
  results/<name>_per_class.csv    sinif bazinda F1 (her model, her tur ortalamasi, nihai topluluk)
  figures/<name>_per_class.png    sinif bazinda F1 karsilastirmasi
  val_probs_<name>.npy            nihai olasiliklar (kontrol seti)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
K, LABELS = len(CLASSES), list(range(11))
MEL, NV = CLASSES.index("MEL"), CLASSES.index("NV")
MAL_IDX = [CLASSES.index(c) for c in ["BCC", "MAL_OTH", "MEL", "SCCKA"]]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLS = ["#2a78d6", "#eb6834", "#1baf7a"]


def macro(y, p):
    return f1_score(y, p.argmax(1), average="macro", labels=LABELS, zero_division=0)


def load(spec):
    d, run = spec.split(":")
    p = np.load(Path(d) / f"val_probs_{run}.npy")
    v = pd.read_csv(Path(d) / f"val_ids_{run}.csv")
    return run, p, v


def metrics(y, p, mal=None):
    yh = p.argmax(1)
    mel = y == MEL
    mm = np.isin(y, [MEL, NV])
    out = {
        "macro_f1": round(float(macro(y, p)), 4),
        "accuracy": round(float((yh == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, yh)), 4),
        "macro_auc_ovr": round(float(roc_auc_score(np.eye(K)[y], p, average="macro")), 4),
        "mel_recall": round(float((yh[mel] == MEL).mean()), 4),
        "mel_precision": round(float((y[yh == MEL] == MEL).mean()), 4),
        "mel_predicted_as_nv": int((mel & (yh == NV)).sum()),
        "mel_vs_nv_auc": round(float(roc_auc_score(y[mm] == MEL, p[mm, MEL] / (p[mm, MEL] + p[mm, NV]))), 4),
    }
    if mal is not None:
        out["malignant_vs_rest_auc"] = round(float(roc_auc_score(mal, p[:, MAL_IDX].sum(1))), 4)
    return out


def boot_gain(y, pa, pb, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(macro(y[i], pa[i]) - macro(y[i], pb[i]))
    d = np.array(d)
    return {"mean": round(float(d.mean()), 4), "ci95": [round(float(x), 4) for x in np.percentile(d, [2.5, 97.5])],
            "share_positive": round(float((d > 0).mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derm", nargs="+", required=True)
    ap.add_argument("--clin", nargs="+", required=True)
    ap.add_argument("--name", default="ensemble")
    ap.add_argument("--archive", default=None, help="ISIC archive metadata (diagnosis_1) -> malign AUC")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    runs = {"derm": [load(s) for s in a.derm], "clin": [load(s) for s in a.clin]}
    ref = runs["derm"][0][2]
    for lst in runs.values():
        for _, _, v in lst:
            assert (v.lesion_id.values == ref.lesion_id.values).all(), "kontrol setleri ayni sirada degil"
    y = ref.label.values
    mal = None
    if a.archive:
        arc = pd.read_csv(a.archive)
        d1 = arc[arc.image_type == "dermoscopic"].set_index("lesion_id").diagnosis_1
        mal = (d1.reindex(ref.lesion_id).values == "Malignant")

    single = {run: p for lst in runs.values() for run, p, _ in lst}
    derm = np.mean([p for _, p, _ in runs["derm"]], axis=0)
    clin = np.mean([p for _, p, _ in runs["clin"]], axis=0)
    final = (derm + clin) / 2

    table = {**{r: metrics(y, p, mal) for r, p in single.items()},
             "derm_mean": metrics(y, derm, mal), "clin_mean": metrics(y, clin, mal),
             a.name: metrics(y, final, mal)}
    first_d, first_c = runs["derm"][0][1], runs["clin"][0][1]
    summary = {
        "rule": "mean within image type, then equal-weight mean of derm and clin; no weights tuned",
        "derm_models": [r for r, _, _ in runs["derm"]], "clin_models": [r for r, _, _ in runs["clin"]],
        "metrics": table,
        "gain_vs_first_derm_model": boot_gain(y, final, first_d),
        "gain_vs_first_pair_average": boot_gain(y, final, (first_d + first_c) / 2),
    }
    with open(out / "results" / f"{a.name}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    np.save(out / f"val_probs_{a.name}.npy", final)

    per = pd.DataFrame({n: np.round(f1_score(y, p.argmax(1), average=None, labels=LABELS, zero_division=0), 4)
                        for n, p in {**single, "derm_mean": derm, "clin_mean": clin, a.name: final}.items()},
                       index=CLASSES)
    per["support"] = np.bincount(y, minlength=K)
    per.to_csv(out / "results" / f"{a.name}_per_class.csv")

    show = {"Dermoscopic models (mean)": derm, "Clinical models (mean)": clin, "Final ensemble": final}
    order = np.argsort(-per["support"].values)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    w, x = 0.26, np.arange(K)
    for k, (n, p) in enumerate(show.items()):
        f = f1_score(y, p.argmax(1), average=None, labels=LABELS, zero_division=0)
        ax.bar(x + (k - 1) * w, f[order], w * 0.92, color=COLS[k], label=f"{n} (Macro F1 {f.mean():.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLASSES[i]}\nn={per['support'].values[i]}" for i in order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1", color=INK2)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("Per-class F1 on the validation set (n = 1,048 lesions)", loc="left", fontweight="bold", pad=20)
    ax.text(0, 1.01, f"Derm: {', '.join(summary['derm_models'])}  |  Clin: {', '.join(summary['clin_models'])}",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    fig.savefig(out / "figures" / f"{a.name}_per_class.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(pd.DataFrame(table).T.to_string())
    print(json.dumps({k: summary[k] for k in ["gain_vs_first_derm_model", "gain_vs_first_pair_average"]}, indent=1))


if __name__ == "__main__":
    main()
