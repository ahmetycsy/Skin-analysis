"""
A1 (derm) ve A2 (clin) olasiliklarinin duz ortalamasi (late fusion, agirlik/ayar YOK).

Calistirma:
  python fuse_avg.py --derm-dir <A1 zip klasoru> --clin-dir <A2 zip klasoru> [--out .]

Uretir:
  results/avg_derm_clin.json            Macro F1, dogruluk, AUC, A1/A2'ye gore fark ve bootstrap araligi
  results/avg_derm_clin_per_class.csv   sinif bazinda F1: A1, A2, ortalama
  val_probs_avg_derm_clin.npy           ortalama olasiliklar (kontrol seti)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
LABELS = list(range(len(CLASSES)))


def macro(y, p):
    return f1_score(y, p.argmax(1), average="macro", labels=LABELS, zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derm-dir", required=True)
    ap.add_argument("--clin-dir", required=True)
    ap.add_argument("--derm-run", default="derm_efficientnet_b3")
    ap.add_argument("--clin-run", default="clin_efficientnet_b3")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    dd, cd, out = Path(a.derm_dir), Path(a.clin_dir), Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)

    pd_ = np.load(dd / f"val_probs_{a.derm_run}.npy")
    pc = np.load(cd / f"val_probs_{a.clin_run}.npy")
    vd = pd.read_csv(dd / f"val_ids_{a.derm_run}.csv")
    vc = pd.read_csv(cd / f"val_ids_{a.clin_run}.csv")
    assert (vd.lesion_id.values == vc.lesion_id.values).all(), "Iki calistirmanin kontrol seti ayni sirada degil"
    y = vd.label.values
    pf = (pd_ + pc) / 2

    runs = {"A1_derm": pd_, "A2_clin": pc, "A1_A2_avg": pf}
    metrics = {}
    for name, p in runs.items():
        yh = p.argmax(1)
        metrics[name] = {
            "macro_f1": round(float(macro(y, p)), 4),
            "accuracy": round(float((y == yh).mean()), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y, yh)), 4),
            "macro_auc_ovr": round(float(roc_auc_score(np.eye(len(CLASSES))[y], p, average="macro")), 4),
        }

    rng = np.random.default_rng(42)
    d1, d2 = [], []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        f = macro(y[i], pf[i])
        d1.append(f - macro(y[i], pd_[i]))
        d2.append(f - macro(y[i], pc[i]))
    a_ok, c_ok = pd_.argmax(1) == y, pc.argmax(1) == y

    summary = {
        "method": "unweighted mean of A1 and A2 softmax probabilities, argmax; no tuning on validation",
        "n_val": int(len(y)),
        "metrics": metrics,
        "gain_vs_A1": {"mean": round(float(np.mean(d1)), 4),
                       "ci95": [round(float(x), 4) for x in np.percentile(d1, [2.5, 97.5])],
                       "share_of_bootstraps_positive": round(float(np.mean(np.array(d1) > 0)), 3)},
        "gain_vs_A2": {"mean": round(float(np.mean(d2)), 4),
                       "ci95": [round(float(x), 4) for x in np.percentile(d2, [2.5, 97.5])],
                       "share_of_bootstraps_positive": round(float(np.mean(np.array(d2) > 0)), 3)},
        "agreement": {"both_correct": round(float((a_ok & c_ok).mean()), 4),
                      "only_A1_correct": round(float((a_ok & ~c_ok).mean()), 4),
                      "only_A2_correct": round(float((~a_ok & c_ok).mean()), 4),
                      "both_wrong": round(float((~a_ok & ~c_ok).mean()), 4)},
        "mel_predicted_as_nv": {n: int(((y == 7) & (p.argmax(1) == 8)).sum()) for n, p in runs.items()},
    }

    per_class = pd.DataFrame({n: np.round(f1_score(y, p.argmax(1), average=None, labels=LABELS, zero_division=0), 4)
                              for n, p in runs.items()}, index=CLASSES)
    per_class["support"] = np.bincount(y, minlength=len(CLASSES))
    per_class.to_csv(out / "results" / "avg_derm_clin_per_class.csv")
    np.save(out / "val_probs_avg_derm_clin.npy", pf)
    with open(out / "results" / "avg_derm_clin.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(per_class.to_string())


if __name__ == "__main__":
    main()
