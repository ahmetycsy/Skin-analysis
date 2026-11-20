"""
Arayuz icin models/model_config.json uretir (hekim modu + hasta modu).

  hekim modu : dermoskopik + klinik goruntu. Her goruntu turunun modelleri kendi icinde ortalanir,
               sonra derm ve klinik esit agirlikla ortalanir.
  hasta modu : yalnizca klinik goruntu. Klinik modellerin ortalamasi.

Her modun kendi kalibrasyon sicakligi (T) ve conformal esigi vardir. Bunlar kontrol setindeki
olasiliklardan hesaplanir. 5 katli CV bitince ayni script, 5.240 lezyonluk out-of-fold olasiliklarla
tekrar calistirilir; arayuz kodu degismez.

Calistirma:
  python build_config.py --run derm=<klasor>:derm_efficientnet_b3 --run derm=<klasor>:derm_tf_efficientnetv2_s \
                         --run clin=<klasor>:clin_efficientnet_b3 --run clin=<klasor>:clin_tf_efficientnetv2_s \
                         --gradcam-derm derm_efficientnet_b3 --gradcam-clin clin_efficientnet_b3 \
                         --out models/model_config.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
MALIGNANT = ["BCC", "MAL_OTH", "MEL", "SCCKA"]
PREMALIGNANT = ["AKIEC"]
BENIGN = ["BEN_OTH", "BKL", "DF", "INF", "NV", "VASC"]


def fit_temperature(p, y):
    logit = np.log(p + 1e-9)
    Ts = np.exp(np.linspace(np.log(0.3), np.log(5), 200))
    best, bestT = np.inf, 1.0
    for T in Ts:
        z = logit / T
        z = z - z.max(1, keepdims=True)
        q = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        nll = -np.log(q[np.arange(len(y)), y] + 1e-12).mean()
        if nll < best:
            best, bestT = nll, T
    return float(bestT)


def apply_T(p, T):
    z = np.log(p + 1e-9) / T
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def mode_stats(p, y, alpha):
    T = fit_temperature(p, y)
    pT = apply_T(p, T)
    n = len(y)
    s = 1 - pT[np.arange(n), y]
    q = float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
    sets = pT >= 1 - q
    return {
        "temperature": round(T, 4),
        "conformal_include_if_prob_at_least": round(1 - q, 6),
        "validation": {
            "n": int(n),
            "macro_f1": round(float(f1_score(y, p.argmax(1), average="macro", labels=range(len(CLASSES)),
                                             zero_division=0)), 4),
            "accuracy": round(float((p.argmax(1) == y).mean()), 4),
            "conformal_coverage": round(float(sets[np.arange(n), y].mean()), 4),
            "conformal_mean_set_size": round(float(sets.sum(1).mean()), 3),
        },
    }


def model_entry(run_dir, run):
    import timm
    res = json.load(open(Path(run_dir) / "results" / f"{run}.json", encoding="utf-8"))
    cfg = timm.create_model(res["model"], pretrained=False).pretrained_cfg
    return {
        "file": f"{run}.pt",
        "timm_name": res["model"],
        "img_size": int(res["img_size"]),
        "mean": [float(v) for v in cfg.get("mean", (0.485, 0.456, 0.406))],
        "std": [float(v) for v in cfg.get("std", (0.229, 0.224, 0.225))],
        "tta_hflip": True,
        "val_macro_f1": res["val_macro_f1"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="derm=<klasor>:<run> veya clin=<klasor>:<run>")
    ap.add_argument("--gradcam-derm", required=True)
    ap.add_argument("--gradcam-clin", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--cost-ratio", type=float, default=1.0)
    ap.add_argument("--version", default="EfficientNet-B3 + EfficientNetV2-S (4 model)")
    ap.add_argument("--out", default="models/model_config.json")
    a = ap.parse_args()

    models, members, probs, ref = {}, {"derm": [], "clin": []}, {}, None
    for spec in a.run:
        kind, rest = spec.split("=", 1)
        d, run = rest.split(":")
        models[run] = model_entry(d, run)
        members[kind].append(run)
        probs[run] = np.load(Path(d) / f"val_probs_{run}.npy")
        v = pd.read_csv(Path(d) / f"val_ids_{run}.csv")
        if ref is None:
            ref = v
        assert (v.lesion_id.values == ref.lesion_id.values).all(), "kontrol setleri ayni sirada degil"
    y = ref.label.values

    derm = np.mean([probs[r] for r in members["derm"]], axis=0)
    clin = np.mean([probs[r] for r in members["clin"]], axis=0)
    modes = {
        "hekim": {"description": "dermoskopik + klinik goruntu", "inputs": ["derm", "clin"],
                  "members": members, **mode_stats((derm + clin) / 2, y, a.alpha)},
        "hasta": {"description": "yalnizca klinik goruntu", "inputs": ["clin"],
                  "members": {"clin": members["clin"]}, **mode_stats(clin, y, a.alpha)},
    }
    config = {
        "version": a.version,
        "classes": CLASSES,
        "groups": {"malignant": MALIGNANT, "premalignant": PREMALIGNANT, "benign": BENIGN},
        "fusion_rule": "mean within image type, then equal-weight mean of derm and clin",
        "models": models,
        "modes": modes,
        "gradcam": {"derm": a.gradcam_derm, "clin": a.gradcam_clin},
        "conformal_alpha": a.alpha,
        "cost_ratio": a.cost_ratio,
        "note": ("Temperature and conformal thresholds come from the 1,048-lesion validation split (fold 0); "
                 "they will be recomputed from 5-fold out-of-fold predictions."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
    print(json.dumps({m: {k: v for k, v in s.items() if k != "members"} for m, s in modes.items()}, indent=2))


if __name__ == "__main__":
    main()
