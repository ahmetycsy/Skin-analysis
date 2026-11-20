"""
5 katli CV sonuclarini birlestirir (kat 0 = ilk egitimler, kat 1-4 = CV notebook'u).

Uretir (--out altinda):
  results/cv_table.csv            model / topluluk bazinda kat sonuclari, ortalama +- sapma, 5.240 lezyonda OOF sonuc
  results/cv_gains.csv            katlar arasi eslestirilmis kazanc karsilastirmalari
  results/cv_per_class_f1.csv     OOF sinif bazinda F1
  results/cv_decision_<mod>.csv   capraz uydurulmus kalibrasyon + maliyet taramasi (hekim / hasta)
  results/cv_conformal.csv        capraz uydurulmus conformal kapsama ve kume boyutu
  results/cv_summary.json
  figures/cv_folds.png            kat sonuclari (nokta) ve ortalama
  oof/<run>/                      build_config.py'nin okuyabilecegi OOF dosyalari (5.240 lezyon)

Capraz uydurma: kat k'nin esikleri (T, conformal esigi) diger 4 katta hesaplanir, kat k'de olculur.
Boylece kalibrasyon/conformal sonuclari da kontrol setine bakilarak sisirilmez.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
K, L = 11, list(range(11))
MAL = [CLASSES.index(c) for c in ["BCC", "MAL_OTH", "MEL", "SCCKA"]]
PRE = [CLASSES.index("AKIEC")]
BEN = [CLASSES.index(c) for c in ["BEN_OTH", "BKL", "DF", "INF", "NV", "VASC"]]
MEL = CLASSES.index("MEL")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

RUNS = {  # run adi: (kat 0 klasoru, CV klasoru)
    "derm_efficientnet_b3": ("/home/claude/res_derm", "/home/claude/cv_res/derm_b3"),
    "clin_efficientnet_b3": ("/home/claude/res_clin", "/home/claude/cv_res/clin_b3"),
    "derm_tf_efficientnetv2_s": ("/home/claude/res_derm_v2", "/home/claude/cv_res/derm_v2"),
    "clin_tf_efficientnetv2_s": ("/home/claude/res_clin_v2", "/home/claude/cv_res/clin_v2"),
}
LABEL = {"derm_efficientnet_b3": "A1  derm B3", "clin_efficientnet_b3": "A2  clin B3",
         "derm_tf_efficientnetv2_s": "A1-v2  derm V2-S", "clin_tf_efficientnetv2_s": "A2-v2  clin V2-S"}


def mf1(y, p):
    return f1_score(y, p.argmax(1), average="macro", labels=L, zero_division=0)


def apply_T(p, T):
    z = np.log(p + 1e-9) / T
    z = z - z.max(1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(1, keepdims=True)


def fit_T(p, y):
    Ts = np.exp(np.linspace(np.log(0.3), np.log(5), 120))
    nll = [-np.log(apply_T(p, T)[np.arange(len(y)), y] + 1e-12).mean() for T in Ts]
    return float(Ts[int(np.argmin(nll))])


def conf_q(pT, y, alpha):
    s = 1 - pT[np.arange(len(y)), y]
    n = len(y)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))


def cost_pred(p, c):
    C = np.ones((K, K))
    np.fill_diagonal(C, 0)
    for i in MAL:
        C[i, BEN] = c
    for i in PRE:
        C[i, BEN] = max(1.0, c / 2)
    return (p @ C).argmin(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/claude/cv_res/summary")
    a = ap.parse_args()
    out = Path(a.out)
    for d in ["results", "figures", "oof"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    split = pd.read_csv("/home/claude/cv_res/derm_b3/split_cv5_seed42.csv")
    order = split.lesion_id.values
    fold = split.set_index("lesion_id").cv_fold
    y = split.label.values
    folds = fold.reindex(order).values

    oof, last_epoch = {}, {}
    for run, (d0, dcv) in RUNS.items():
        P = pd.DataFrame(index=order, columns=CLASSES, dtype=float)
        v0 = pd.read_csv(f"{d0}/val_ids_{run}.csv")
        P.loc[v0.lesion_id.values] = np.load(f"{d0}/val_probs_{run}.npy")
        r0 = json.load(open(f"{d0}/results/{run}.json"))
        last = [r0["history"][-1]["val_macro_f1"]] if r0.get("history") else []
        for k in range(1, 5):
            vk = pd.read_csv(f"{dcv}/val_ids_{run}_f{k}.csv")
            P.loc[vk.lesion_id.values] = np.load(f"{dcv}/val_probs_{run}_f{k}.npy")
            last.append(json.load(open(f"{dcv}/results/{run}_f{k}.json"))["history"][-1]["val_macro_f1"])
        assert not P.isna().any().any(), run
        oof[run] = P.values
        last_epoch[run] = last
        od = out / "oof" / run
        (od / "results").mkdir(parents=True, exist_ok=True)
        np.save(od / f"val_probs_{run}.npy", P.values)
        pd.DataFrame({"lesion_id": order, "label": y}).to_csv(od / f"val_ids_{run}.csv", index=False)
        shutil.copy(f"{d0}/results/{run}.json", od / "results" / f"{run}.json")

    derm = (oof["derm_efficientnet_b3"] + oof["derm_tf_efficientnetv2_s"]) / 2
    clin = (oof["clin_efficientnet_b3"] + oof["clin_tf_efficientnetv2_s"]) / 2
    ens = {
        "B3 pair (A1+A2)": (oof["derm_efficientnet_b3"] + oof["clin_efficientnet_b3"]) / 2,
        "V2 pair (A1-v2+A2-v2)": (oof["derm_tf_efficientnetv2_s"] + oof["clin_tf_efficientnetv2_s"]) / 2,
        "Derm mean (B3+V2)": derm,
        "Patient mode: clin mean (B3+V2)": clin,
        "Doctor mode: 4-model ensemble": (derm + clin) / 2,
    }
    allp = {**{LABEL[r]: p for r, p in oof.items()}, **ens}

    rows, per_class = [], {}
    for name, p in allp.items():
        fk = [mf1(y[folds == k], p[folds == k]) for k in range(5)]
        rows.append({"model": name, **{f"fold{k}": round(fk[k], 4) for k in range(5)},
                     "mean": round(float(np.mean(fk)), 4), "sd": round(float(np.std(fk, ddof=1)), 4),
                     "min": round(float(np.min(fk)), 4),
                     "oof_macro_f1": round(float(mf1(y, p)), 4),
                     "oof_accuracy": round(float((p.argmax(1) == y).mean()), 4),
                     "oof_macro_auc": round(float(roc_auc_score(np.eye(K)[y], p, average="macro")), 4)})
        per_class[name] = np.round(f1_score(y, p.argmax(1), average=None, labels=L, zero_division=0), 4)
    for r in oof:
        row = next(x for x in rows if x["model"] == LABEL[r])
        row["last_epoch_mean"] = round(float(np.mean(last_epoch[r])), 4)
    table = pd.DataFrame(rows)
    table.to_csv(out / "results" / "cv_table.csv", index=False)
    pc = pd.DataFrame(per_class, index=CLASSES)
    pc["n"] = np.bincount(y, minlength=K)
    pc.to_csv(out / "results" / "cv_per_class_f1.csv")

    def gain(a_, b_):
        d = [mf1(y[folds == k], allp[a_][folds == k]) - mf1(y[folds == k], allp[b_][folds == k]) for k in range(5)]
        return {"comparison": f"{a_}  vs  {b_}", **{f"fold{k}": round(d[k], 4) for k in range(5)},
                "mean": round(float(np.mean(d)), 4), "folds_positive": int(sum(x > 0 for x in d))}
    gains = pd.DataFrame([
        gain("A1-v2  derm V2-S", "A1  derm B3"),
        gain("A2-v2  clin V2-S", "A2  clin B3"),
        gain("B3 pair (A1+A2)", "A1  derm B3"),
        gain("Doctor mode: 4-model ensemble", "B3 pair (A1+A2)"),
        gain("Doctor mode: 4-model ensemble", "A1  derm B3"),
        gain("Patient mode: clin mean (B3+V2)", "A2  clin B3"),
    ])
    gains.to_csv(out / "results" / "cv_gains.csv", index=False)

    # Capraz uydurulmus kalibrasyon, conformal ve maliyet
    conf_rows, dec_all = [], {}
    for mode, p in [("hekim", ens["Doctor mode: 4-model ensemble"]), ("hasta", clin)]:
        pT = np.zeros_like(p)
        sets = {0.10: np.zeros_like(p, dtype=bool), 0.05: np.zeros_like(p, dtype=bool)}
        Ts = []
        for k in range(5):
            tr, te = folds != k, folds == k
            T = fit_T(p[tr], y[tr])
            Ts.append(T)
            pT[te] = apply_T(p[te], T)
            ptr = apply_T(p[tr], T)
            for al in sets:
                sets[al][te] = pT[te] >= 1 - conf_q(ptr, y[tr], al)
        for al, S in sets.items():
            hit = S[np.arange(len(y)), y]
            conf_rows.append({"mode": mode, "target": 1 - al, "coverage": round(float(hit.mean()), 4),
                              "mean_set_size": round(float(S.sum(1).mean()), 3),
                              "share_single": round(float((S.sum(1) == 1).mean()), 4),
                              "mel_coverage": round(float(hit[y == MEL].mean()), 4)})
        dec = []
        is_mal, is_ben = np.isin(y, MAL), np.isin(y, BEN)
        for c in [1, 2, 3, 5]:
            pr = cost_pred(pT, c)
            said = np.isin(pr, MAL + PRE)
            dec.append({"cost_ratio": c, "macro_f1": round(float(f1_score(y, pr, average="macro", labels=L,
                                                                         zero_division=0)), 4),
                        "malignant_sensitivity": round(float(said[is_mal].mean()), 4),
                        "benign_specificity": round(float((~said)[is_ben].mean()), 4),
                        "mel_recall": round(float((pr[y == MEL] == MEL).mean()), 4),
                        "malignant_missed": int((is_mal & np.isin(pr, BEN)).sum()),
                        "benign_false_alarm": int((is_ben & said).sum())})
        dec_all[mode] = dec
        pd.DataFrame(dec).to_csv(out / "results" / f"cv_decision_{mode}.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(out / "results" / "cv_conformal.csv", index=False)

    # Grafik: kat noktalari + ortalama
    show = ["A1  derm B3", "A1-v2  derm V2-S", "A2  clin B3", "A2-v2  clin V2-S", "Patient mode: clin mean (B3+V2)",
            "B3 pair (A1+A2)", "V2 pair (A1-v2+A2-v2)", "Doctor mode: 4-model ensemble"]
    t = table.set_index("model").loc[show]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, (name, r) in enumerate(t.iterrows()):
        yy = len(show) - 1 - i
        col = ORANGE if name.startswith(("Doctor", "Patient")) else BLUE
        ax.scatter([r[f"fold{k}"] for k in range(5)], [yy] * 5, s=22, color=col, alpha=0.55, zorder=3)
        ax.plot([r["mean"] - r["sd"], r["mean"] + r["sd"]], [yy, yy], color=col, lw=2, zorder=2)
        ax.scatter([r["mean"]], [yy], s=70, color=col, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(0.585, yy, f"{r['mean']:.3f} ± {r['sd']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show[::-1])
    ax.axvline(0.45, color=INK2, ls="--", lw=1)
    ax.set_xlim(0.38, 0.64)
    ax.set_xlabel("Macro F1 (each dot = one validation fold of 1,048 lesions)")
    ax.set_title("5-fold cross-validation: single models and ensembles", loc="left", fontweight="bold", pad=20)
    ax.text(0, 1.01, "Large dot = mean, line = ±1 SD; dashed line = 0.45 target; orange = deployed modes",
            transform=ax.transAxes, color=INK2, fontsize=9)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    fig.savefig(out / "figures" / "cv_folds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    json.dump({"table": table.to_dict(orient="records"), "gains": gains.to_dict(orient="records"),
               "conformal_crossfit": conf_rows, "decision_crossfit": dec_all},
              open(out / "results" / "cv_summary.json", "w"), indent=2, ensure_ascii=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    print(gains.to_string(index=False))
    print(pd.DataFrame(conf_rows).to_string(index=False))
    for m, d in dec_all.items():
        print(m)
        print(pd.DataFrame(d).to_string(index=False))


if __name__ == "__main__":
    main()
