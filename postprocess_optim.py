"""
A1 (derm) + A2 (clin) cikislari uzerinde GPU gerektirmeyen uc yontem:

  1) Birlestirme yontemleri: sade ortalama, sinif esikleri, metadata olabilirlik orani,
                 lojistik regresyon (stacking) A1/A2 olasiliklari (+ metadata) ile.
                 Ic ice capraz dogrulama: dis 5 kat x 3 tekrar (degerlendirme),
                 ic 3 kat (C ve class_weight secimi). Kontrol setine bakip ayar YAPILMAZ;
                 her lezyonun tahmini, o lezyonu gormemis bir modelden gelir (out-of-fold).
  2) Kalibrasyon + maliyete duyarli karar (Bayes karar kurali):
                 tahmin = argmin_j  sum_i  C[i, j] * p(i | x)
                 "malign lezyonu benign demek" maliyeti c = 1..20 taranir.
  3) Conformal prediction (split conformal, LAC skoru):
                 %90 / %95 kapsama garantili olasi tani kumeleri.

Calistirma:
  python postprocess_optim.py --derm-dir <A1 klasoru> --clin-dir <A2 klasoru> \
      --meta MILK10k_Training_Metadata.csv [--out .]

Uretir (figures/, results/):
  stack_variants.png, stack_results.csv
  cost_tradeoff.png, cost_tradeoff.csv, calibration.csv
  conformal_sets.png, conformal_summary.csv, conformal_per_class.csv
  oof_probs_avg_calibrated.npy, postprocess_summary.json
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
K = len(CLASSES)
LABELS = list(range(K))
MALIGNANT = [CLASSES.index(c) for c in ["BCC", "MAL_OTH", "MEL", "SCCKA"]]
PREMALIGNANT = [CLASSES.index("AKIEC")]            # aktinik keratoz / intraepidermal karsinom
BENIGN = [CLASSES.index(c) for c in ["BEN_OTH", "BKL", "DF", "INF", "NV", "VASC"]]
MEL = CLASSES.index("MEL")
SEED = 42

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlelocation": "left", "axes.titleweight": "bold",
                     "axes.titlepad": 20, "savefig.dpi": 200, "savefig.bbox": "tight"})


def style(ax, grid="y"):
    ax.grid(axis=grid, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)


# ------------------------------------------------------------------ veri
def load(a):
    dd, cd = Path(a.derm_dir), Path(a.clin_dir)
    pdm = np.load(dd / f"val_probs_{a.derm_run}.npy")
    pcl = np.load(cd / f"val_probs_{a.clin_run}.npy")
    vd = pd.read_csv(dd / f"val_ids_{a.derm_run}.csv")
    vc = pd.read_csv(cd / f"val_ids_{a.clin_run}.csv")
    assert (vd.lesion_id.values == vc.lesion_id.values).all()
    y = vd.label.values

    meta = pd.read_csv(a.meta)
    derm = meta[meta.image_type == "dermoscopic"].set_index("lesion_id")
    clin = meta[meta.image_type != "dermoscopic"].set_index("lesion_id")
    monet = [c for c in meta.columns if c.startswith("MONET_")]
    X = pd.DataFrame(index=vd.lesion_id.values)
    X["age"] = derm["age_approx"].reindex(X.index).values
    X["skin_tone"] = derm["skin_tone_class"].reindex(X.index).values
    X["sex"] = derm["sex"].reindex(X.index).fillna("missing").values
    X["site"] = derm["site"].reindex(X.index).fillna("missing").values
    for c in monet:
        X[f"derm_{c}"] = derm[c].reindex(X.index).values
        X[f"clin_{c}"] = clin[c].reindex(X.index).values
    eps = 1e-6
    for k, c in enumerate(CLASSES):
        X[f"A1_{c}"] = np.log(pdm[:, k] + eps)
        X[f"A2_{c}"] = np.log(pcl[:, k] + eps)
    return y, pdm, pcl, X, monet


def make_stacker(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=4000))])
    grid = {"lr__C": [0.01, 0.03, 0.1, 0.3, 1.0], "lr__class_weight": [None, "balanced"]}
    return GridSearchCV(pipe, grid, scoring="f1_macro",
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=-1)


def full_proba(model, Xte):
    """Egitim katinda olmayan sinif varsa 11 sutuna tamamla."""
    p = np.zeros((len(Xte), K))
    p[:, model.classes_] = model.predict_proba(Xte)
    return p


# ------------------------------------------------------------------ 1) stacking
def fit_bias(logp, yy):
    """Sinif bazinda kaydirma (esik ayari): Macro F1'i en buyuten b (koordinat arama)."""
    b, grid = np.zeros(K), np.linspace(-2, 2, 21)
    for _ in range(2):
        for k in range(K):
            best = (macro_f1(yy, (logp + b).argmax(1)), b[k])
            for g in grid:
                b2 = b.copy()
                b2[k] = g
                sc = macro_f1(yy, (logp + b2).argmax(1))
                if sc > best[0] + 1e-9:
                    best = (sc, g)
            b[k] = best[1]
    return b


def inner_oof(make, Xtr, ytr):
    """Egitim katinin icinde out-of-fold olasiliklar (ic CV)."""
    p = np.zeros((len(ytr), K))
    for itr, ite in StratifiedKFold(3, shuffle=True, random_state=SEED).split(Xtr, ytr):
        m = make().fit(Xtr.iloc[itr], ytr[itr])
        p[ite] = full_proba(m.best_estimator_, Xtr.iloc[ite])
    return p


def run_stacking(y, pdm, pcl, X, monet):
    prob_cols = [c for c in X.columns if c.startswith(("A1_", "A2_"))]
    meta_num = ["age", "skin_tone"] + [c for c in X.columns if "MONET_" in c]
    meta_cat = ["sex", "site"]
    avg = (pdm + pcl) / 2
    lavg = np.log(avg + 1e-9)

    names = ["Average A1+A2 (no learning)",
             "Average + per-class thresholds",
             "Average x metadata likelihood ratio",
             "LR: A1+A2 probabilities",
             "LR: A1+A2 probabilities + metadata",
             "LR: metadata only"]
    n_rep = 3
    oof = {v: np.zeros((n_rep, len(y), K)) for v in names}
    chosen = {v: [] for v in names}
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED).split(X, y))
    mk_meta = lambda: make_stacker(meta_num, meta_cat)
    for s, (tr, te) in enumerate(splits):
        r = s // 5
        Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y[tr]
        oof[names[0]][r, te] = avg[te]

        b = fit_bias(lavg[tr], ytr)
        oof[names[1]][r, te] = np.exp(lavg[te] + b)
        chosen[names[1]].append(np.round(b, 1).tolist())

        m_meta = mk_meta().fit(Xtr, ytr)
        p_meta_te = full_proba(m_meta.best_estimator_, Xte)
        oof[names[5]][r, te] = p_meta_te
        chosen[names[5]].append(m_meta.best_params_)
        prior = np.bincount(ytr, minlength=K) / len(ytr) + 1e-6
        p_meta_in = inner_oof(mk_meta, Xtr, ytr)
        best = (-1, 0.0)
        for w in [0, 0.25, 0.5, 0.75, 1.0]:
            sc = macro_f1(ytr, (lavg[tr] + w * (np.log(p_meta_in + 1e-9) - np.log(prior))).argmax(1))
            if sc > best[0]:
                best = (sc, w)
        w = best[1]
        oof[names[2]][r, te] = np.exp(lavg[te] + w * (np.log(p_meta_te + 1e-9) - np.log(prior)))
        chosen[names[2]].append({"w_meta": w})

        for v, (num, cat) in {names[3]: (prob_cols, []), names[4]: (prob_cols + meta_num, meta_cat)}.items():
            m = make_stacker(num, cat).fit(Xtr, ytr)
            oof[v][r, te] = full_proba(m.best_estimator_, Xte)
            chosen[v].append(m.best_params_)
        print(f"   dis kat {s + 1}/{len(splits)}", flush=True)

    rows, oof_mean = [], {}
    for v in names:
        sc = [macro_f1(y, oof[v][r].argmax(1)) for r in range(n_rep)]
        p = oof[v].mean(0)
        p = p / p.sum(1, keepdims=True)
        oof_mean[v] = p
        rows.append({"variant": v, "macro_f1_mean": round(float(np.mean(sc)), 4),
                     "macro_f1_sd_over_repeats": round(float(np.std(sc)), 4),
                     "accuracy": round(float((p.argmax(1) == y).mean()), 4),
                     "macro_auc": round(float(roc_auc_score(np.eye(K)[y], p, average="macro")), 4)})
    res = pd.DataFrame(rows)
    per_class = pd.DataFrame({v: np.round(f1_score(y, oof_mean[v].argmax(1), average=None, labels=LABELS,
                                                   zero_division=0), 4) for v in names}, index=CLASSES)
    params = {v: pd.Series([str(c) for c in chosen[v]]).value_counts().head(5).to_dict() for v in names if chosen[v]}
    return res, per_class, oof_mean, params, avg


def plot_stack(res, out):
    r = res.sort_values("macro_f1_mean")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = [GREY if v.startswith("LR: metadata") else BLUE for v in r.variant]
    ax.barh(r.variant, r.macro_f1_mean, color=colors, height=0.55,
            xerr=r.macro_f1_sd_over_repeats, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    for i, (m, s) in enumerate(zip(r.macro_f1_mean, r.macro_f1_sd_over_repeats)):
        ax.text(m + s + 0.01, i, f"{m:.3f}", va="center", color=INK, fontsize=9)
    ax.axvline(0.45, color=INK2, ls="--", lw=1)
    ax.text(0.455, 0, "target 0.45", color=INK2, fontsize=8.5, va="center")
    ax.set_xlim(0, r.macro_f1_mean.max() + 0.12)
    ax.set_xlabel("Macro F1 (out-of-fold, validation set n = 1,048)")
    ax.set_title("Ways to combine A1, A2 and metadata, tested without peeking")
    ax.text(0, 1.01, "Nested CV: outer 5 folds × 3 repeats, every setting chosen in inner folds; bar = SD across repeats",
            transform=ax.transAxes, color=INK2, fontsize=9)
    style(ax, "x")
    fig.savefig(out / "stack_variants.png")
    plt.close(fig)


# ------------------------------------------------------------------ 2) kalibrasyon + maliyet
def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e


def temperature_oof(p, y, seed=SEED):
    """Ortalama olasiliklar icin out-of-fold temperature scaling."""
    logit = np.log(p + 1e-9)
    out = np.zeros_like(p)
    Ts = np.exp(np.linspace(np.log(0.3), np.log(5), 60))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y):
        nll = []
        for T in Ts:
            z = logit[tr] / T
            z = z - z.max(1, keepdims=True)
            q = np.exp(z) / np.exp(z).sum(1, keepdims=True)
            nll.append(-np.log(q[np.arange(len(tr)), y[tr]] + 1e-12).mean())
        T = Ts[int(np.argmin(nll))]
        z = logit[te] / T
        z = z - z.max(1, keepdims=True)
        out[te] = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return out


def cost_matrix(c):
    """C[i, j] = gercek i iken j demenin maliyeti."""
    C = np.ones((K, K))
    np.fill_diagonal(C, 0)
    for i in MALIGNANT:
        for j in BENIGN:
            C[i, j] = c                # kotu huyluyu iyi huylu demek
    for i in PREMALIGNANT:
        for j in BENIGN:
            C[i, j] = max(1.0, c / 2)  # AKIEC'i iyi huylu demek
    return C


def decision_metrics(p, y, c):
    pred = (p @ cost_matrix(c)).argmin(1)
    is_mal = np.isin(y, MALIGNANT)
    said_mal = np.isin(pred, MALIGNANT + PREMALIGNANT)
    is_ben = np.isin(y, BENIGN)
    return {"cost_ratio": c,
            "macro_f1": round(float(macro_f1(y, pred)), 4),
            "malignant_sensitivity": round(float(said_mal[is_mal].mean()), 4),
            "benign_specificity": round(float((~said_mal)[is_ben].mean()), 4),
            "mel_recall": round(float((pred[y == MEL] == MEL).mean()), 4),
            "malignant_called_benign": int((is_mal & np.isin(pred, BENIGN)).sum()),
            "benign_called_malignant": int((is_ben & said_mal).sum())}


def plot_cost(df, out):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    series = [("malignant_sensitivity", "Malignant sensitivity", BLUE),
              ("benign_specificity", "Benign specificity", ORANGE),
              ("mel_recall", "Melanoma recall", AQUA),
              ("macro_f1", "Macro F1", YELLOW)]
    for col, lab, colr in series:
        ax.plot(df.cost_ratio, df[col], color=colr, lw=2, marker="o", ms=4, label=lab)
        ax.text(df.cost_ratio.iloc[-1] + 0.4, df[col].iloc[-1], lab, va="center", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks(df.cost_ratio)
    ax.set_xticklabels([str(int(c)) for c in df.cost_ratio])
    ax.set_xlim(0.9, df.cost_ratio.max() * 2.4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cost of calling a malignant lesion benign (relative to any other error = 1)")
    ax.set_title("Cost-sensitive decision rule: trading false alarms for missed cancers")
    ax.text(0, 1.01, "Bayes rule: predict the class with the lowest expected cost; calibrated A1+A2 average",
            transform=ax.transAxes, color=INK2, fontsize=9)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    style(ax)
    fig.savefig(out / "cost_tradeoff.png")
    plt.close(fig)


# ------------------------------------------------------------------ 3) conformal
def conformal(p, y, alpha, n_rep=50, seed=SEED, class_conditional=False):
    """Split conformal (LAC). Kontrol seti rastgele ikiye: kalibrasyon / test; 50 tekrar.
    class_conditional=True: her sinif icin ayri esik (Mondrian); kalibrasyonda < 20 ornegi olan
    siniflar genel esigi (ya da kendi en buyuk skorunu, hangisi buyukse) kullanir."""
    rng = np.random.default_rng(seed)
    cov, size, single, per_cls_hit, per_cls_n, sizes_all = [], [], [], np.zeros(K), np.zeros(K), []
    for _ in range(n_rep):
        idx = rng.permutation(len(y))
        ca, te = idx[: len(y) // 2], idx[len(y) // 2:]
        s = 1 - p[ca, y[ca]]
        n = len(ca)
        q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        if class_conditional:
            qk = np.full(K, q)
            for k in LABELS:
                sk = s[y[ca] == k]
                nk = len(sk)
                if nk >= 20:
                    qk[k] = np.quantile(sk, min(1, np.ceil((nk + 1) * (1 - alpha)) / nk), method="higher")
                elif nk:
                    qk[k] = max(q, sk.max())
            q = qk[None, :]
        S = p[te] >= 1 - q
        hit = S[np.arange(len(te)), y[te]]
        cov.append(hit.mean())
        size.append(S.sum(1).mean())
        single.append((S.sum(1) == 1).mean())
        sizes_all.append(S.sum(1))
        for k in LABELS:
            m = y[te] == k
            per_cls_hit[k] += hit[m].sum()
            per_cls_n[k] += m.sum()
    summ = {"method": "class-conditional" if class_conditional else "marginal", "target_coverage": 1 - alpha, "empirical_coverage": round(float(np.mean(cov)), 4),
            "mean_set_size": round(float(np.mean(size)), 3),
            "share_single_diagnosis": round(float(np.mean(single)), 4)}
    per_cls = np.where(per_cls_n > 0, per_cls_hit / np.maximum(per_cls_n, 1), np.nan)
    return summ, per_cls, np.concatenate(sizes_all)


def plot_conformal(sizes90, sizes95, out):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    mx = int(max(sizes90.max(), sizes95.max()))
    x = np.arange(1, mx + 1)
    for k, (s, lab, col) in enumerate([(sizes90, "90% coverage", BLUE), (sizes95, "95% coverage", ORANGE)]):
        share = np.array([(s == i).mean() for i in x])
        ax.bar(x + (k - 0.5) * 0.38, share, 0.36, color=col, label=f"{lab} (mean {s.mean():.2f} diagnoses)")
    ax.set_xticks(x)
    ax.set_xlabel("Number of diagnoses in the prediction set")
    ax.set_ylabel("Share of lesions", color=INK2)
    ax.set_title("Conformal prediction: how many diagnoses the system needs to name")
    ax.text(0, 1.01, "Split conformal (LAC) on calibrated A1+A2 average; 50 random calibration/test splits",
            transform=ax.transAxes, color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.savefig(out / "conformal_sets.png")
    plt.close(fig)


# ------------------------------------------------------------------ ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derm-dir", required=True)
    ap.add_argument("--clin-dir", required=True)
    ap.add_argument("--derm-run", default="derm_efficientnet_b3")
    ap.add_argument("--clin-run", default="clin_efficientnet_b3")
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    fig_dir, res_dir = out / "figures", out / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    y, pdm, pcl, X, monet = load(a)

    print("1) Stacking (ic ice CV) ...")
    res, per_class, oof, params, avg = run_stacking(y, pdm, pcl, X, monet)
    res.to_csv(res_dir / "stack_results.csv", index=False)
    per_class.to_csv(res_dir / "stack_per_class_f1.csv")
    plot_stack(res, fig_dir)
    print(res.to_string(index=False))

    best = res.sort_values("macro_f1_mean").iloc[-1].variant

    print("2) Kalibrasyon + maliyet ...")
    p_avg_T = temperature_oof(avg, y)
    # Sonraki adimlar icin: sade ortalama + OOF temperature scaling (argmax degismez, olasiliklar kalibre)
    p_best = p_avg_T
    np.save(out / "oof_probs_avg_calibrated.npy", p_best)
    calib = pd.DataFrame([
        {"probabilities": "A1", "ece": ece(pdm, y)},
        {"probabilities": "A2", "ece": ece(pcl, y)},
        {"probabilities": "Average A1+A2", "ece": ece(avg, y)},
        {"probabilities": "Average A1+A2, temperature-scaled (OOF)", "ece": ece(p_avg_T, y)},
        {"probabilities": "LR: A1+A2 probabilities (OOF)", "ece": ece(oof["LR: A1+A2 probabilities"], y)},
    ]).round(4)
    calib.to_csv(res_dir / "calibration.csv", index=False)
    print(calib.to_string(index=False))

    costs = [1, 2, 3, 5, 8, 12, 20]
    cost_df = pd.DataFrame([decision_metrics(p_best, y, c) for c in costs])
    cost_df.to_csv(res_dir / "cost_tradeoff.csv", index=False)
    plot_cost(cost_df, fig_dir)
    print(cost_df.to_string(index=False))

    print("3) Conformal ...")
    s90, pc90, z90 = conformal(p_best, y, 0.10)
    s95, pc95, z95 = conformal(p_best, y, 0.05)
    m90, pm90, _ = conformal(p_best, y, 0.10, class_conditional=True)
    pd.DataFrame([s90, s95, m90]).to_csv(res_dir / "conformal_summary.csv", index=False)
    pd.DataFrame({"class": CLASSES, "support": np.bincount(y, minlength=K),
                  "coverage_at_90": np.round(pc90, 3), "coverage_at_95": np.round(pc95, 3),
                  "coverage_at_90_class_conditional": np.round(pm90, 3)}).to_csv(
        res_dir / "conformal_per_class.csv", index=False)
    plot_conformal(z90, z95, fig_dir)
    print(s90, s95, m90)

    summary = {
        "note": "All numbers are out-of-fold on the 1,048-lesion validation split (seed 42). "
                "A1/A2 best epochs were chosen on this split, so absolute values remain somewhat optimistic.",
        "stacking": res.to_dict(orient="records"),
        "best_combiner": best,
        "downstream_probabilities": "Average A1+A2, temperature-scaled out-of-fold",
        "chosen_hyperparameters": params,
        "calibration_ece": calib.to_dict(orient="records"),
        "cost_tradeoff": cost_df.to_dict(orient="records"),
        "conformal": [s90, s95, m90],
    }
    with open(res_dir / "postprocess_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print("Bitti.")


if __name__ == "__main__":
    main()
