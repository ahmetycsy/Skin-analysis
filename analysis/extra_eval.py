"""Teklifte söz verilen ek değerlendirmeler (yeniden eğitim gerekmez, kayıtlı doğrulama tahminlerinden):
  1) Ablasyon: yalnız dermoskopik / yalnız klinik / iki görüntü / iki görüntü + hasta bilgisi
     (hem yalnız MILK10k ile eğitilen ilk sürüm hem de ek veriyle eğitilen son sürüm)
  2) ROC-AUC (tür bazında ve makro, bire-karşı-hepsi)
  3) Tür bazında karar eşiği (log-olasılığa eklenen sapma) ayarı — 5 katlı tabakalı çapraz doğrulama, 3 tekrar;
     ayar eğitim katlarında yapılır, yalnızca dışarıda kalan katta ölçülür (iyimser olmasın diye).

Hasta bilgisi katkısı da çapraz doğrulamayla ölçülür: aynı lojistik regresyon, bir kez yalnız görüntü olasılıklarıyla,
bir kez + yaş, cinsiyet, ten rengi, bölge ile eğitilir; fark hasta bilgisinin katkısıdır.

Kullanım: PYTHONPATH=<scikit-learn yolu> python3 analysis/extra_eval.py   → analysis/extra_eval_sonuc.json + ekran özeti
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from predictor import CLASSES, GROUPS, MALIGN  # noqa: E402

S = os.path.expanduser("~/Downloads/sonuclar")
cfg = json.load(open(os.path.join(ROOT, "models", "model_config.json")))
TH, TP = cfg["modes"]["hekim"]["temperature"], cfg["modes"]["hasta"]["temperature"]
K = len(CLASSES)


def norm(p):
    p = np.clip(p, 1e-12, None)
    return p / p.sum(1, keepdims=True)


def calib(p, T):
    z = np.log(p + 1e-9) / T
    z -= z.max(1, keepdims=True)
    return norm(np.exp(z))


def f1s(y, pred):
    out = []
    for c in range(K):
        tp, fp, fn = ((pred == c) & (y == c)).sum(), ((pred == c) & (y != c)).sum(), ((pred != c) & (y == c)).sum()
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.array(out)


def macro_f1(y, pred):
    return float(f1s(y, pred).mean())


def bal_acc(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(K) if (y == c).any()]))


def aucs(y, P):
    res = {}
    for c in range(K):
        yy = (y == c).astype(int)
        res[CLASSES[c]] = float(roc_auc_score(yy, P[:, c])) if 0 < yy.sum() < len(yy) else None
    vals = [v for v in res.values() if v is not None]
    return res, float(np.mean(vals))


def summary(y, P):
    pred = P.argmax(1)
    mel = CLASSES.index("MEL")
    _, mauc = aucs(y, P)
    return {"macro_f1": macro_f1(y, pred), "acc": float((pred == y).mean()), "bal_acc": bal_acc(y, pred),
            "macro_auc": mauc, "mel_recall": float((pred[y == mel] == mel).mean())}


def load(fname):
    v = pd.read_csv(os.path.join(S, fname)).set_index("lesion_id")
    H, P = v[[f"h_{c}" for c in CLASSES]].values, v[[f"p_{c}" for c in CLASSES]].values
    clin = norm(P ** TP)
    derm = norm(2 * norm(H ** TH) - clin)
    return v, clin, derm, H


old_v, clin_old, derm_old, H_old = load("val_ensemble_predictions_eski.csv")
cur_v, clin_cur, derm_cur, H_cur = load("val_ensemble_predictions.csv")
cur_v = cur_v.reindex(old_v.index)
clin_cur, derm_cur, H_cur = (pd.DataFrame(a, index=load("val_ensemble_predictions.csv")[0].index).reindex(old_v.index).values
                             for a in (clin_cur, derm_cur, H_cur))
y = old_v.label.map(CLASSES.index).values
assert (cur_v.label.map(CLASSES.index).values == y).all()


# ---------------------------------------------------------------- hasta bilgisi (meta) özellikleri
def meta_feats(v):
    age = v.age.fillna(v.age.median()).values[:, None] / 100.0
    sex = pd.get_dummies(v.sex.fillna("unk")).values.astype(float)
    tone = v.skin_tone.fillna(3).values[:, None] / 5.0
    site = pd.get_dummies(v.site.fillna("unk")).values.astype(float)
    return np.hstack([age, sex, tone, site])


M = meta_feats(old_v)


def cv_stack(Pimg, use_meta, seeds=(0, 1, 2)):
    """Lojistik regresyonla katlar dışı (out-of-fold) tahmin; görüntü log-olasılıkları (+ hasta bilgisi)."""
    X = np.log(np.clip(Pimg, 1e-9, 1))
    if use_meta:
        X = np.hstack([X, M])
    res = []
    for sd in seeds:
        oof = np.zeros_like(Pimg)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, np.minimum(y, 9)):
            lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
            lr.fit(X[tr], y[tr])
            pr = np.zeros((len(te), K))
            pr[:, lr.classes_] = lr.predict_proba(X[te])
            oof[te] = pr
        res.append(summary(y, oof))
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


# ---------------------------------------------------------------- tür bazında eşik (sapma) ayarı
GRID = np.linspace(-3, 3, 25)


def fit_bias(L, yy, rounds=3):
    b = np.zeros(K)
    for _ in range(rounds):
        for c in range(K):
            best, bv = -1, b[c]
            for g in GRID:
                b[c] = g
                s = macro_f1(yy, (L + b).argmax(1))
                if s > best + 1e-9:
                    best, bv = s, g
            b[c] = bv
    return b


def cv_bias(P, seeds=(0, 1, 2)):
    L = np.log(np.clip(P, 1e-9, 1))
    base, tuned, mal_base, mal_tuned = [], [], [], []
    for sd in seeds:
        pb, pt = np.zeros(len(y), int), np.zeros(len(y), int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(L, np.minimum(y, 9)):
            b = fit_bias(L[tr], y[tr])
            pb[te] = L[te].argmax(1)
            pt[te] = (L[te] + b).argmax(1)
        base.append(macro_f1(y, pb))
        tuned.append(macro_f1(y, pt))
    full_b = fit_bias(L, y)
    return {"cv_base": float(np.mean(base)), "cv_tuned": float(np.mean(tuned)), "gain": float(np.mean(tuned) - np.mean(base)),
            "gain_per_seed": [float(t - b) for t, b in zip(tuned, base)],
            "same_set_tuned": macro_f1(y, (L + full_b).argmax(1)), "bias_full": dict(zip(CLASSES, map(float, full_b)))}


out = {"n": int(len(y)), "ablation": {}, "auc": {}, "bias": {}}
for tag, clin, derm, H in (("milk_only", clin_old, derm_old, H_old), ("extra_data", clin_cur, derm_cur, H_cur)):
    out["ablation"][tag] = {
        "derm_only": summary(y, derm), "clin_only": summary(y, calib(clin, TP)), "dual": summary(y, H),
        "cv_dual_lr": cv_stack(H, False), "cv_dual_lr_meta": cv_stack(H, True),
    }
    per, mauc = aucs(y, H)
    out["auc"][tag] = {"hekim_macro": mauc, "hekim_per_class": per, "hasta_macro": aucs(y, calib(clin, TP))[1],
                       "hasta_per_class": aucs(y, calib(clin, TP))[0]}
    out["bias"][tag] = {"hekim": cv_bias(H), "hasta": cv_bias(calib(clin, TP))}

json.dump(out, open(os.path.join(ROOT, "analysis", "extra_eval_sonuc.json"), "w"), indent=1, ensure_ascii=False)

f = lambda x: f"{x:.3f}"  # noqa: E731
for tag in out["ablation"]:
    print(f"\n=== ABLASYON · {tag} (1.048 kontrol lezyonu) ===")
    print(f"{'':28s}{'MacroF1':>9s}{'Doğr.':>8s}{'DengeliD':>10s}{'MakroAUC':>10s}{'MEL duy.':>9s}")
    for k, lab in (("derm_only", "Yalnız dermoskopik"), ("clin_only", "Yalnız klinik"), ("dual", "İki görüntü (sistem)"),
                   ("cv_dual_lr", "İki görüntü + LR (ÇD)"), ("cv_dual_lr_meta", "İki görüntü + hasta bilgisi (ÇD)")):
        r = out["ablation"][tag][k]
        print(f"{lab:28s}{f(r['macro_f1']):>9s}{f(r['acc']):>8s}{f(r['bal_acc']):>10s}{f(r['macro_auc']):>10s}{f(r['mel_recall']):>9s}")
    a = out["auc"][tag]
    print("ROC-AUC tür bazında (hekim):", {k: (round(v, 3) if v else None) for k, v in a["hekim_per_class"].items()})
    for mode in ("hekim", "hasta"):
        b = out["bias"][tag][mode]
        print(f"Eşik ayarı {mode}: ÇD önce {f(b['cv_base'])} sonra {f(b['cv_tuned'])} kazanç {b['gain']:+.3f} "
              f"(tohumlar {[round(g, 3) for g in b['gain_per_seed']]}) | aynı kümede ayarlanırsa {f(b['same_set_tuned'])}")
print("\nYazıldı: analysis/extra_eval_sonuc.json")
