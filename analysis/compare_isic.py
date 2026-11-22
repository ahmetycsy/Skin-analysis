"""Eski dermoskopik modeller ile ISIC 2019 eklenerek eğitilen yeni dermoskopik modelleri karşılaştırır (hekim modu).

Kullanım:  python3 analysis/compare_isic.py ~/Downloads/sonuclar/isic   (Kaggle'dan inen isic_derm_sonuclar.zip'in açıldığı klasör)

Karşılaştırma aynı 1.048 MILK10k kontrol lezyonunda, eşleştirilmiş:
  eski = şu anki sistem: eski dermoskopik + (PAD-UFES-20 ile eğitilmiş) klinik  → val_ensemble_predictions.csv h_*
  yeni = yeni dermoskopik (ISIC 2019) + aynı klinik
Eski dermoskopik ham olasılık, val_ensemble_predictions_eski.csv'den kalibrasyon tersine çevrilerek geri hesaplanır.
Hasta modu bu değişiklikten etkilenmez (yalnızca klinik model).
Karar kuralı (önceden belirlendi): hekim Macro F1 en az +0,01, kaçırılan kötü huylu artmaz, ECE ≤ 0,03.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from predictor import (CLASSES, GROUPS, MALIGN, PREMALIGN, REFER_MALIGN_OR_PRE, REFER_URGENT_MALIGN,  # noqa: E402
                       UNCERTAIN_GAP, UNCERTAIN_TOP)

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/sonuclar/isic")
S = os.path.expanduser("~/Downloads/sonuclar")
cfg = json.load(open(os.path.join(ROOT, "models", "model_config.json")))
TH, TP = cfg["modes"]["hekim"]["temperature"], cfg["modes"]["hasta"]["temperature"]
MAL = [CLASSES.index(c) for c in CLASSES if GROUPS[c] == MALIGN]
RISK = [CLASSES.index(c) for c in CLASSES if GROUPS[c] in (MALIGN, PREMALIGN)]
rng = np.random.default_rng(42)


def norm(p):
    p = np.clip(p, 1e-12, None)
    return p / p.sum(1, keepdims=True)


def calib(p, T):
    z = np.log(p + 1e-9) / T
    z -= z.max(1, keepdims=True)
    return norm(np.exp(z))


def macro_f1(y, pred):
    f = []
    for c in range(len(CLASSES)):
        tp, fp, fn = ((pred == c) & (y == c)).sum(), ((pred == c) & (y != c)).sum(), ((pred != c) & (y == c)).sum()
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def metrics(P, y):
    mal, risk = P[:, MAL].sum(1), P[:, RISK].sum(1)
    s = np.sort(P, 1)
    unc = (s[:, -1] < UNCERTAIN_TOP) | ((s[:, -1] - s[:, -2]) < UNCERTAIN_GAP)
    referred = (mal >= REFER_URGENT_MALIGN) | (risk >= REFER_MALIGN_OR_PRE) | unc
    ym, yb = np.isin(y, MAL), ~np.isin(y, RISK)
    conf, ok = P.max(1), P.argmax(1) == y
    bins = [(conf >= lo) & ((conf < lo + .1) if lo < .9 else (conf <= 1)) for lo in np.arange(0, 1, .1)]
    ece = sum(abs(conf[b].mean() - ok[b].mean()) * b.sum() for b in bins if b.any()) / len(y)
    return {"macro_f1": macro_f1(y, P.argmax(1)), "acc": float(ok.mean()), "mal_caught": float(referred[ym].mean()),
            "mal_missed": int((~referred & ym).sum()), "benign_referred": float(referred[yb].mean()), "ece": float(ece)}


def boot(Pa, Pb, y, B=2000):
    d = []
    for _ in range(B):
        ix = rng.choice(len(y), len(y))
        d.append(macro_f1(y[ix], Pb[ix].argmax(1)) - macro_f1(y[ix], Pa[ix].argmax(1)))
    return np.percentile(d, [2.5, 97.5])


def show(title, a, b, ci):
    print(f"\n=== {title} ===\n{'':24s}{'eski':>10s}{'yeni':>10s}{'fark':>10s}")
    for k, lab, pct in (("macro_f1", "Macro F1", False), ("acc", "Doğruluk", True), ("mal_caught", "Kötü huylu yakalama", True),
                        ("mal_missed", "Kaçırılan kötü huylu", False), ("benign_referred", "İyi huylu sevk", True), ("ece", "ECE", False)):
        fa, fb = a[k], b[k]
        if pct:
            print(f"{lab:24s}{fa * 100:>9.1f}%{fb * 100:>9.1f}%{(fb - fa) * 100:>+9.1f}p")
        elif isinstance(fa, int):
            print(f"{lab:24s}{fa:>10d}{fb:>10d}{fb - fa:>+10d}")
        else:
            print(f"{lab:24s}{fa:>10.3f}{fb:>10.3f}{fb - fa:>+10.3f}")
    print(f"Macro F1 farkı %95 GA: [{ci[0]:+.3f}, {ci[1]:+.3f}]" + ("  → anlamlı" if ci[0] > 0 or ci[1] < 0 else "  → anlamlı değil"))


old = pd.read_csv(os.path.join(S, "val_ensemble_predictions_eski.csv")).set_index("lesion_id")   # ilk sürüm (eski derm + eski klinik)
cur = pd.read_csv(os.path.join(S, "val_ensemble_predictions.csv")).set_index("lesion_id").reindex(old.index)  # şu an (eski derm + PAD klinik)
y = old.label.map(CLASSES.index).values
ho, po = old[[f"h_{c}" for c in CLASSES]].values, old[[f"p_{c}" for c in CLASSES]].values
derm_old = norm(2 * norm(ho ** TH) - norm(po ** TP))
clin_cur = norm(cur[[f"p_{c}" for c in CLASSES]].values ** TP)
H_cur = cur[[f"h_{c}" for c in CLASSES]].values
print("Geri hesaplama kontrolü (0'a yakın olmalı):", float(np.abs(calib((derm_old + clin_cur) / 2, TH) - H_cur).max()))

runs = sorted(f[6:-4] for f in os.listdir(D) if f.startswith("preds_") and f.endswith(".csv"))
print("Yeni dermoskopik modeller:", runs)
for r in runs:
    res = json.load(open(os.path.join(D, f"result_{r}.json")))
    print(f"  {r}: en iyi epoch {res['best_epoch']}, tek model MILK kontrol Macro F1 {res['milk_val_macro_f1']}")
rep = os.path.join(D, "dedup_report.json")
if os.path.exists(rep):
    r = json.load(open(rep))
    print(f"Kopya temizliği: aynı numara {r['removed_same_isic_id']}, benzer görüntü {r['removed_near_duplicate']}, "
          f"kullanılan ISIC 2019 görüntüsü {r['isic_used']}")
new = [pd.read_csv(os.path.join(D, f"preds_{r}.csv")).set_index("lesion_id").reindex(old.index)[CLASSES].values for r in runs]
assert not np.isnan(np.array(new)).any(), "Kontrol lezyonları eşleşmedi"
derm_new = norm(np.mean(new, 0))
H_new = calib((derm_new + clin_cur) / 2, TH)

show("YALNIZCA DERMOSKOPİK (kalibrasyonsuz, bilgi için)", metrics(derm_old, y), metrics(derm_new, y), boot(derm_old, derm_new, y))
a, b = metrics(H_cur, y), metrics(H_new, y)
show("HEKİM MODU · MILK10k kontrol (1.048)", a, b, boot(H_cur, H_new, y))
ok = (b["macro_f1"] - a["macro_f1"] >= 0.01) and (b["mal_missed"] <= a["mal_missed"]) and (b["ece"] <= 0.03)
print("\nKarar kuralı (F1 ≥ +0,01, kaçırılan kötü huylu artmaz, ECE ≤ 0,03):", "SAĞLANDI → yayına alınabilir" if ok else "SAĞLANMADI → eski kalsın")
