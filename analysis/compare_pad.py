"""Eski klinik modeller ile PAD-UFES-20 eklenerek eğitilen yeni klinik modelleri karşılaştırır (adil, eşleştirilmiş).

Kullanım:
  python3 analysis/compare_pad.py ~/Downloads/sonuclar/pad      (Kaggle'dan inen pad_clin_sonuclar.zip'in açıldığı klasör)

İki sınav:
  1) MILK10k kontrol kümesi (1.048 lezyon; yayındaki modellerin ölçüldüğü aynı küme). Eski modellerin olasılıkları
     val_ensemble_predictions.csv'den; yeni klinik modellerinkiler Kaggle'ın preds_*.csv dosyalarından (aynı ön işleme).
     Hekim modu için dermoskopik kısım değişmez; eski birleşik çıktıdan geri hesaplanır (kalibrasyon tersine çevrilerek).
  2) PAD test (hasta bazında ayrılmış, eğitimde hiç kullanılmamış telefon fotoğrafları). Eski ve yeni modeller
     aynı görüntüler üzerinde yerelde çalıştırılır.
Farklar için %95 güven aralığı: eşleştirilmiş bootstrap (PAD'de hasta bazında).
Yalnızca özet yazdırır; lezyon bazlı veri repo'ya yazılmaz.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from predictor import CLASSES, GROUPS, MALIGN, PREMALIGN, REFER_MALIGN_OR_PRE, REFER_URGENT_MALIGN, UNCERTAIN_GAP, UNCERTAIN_TOP  # noqa: E402

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/sonuclar/pad")
VAL = os.path.expanduser("~/Downloads/sonuclar/val_ensemble_predictions.csv")
cfg = json.load(open(os.path.join(ROOT, "models", "model_config.json")))
T_H, T_P = cfg["modes"]["hekim"]["temperature"], cfg["modes"]["hasta"]["temperature"]
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


def uncalib(q, T):  # calib'in tersi (olasılık vektörüne kadar)
    return norm(q ** T)


def macro_f1(y, pred, labels):
    f = []
    for c in labels:
        tp = ((pred == c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        f.append(0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def metrics(P, y, labels):
    mal, risk = P[:, MAL].sum(1), P[:, RISK].sum(1)
    s = np.sort(P, 1)
    unc = (s[:, -1] < UNCERTAIN_TOP) | ((s[:, -1] - s[:, -2]) < UNCERTAIN_GAP)
    referred = (mal >= REFER_URGENT_MALIGN) | (risk >= REFER_MALIGN_OR_PRE) | unc
    ym = np.isin(y, MAL)
    yb = ~np.isin(y, RISK)
    conf, ok = P.max(1), P.argmax(1) == y
    ece = sum(abs(conf[(conf >= lo) & (conf < lo + .1)].mean() - ok[(conf >= lo) & (conf < lo + .1)].mean())
              * ((conf >= lo) & (conf < lo + .1)).sum() for lo in np.arange(0, 1, .1) if ((conf >= lo) & (conf < lo + .1)).any()) / len(y)
    return {"macro_f1": macro_f1(y, P.argmax(1), labels), "acc": float(ok.mean()),
            "mal_caught": float(referred[ym].mean()) if ym.any() else np.nan, "mal_missed": int((~referred & ym).sum()),
            "n_mal": int(ym.sum()), "benign_referred": float(referred[yb].mean()) if yb.any() else np.nan, "ece": float(ece)}


def boot_diff(Pa, Pb, y, labels, groups=None, B=2000):
    ids = np.arange(len(y)) if groups is None else None
    ug = None if groups is None else np.unique(groups)
    idx_of = None if groups is None else {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(B):
        if groups is None:
            ix = rng.choice(ids, len(ids))
        else:
            ix = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        d.append(macro_f1(y[ix], Pb[ix].argmax(1), labels) - macro_f1(y[ix], Pa[ix].argmax(1), labels))
    return np.percentile(d, [2.5, 97.5])


def show(title, a, b, ci):
    print(f"\n=== {title} ===")
    print(f"{'':24s}{'eski':>10s}{'yeni':>10s}{'fark':>10s}")
    for k, lab, pct in (("macro_f1", "Macro F1", False), ("acc", "Doğruluk", True), ("mal_caught", "Kötü huylu yakalama", True),
                        ("mal_missed", "Kaçırılan kötü huylu", False), ("benign_referred", "İyi huylu sevk", True), ("ece", "ECE", False)):
        fa, fb = a[k], b[k]
        f = (lambda v: f"%{v * 100:.1f}") if pct else (lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))
        diff = fb - fa
        print(f"{lab:24s}{f(fa):>10s}{f(fb):>10s}{(f'{diff * 100:+.1f}p' if pct else (f'{diff:+.3f}' if isinstance(diff, float) else f'{diff:+d}')):>10s}")
    print(f"Macro F1 farkı %95 GA: [{ci[0]:+.3f}, {ci[1]:+.3f}]" + ("  → anlamlı" if ci[0] > 0 or ci[1] < 0 else "  → anlamlı değil"))


runs = sorted(f[6:-4] for f in os.listdir(D) if f.startswith("preds_") and f.endswith(".csv"))
print("Yeni modeller:", runs)
preds = {r: pd.read_csv(os.path.join(D, f"preds_{r}.csv")) for r in runs}
for r in runs:
    print(r, json.load(open(os.path.join(D, f"result_{r}.json"))).get("best_epoch"), "en iyi epoch")

# ---------------------------------------------------------------- 1) MILK10k kontrol kümesi
val = pd.read_csv(VAL).set_index("lesion_id")
new_val = [p[p.fold == "val"].set_index("lesion_id").reindex(val.index)[CLASSES].values for p in preds.values()]
assert not np.isnan(new_val[0]).any(), "Kontrol lezyonları eşleşmedi (bölünme farklı?)"
y = val.label.map(CLASSES.index).values
H_old, P_old = val[[f"h_{c}" for c in CLASSES]].values, val[[f"p_{c}" for c in CLASSES]].values
clin_old = uncalib(P_old, T_P)
derm = norm(2 * uncalib(H_old, T_H) - clin_old)          # eski birleşik = (derm + klinik) / 2
clin_new = norm(np.mean(new_val, 0))
P_new, H_new = calib(clin_new, T_P), calib((derm + clin_new) / 2, T_H)
print("Geri hesaplama kontrolü (0'a yakın olmalı):", float(np.abs(calib((derm + clin_old) / 2, T_H) - H_old).max()))
labels11 = list(range(len(CLASSES)))
show("HASTA MODU · MILK10k kontrol (1.048)", metrics(P_old, y, labels11), metrics(P_new, y, labels11),
     boot_diff(P_old, P_new, y, labels11))
show("HEKİM MODU · MILK10k kontrol (1.048)", metrics(H_old, y, labels11), metrics(H_new, y, labels11),
     boot_diff(H_old, H_new, y, labels11))

# ---------------------------------------------------------------- 2) PAD test (telefon fotoğrafları)
from PIL import Image  # noqa: E402

from gradcam import load_image_model, preprocess  # noqa: E402
import torch  # noqa: E402

split = pd.read_csv(os.path.join(D, "pad_split.csv"))
te = split[split.fold == "pad_test"].reset_index(drop=True)
imgs = [Image.open(os.path.join(D, "pad_test", f"{k}.jpg")).convert("RGB") for k in te.img_key]
pad_meta_groups = te.patient_id.astype(str).values  # bootstrap hasta bazında


def run_models(specs):
    out = []
    for path, m in specs:
        net = load_image_model(path, m["timm_name"], len(CLASSES))
        ps = []
        for im in imgs:
            x, _ = preprocess(im, m["img_size"], m["mean"], m["std"])
            with torch.no_grad():
                p = (net(x).softmax(1) + net(torch.flip(x, dims=[3])).softmax(1)) / 2
            ps.append(p[0].numpy())
        out.append(np.array(ps))
        print("  çalıştı:", os.path.basename(path))
    return norm(np.mean(out, 0))


old_specs = [(os.path.join(ROOT, "models", cfg["models"][n]["file"]), cfg["models"][n]) for n in cfg["modes"]["hasta"]["members"]["clin"]]
new_specs = []
for r in runs:
    res = json.load(open(os.path.join(D, f"result_{r}.json")))
    new_specs.append((os.path.join(D, f"{r}.pt"), {"timm_name": res["timm"], "img_size": res["img_size"], "mean": res["mean"], "std": res["std"]}))
print(f"\nPAD test: {len(te)} görüntü, {len(set(pad_meta_groups))} hasta")
Po, Pn = calib(run_models(old_specs), T_P), calib(run_models(new_specs), T_P)
yt = te.label.values
labels6 = sorted(set(yt))
show("HASTA MODU · PAD test (telefon fotoğrafları, 6 tür)", metrics(Po, yt, labels6), metrics(Pn, yt, labels6),
     boot_diff(Po, Pn, yt, labels6, groups=pad_meta_groups))
print("\nNot: PAD testinde Macro F1 yalnızca PAD'de bulunan 6 tür üzerinden hesaplanır. Kalibrasyon sıcaklığı eski "
      "modelden alındı; yeni model seçilirse ECE'ye bakılmalı.")
