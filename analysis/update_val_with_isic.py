"""Dermoskopik modeller ISIC 2019 ile yenilenince doğrulama tahmin dosyasının hekim (h_*) sütunlarını yeniden üretir.

Klinik kısım (PAD-UFES-20 modelleri) şu anki dosyadan, kalibrasyon tersine çevrilerek alınır; yeni dermoskopik olasılıklar
Kaggle'ın preds_*.csv dosyalarından gelir. Hasta (p_*) sütunları değişmez. Yedek: val_ensemble_predictions_onceki_isic.csv.
Ayrıca hekim modunda en iyi sıcaklığı ve conformal kapsamayı yazdırır (kontrol için).

Kullanım: python3 analysis/update_val_with_isic.py ~/Downloads/sonuclar/isic
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from predictor import CLASSES  # noqa: E402

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/sonuclar/isic")
VAL = os.path.expanduser("~/Downloads/sonuclar/val_ensemble_predictions.csv")
BAK = VAL.replace(".csv", "_onceki_isic.csv")
cfg = json.load(open(os.path.join(ROOT, "models", "model_config.json")))
TH, TP = cfg["modes"]["hekim"]["temperature"], cfg["modes"]["hasta"]["temperature"]
Q = cfg["modes"]["hekim"]["conformal_include_if_prob_at_least"]


def norm(p):
    p = np.clip(p, 1e-12, None)
    return p / p.sum(1, keepdims=True)


def calib(p, T):
    z = np.log(p + 1e-9) / T
    z -= z.max(1, keepdims=True)
    return norm(np.exp(z))


if not os.path.exists(BAK):
    shutil.copy(VAL, BAK)
val = pd.read_csv(BAK).set_index("lesion_id")
y = val.label.map(CLASSES.index).values
clin = norm(val[[f"p_{c}" for c in CLASSES]].values ** TP)
runs = cfg["modes"]["hekim"]["members"]["derm"]
new = [pd.read_csv(os.path.join(D, f"preds_{r}.csv")).set_index("lesion_id").reindex(val.index)[CLASSES].values for r in runs]
assert not np.isnan(np.array(new)).any()
raw = (norm(np.mean(new, 0)) + clin) / 2
best = min(np.arange(0.3, 1.21, 0.01), key=lambda T: -np.log(calib(raw, T)[np.arange(len(y)), y] + 1e-12).mean())
H = calib(raw, TH)
cov = (H[np.arange(len(y)), y] >= Q).mean()
print(f"hekim: en iyi sıcaklık {best:.2f} (kullanılan {TH}), conformal kapsama %{cov * 100:.1f}, ort. küme {(H >= Q).sum(1).mean():.2f}")
val[[f"h_{c}" for c in CLASSES]] = H
val.reset_index().to_csv(VAL, index=False)
print(f"hekim doğruluk %{(H.argmax(1) == y).mean() * 100:.1f} | yazıldı: {VAL} | yedek: {BAK}")
