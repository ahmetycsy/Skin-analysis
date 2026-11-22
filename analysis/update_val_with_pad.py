"""Klinik modeller PAD-UFES-20 ile yenilenince doğrulama tahmin dosyasını (val_ensemble_predictions.csv) yeniden üretir.

Modelleri yeniden çalıştırmaya gerek yoktur (CPU'da ~2 saat sürerdi):
  - dermoskopik kısım değişmedi: eski birleşik çıktıdan kalibrasyon tersine çevrilerek geri hesaplanır,
  - yeni klinik olasılıklar Kaggle'ın preds_*.csv dosyalarından gelir (aynı ön işleme ve TTA).
Eski dosya val_ensemble_predictions_eski.csv olarak saklanır.

Kullanım: python3 analysis/update_val_with_pad.py ~/Downloads/sonuclar/pad
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

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/sonuclar/pad")
VAL = os.path.expanduser("~/Downloads/sonuclar/val_ensemble_predictions.csv")
OLD = VAL.replace(".csv", "_eski.csv")
cfg_old = json.load(open(os.path.expanduser("~/Downloads/ahmet_models_guncel/models/model_config_eski.json")))
cfg = json.load(open(os.path.join(ROOT, "models", "model_config.json")))
assert cfg["modes"]["hasta"]["temperature"] == cfg_old["modes"]["hasta"]["temperature"]
TH, TP = cfg["modes"]["hekim"]["temperature"], cfg["modes"]["hasta"]["temperature"]


def norm(p):
    p = np.clip(p, 1e-12, None)
    return p / p.sum(1, keepdims=True)


def calib(p, T):
    z = np.log(p + 1e-9) / T
    z -= z.max(1, keepdims=True)
    return norm(np.exp(z))


if not os.path.exists(OLD):
    shutil.copy(VAL, OLD)
val = pd.read_csv(OLD).set_index("lesion_id")
H, P = val[[f"h_{c}" for c in CLASSES]].values, val[[f"p_{c}" for c in CLASSES]].values
clin_old = norm(P ** TP)
derm = norm(2 * norm(H ** TH) - clin_old)
runs = cfg["modes"]["hasta"]["members"]["clin"]
new = [pd.read_csv(os.path.join(D, f"preds_{r}.csv")).query("fold == 'val'").set_index("lesion_id").reindex(val.index)[CLASSES].values
       for r in runs]
assert not np.isnan(np.array(new)).any()
clin = norm(np.mean(new, 0))
val[[f"h_{c}" for c in CLASSES]] = calib((derm + clin) / 2, TH)
val[[f"p_{c}" for c in CLASSES]] = calib(clin, TP)
val.reset_index().to_csv(VAL, index=False)
y = val.label.map(CLASSES.index).values
for m in ("h", "p"):
    acc = (val[[f"{m}_{c}" for c in CLASSES]].values.argmax(1) == y).mean()
    print(f"{'hekim' if m == 'h' else 'hasta'} doğruluk %{acc * 100:.1f}")
print("yazıldı:", VAL, "| yedek:", OLD)
