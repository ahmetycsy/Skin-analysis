"""Doğrulama (kontrol) kümesinde kullanılan 4 modelin tahminlerini çıkarır: eşik ayarı, performans panosu ve eşitlik analizi için.

Kontrol kümesi = split_seed42.csv'deki 'val' lezyonları (1.048). Yayındaki ağırlıklar kat 0 modelleri olduğundan bu lezyonları
eğitimde görmediler; yani tahminler dürüsttür.
Kullanım: python3 analysis/val_predictions.py ~/Downloads   (MILK10k eğitim dosyalarının ve sonuclar/split_seed42.csv'nin olduğu klasör)
Çıktı: ~/Downloads/sonuclar/val_ensemble_predictions.csv (repo DIŞINDA; lezyon bazlı veri içerir)
"""
import os
import sys
import time

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from predictor import CLASSES, RealPredictor  # noqa: E402

D = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads")
split = pd.read_csv(f"{D}/sonuclar/split_seed42.csv")
gt = pd.read_csv(f"{D}/MILK10k_Training_GroundTruth.csv").set_index("lesion_id")
meta = pd.read_csv(f"{D}/MILK10k_Training_Metadata.csv")
val = split[split.fold == "val"].lesion_id.tolist()
p = RealPredictor(os.path.join(ROOT, "models"))
print("model:", p.name, "| lezyon:", len(val), flush=True)
rows, t0 = [], time.time()
for i, lid in enumerate(val):
    g = meta[meta.lesion_id == lid]
    d = g[g.image_type.str.contains("dermoscop", case=False)].iloc[0]
    c = g[~g.image_type.str.contains("dermoscop", case=False)].iloc[0]
    clin = Image.open(f"{D}/MILK10k_Training_Input/{lid}/{c.isic_id}.jpg").convert("RGB")
    derm = Image.open(f"{D}/MILK10k_Training_Input/{lid}/{d.isic_id}.jpg").convert("RGB")
    ph, pp = p.predict(clin, derm, None), p.predict(clin, None, None)
    row = {"lesion_id": lid, "label": gt.loc[lid].idxmax(), "skin_tone": c.skin_tone_class, "sex": c.sex,
           "age": c.age_approx, "site": c.site}
    row.update({f"h_{k}": v for k, v in ph.items()})
    row.update({f"p_{k}": v for k, v in pp.items()})
    rows.append(row)
    if (i + 1) % 100 == 0:
        print(f"{i + 1} lezyon, {time.time() - t0:.0f} sn", flush=True)
out = f"{D}/sonuclar/val_ensemble_predictions.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("tamam:", out, flush=True)
