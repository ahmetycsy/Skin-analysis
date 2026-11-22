"""MILK10k Benchmark (ISIC gizli test seti, 479 lezyon) için tahmin dosyası üretir.

Kullanım: python3 isic_benchmark/predict_benchmark.py <test_görüntü_klasörü> <MILK10k_Test_Metadata.csv> [ek_ad]
(ek_ad verilirse dosya adlarının sonuna eklenir, eski gönderim dosyaları korunur. ISIC puanlayıcısı `lesion_id` sütununu tanır.)
Çıktılar (isic_benchmark/ içinde; .csv dosyaları .gitignore'da):
  - benchmark_probs_raw.csv   : modelin kalibre olasılıkları (toplamı 1), rapor ve analiz için
  - milk10k_benchmark_submission.csv : ISIC'e gönderilecek dosya

ISIC her sınıfı ayrı ayrı ">= 0,5 ise pozitif" diye ikili hale getirip Macro F1 hesaplar. Bizim olasılıklar toplamı 1
olduğundan en olası tür bazen 0,5'in altında kalır ve o lezyon için hiçbir tür seçilmemiş sayılırdı. Bu yüzden her
lezyonda en olası türün değeri 1,0 yapılır (arayüzdeki "en olası tür" kararıyla aynı), diğerleri olduğu gibi kalır
(hepsi < 0,5 olduğu için negatif sayılır). Yani her lezyona tam olarak bir tanı verilmiş olur.
"""
import glob
import os
import sys
import time

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from predictor import CLASSES, RealPredictor  # noqa: E402

img_dir, meta_csv = sys.argv[1], sys.argv[2]
sfx = f"_{sys.argv[3]}" if len(sys.argv) > 3 else ""
out_dir = os.path.join(ROOT, "isic_benchmark")
meta = pd.read_csv(meta_csv)
paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)}
print("metadata satırı:", len(meta), "| bulunan görüntü:", len(paths), flush=True)

p = RealPredictor(os.path.join(ROOT, "models"))
print("model:", p.name, flush=True)
rows, t0 = [], time.time()
for i, (lid, g) in enumerate(meta.groupby("lesion_id", sort=True)):
    derm_id = g[g.image_type.str.contains("dermoscop", case=False)].isic_id.iloc[0]
    clin_id = g[~g.image_type.str.contains("dermoscop", case=False)].isic_id.iloc[0]
    probs = p.predict(Image.open(paths[clin_id]).convert("RGB"), Image.open(paths[derm_id]).convert("RGB"), None)
    rows.append({"lesion_id": lid, **probs})
    if (i + 1) % 50 == 0:
        print(f"{i + 1} lezyon, {time.time() - t0:.0f} sn", flush=True)

raw = pd.DataFrame(rows)[["lesion_id"] + CLASSES]
raw.to_csv(os.path.join(out_dir, f"benchmark_probs_raw{sfx}.csv"), index=False)
sub = raw.copy()
top = sub[CLASSES].idxmax(axis=1)
for c in CLASSES:
    sub.loc[top == c, c] = 1.0
assert len(sub) == meta.lesion_id.nunique() and sub[CLASSES].ge(0.5).sum(axis=1).eq(1).all()
sub.to_csv(os.path.join(out_dir, f"milk10k_benchmark_submission{sfx}.csv"), index=False, float_format="%.6f")
print("tamam:", len(sub), "lezyon | tahmin edilen tür dağılımı:", top.value_counts().to_dict())
print("en olası türün olasılığı < 0,5 olan lezyon:", int((raw[CLASSES].max(axis=1) < 0.5).sum()))
