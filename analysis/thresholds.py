"""Sevk eşiklerini doğrulama kümesi tahminleriyle inceler (val_predictions.py çıktısı).

Kullanım: python3 analysis/thresholds.py ~/Downloads/sonuclar/val_ensemble_predictions.csv
Yalnızca özet sayılar yazdırır (lezyon bazlı veri yazmaz).

Kural (arayüzdeki assess() ile aynı):
  öncelikli  : kötü huylu toplam olasılık >= U
  değerlendirme: (kötü huylu + kansere dönüşebilir) toplam >= R   ya da model kararsız
  iyi huylu görünüm: geri kalan
Hedef: kötü huylu ve kansere dönüşebilir lezyonların çok büyük kısmı en azından "değerlendirme"ye yönlendirilsin
(kaçırılan kanser az olsun), ama iyi huylu lezyonların hepsi de sevk edilmesin.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from predictor import CLASSES, GROUPS, MALIGN, PREMALIGN  # noqa: E402

df = pd.read_csv(sys.argv[1])
MAL = [c for c in CLASSES if GROUPS[c] == MALIGN]
PRE = [c for c in CLASSES if GROUPS[c] == PREMALIGN]
y_mal = df.label.isin(MAL).values
y_risk = df.label.isin(MAL + PRE).values
print(f"lezyon: {len(df)} | kötü huylu: {y_mal.sum()} | kansere dönüşebilir: {(df.label.isin(PRE)).sum()} | "
      f"iyi huylu: {(~y_risk).sum()}")

for mode in ("h", "p"):
    P = df[[f"{mode}_{c}" for c in CLASSES]].values
    mal = P[:, [CLASSES.index(c) for c in MAL]].sum(1)
    risk = P[:, [CLASSES.index(c) for c in MAL + PRE]].sum(1)
    top = P.max(1)
    srt = np.sort(P, 1)
    gap = srt[:, -1] - srt[:, -2]
    acc = (np.array(CLASSES)[P.argmax(1)] == df.label.values).mean()
    print(f"\n=== {'HEKİM' if mode == 'h' else 'HASTA'} MODU (doğruluk %{acc*100:.1f}) ===")

    print("Öncelikli eşiği U (kötü huylu toplam ≥ U): kötü huyluları yakalama / iyi huyluyu yanlış öncelikli")
    for u in (0.2, 0.3, 0.4, 0.5, 0.6):
        flag = mal >= u
        print(f"  U={u:.1f}: duyarlılık %{flag[y_mal].mean()*100:5.1f} | iyi huylu yanlış alarm %{flag[~y_risk].mean()*100:5.1f}"
              f" | öncelikli oranı %{flag.mean()*100:5.1f}")

    for top_t, gap_t in ((0.40, 0.10), (0.50, 0.15), (0.30, 0.05)):
        unc = (top < top_t) | (gap < gap_t)
        print(f"Kararsız (en yüksek < {top_t} ya da fark < {gap_t}): oran %{unc.mean()*100:.1f} | "
              f"bu vakalarda doğruluk %{(np.array(CLASSES)[P.argmax(1)] == df.label.values)[unc].mean()*100:.1f} | "
              f"diğerlerinde %{(np.array(CLASSES)[P.argmax(1)] == df.label.values)[~unc].mean()*100:.1f}")

    print("Sevk eşiği R (kötü huylu+kansere dönüşebilir ≥ R ya da kararsız[0.40/0.10]): riskli lezyonları yakalama / iyi huylu sevk")
    unc = (top < 0.40) | (gap < 0.10)
    for r in (0.1, 0.2, 0.3, 0.4, 0.5):
        ref = (risk >= r) | unc
        missed_mal = (~ref & y_mal).sum()
        print(f"  R={r:.1f}: riskli yakalama %{ref[y_risk].mean()*100:5.1f} | kaçırılan kötü huylu {missed_mal:3d}/{y_mal.sum()}"
              f" | iyi huylu sevk %{ref[~y_risk].mean()*100:5.1f} | toplam sevk %{ref.mean()*100:5.1f}")
