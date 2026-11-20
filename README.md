# Tuğalp: veri, tek görüntülü modeller ve değerlendirme

Model ağırlıkları (.pt), MILK10k verisi ve lezyon bazlı dosyalar (bölünme listeleri, olasılıklar) bu klasörde **yok**: repo herkese açık ve `.gitignore` kuralları gereği (CC-BY-NC). Bunlar ekip içinde ayrıca paylaşılır.

| Klasör | İçerik |
|---|---|
| `kaggle/` | `milk10k_train_v2.ipynb` (EfficientNetV2 destekli eğitim), `milk10k_train_cv.ipynb` (5 katlı CV) |
| `analysis/` | EDA, model analizi, topluluk, kalibrasyon/maliyet/conformal ve CV özet scriptleri |
| `tablolar/` | EDA, model, CV, kalibrasyon/maliyet/conformal tabloları (yalnızca özet; lezyon bazlı veri yok) |
| `figures/` | Rapor grafikleri (EDA, karışıklık matrisleri, öğrenme eğrileri, CV) |
| `integration/` | Arayüz için `RealPredictor`, parametrik `gradcam.py`, `model_config.json` ve kurulum notu (`README_AHMET.md`) |

## Ana sonuçlar (5 katlı CV, 5.240 lezyon, Macro F1 ortalama ± sapma)
| Model | Macro F1 |
|---|---|
| A1 derm EfficientNet-B3 | 0,484 ± 0,017 |
| A1-v2 derm EfficientNetV2-S | 0,506 ± 0,035 |
| A2 klinik EfficientNet-B3 | 0,445 ± 0,020 |
| A2-v2 klinik EfficientNetV2-S | 0,436 ± 0,023 |
| Hasta modu (2 klinik model) | 0,463 ± 0,025 |
| **Hekim modu (4 model)** | **0,542 ± 0,024** |

Birleştirme kuralı önceden sabit: her görüntü türünün modelleri kendi içinde ortalanır, sonra dermoskopik ve klinik eşit ağırlıkla ortalanır.
`integration/` altındaki dosyalar ana koda henüz uygulanmadı; uygulama `README_AHMET.md`'de.
