"""Model arayüzü: arayüz (app.py) yalnızca bu dosyadaki `load_predictor()` fonksiyonunu bilir.

Gerçek model hazır olunca SADECE `RealPredictor` doldurulur; app.py değişmez.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

# MILK10k'in 11 tanı kategorisi (ISIC sayfasındaki sütun adları)
CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]

CLASS_NAMES = {
    "AKIEC": "Aktinik keratoz / intraepidermal karsinom",
    "BCC": "Bazal hücreli karsinom",
    "BEN_OTH": "Diğer iyi huylu proliferasyonlar",
    "BKL": "İyi huylu keratinositik lezyon",
    "DF": "Dermatofibrom",
    "INF": "İnflamatuvar ve enfeksiyöz durumlar",
    "MAL_OTH": "Diğer kötü huylu proliferasyonlar",
    "MEL": "Melanom",
    "NV": "Melanositik nevüs (ben)",
    "SCCKA": "Skuamöz hücreli karsinom / keratoakantom",
    "VASC": "Vasküler lezyonlar ve kanama",
}

# Ekranda gösterilen sade adlar (tıbbi terim bilmeyen biri de anlasın diye)
PLAIN_NAMES = {
    "AKIEC": "Güneş lekesi (kansere dönüşebilir)",
    "BCC": "Yavaş büyüyen deri kanseri",
    "BEN_OTH": "Diğer iyi huylu oluşumlar",
    "BKL": "İyi huylu leke / siğil benzeri oluşum",
    "DF": "Deri altında sert küçük şişlik",
    "INF": "İltihap veya enfeksiyon",
    "MAL_OTH": "Diğer kötü huylu oluşumlar",
    "MEL": "Melanom (tehlikeli ben kanseri)",
    "NV": "Ben",
    "SCCKA": "Pullu deri kanseri",
    "VASC": "Damar kaynaklı oluşum",
}

# Tek cümlelik sade açıklamalar
DESCRIPTIONS = {
    "AKIEC": "Uzun süre güneş görmüş deride çıkan pürüzlü, kabuklu leke. Zamanla kansere dönüşebildiği için takip ve tedavi gerekir.",
    "BCC": "En sık görülen deri kanseri. Yavaş büyür ve nadiren yayılır, ancak tedavi edilmesi gerekir.",
    "BEN_OTH": "Listedeki diğer türlere girmeyen, zararsız deri oluşumları.",
    "BKL": "Genellikle yaşla birlikte çıkan, zararsız kahverengi lekeler ve siğil benzeri kabarıklıklar.",
    "DF": "Çoğunlukla bacakta görülen, dokununca sert hissedilen küçük ve zararsız bir şişlik.",
    "INF": "Kanser değildir. Deride iltihap ya da mikrop kaynaklı bir sorunu gösterir.",
    "MAL_OTH": "Listedeki diğer türlere girmeyen, daha nadir görülen kötü huylu deri oluşumları.",
    "MEL": "Benlerden ya da sağlam deriden gelişebilen, yayılma riski yüksek bir deri kanseri. Erken fark edilmesi çok önemlidir.",
    "NV": "Sıradan, zararsız ben.",
    "SCCKA": "Çoğunlukla güneş gören bölgelerde çıkan, kabuklu ve büyüyebilen bir deri kanseri.",
    "VASC": "Kırmızı-mor renkli, damarlardan oluşan ve genellikle zararsız bir oluşum.",
}

# Sonuç ekranındaki renk grupları (sade adlarla)
MALIGN = "Kötü huylu"
PREMALIGN = "Kansere dönüşebilir"
BENIGN = "İyi huylu"
GROUP_ORDER = [MALIGN, PREMALIGN, BENIGN]
GROUPS = {
    "MEL": MALIGN,
    "BCC": MALIGN,
    "SCCKA": MALIGN,
    "MAL_OTH": MALIGN,
    "AKIEC": PREMALIGN,
    "NV": BENIGN,
    "BKL": BENIGN,
    "DF": BENIGN,
    "BEN_OTH": BENIGN,
    "INF": BENIGN,
    "VASC": BENIGN,
}

# Sonuç ekranındaki yönlendirme ve "kararsız" uyarısı için eşikler.
# GEÇİCİ değerlerdir: gerçek model gelince Bedirhan'ın validation sonuçlarına göre ayarlanmalı.
REFER_URGENT_MALIGN = 0.30      # kötü huylu toplamı bunun üstündeyse: öncelikli değerlendirme
REFER_MALIGN_OR_PRE = 0.30      # kötü huylu + kansere dönüşebilir toplamı bunun üstündeyse: değerlendirme önerilir
UNCERTAIN_TOP = 0.40            # en yüksek olasılık bunun altındaysa: kararsız
UNCERTAIN_GAP = 0.10            # ilk iki sonuç arasındaki fark bunun altındaysa: kararsız

# Modele giden değer (MILK10k'taki yazım) -> ekranda görünen Türkçe ad
SITES = {
    "head/neck": "Baş ve boyun",
    "upper extremity": "Kol ve el (omuz dahil)",
    "lower extremity": "Bacak ve ayak (kalça dahil)",
    "anterior torso": "Göğüs ve karın",
    "posterior torso": "Sırt",
    "lateral torso": "Gövdenin yan tarafı",
    "palms/soles": "Avuç içi veya ayak tabanı",
    "oral/genital": "Ağız veya genital bölge",
}

# MILK10k yaşı 5 yıllık aralıklarla tutar. Aralığın alt sınırı modele gider.
# (Eğitimde veri setindeki değer biçimi neyse burada da aynısı kullanılmalı; Bedirhan ile teyit edin.)
AGE_BANDS = [(a, a + 4) for a in range(0, 90, 5)] + [(90, None)]


def age_band_label(lower: int) -> str:
    for lo, hi in AGE_BANDS:
        if lo == lower:
            return f"{lo}+" if hi is None else f"{lo}–{hi}"
    return str(lower)


@dataclass
class Meta:
    age: int          # yaş aralığının alt sınırı (0, 5, 10, ... 90)
    sex: str          # "female" | "male"
    skin_tone: int    # 0 (çok koyu) ... 5 (çok açık)
    site: str         # SITES anahtarlarından biri


class MockPredictor:
    """Sahte model: gerçek model gelene kadar arayüzü geliştirmek için.
    Aynı görseller her zaman aynı sonucu verir (görsellerin hash'inden üretilir)."""

    name = "Sahte model (demo)"
    is_mock = True

    def predict(self, clinical: Image.Image, dermoscopic: Image.Image, meta: Meta) -> dict[str, float]:
        h = hashlib.sha256(clinical.tobytes()[:50000] + dermoscopic.tobytes()[:50000]).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        logits = rng.normal(size=len(CLASSES)) * 1.5
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return dict(zip(CLASSES, probs.tolist()))


class RealPredictor:
    """Gerçek model. Bedirhan'ın teslim ettiği dosya biçimi netleşince burası doldurulur.

    Beklenen sözleşme (Bedirhan ile bu şekilde anlaşın):
      - `models/` klasöründe model dosyası/dosyaları
      - girdi: klinik görsel, dermoskopik görsel, Meta
      - çıktı: CLASSES sırasında 11 olasılık (toplamı 1)

    İsteğe bağlı: `explain(clinical, dermoscopic, meta)` metodu eklenirse arayüz Grad-CAM ısı haritasını gösterir.
    Dönüş: (ısı haritası bindirilmiş PIL görüntü, sınıf kodu) ya da None. Hazır kod: gradcam.py
    (load_image_model, preprocess, gradcam, overlay). MockPredictor'da bu metot yoktur; demo modunda harita gösterilmez.
    """

    name = "Gerçek model"
    is_mock = False

    def __init__(self, model_dir: str):
        raise NotImplementedError(
            "Gerçek model henüz bağlanmadı. Model dosya biçimi netleşince bu sınıf doldurulacak."
        )

    def predict(self, clinical, dermoscopic, meta):  # pragma: no cover
        raise NotImplementedError


def load_predictor():
    """models/ içinde `READY` dosyası varsa gerçek modeli, yoksa sahte modeli yükler."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    if os.path.exists(os.path.join(model_dir, "READY")):
        return RealPredictor(model_dir)
    return MockPredictor()
