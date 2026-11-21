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

# Sonuç ekranındaki yönlendirme ve "kararsız" uyarısı için eşikler (olasılıklar kalibre).
# 21 Kasım 2026'da kontrol kümesindeki 1.048 lezyonla seçildi (analysis/thresholds.py; hekim modu):
#   öncelikli  (kötü huylu toplam ≥ 0,30): kötü huyluların %93,1'i öncelikli, iyi huyluların %32'si yanlış alarm
#   değerlendirme (kötü+kansere dönüşebilir ≥ 0,20 ya da kararsız): riskli lezyonların %97,2'si en az "değerlendirme"ye
#     yönlendirilir; 691 kötü huyludan 19'u (%2,7) "iyi huylu görünüm" alır; iyi huyluların %48,6'sı sevk edilir.
#   kararsız (en yüksek < 0,40 ya da ilk iki fark < 0,10): vakaların %7,7'si; bu vakalarda doğruluk %41 (diğerlerinde %80).
# Eşikler aynı kümede seçildiği için bu oranlar biraz iyimserdir. Tarama aracı olduğu için kanseri kaçırmamak öncelikli
# tutuldu; son karar (Tuğalp'in maliyet katsayısı c dahil) ekip ve danışman tarafından verilmelidir.
REFER_URGENT_MALIGN = 0.30      # kötü huylu toplamı bunun üstündeyse: öncelikli değerlendirme
REFER_MALIGN_OR_PRE = 0.20      # kötü huylu + kansere dönüşebilir toplamı bunun üstündeyse: değerlendirme önerilir
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
    load_error = None  # gerçek model denendi ama yüklenemediyse hata metni

    def predict(self, clinical: Image.Image, dermoscopic, meta: Meta) -> dict[str, float]:
        derm_bytes = dermoscopic.tobytes()[:50000] if dermoscopic is not None else b"hasta"
        h = hashlib.sha256(clinical.tobytes()[:50000] + derm_bytes).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        logits = rng.normal(size=len(CLASSES)) * 1.5
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return dict(zip(CLASSES, probs.tolist()))


class RealPredictor:
    """Gerçek model: EfficientNet-B3 + EfficientNetV2-S topluluğu. Ayarların hepsi models/model_config.json'da.

    - Hekim modu: dermoskopik + klinik görüntü verilir (4 model). Her görüntü türünün modelleri kendi içinde
      ortalanır, sonra dermoskopik ve klinik eşit ağırlıkla ortalanır.
    - Hasta modu: dermoscopic=None verilirse yalnızca klinik modeller kullanılır (2 model).
    Çıktı: CLASSES sırasında 11 kalibre olasılık (toplamı 1). Hasta bilgisi (Meta) bu modellere girmez.

    Model dosyaları models/ klasöründe yoksa ve MODEL_REPO (Streamlit Secrets) ya da config'teki "hf_repo"
    tanımlıysa Hugging Face'ten indirilir.
    """

    is_mock = False

    def __init__(self, model_dir: str):
        import json

        import torch
        from gradcam import load_image_model

        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        cfg_path = os.path.join(model_dir, "model_config.json")
        if not os.path.exists(cfg_path) and model_repo():  # yayında config de Hugging Face model reposundan gelir
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(repo_id=model_repo(), filename="model_config.json", local_dir=model_dir)
        with open(cfg_path, encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        if self.cfg["classes"] != CLASSES:
            raise ValueError("model_config.json içindeki sınıf sırası CLASSES ile aynı değil")
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.models = {name: load_image_model(self._weights(m["file"]), m["timm_name"], len(CLASSES))
                       for name, m in self.cfg["models"].items()}
        self.name = f"Gerçek model ({self.cfg['version']})"

    def _weights(self, fname: str) -> str:
        path = os.path.join(self.model_dir, fname)
        if os.path.exists(path):
            return path
        repo = model_repo() or self.cfg.get("hf_repo")
        if not repo:
            raise FileNotFoundError(f"{fname} bulunamadı. models/ klasörüne koyun ya da MODEL_REPO tanımlayın.")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, filename=fname, local_dir=self.model_dir)

    def _model_probs(self, name: str, img: Image.Image) -> np.ndarray:
        import torch
        from gradcam import preprocess

        m = self.cfg["models"][name]
        x, _ = preprocess(img, m["img_size"], m["mean"], m["std"])
        net = self.models[name]
        with torch.no_grad():
            p = net(x).softmax(1)
            if m.get("tta_hflip", True):                       # eğitimdeki gibi: aynı görüntü + yatay çevrilmişi
                p = (p + net(torch.flip(x, dims=[3])).softmax(1)) / 2
        return p[0].numpy().astype(np.float64)

    @staticmethod
    def mode_for(dermoscopic) -> str:
        return "hekim" if dermoscopic is not None else "hasta"

    def predict(self, clinical: Image.Image, dermoscopic: Image.Image | None, meta: Meta | None = None) -> dict[str, float]:
        spec = self.cfg["modes"][self.mode_for(dermoscopic)]
        imgs = {"clin": clinical, "derm": dermoscopic}
        parts = [np.mean([self._model_probs(run, imgs[kind]) for run in runs], axis=0)
                 for kind, runs in spec["members"].items()]
        p = np.mean(parts, axis=0)
        z = np.log(p + 1e-9) / spec["temperature"]          # kalibrasyon (temperature scaling)
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        return dict(zip(CLASSES, p.tolist()))

    def prediction_set(self, probs: dict[str, float], dermoscopic_given: bool = True) -> list[str]:
        """%90 güvenle gerçek tanıyı içeren olası tanılar listesi (conformal prediction), olasılığa göre sıralı.
        İsteğe bağlı: arayüzde 'ayırıcı tanı' ve güven seviyesi göstermek için."""
        thr = self.cfg["modes"]["hekim" if dermoscopic_given else "hasta"]["conformal_include_if_prob_at_least"]
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        return [c for c, p in ranked if p >= thr] or [ranked[0][0]]

    def explain(self, clinical: Image.Image, dermoscopic: Image.Image | None, meta: Meta | None = None):
        """Grad-CAM: hekim modunda dermoskopik B3 modelinden, hasta modunda klinik B3 modelinden.
        Harita, o modelin en olası bulduğu sınıf için çizilir; dönüşteki sınıf kodu bu sınıftır."""
        from gradcam import gradcam, overlay, preprocess

        kind = "derm" if dermoscopic is not None else "clin"
        name = self.cfg["gradcam"][kind]
        m = self.cfg["models"][name]
        x, crop = preprocess(dermoscopic if kind == "derm" else clinical, m["img_size"], m["mean"], m["std"])
        cam, probs = gradcam(self.models[name], x)
        return overlay(crop, cam), CLASSES[int(np.argmax(probs))]


def model_repo():
    """Hugging Face model reposu (ör. "kullanici/skin-cds-models"). Ortam değişkeni MODEL_REPO'dan okunur
    (Hugging Face Spaces'te "Variables and secrets" bölümüne girilir). Gizli repo için HF_TOKEN da tanımlı olmalı."""
    return os.environ.get("MODEL_REPO", "").strip() or None


def load_predictor():
    """Gerçek model şu durumlarda yüklenir: models/READY dosyası var (bilgisayarda) ya da MODEL_REPO tanımlı (yayında).
    Gerçek model yüklenemezse uygulama çökmez; demo moduna geçer ve hatayı `load_error` içinde bildirir
    (arayüzde gösterilir, böylece sessizce demo moduna düşmez)."""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    if os.path.exists(os.path.join(model_dir, "READY")) or model_repo():
        try:
            return RealPredictor(model_dir)
        except Exception as exc:  # indirme, eksik dosya, bellek vb.
            mock = MockPredictor()
            mock.load_error = f"{type(exc).__name__}: {exc}"
            return mock
    return MockPredictor()
