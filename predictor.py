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
    "AKIEC": "Aktinik keratoz (güneşe bağlı pürüzlü leke, kansere dönüşebilir)",
    "BCC": "Yavaş büyüyen deri kanseri",
    "BEN_OTH": "Diğer iyi huylu oluşumlar",
    "BKL": "İyi huylu leke (güneş lentigosu, seboreik keratoz)",
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
    "AKIEC": "Uzun süre güneş görmüş deride çıkan, dokununca zımpara gibi pürüzlü, çoğu zaman pembe-kırmızı kabuklu leke. Halk arasındaki zararsız kahverengi \"güneş lekesi\" ile karıştırılmamalı. Zamanla kansere dönüşebildiği için takip ve tedavi gerekir.",
    "BCC": "En sık görülen deri kanseri. Yavaş büyür ve nadiren yayılır, ancak tedavi edilmesi gerekir.",
    "BEN_OTH": "Listedeki diğer türlere girmeyen, zararsız deri oluşumları.",
    "BKL": "Yaşla ve güneşle birlikte çıkan, zararsız düz kahverengi lekeler (halk arasında güneş/yaşlılık lekesi) ve siğil benzeri kabarıklıklar.",
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
# 21 Kasım 2026'da kontrol kümesindeki 1.048 lezyonla seçildi (analysis/thresholds.py; hekim modu). 22 Kasım'da klinik
# modeller PAD-UFES-20 ile yenilenince yeniden kontrol edildi, eşikler aynı kaldı (parantez içinde önceki değerler):
#   öncelikli  (kötü huylu toplam ≥ 0,30): kötü huyluların %95,2'si (%93,1) öncelikli, iyi huyluların %36,5'i (%32) yanlış alarm
#   değerlendirme (kötü+kansere dönüşebilir ≥ 0,20 ya da kararsız): riskli lezyonların %97,5'i (%97,2) en az "değerlendirme"ye
#     yönlendirilir; 691 kötü huyludan 18'i (19) "iyi huylu görünüm" alır; iyi huyluların %49'u (%48,6) sevk edilir.
#     Hasta modunda: 691'den 20'si (24) kaçırılır, iyi huyluların %58'i sevk edilir.
#   kararsız (en yüksek < 0,40 ya da ilk iki fark < 0,10): vakaların %7,3'ü; bu vakalarda doğruluk %34 (diğerlerinde %81).
# 22 Kasım'da dermoskopik modeller ISIC 2019 ile yenilenince tekrar kontrol edildi (hekim modu), eşikler yine aynı:
#   öncelikli: kötü huyluların %96,4'ü, iyi huylu yanlış alarm %31,1; değerlendirme: 691 kötü huyludan 14'ü kaçırılır,
#   iyi huyluların %44,6'sı sevk edilir; kararsız: vakaların %5,9'u (doğruluk %34, diğerlerinde %83).
# Eşikler aynı kümede seçildiği için bu oranlar biraz iyimserdir. Tarama aracı olduğu için kanseri kaçırmamak öncelikli
# tutuldu; son karar (Tuğalp'in maliyet katsayısı c dahil) ekip ve danışman tarafından verilmelidir.
REFER_URGENT_MALIGN = 0.30      # kötü huylu toplamı bunun üstündeyse: öncelikli değerlendirme
REFER_MALIGN_OR_PRE = 0.20      # kötü huylu + kansere dönüşebilir toplamı bunun üstündeyse: değerlendirme önerilir
UNCERTAIN_TOP = 0.40            # en yüksek olasılık bunun altındaysa: kararsız
UNCERTAIN_GAP = 0.10            # ilk iki sonuç arasındaki fark bunun altındaysa: kararsız
# Melanom güvenlik kuralı (24 Kasım 2026, danışman görüşmesi sonrası): diğer eşiklerle "iyi huylu görünüm" alan bir lezyonda
# melanom olasılığı bu değerin üstündeyse yine de dermatolog değerlendirmesi önerilir. Kaçırılan melanomların hepsi "ben"
# sanılıyordu. 1.048 kontrol lezyonunda (aynı kümede seçildi, biraz iyimser):
#   hekim %10: kaçırılan melanom 12→7 (90'da), kaçırılan kötü huylu 14→9, iyi huylu sevk %44,6→%51,0
#   hasta %15: kaçırılan melanom 13→11, kaçırılan kötü huylu 20→18, iyi huylu sevk %58,1→%59,5 (sağlık kaygısı için temkinli)
MEL_SAFETY = {"hekim": 0.10, "hasta": 0.15}

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

    @staticmethod
    def _fake(img: Image.Image) -> np.ndarray:
        h = hashlib.sha256(img.tobytes()[:50000]).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        logits = rng.normal(size=len(CLASSES)) * 1.5
        p = np.exp(logits - logits.max())
        return p / p.sum()

    def predict(self, clinical: Image.Image, dermoscopic, meta: Meta) -> dict[str, float]:
        return self.predict_detail(clinical, dermoscopic, meta)[0]

    def predict_detail(self, clinical: Image.Image, dermoscopic, meta=None):
        parts = {"clin": self._fake(clinical)}
        if dermoscopic is not None:
            parts["derm"] = self._fake(dermoscopic)
        p = np.mean(list(parts.values()), axis=0)
        return dict(zip(CLASSES, p.tolist())), {k: dict(zip(CLASSES, v.tolist())) for k, v in parts.items()}


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
        if model_repo():  # yayında config Hugging Face model reposundan gelir; her açılışta güncel sürüm denetlenir
            try:
                from huggingface_hub import hf_hub_download
                cfg_path = hf_hub_download(repo_id=model_repo(), filename="model_config.json", local_dir=model_dir,
                                           force_download=True)
            except Exception:
                if not os.path.exists(cfg_path):  # internet/HF sorunu: elde config yoksa devam edilemez
                    raise
        with open(cfg_path, encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        if self.cfg["classes"] != CLASSES:
            raise ValueError("model_config.json içindeki sınıf sırası CLASSES ile aynı değil")
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        # Yalnızca modlarda ve Grad-CAM'de kullanılan modeller yüklenir (config'te geri dönüş için eski modeller de
        # listelenebilir; onları yüklemek ücretsiz sunucunun belleğini boşuna doldurur).
        used = {n for spec in self.cfg["modes"].values() for runs in spec["members"].values() for n in runs}
        used |= set(self.cfg.get("gradcam", {}).values())
        self.models = {name: load_image_model(self._weights(m["file"]), m["timm_name"], len(CLASSES))
                       for name, m in self.cfg["models"].items() if name in used}
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
        return self.predict_detail(clinical, dermoscopic, meta)[0]

    def predict_detail(self, clinical: Image.Image, dermoscopic: Image.Image | None, meta: Meta | None = None):
        """(birleşik kalibre olasılıklar, görüntü türü başına ham olasılıklar). İkincisi "hangi görüntü ne dedi" gösterimi için;
        kalibrasyon yalnızca birleşik sonuca uygulanır (config böyle ayarlandı), görüntü başına değerler kalibrasyon öncesidir."""
        spec = self.cfg["modes"][self.mode_for(dermoscopic)]
        imgs = {"clin": clinical, "derm": dermoscopic}
        parts = {kind: np.mean([self._model_probs(run, imgs[kind]) for run in runs], axis=0)
                 for kind, runs in spec["members"].items()}
        p = np.mean(list(parts.values()), axis=0)
        z = np.log(p + 1e-9) / spec["temperature"]          # kalibrasyon (temperature scaling)
        z -= z.max()
        p = np.exp(z) / np.exp(z).sum()
        return dict(zip(CLASSES, p.tolist())), {k: dict(zip(CLASSES, v.tolist())) for k, v in parts.items()}

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
