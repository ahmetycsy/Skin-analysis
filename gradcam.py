"""Grad-CAM: modelin görüntüde hangi bölgelere bakarak karar verdiğini gösteren ısı haritası.

Kaggle notebook'unda (kaggle/milk10k_train.py) eğitilen timm görüntü modelleriyle çalışır.
Ön işleme eğitimdeki doğrulama (val) adımıyla birebir aynıdır:
ortadan kare kırp -> IMG_SIZE*1.15 boyutuna küçült -> ortadan IMG_SIZE kırp -> normalize.

Not: Isı haritası "modelin dikkat ettiği yer"dir, lezyonun sınırı ya da tanı kanıtı değildir.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_image_model(weights_path: str, model_name: str = "efficientnet_b3", num_classes: int = 11):
    """Notebook'un kaydettiği state_dict dosyasından modeli kurar (CPU, eval modunda)."""
    import timm

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(img: Image.Image, img_size: int = 300, mean=MEAN, std=STD) -> tuple[torch.Tensor, Image.Image]:
    """Görüntüyü modele hazırlar. (tensör, modelin gördüğü kırpılmış RGB görüntü) döner.

    img_size, mean ve std modele göre verilmelidir (models/model_config.json):
    EfficientNet-B3 = 300 px, ImageNet ortalama/sapma; EfficientNetV2-S = 384 px, 0.5/0.5.
    Varsayılanlar B3 içindir."""
    im = img.convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    cache = int(img_size * 1.15)
    im = im.resize((cache, cache), Image.BILINEAR)
    off = (cache - img_size) // 2
    im = im.crop((off, off, off + img_size, off + img_size))
    arr = (np.asarray(im, dtype=np.float32) / 255.0 - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    x = torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32)).unsqueeze(0)
    return x, im


def _target_layer(model):
    # timm EfficientNet: son blok grubu. 20 klinik + 20 dermoskopik görüntüde blocks[-2] ile karşılaştırıldı:
    # haritanın gösterdiği %20'lik alan kapatılınca olasılık daha çok düşüyor (klinik 0,42 / 0,30; dermoskopik 0,31 / 0,26),
    # ısı ortada toplanıyor ve en sıcak nokta hiç kenara kaymıyor (blocks[-2]'de %30–45). En son katman (bn2) kenara kayıyor.
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    return list(model.children())[-3]


def gradcam(model, x: torch.Tensor, class_idx: int | None = None, targets=None) -> tuple[np.ndarray, np.ndarray]:
    """(ısı haritası [0-1], softmax olasılıkları) döner.
    targets: sınıf indeksleri listesi verilirse harita bu sınıfların birlikte olasılığı (logsumexp) için çizilir —
    örneğin sevki tetikleyen türler (melanom + SCC). Verilmezse class_idx, o da yoksa en olası sınıf kullanılır."""
    store = {}
    layer = _target_layer(model)
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.zero_grad()
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().numpy()
        if targets:
            score = torch.logsumexp(logits[0, list(targets)], dim=0)
        else:
            if class_idx is None:
                class_idx = int(probs.argmax())
            score = logits[0, class_idx]
        score.backward()
    finally:
        h1.remove()
        h2.remove()

    act, grad = store["act"][0].detach(), store["grad"][0].detach()
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(0)).numpy()
    # Köşe hücreleri sabit bir yapay etki üretebiliyor (kenar dolgusu): boş gri görüntüde de yanıyorlar ve lezyon kareyi
    # doldurduğunda haritayı tek başına bir köşeye çekebiliyorlar. Dört köşe hesaba katılmaz.
    cam[0, 0] = cam[0, -1] = cam[-1, 0] = cam[-1, -1] = 0.0
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, probs


def gradcam_flip(model, x: torch.Tensor, targets=None) -> tuple[np.ndarray, np.ndarray]:
    """Aynalı ortalama: görüntü ve yatay aynası için haritalar hesaplanır, ayna geri çevrilip ortalanır.
    Rastgele lekeleri ve kenar etkilerini azaltır (modeller eğitimde ve tahminde de aynalı görüntüyle çalışıyor)."""
    cam1, probs = gradcam(model, x, targets=targets)
    cam2, _ = gradcam(model, torch.flip(x, dims=[3]), targets=targets if targets else [int(probs.argmax())])
    cam = (cam1 + cam2[:, ::-1]) / 2
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, probs


def central_share(cam: np.ndarray, frac: float = 0.5) -> float:
    """Isının ne kadarı görüntünün ortasındaki kare bölgede (kenarın frac'ı kadar)? Düzgün dağılımda ≈ frac².
    Hasta modunda kırpma zorunlu olduğu için lezyon ortadadır; bu oran düşükse model arka plana bakıyor olabilir."""
    h, w = cam.shape
    a, b = int(round(h * (1 - frac) / 2)), int(round(w * (1 - frac) / 2))
    total = float(cam.sum())
    return float(cam[a:h - a, b:w - b].sum()) / total if total > 0 else 0.0


# koyu mor → kırmızı → turuncu → açık sarı (parlaklık tek yönde arttığı için renk körlüğünde de okunur)
HEAT_STOPS = ((0.0, (66, 10, 104)), (0.25, (120, 28, 109)), (0.5, (188, 55, 84)), (0.7, (237, 105, 37)),
              (0.85, (251, 165, 10)), (1.0, (252, 236, 130)))


def _ramp(v: np.ndarray) -> np.ndarray:
    xs = [p for p, _ in HEAT_STOPS]
    return np.stack([np.interp(v, xs, [c[i] for _, c in HEAT_STOPS]) for i in range(3)], -1).astype(np.float32)


def _smooth_field(cam: np.ndarray, n: int) -> np.ndarray:
    """Kaba ızgarayı yumuşakça büyütür (bikübik + hafif Gauss), böylece hücre köşeleri görünmez."""
    up = Image.fromarray(cam.astype(np.float32)).resize((n, n), Image.BICUBIC)
    a = np.clip(np.asarray(up, dtype=np.float32), 0, 1)
    sigma = n / 45
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    for axis in (0, 1):
        a = np.apply_along_axis(lambda v: np.convolve(np.pad(v, r, mode="edge"), k, mode="valid"), axis, a)
    return (a / max(float(a.max()), 1e-6)).astype(np.float32)


def _contour(field: np.ndarray, level: float, width: float) -> np.ndarray:
    """Kenarı yumuşatılmış eş düzey çizgisi (0-1 örtü). Her pikselin çizgiye uzaklığı ≈ |değer − düzey| / eğim."""
    gy, gx = np.gradient(field)
    dist = np.abs(field - level) / np.maximum(np.hypot(gx, gy), 1e-6)
    return np.clip(width / 2 + 0.5 - dist, 0, 1)


def _blend(base: np.ndarray, color, a: np.ndarray) -> np.ndarray:
    return base * (1 - a[..., None]) + np.asarray(color, np.float32) * a[..., None]


def overlay(img: Image.Image, cam: np.ndarray, size: int = 600) -> Image.Image:
    """Isı haritasını görüntünün üstüne bindirir: ilgi görmeyen alan hafifçe karartılır, ilgi gören bölge yarı saydam
    renklendirilir, iki ince eş düzey çizgisi (%50 ve %75) yoğunluğun şeklini gösterir. Lezyon dokusu görünür kalır."""
    base = np.asarray(img.convert("RGB").resize((size, size), Image.LANCZOS), dtype=np.float32)
    c = _smooth_field(cam, size)
    out = base * (0.66 + 0.34 * np.clip((c - 0.20) / 0.40, 0, 1))[..., None]
    out = _blend(out, _ramp(c), 0.48 * np.clip((c - 0.30) / 0.70, 0, 1) ** 0.9)
    for level, width, strength in ((0.50, 1.6, 0.60), (0.75, 2.2, 0.95)):
        out = _blend(out, (0, 0, 0), 0.22 * strength * _contour(c, level, width + 3))   # çizginin altında ince koyu gölge
        out = _blend(out, (255, 255, 255), _contour(c, level, width) * strength)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
