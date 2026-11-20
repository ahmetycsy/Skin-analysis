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
    # timm EfficientNet: sondan ikinci blok grubu. En son katman (bn2) denendi; 20 doğrulama görüntüsünün
    # 19'unda en sıcak nokta görüntü köşesine kayıyordu (kenar dolgusu etkisi). blocks[-2]'de bu 6/20'ye düşüyor.
    if hasattr(model, "blocks"):
        return model.blocks[-2]
    return list(model.children())[-3]


def gradcam(model, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(ısı haritası [0-1], softmax olasılıkları) döner. class_idx verilmezse en olası sınıf kullanılır."""
    store = {}
    layer = _target_layer(model)
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.zero_grad()
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].detach().numpy()
        if class_idx is None:
            class_idx = int(probs.argmax())
        logits[0, class_idx].backward()
    finally:
        h1.remove()
        h2.remove()

    act, grad = store["act"][0].detach(), store["grad"][0].detach()
    weights = grad.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * act).sum(0)).numpy()
    # Sol üst hücre sabit bir yapay etkidir: düz gri, boş bir görüntüde bile tek yanan hücre budur
    # (köşedeki asimetrik kenar dolgusu). Görüntü içeriğiyle ilgisi olmadığı için hesaba katılmaz.
    cam[0, 0] = 0.0
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, probs


def _jet(v: np.ndarray) -> np.ndarray:
    """0-1 aralığındaki değerleri mavi→yeşil→sarı→kırmızı renklere çevirir (matplotlib gerekmez)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Isı haritasını görüntünün üstüne yarı saydam bindirir. Zayıf bölgeler neredeyse şeffaf kalır."""
    size = img.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
    c = np.asarray(cam_img, dtype=np.float32) / 255.0
    heat = _jet(c).astype(np.float32)
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    a = (alpha * np.clip(c * 1.4, 0, 1))[..., None]
    out = base * (1 - a) + heat * a
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
