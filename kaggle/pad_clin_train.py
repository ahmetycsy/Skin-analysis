# %% [markdown]
# # Klinik model: MILK10k + PAD-UFES-20 (telefon fotoğrafları)
#
# Amaç: hasta modunu (yalnızca klinik / telefon fotoğrafı) güçlendirmek. Yayındaki iki klinik modelin
# (EfficientNet-B3 ve EfficientNetV2-S) **aynı ayarlarla** yeniden eğitilir; tek fark eğitime
# PAD-UFES-20 telefon fotoğraflarının ve telefon benzeri görüntü bozulmalarının eklenmesidir.
#
# **Adil karşılaştırma kuralları (değiştirmeyin):**
# - MILK10k bölünmesi yayındaki modellerle aynı (seed 42, %80/%20, 1.048 kontrol lezyonu). PAD bu kontrol kümesine girmez.
# - PAD'in %20'si **hasta bazında** ayrılır (aynı hastanın tüm fotoğrafları aynı tarafta) ve eğitimde, epoch seçiminde,
#   eşik ayarında **hiç kullanılmaz**: en sonda bir kez "telefon fotoğrafı testi" olarak değerlendirilir.
# - En iyi epoch, eski modellerde olduğu gibi MILK10k kontrol kümesinin Macro F1'ine göre seçilir.
#
# **Nasıl çalıştırılır:** Sağ panel → *Accelerator → GPU T4 x2* (ya da P100), *Internet → On*.
# *Add Data* ile iki veri seti ekleyin: MILK10k (daha önce kullandığınız) ve **PAD-UFES-20**
# (aramada "PAD-UFES-20" yazın; ~2.298 görüntülü, metadata.csv içeren veri seti). Sonra *Run All*.
# Süre: B3 ~20 dk, V2-S ~45 dk. En sonda çıkan linkten zip'i indirin.
#
# PAD-UFES-20: Pacheco ve ark., Data in Brief 2020, CC BY 4.0. MILK10k: CC BY-NC 4.0.

# %%
# ===================== AYARLAR =====================
MODELS = [  # yayındaki klinik modellerle aynı mimari, çözünürlük ve eğitim ayarları
    {"run": "clin_efficientnet_b3_pad", "timm": "efficientnet_b3", "img": 300},
    {"run": "clin_tf_efficientnetv2_s_pad", "timm": "tf_efficientnetv2_s.in21k_ft_in1k", "img": 384},
]
EPOCHS = 12
BATCH_SIZE = 24
LR = 3e-4
SEED = 42                 # yayındaki bölünmeyle aynı olması için değiştirmeyin
VAL_FRACTION = 0.2
PAD_TEST_FRACTION = 0.2   # PAD'in hasta bazında ayrılan, hiç dokunulmayan test kısmı
PHONE_AUG = True          # telefon benzeri bulanıklık / JPEG sıkıştırma / ışık değişimi
# ===================================================

import os
DATA_ROOT = os.environ.get("MILK_DATA_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("MILK_OUT_DIR", "/kaggle/working")
DEBUG_LIMIT = int(os.environ.get("DEBUG_LIMIT", "0"))   # yalnızca yerel deneme için (0 = hepsi)
if DEBUG_LIMIT:
    EPOCHS = int(os.environ.get("DEBUG_EPOCHS", "1"))
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# 1) Dosyaları bul
import glob, io, json, re, random, time, zipfile
import numpy as np
import pandas as pd

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
PAD_MAP = {"BCC": "BCC", "SCC": "SCCKA", "ACK": "AKIEC", "SEK": "BKL", "MEL": "MEL", "NEV": "NV"}
IMG_EXT = (".jpg", ".jpeg", ".png")


def walk(root):
    csvs, imgs, zips = [], [], []
    for d, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            p, fl = os.path.join(d, f), f.lower()
            if fl.endswith(".csv"):
                csvs.append(p)
            elif fl.endswith(IMG_EXT):
                imgs.append(p)
            elif fl.endswith(".zip"):
                zips.append(p)
    return sorted(csvs), imgs, zips


csv_files, img_files, zip_files = walk(DATA_ROOT)
if zip_files and not any("ISIC_" in os.path.basename(p) for p in img_files):  # MILK10k görüntüleri zip içinde kaldıysa
    unz = os.path.join(OUT_DIR, "unzipped")
    for z in zip_files:
        if "test" not in os.path.basename(z).lower():
            print("Zip açılıyor:", z)
            zipfile.ZipFile(z).extractall(unz)
    c2, i2, _ = walk(unz)
    csv_files += c2
    img_files += i2
print(f"{len(img_files)} görüntü, {len(csv_files)} CSV")
tables = {p: pd.read_csv(p, low_memory=False) for p in csv_files}
img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}

# %%
# 2) MILK10k: lezyon tablosu ve yayındakiyle AYNI bölünme (kod eski notebook'tan birebir)
def is_test(p):
    return "test" in p.lower().replace(DATA_ROOT.lower(), "")


def find_col(df, pattern):
    for c in df.columns:
        if re.search(pattern, c.lower()):
            return c
    return None


gt_path = next(p for p, df in tables.items() if all(c in df.columns for c in CLASSES) and not is_test(p))
gt = tables[gt_path].copy()
meta_path = TYPE_COL = None
for p, df in tables.items():
    if p == gt_path or is_test(p) or any("diagnos" in c.lower() for c in df.columns):
        continue
    for c in df.columns:
        if df[c].astype(str).str.lower().str.contains("dermoscop").any():
            meta_path, TYPE_COL = p, c
            break
    if meta_path:
        break
meta = tables[meta_path].copy()
IMG_ID, LES_ID = find_col(meta, r"isic_id|image_id|image_name|^image$"), find_col(meta, r"lesion")
GT_ID = next(c for c in gt.columns if c not in CLASSES)
meta[IMG_ID] = meta[IMG_ID].astype(str)
meta[LES_ID] = meta[LES_ID].astype(str)
t = meta[TYPE_COL].astype(str).str.lower()
meta["_type"] = np.where(t.str.contains("dermoscop"), "derm", np.where(t.str.contains("clinic|close"), "clin", "other"))
meta["_path"] = meta[IMG_ID].map(img_map)
gt[GT_ID] = gt[GT_ID].astype(str)
gt["lesion_id"] = gt[GT_ID] if gt[GT_ID].isin(meta[LES_ID]).mean() > 0.5 else gt[GT_ID].map(dict(zip(meta[IMG_ID], meta[LES_ID])))
gt = gt.dropna(subset=["lesion_id"]).drop_duplicates("lesion_id")
rows = []
for les, g in meta.groupby(LES_ID):
    c = g[g._type == "clin"]
    rows.append({"lesion_id": les, "clin_path": c._path.dropna().iloc[0] if c._path.notna().any() else None})
milk = pd.DataFrame(rows).merge(gt[["lesion_id"] + CLASSES], on="lesion_id", how="inner")
milk["label"] = milk[CLASSES].values.argmax(1)
milk = milk.sort_values("lesion_id").reset_index(drop=True)

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
random.seed(SEED)
np.random.seed(SEED)
_, va_idx = train_test_split(np.arange(len(milk)), test_size=VAL_FRACTION, stratify=milk.label, random_state=SEED)
milk["fold"] = "train"
milk.loc[va_idx, "fold"] = "val"
milk["source"] = "milk"
milk["img_key"] = milk.lesion_id
print("MILK10k:", milk.groupby("fold").size().to_dict(), "| klinik görüntüsü olmayan:", milk.clin_path.isna().sum())
split_file = glob.glob(f"{DATA_ROOT}/**/split_seed42.csv", recursive=True)
if split_file:  # elinizde yayındaki bölünme dosyası varsa birebir aynı mı kontrol edilir
    ref = pd.read_csv(split_file[0]).set_index("lesion_id").fold
    same = (milk.set_index("lesion_id").fold == ref.reindex(milk.lesion_id).values).mean()
    print(f"Yayındaki bölünmeyle uyum: %{same * 100:.1f}")
    assert same > 0.999, "Bölünme yayındakiyle aynı değil!"

# %%
# 3) PAD-UFES-20: türleri eşle, hasta bazında %20'yi test olarak ayır
pad_meta_path = next((p for p, df in tables.items() if {"diagnostic", "img_id", "patient_id"} <= set(df.columns)), None)
assert pad_meta_path, "PAD-UFES-20 metadata.csv bulunamadı. Add Data ile PAD-UFES-20 veri setini ekleyin."
pad = tables[pad_meta_path].copy()
pad["label_code"] = pad.diagnostic.map(PAD_MAP)
pad = pad.dropna(subset=["label_code"])
pad["label"] = pad.label_code.map(CLASSES.index)
pad["clin_path"] = pad.img_id.astype(str).map(lambda s: img_map.get(os.path.splitext(s)[0]))
print("PAD görüntüsü bulunamayan:", pad.clin_path.isna().sum())
pad = pad.dropna(subset=["clin_path"]).reset_index(drop=True)
sgkf = StratifiedGroupKFold(n_splits=round(1 / PAD_TEST_FRACTION), shuffle=True, random_state=SEED)
_, te_idx = next(sgkf.split(pad, pad.label, groups=pad.patient_id))
pad["fold"] = "train"
pad.loc[te_idx, "fold"] = "pad_test"
pad["source"] = "pad"
pad["img_key"] = pad.img_id.astype(str)
assert not set(pad[pad.fold == "train"].patient_id) & set(pad[pad.fold == "pad_test"].patient_id), "hasta sızıntısı!"
print("PAD:", pad.groupby("fold").size().to_dict(), "| hasta:", pad.patient_id.nunique())
print(pad.groupby(["fold", "label_code"]).size().unstack(0).to_string())

milk["patient_id"] = None  # MILK10k'te hasta kimliği yok
cols = ["img_key", "lesion_id", "patient_id", "clin_path", "label", "fold", "source"]
data = pd.concat([milk[milk.clin_path.notna()][cols], pad[cols]], ignore_index=True)
if DEBUG_LIMIT:
    data = data.groupby(["source", "fold"], group_keys=False).apply(lambda g: g.head(DEBUG_LIMIT)).reset_index(drop=True)
data.to_csv(f"{OUT_DIR}/pad_split.csv", index=False)
print("Eğitim:", (data.fold == "train").sum(), "| MILK kontrol:", (data.fold == "val").sum(), "| PAD test:", (data.fold == "pad_test").sum())

# %%
# 4) Eğitim (her model için aynı döngü)
import torch
import torch.nn as nn
import timm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.metrics import f1_score, balanced_accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", device)


class PhoneLike:
    """Telefon fotoğrafına benzeyen bozulmalar: hafif bulanıklık, JPEG sıkıştırma (rastgele)."""

    def __call__(self, im):
        if random.random() < 0.3:
            im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.5)))
        if random.random() < 0.4:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=random.randint(35, 90))
            im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return im


def train_one(spec):
    run, size = spec["run"], spec["img"]
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    cfg = timm.create_model(spec["timm"], pretrained=False).pretrained_cfg
    mean, std = tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    cache_px = int(size * 1.15)

    def load_resized(path):  # sitedeki preprocess() ile aynı: ortadan kare kırp, büyüt/küçült
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.asarray(im.resize((cache_px, cache_px), Image.BILINEAR))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 2) * 2) as pool:
        cache = list(pool.map(load_resized, data.clin_path.tolist()))
    print(f"[{run}] {len(cache)} görüntü belleğe alındı ({time.time() - t0:.0f} sn)")

    aug = [T.RandomResizedCrop(size, scale=(0.5 if PHONE_AUG else 0.7, 1.0)),
           T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(30),
           T.ColorJitter(0.3 if PHONE_AUG else 0.2, 0.3 if PHONE_AUG else 0.2, 0.15 if PHONE_AUG else 0.1, 0.02)]
    if PHONE_AUG:
        aug.insert(0, PhoneLike())
    tf_train = T.Compose(aug + [T.ToTensor(), T.Normalize(mean, std)])
    tf_val = T.Compose([T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, idx, tf):
            self.idx, self.tf = list(idx), tf

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            return self.tf(Image.fromarray(cache[j])), int(data.label[j])

    tr = np.where(data.fold == "train")[0]
    va = np.where(data.fold == "val")[0]
    te = np.where(data.fold == "pad_test")[0]
    nw = min(4, os.cpu_count() or 1) if device == "cuda" else 0
    dl_tr = DataLoader(DS(tr, tf_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True)
    dl_va = DataLoader(DS(va, tf_val), batch_size=BATCH_SIZE * 2, num_workers=nw)
    dl_te = DataLoader(DS(te, tf_val), batch_size=BATCH_SIZE * 2, num_workers=nw)

    counts = np.bincount(data.label[tr], minlength=len(CLASSES)).astype(float)
    w = (counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** 0.5  # eski notebook'la aynı ağırlıklandırma
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32, device=device), label_smoothing=0.05)
    model = timm.create_model(spec["timm"], pretrained=not DEBUG_LIMIT, num_classes=len(CLASSES), drop_rate=0.3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(dl_tr), pct_start=0.1)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def predict(loader):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                p = model(x).float().softmax(1) + model(torch.flip(x, dims=[3])).float().softmax(1)
                out.append((p / 2).cpu().numpy())
        return np.concatenate(out)

    best, hist = (-1, None), []
    y_va = data.label[va].values
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        f1 = f1_score(y_va, predict(dl_va).argmax(1), average="macro")
        hist.append({"epoch": ep, "loss": tot / len(dl_tr), "val_macro_f1": f1})
        flag = ""
        if f1 > best[0]:
            best = (f1, ep)
            torch.save(model.state_dict(), f"{OUT_DIR}/{run}.pt")
            flag = "  ← en iyi, kaydedildi"
        print(f"[{run}] epoch {ep:2d}/{EPOCHS} | kayıp {tot / len(dl_tr):.3f} | MILK kontrol Macro F1 {f1:.4f} | "
              f"{time.time() - t0:.0f} sn{flag}")

    # en iyi ağırlıkla son tahminler: MILK kontrol + PAD test (PAD test burada İLK kez kullanılır)
    model.load_state_dict(torch.load(f"{OUT_DIR}/{run}.pt", map_location=device))
    p_va, p_te = predict(dl_va), predict(dl_te)
    out = data.iloc[np.r_[va, te]][["img_key", "lesion_id", "label", "fold", "source"]].copy()
    out[CLASSES] = np.vstack([p_va, p_te])
    out.to_csv(f"{OUT_DIR}/preds_{run}.csv", index=False)
    y_te = data.label[te].values
    res = {"run": run, "timm": spec["timm"], "img_size": size, "mean": list(mean), "std": list(std),
           "best_epoch": best[1], "milk_val_macro_f1": round(float(best[0]), 4),
           "pad_test_macro_f1_6cls": round(float(f1_score(y_te, p_te.argmax(1), average="macro",
                                                         labels=sorted(set(y_te)))), 4) if len(te) else None,
           "pad_test_balanced_acc": round(float(balanced_accuracy_score(y_te, p_te.argmax(1))), 4) if len(te) else None,
           "phone_aug": PHONE_AUG, "epochs": EPOCHS, "n_train": int(len(tr)), "history": hist}
    json.dump(res, open(f"{OUT_DIR}/result_{run}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))
    del model, cache
    if device == "cuda":
        torch.cuda.empty_cache()


for spec in MODELS:
    train_one(spec)

# %%
# 5) PAD test görüntülerini de pakete koy (yerelde eski ve yeni modeli AYNI görüntülerle karşılaştırmak için).
#    PAD-UFES-20 CC BY 4.0; uzun kenar 1024 px'e küçültülür. Bu dosyalar GitHub'a konmaz.
os.makedirs(f"{OUT_DIR}/pad_test", exist_ok=True)
for _, r in data[data.fold == "pad_test"].iterrows():
    im = Image.open(r.clin_path).convert("RGB")
    im.thumbnail((1024, 1024))
    im.save(f"{OUT_DIR}/pad_test/{r.img_key}.jpg", quality=95)
print("PAD test görüntüleri:", len(os.listdir(f"{OUT_DIR}/pad_test")))

# %%
# 6) Hepsini tek zip'e koy ve indirme linki ver (oturum kapanınca /kaggle/working silinir!)
try:
    from IPython.display import FileLink, display
except ImportError:  # yerel deneme
    FileLink, display = str, print

zip_name = "pad_clin_sonuclar.zip"
os.chdir(OUT_DIR)
with zipfile.ZipFile(zip_name, "w") as z:
    for f in glob.glob("*.pt") + glob.glob("preds_*.csv") + glob.glob("result_*.json") + ["pad_split.csv"] + glob.glob("pad_test/*.jpg"):
        z.write(f)
print("Boyut:", round(os.path.getsize(zip_name) / 1e6), "MB")
display(FileLink(zip_name))
