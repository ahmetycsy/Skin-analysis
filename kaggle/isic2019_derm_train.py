# %% [markdown]
# # Dermoskopik model: MILK10k + ISIC 2019
#
# Amaç: hekim modunu güçlendirmek. Yayındaki iki dermoskopik model (EfficientNet-B3 ve EfficientNetV2-S) **aynı
# ayarlarla** yeniden eğitilir; tek fark eğitime ISIC 2019'un ~25.000 dermoskopik görüntüsünün eklenmesidir.
#
# **Adil karşılaştırma kuralları (değiştirmeyin):**
# - MILK10k bölünmesi yayındaki modellerle aynı (seed 42, %80/%20, 1.048 kontrol lezyonu).
# - **Kopya temizliği:** MILK10k ile ISIC 2019 aynı arşivden gelir. MILK10k'in TÜM görüntülerine karşı
#   (1) aynı ISIC numarası, (2) görüntü benzerliği (256 bit fark özeti, dHash) ile kopyalar ISIC 2019'dan çıkarılır.
#   Böylece kontrol kümesindeki bir lezyon ISIC 2019 üzerinden eğitime sızamaz.
# - En iyi epoch MILK10k kontrol kümesinin Macro F1'ine göre seçilir (eski modellerde olduğu gibi).
# - ISIC 2019'daki ben (NV) sayısı çok fazla (~12.900); MILK10k'in tür dağılımını ezmemesi için NV_CAP ile sınırlanır.
#
# **Nasıl çalıştırılır:** Sağ panel → *Accelerator → GPU T4 x2*, *Internet → On*. *Add Input* ile iki veri seti:
# MILK10k (Bau Capstone) ve **ISIC 2019** (aramada "ISIC 2019" yazın; ~25.331 görüntülü, içinde
# ISIC_2019_Training_GroundTruth.csv olan). Sonra *Save Version → Save & Run All*. Süre ~2–3 saat.
#
# ISIC 2019: Tschandl ve ark. 2018 (HAM10000), Codella ve ark. 2017, Combalia ve ark. 2019 (BCN20000); CC BY-NC 4.0.

# %%
# ===================== AYARLAR =====================
MODELS = [  # yayındaki dermoskopik modellerle aynı mimari, çözünürlük ve eğitim ayarları
    {"run": "derm_efficientnet_b3_isic", "timm": "efficientnet_b3", "img": 300},
    {"run": "derm_tf_efficientnetv2_s_isic", "timm": "tf_efficientnetv2_s.in21k_ft_in1k", "img": 384},
]
EPOCHS = 12
BATCH_SIZE = 24
LR = 3e-4
SEED = 42            # yayındaki bölünmeyle aynı olması için değiştirmeyin
VAL_FRACTION = 0.2
NV_CAP = 4000        # ISIC 2019'dan en çok bu kadar ben (NV) görüntüsü alınır
HASH_DIST = 22       # 256 bitlik dHash farkı bu değer ya da altındaysa "aynı görüntü" sayılır. MILK10k'te denendi:
                     # aynı fotoğraf (%80 küçültme + JPEG 70) en çok 20, farklı fotoğraflar en az 28 uzaklıkta.
# ===================================================

import os
DATA_ROOT = os.environ.get("MILK_DATA_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("MILK_OUT_DIR", "/kaggle/working")
DEBUG_LIMIT = int(os.environ.get("DEBUG_LIMIT", "0"))   # yalnızca yerel deneme için (0 = hepsi)
if DEBUG_LIMIT:
    EPOCHS = int(os.environ.get("DEBUG_EPOCHS", "1"))
os.makedirs(OUT_DIR, exist_ok=True)

# %%
# 1) Dosyaları bul (MILK10k ve ISIC 2019 görüntüleri ayrı tutulur: aynı numara iki sette de olabilir)
import glob, io, json, re, random, time, zipfile
import numpy as np
import pandas as pd

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
ISIC_MAP = {"MEL": "MEL", "NV": "NV", "BCC": "BCC", "AK": "AKIEC", "BKL": "BKL", "DF": "DF", "VASC": "VASC", "SCC": "SCCKA"}
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
milk_imgs = [p for p in img_files if os.path.basename(os.path.dirname(p)).startswith("IL_")]  # MILK10k: IL_xxx/ISIC_xxx.jpg
if not milk_imgs and zip_files:  # MILK10k görüntüleri zip içinde kaldıysa
    unz = os.path.join(OUT_DIR, "unzipped")
    for z in zip_files:
        if "test" not in os.path.basename(z).lower() and "milk" in os.path.basename(z).lower():
            print("Zip açılıyor:", z)
            zipfile.ZipFile(z).extractall(unz)
    c2, i2, _ = walk(unz)
    csv_files += c2
    milk_imgs = [p for p in i2 if os.path.basename(os.path.dirname(p)).startswith("IL_")]
milk_set = set(milk_imgs)
other_imgs = [p for p in img_files if p not in milk_set]
milk_map = {os.path.splitext(os.path.basename(p))[0]: p for p in milk_imgs}
isic_map = {os.path.splitext(os.path.basename(p))[0].replace("_downsampled", ""): p for p in other_imgs}
print(f"MILK10k görüntüsü: {len(milk_map)} | diğer görüntü: {len(isic_map)} | CSV: {len(csv_files)}")
tables = {p: pd.read_csv(p, low_memory=False) for p in csv_files}

# %%
# 2) MILK10k: dermoskopik lezyon tablosu ve yayındakiyle AYNI bölünme (kod eski notebook'tan birebir)
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
meta["_path"] = meta[IMG_ID].map(milk_map)
gt[GT_ID] = gt[GT_ID].astype(str)
gt["lesion_id"] = gt[GT_ID] if gt[GT_ID].isin(meta[LES_ID]).mean() > 0.5 else gt[GT_ID].map(dict(zip(meta[IMG_ID], meta[LES_ID])))
gt = gt.dropna(subset=["lesion_id"]).drop_duplicates("lesion_id")
rows = []
for les, g in meta.groupby(LES_ID):
    d = g[g._type == "derm"]
    rows.append({"lesion_id": les, "path": d._path.dropna().iloc[0] if d._path.notna().any() else None})
milk = pd.DataFrame(rows).merge(gt[["lesion_id"] + CLASSES], on="lesion_id", how="inner")
milk["label"] = milk[CLASSES].values.argmax(1)
milk = milk.sort_values("lesion_id").reset_index(drop=True)

from sklearn.model_selection import train_test_split
random.seed(SEED)
np.random.seed(SEED)
_, va_idx = train_test_split(np.arange(len(milk)), test_size=VAL_FRACTION, stratify=milk.label, random_state=SEED)
milk["fold"] = "train"
milk.loc[va_idx, "fold"] = "val"
milk["source"] = "milk"
milk["img_key"] = milk.lesion_id
print("MILK10k:", milk.groupby("fold").size().to_dict(), "| dermoskopik görüntüsü olmayan:", milk.path.isna().sum())
split_file = glob.glob(f"{DATA_ROOT}/**/split_seed42.csv", recursive=True)
if split_file:
    ref = pd.read_csv(split_file[0]).set_index("lesion_id").fold
    same = (milk.set_index("lesion_id").fold == ref.reindex(milk.lesion_id).values).mean()
    print(f"Yayındaki bölünmeyle uyum: %{same * 100:.1f}")
    assert same > 0.999, "Bölünme yayındakiyle aynı değil!"
milk_all_ids = set(meta[IMG_ID])  # MILK10k'in klinik + dermoskopik tüm görüntü numaraları

# %%
# 3) ISIC 2019: türleri eşle, aynı numaralıları çıkar, ben (NV) sayısını sınırla
isic_gt_path = next((p for p, df in tables.items() if {"MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"} <= set(df.columns)), None)
FOLDER_WORDS = [("squamous", "SCCKA"), ("basal", "BCC"), ("actinic", "AKIEC"), ("melanoma", "MEL"), ("nev", "NV"),
                ("keratos", "BKL"), ("dermatofib", "DF"), ("vascular", "VASC")]  # sıra önemli (ör. "seborrheic keratosis")
if isic_gt_path:  # ISIC'in orijinal cevap dosyası (tek sıcak kodlu sütunlar)
    print("ISIC 2019 etiketleri:", isic_gt_path.replace(DATA_ROOT, ""))
    ig = tables[isic_gt_path].copy()
    id_col = find_col(ig, r"^image$|image_id|image_name|isic_id")
    ig["img_key"] = ig[id_col].astype(str).str.replace("_downsampled", "", regex=False)
    cols = list(ISIC_MAP)
    ig = ig[ig[cols].sum(1) == 1]                     # UNK ve çok etiketliler dışarıda
    ig["label"] = ig[cols].values.argmax(1)
    ig["label"] = ig.label.map(lambda i: CLASSES.index(ISIC_MAP[cols[i]]))
    ig["path"] = ig.img_key.map(isic_map)
else:  # yedek: görüntüler tür adlı klasörlerde (ör. .../AK/ISIC_0024468.jpg ya da .../Melanoma/ISIC_0000002.jpg)
    rows = []
    for key, path in isic_map.items():
        folder = os.path.basename(os.path.dirname(path))
        code = ISIC_MAP.get(folder.strip().upper()) or next((c for w, c in FOLDER_WORDS if w in folder.lower()), None)
        if code and key.startswith("ISIC_"):
            rows.append({"img_key": key, "label": CLASSES.index(code), "path": path})
    assert rows, "ISIC 2019 bulunamadı: ne GroundTruth CSV'si ne de tür adlı klasörler var. Add Input ile ISIC 2019'u ekleyin."
    ig = pd.DataFrame(rows).drop_duplicates("img_key")
    print("ISIC 2019 etiketleri klasör adlarından okundu:",
          sorted({os.path.basename(os.path.dirname(p)) for p in ig.path})[:12])
report = {"isic_rows": int(len(ig)), "isic_missing_image": int(ig.path.isna().sum())}
ig = ig.dropna(subset=["path"])
same_id = ig.img_key.isin(milk_all_ids)
report["removed_same_isic_id"] = int(same_id.sum())
print("MILK10k ile aynı ISIC numaralı görüntü:", int(same_id.sum()), list(ig.img_key[same_id])[:10])
ig = ig[~same_id]
nv = ig[ig.label == CLASSES.index("NV")]
if len(nv) > NV_CAP:
    ig = pd.concat([ig[ig.label != CLASSES.index("NV")], nv.sample(NV_CAP, random_state=SEED)])
ig = ig.sort_values("img_key").reset_index(drop=True)
if DEBUG_LIMIT:  # yerel deneme: MILK'ten az lezyon (+ DEBUG_KEEP ile belirtilenler), ISIC'ten tür başına birkaç görüntü
    keep = os.environ.get("DEBUG_KEEP", "").split(",")
    milk = milk[milk.lesion_id.isin(keep) | milk.groupby("fold").cumcount().lt(DEBUG_LIMIT)].reset_index(drop=True)
    ig = ig[ig.img_key.isin(keep) | ig.groupby("label").cumcount().lt(max(2, DEBUG_LIMIT // 4))].reset_index(drop=True)
ig["fold"], ig["source"], ig["lesion_id"] = "train", "isic2019", ig.img_key
print("ISIC 2019 (kopya temizliği öncesi):", len(ig))
print(ig.label.map(dict(enumerate(CLASSES))).value_counts().to_string())

# %%
# 4) Görüntüleri bir kez oku: MILK10k için ham dizi (eski notebook'la birebir ön işleme), ISIC 2019 için
#    sıkıştırılmış JPEG (bellek), her ikisi için dHash. Sonra ISIC 2019'dan MILK10k'e benzeyenleri çıkar.
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

SIZES = sorted({int(m["img"] * 1.15) for m in MODELS})


def square(im):
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def dhash(im):  # 16x16 = 256 bitlik fark özeti (32 bayt)
    small = im.copy()
    small.thumbnail((256, 256))
    g = np.asarray(small.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def load_milk(path):
    im = square(Image.open(path).convert("RGB"))
    return dhash(im), {s: np.asarray(im.resize((s, s), Image.BILINEAR)) for s in SIZES}


def load_isic(path):
    im = square(Image.open(path).convert("RGB"))
    out = {}
    for s in SIZES:
        buf = io.BytesIO()
        im.resize((s, s), Image.BILINEAR).save(buf, format="JPEG", quality=95)
        out[s] = buf.getvalue()
    return dhash(im), out


t0 = time.time()
milk = milk[milk.path.notna()].reset_index(drop=True)
with ThreadPoolExecutor(max_workers=(os.cpu_count() or 2) * 2) as pool:
    m_res = list(pool.map(load_milk, milk.path.tolist()))
    i_res = list(pool.map(load_isic, ig.path.tolist()))
print(f"{len(m_res) + len(i_res)} görüntü okundu ({time.time() - t0:.0f} sn)")

m_hash = np.stack([h for h, _ in m_res])                     # (MILK, 32) bayt
i_hash = np.stack([h for h, _ in i_res]) if i_res else np.zeros((0, 32), np.uint8)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
min_dist = np.full(len(i_hash), 256)
for a in range(0, len(i_hash), 300):
    x = np.bitwise_xor(i_hash[a:a + 300, None, :], m_hash[None, :, :])
    min_dist[a:a + 300] = POP[x].sum(2).min(1)
for k in (0, 10, 16, 22, 28, 34):
    report[f"near_dup_dist_le_{k}"] = int((min_dist <= k).sum())
print("MILK10k'e en yakın uzaklık dağılımı:", {k: report[f"near_dup_dist_le_{k}"] for k in (0, 10, 16, 22, 28, 34)})
dup = min_dist <= HASH_DIST
report["removed_near_duplicate"] = int(dup.sum())
ig = ig[~dup].reset_index(drop=True)
i_res = [r for r, d in zip(i_res, dup) if not d]
report["isic_used"] = int(len(ig))
report["isic_used_per_class"] = ig.label.map(dict(enumerate(CLASSES))).value_counts().to_dict()
json.dump(report, open(f"{OUT_DIR}/dedup_report.json", "w"), indent=2)
print("Kopya temizliği:", json.dumps(report, indent=1))

data = pd.concat([milk[["img_key", "lesion_id", "label", "fold", "source"]], ig[["img_key", "lesion_id", "label", "fold", "source"]]],
                 ignore_index=True)
cache = {s: [r[1][s] for r in m_res] + [r[1][s] for r in i_res] for s in SIZES}   # dizi (MILK) ya da JPEG baytı (ISIC)
del m_res, i_res
data[["img_key", "label", "fold", "source"]].to_csv(f"{OUT_DIR}/isic_split.csv", index=False)
print("Eğitim:", (data.fold == "train").sum(), "| MILK kontrol:", (data.fold == "val").sum())

# %%
# 5) Eğitim (her model için aynı döngü; ayarlar eski notebook'la aynı)
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from sklearn.metrics import f1_score, balanced_accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Cihaz:", device)


def train_one(spec):
    run, size = spec["run"], spec["img"]
    store = cache[int(size * 1.15)]
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    cfg = timm.create_model(spec["timm"], pretrained=False).pretrained_cfg
    mean, std = tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    tf_train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(30),
        T.ColorJitter(0.2, 0.2, 0.1, 0.02),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    tf_val = T.Compose([T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, idx, tf):
            self.idx, self.tf = list(idx), tf

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            v = store[j]
            im = Image.fromarray(v) if isinstance(v, np.ndarray) else Image.open(io.BytesIO(v)).convert("RGB")
            return self.tf(im), int(data.label[j])

    tr = np.where(data.fold == "train")[0]
    va = np.where(data.fold == "val")[0]
    nw = min(4, os.cpu_count() or 1) if device == "cuda" else 0
    dl_tr = DataLoader(DS(tr, tf_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, drop_last=True, pin_memory=True)
    dl_va = DataLoader(DS(va, tf_val), batch_size=BATCH_SIZE * 2, num_workers=nw)

    counts = np.bincount(data.label[tr], minlength=len(CLASSES)).astype(float)
    w = (counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** 0.5
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

    model.load_state_dict(torch.load(f"{OUT_DIR}/{run}.pt", map_location=device))
    out = data.iloc[va][["img_key", "lesion_id", "label", "fold", "source"]].copy()
    out[CLASSES] = predict(dl_va)
    out.to_csv(f"{OUT_DIR}/preds_{run}.csv", index=False)
    res = {"run": run, "timm": spec["timm"], "img_size": size, "mean": list(mean), "std": list(std),
           "best_epoch": best[1], "milk_val_macro_f1": round(float(best[0]), 4),
           "milk_val_balanced_acc": round(float(balanced_accuracy_score(y_va, out[CLASSES].values.argmax(1))), 4),
           "epochs": EPOCHS, "n_train": int(len(tr)), "nv_cap": NV_CAP, "hash_dist": HASH_DIST, "history": hist}
    json.dump(res, open(f"{OUT_DIR}/result_{run}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1))
    del model
    if device == "cuda":
        torch.cuda.empty_cache()


for spec in MODELS:
    train_one(spec)

# %%
# 6) Hepsini tek zip'e koy ve indirme linki ver (oturum kapanınca /kaggle/working silinir!)
try:
    from IPython.display import FileLink, display
except ImportError:  # yerel deneme
    FileLink, display = str, print

zip_name = "isic_derm_sonuclar.zip"
os.chdir(OUT_DIR)
with zipfile.ZipFile(zip_name, "w") as z:
    for f in glob.glob("*.pt") + glob.glob("preds_*.csv") + glob.glob("result_*.json") + ["isic_split.csv", "dedup_report.json"]:
        z.write(f)
print("Boyut:", round(os.path.getsize(zip_name) / 1e6), "MB")
display(FileLink(zip_name))
