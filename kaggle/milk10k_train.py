# %% [markdown]
# # MILK10k eğitim notebook'u (ortak)
#
# Bu notebook MILK10k verisini okur, herkes için **aynı** eğitim/kontrol bölünmesini yapar (seed 42)
# ve seçilen **tek bir** modeli eğitir:
#
# | MODE | Ne eğitir | Kim kullanır |
# |---|---|---|
# | `"derm"` | Yalnız dermoskopik görüntü modeli (EfficientNet) | Tuğalp |
# | `"clin"` | Yalnız klinik yakın çekim modeli (EfficientNet) | Tuğalp |
# | `"meta"` | Yalnız hasta bilgisi modeli (lojistik regresyon) | Bedirhan |
#
# **Nasıl çalıştırılır:** Sağ panelden *Accelerator → GPU T4*, *Internet → On* yapın. Aşağıdaki
# AYARLAR hücresinde `MODE`'u seçin, sonra *Run All*.
#
# **Çıktılar** (`/kaggle/working`): `results/<isim>.json` (Macro F1 + sınıf bazlı sonuçlar),
# `val_probs_<isim>.npy` (kontrol setindeki olasılıklar, füzyon için), `val_ids_<isim>.csv`,
# model dosyası ve `split_seed42.csv` (bölünme listesi).
#
# Not: "supplemental" dosyasındaki ayrıntılı tanı bilgisi cevabın kendisidir; modele **verilmez**.

# %%
# ===================== AYARLAR (yalnızca burayı değiştirin) =====================
MODE = "derm"                    # "derm", "clin" veya "meta"
MODEL_NAME = "efficientnet_b3"   # GPU belleği yetmezse "efficientnet_b0"
PRETRAINED = True                # ImageNet ön eğitimli ağırlıklar (Internet açık olmalı)
IMG_SIZE = 300                   # B3 için 300, B0 için 224
EPOCHS = 12
BATCH_SIZE = 24
LR = 3e-4
SEED = 42                        # herkes aynı seed'i kullanmalı, değiştirmeyin
VAL_FRACTION = 0.2
# ===============================================================================

import os
DATA_ROOT = os.environ.get("MILK_DATA_ROOT", "/kaggle/input")
OUT_DIR = os.environ.get("MILK_OUT_DIR", "/kaggle/working")
RUN_NAME = "meta_logreg" if MODE == "meta" else f"{MODE}_{MODEL_NAME}"
assert MODE in ("derm", "clin", "meta"), "MODE 'derm', 'clin' veya 'meta' olmalı"
print("Çalıştırma adı:", RUN_NAME)

# %%
# 1) Dosyaları bul ve göster (sütun adlarını buradan kontrol edebilirsiniz)
import glob, json, re, random, time, zipfile
import numpy as np
import pandas as pd

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
IMG_EXT = (".jpg", ".jpeg", ".png")
os.makedirs(OUT_DIR, exist_ok=True)


def walk(root):
    csvs, imgs, zips = [], [], []
    for d, _, fs in os.walk(root):
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

# Görüntüler zip içinde kaldıysa bir kere aç
if not img_files and zip_files:
    unz = os.path.join(OUT_DIR, "unzipped")
    for z in zip_files:
        if "test" in os.path.basename(z).lower():
            continue
        print("Zip açılıyor:", z)
        zipfile.ZipFile(z).extractall(unz)
    c2, img_files, _ = walk(unz)
    csv_files += c2

print(f"{len(img_files)} görüntü, {len(csv_files)} CSV bulundu\n")
for p in csv_files:
    head = pd.read_csv(p, nrows=3)
    print("•", p.replace(DATA_ROOT, ""), "→ sütunlar:", list(head.columns))

# %%
# 2) Lezyon tablosunu kur (her lezyon: klinik + dermoskopik görüntü + hasta bilgisi + doğru sınıf)
tables = {p: pd.read_csv(p) for p in csv_files}


def is_test(p):
    return "test" in p.lower().replace(DATA_ROOT.lower(), "")


def find_col(df, pattern):
    for c in df.columns:
        if re.search(pattern, c.lower()):
            return c
    return None


# Doğru cevaplar: 11 sınıf sütununun hepsini içeren tablo
gt_path = next(p for p, df in tables.items() if all(c in df.columns for c in CLASSES) and not is_test(p))
gt = tables[gt_path]

# Hasta bilgisi tablosu: değerlerinde "dermoscop" geçen bir sütunu olan, tanı bilgisi İÇERMEYEN tablo
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
assert meta_path, "Hasta bilgisi tablosu bulunamadı; yukarıdaki sütun listesini kontrol edin."
meta = tables[meta_path].copy()

IMG_ID = find_col(meta, r"isic_id|image_id|image_name|^image$")
LES_ID = find_col(meta, r"lesion")
PAT_ID = find_col(meta, r"patient")
AGE_COL = find_col(meta, r"(^|_)age(_|$)")
SEX_COL = find_col(meta, r"(^|_)(sex|gender)(_|$)")
SKIN_COL = find_col(meta, r"skin|fitz")
SITE_COL = find_col(meta, r"site|anatom|location")
GT_ID = next(c for c in gt.columns if c not in CLASSES)

print("Doğru cevaplar :", gt_path.replace(DATA_ROOT, ""), "| kimlik sütunu:", GT_ID)
print("Hasta bilgisi  :", meta_path.replace(DATA_ROOT, ""))
print("Algılanan sütunlar:", dict(görüntü=IMG_ID, lezyon=LES_ID, hasta=PAT_ID, tür=TYPE_COL,
                                  yaş=AGE_COL, cinsiyet=SEX_COL, ten=SKIN_COL, bölge=SITE_COL))
assert IMG_ID and LES_ID, "Görüntü veya lezyon kimliği sütunu bulunamadı."

meta[IMG_ID] = meta[IMG_ID].astype(str)
meta[LES_ID] = meta[LES_ID].astype(str)
t = meta[TYPE_COL].astype(str).str.lower()
meta["_type"] = np.where(t.str.contains("dermoscop"), "derm", np.where(t.str.contains("clinic|close"), "clin", "other"))

img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_files}
meta["_path"] = meta[IMG_ID].map(img_map)

# Doğru cevap tablosu lezyon mu görüntü mü bazlı?
gt = gt.copy()
gt[GT_ID] = gt[GT_ID].astype(str)
if gt[GT_ID].isin(meta[LES_ID]).mean() > 0.5:
    gt["lesion_id"] = gt[GT_ID]
else:
    gt["lesion_id"] = gt[GT_ID].map(dict(zip(meta[IMG_ID], meta[LES_ID])))
gt = gt.dropna(subset=["lesion_id"]).drop_duplicates("lesion_id")

rows = []
for les, g in meta.groupby(LES_ID):
    c, d = g[g._type == "clin"], g[g._type == "derm"]
    base = (d if len(d) else g).iloc[0]
    rows.append({
        "lesion_id": les,
        "patient_id": base[PAT_ID] if PAT_ID else None,
        "clin_path": c._path.dropna().iloc[0] if c._path.notna().any() else None,
        "derm_path": d._path.dropna().iloc[0] if d._path.notna().any() else None,
        "age": pd.to_numeric(base[AGE_COL], errors="coerce") if AGE_COL else np.nan,
        "sex": str(base[SEX_COL]) if SEX_COL else "unknown",
        "skin_tone": str(base[SKIN_COL]) if SKIN_COL else "unknown",
        "site": str(base[SITE_COL]) if SITE_COL else "unknown",
    })
les_df = pd.DataFrame(rows).merge(gt[["lesion_id"] + CLASSES], on="lesion_id", how="inner")
les_df["label"] = les_df[CLASSES].values.argmax(1)
les_df = les_df.sort_values("lesion_id").reset_index(drop=True)

print(f"\n{len(les_df)} lezyon | klinik görüntüsü eksik: {les_df.clin_path.isna().sum()} | "
      f"dermoskopik görüntüsü eksik: {les_df.derm_path.isna().sum()}")
print("\nSınıf dağılımı:")
print(les_df.label.map(dict(enumerate(CLASSES))).value_counts().to_string())

# %%
# 3) Eğitim / kontrol bölünmesi (seed 42, sınıf oranları korunur, aynı hasta iki tarafa düşmez)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

random.seed(SEED)
np.random.seed(SEED)
n_splits = round(1 / VAL_FRACTION)
if PAT_ID and les_df.patient_id.notna().all():
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    _, va_idx = next(sgkf.split(les_df, les_df.label, groups=les_df.patient_id))
    print("Bölünme hasta bazında yapıldı.")
else:
    _, va_idx = train_test_split(np.arange(len(les_df)), test_size=VAL_FRACTION,
                                 stratify=les_df.label, random_state=SEED)
    print("Hasta kimliği yok; bölünme lezyon bazında yapıldı.")
les_df["fold"] = "train"
les_df.loc[va_idx, "fold"] = "val"
les_df[["lesion_id", "label", "fold"]].to_csv(f"{OUT_DIR}/split_seed{SEED}.csv", index=False)
print(les_df.groupby("fold").size().to_string())

# %%
# 4a) MODE == "meta": yalnız hasta bilgisiyle model (lojistik regresyon)
from sklearn.metrics import f1_score

val_probs = None
if MODE == "meta":
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    feats = ["age", "sex", "skin_tone", "site"]
    tr, va = les_df[les_df.fold == "train"], les_df[les_df.fold == "val"]
    pre = ColumnTransformer([
        ("yas", Pipeline([("doldur", SimpleImputer(strategy="median")), ("olcek", StandardScaler())]), ["age"]),
        ("kategori", OneHotEncoder(handle_unknown="ignore"), ["sex", "skin_tone", "site"]),
    ])
    clf = Pipeline([("on_isleme", pre),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))])
    clf.fit(tr[feats], tr.label)
    p = clf.predict_proba(va[feats])
    val_probs = np.zeros((len(va), len(CLASSES)))
    val_probs[:, clf.classes_] = p
    val_df = va
    best_epoch = None
    joblib.dump({"model": clf, "features": feats, "classes": CLASSES}, f"{OUT_DIR}/{RUN_NAME}.joblib")
    print("Kontrol seti Macro F1:", round(f1_score(va.label, val_probs.argmax(1), average="macro"), 4))

# %%
# 4b) MODE == "derm" veya "clin": görüntü modeli (EfficientNet, transfer öğrenme)
if MODE in ("derm", "clin"):
    import torch
    import torch.nn as nn
    import timm
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T

    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device, "| GPU yoksa eğitim çok yavaş olur; Accelerator → GPU seçin.")

    col = "derm_path" if MODE == "derm" else "clin_path"
    df = les_df[les_df[col].notna()].reset_index(drop=True)
    if len(df) < len(les_df):
        print(f"Uyarı: {len(les_df) - len(df)} lezyonun bu görüntüsü yok, dışarıda bırakıldı.")
    CACHE = int(IMG_SIZE * 1.15)

    def load_resized(path):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.asarray(im.resize((CACHE, CACHE), Image.BILINEAR))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=(os.cpu_count() or 2) * 2) as pool:
        cache = list(pool.map(load_resized, df[col].tolist()))
    print(f"{len(cache)} görüntü belleğe alındı ({time.time() - t0:.0f} sn)")

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf_train = T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(30),
        T.ColorJitter(0.2, 0.2, 0.1, 0.02),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    tf_val = T.Compose([T.CenterCrop(IMG_SIZE), T.ToTensor(), T.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, idx, tf):
            self.idx, self.tf = list(idx), tf

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            return self.tf(Image.fromarray(cache[j])), int(df.label[j])

    tr_idx = np.where(df.fold == "train")[0]
    va_idx = np.where(df.fold == "val")[0]
    nw = min(4, os.cpu_count() or 1) if device == "cuda" else 0
    dl_tr = DataLoader(DS(tr_idx, tf_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=nw,
                       drop_last=True, pin_memory=True)
    dl_va = DataLoader(DS(va_idx, tf_val), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=nw)

    counts = np.bincount(df.label[tr_idx], minlength=len(CLASSES)).astype(float)
    w = (counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** 0.5   # nadir sınıflara daha çok ağırlık
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32, device=device),
                                    label_smoothing=0.05)

    model = timm.create_model(MODEL_NAME, pretrained=PRETRAINED, num_classes=len(CLASSES), drop_rate=0.3).to(device)
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

    best_f1, best_epoch, val_probs = -1, None, None
    y_va = df.label[va_idx].values
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
        p = predict(dl_va)
        f1 = f1_score(y_va, p.argmax(1), average="macro")
        flag = ""
        if f1 > best_f1:
            best_f1, best_epoch, val_probs = f1, ep, p
            torch.save(model.state_dict(), f"{OUT_DIR}/{RUN_NAME}.pt")
            flag = "  ← en iyi, kaydedildi"
        print(f"Epoch {ep:2d}/{EPOCHS} | kayıp {tot / len(dl_tr):.3f} | kontrol Macro F1 {f1:.4f} | "
              f"{time.time() - t0:.0f} sn{flag}")
    val_df = df.iloc[va_idx]

# %%
# 5) Sonuçları kaydet (Macro F1, sınıf bazlı sonuçlar, kontrol olasılıkları)
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

y_true, y_pred = val_df.label.values, val_probs.argmax(1)
macro = f1_score(y_true, y_pred, average="macro")
per_class = f1_score(y_true, y_pred, average=None, labels=range(len(CLASSES)), zero_division=0)
print(classification_report(y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES, zero_division=0))

os.makedirs(f"{OUT_DIR}/results", exist_ok=True)
result = {
    "run_name": RUN_NAME, "mode": MODE,
    "model": None if MODE == "meta" else MODEL_NAME,
    "img_size": None if MODE == "meta" else IMG_SIZE,
    "epochs": None if MODE == "meta" else EPOCHS, "best_epoch": best_epoch,
    "seed": SEED, "n_train": int((les_df.fold == "train").sum()), "n_val": int(len(val_df)),
    "val_macro_f1": round(float(macro), 4),
    "val_balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
    "per_class_f1": {c: round(float(v), 4) for c, v in zip(CLASSES, per_class)},
    "classes": CLASSES,
    "created": time.strftime("%Y-%m-%d %H:%M"),
}
json.dump(result, open(f"{OUT_DIR}/results/{RUN_NAME}.json", "w"), indent=2, ensure_ascii=False)
np.save(f"{OUT_DIR}/val_probs_{RUN_NAME}.npy", val_probs)
val_df[["lesion_id", "label"]].to_csv(f"{OUT_DIR}/val_ids_{RUN_NAME}.csv", index=False)

cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
print("Karışıklık matrisi (satır = gerçek, sütun = tahmin):")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
print(f"\nMacro F1 = {macro:.4f}  (hedef ≥ 0,45)")
print("Kaydedilen dosyalar:", sorted(os.listdir(OUT_DIR)))

# %%
# 6) Sonuçları tek zip'e koy ve indirme linki ver (oturum kapanınca /kaggle/working silinir!)
import zipfile
from IPython.display import FileLink, display

zip_name = f"sonuclar_{RUN_NAME}.zip"
os.chdir(OUT_DIR)
with zipfile.ZipFile(zip_name, "w") as z:
    for f in glob.glob(f"*{RUN_NAME}*") + glob.glob("split_seed*.csv") + glob.glob(f"results/{RUN_NAME}.json"):
        if f != zip_name:
            z.write(f)
            print("eklendi:", f)
print("\nAşağıdaki linke tıklayarak indirin:")
display(FileLink(zip_name))
