"""
MILK10k EDA - Tugalp (veri + tek goruntulu modeller paketi)

Ne yapar:
  1) Veri dogrulama: 5.240 lezyon, 10.480 goruntu, 11 sinif, her lezyonda 1 klinik + 1 dermoskopik
  2) Grafikler (figures/):
       eda_class_dist.png      sinif dagilimi
       eda_demographics.png    yas, cinsiyet, anatomik bolge
       eda_age_by_class.png    sinif bazinda yas
       eda_site_x_class.png    bolge x sinif isi haritasi
       eda_missing.png         eksik metadata
       eda_skin_tone.png       ten rengi 0-5      (yalnizca --train-meta verilirse)
       eda_tone_x_class.png    ten x sinif        (yalnizca --train-meta verilirse)
       eda_samples.png         sinif basina ornek cift (yalnizca --img-dir verilirse)
  3) Tablolar (results/): sinif sayilari, eksik veri, sinif bazinda demografi,
     bolge x sinif, (varsa) ten x sinif, ince tani -> sinif eslemesi, ozet json

Calistirma (yerelde veya Kaggle'da):
  python eda_milk10k.py --gt MILK10k_Training_GroundTruth.csv --meta milk10k.csv \
      [--train-meta MILK10k_Training_Metadata.csv] [--img-dir /kaggle/input/.../images] [--out .]

--meta        : ISIC Archive'dan inen goruntu bazli metadata (isic_id, lesion_id, image_type,
                age_approx, sex, anatom_site_general, diagnosis_1..4 ...)
--train-meta  : Challenge paketindeki model girdisi metadata'si (ten rengi sutunu burada).
                Ten rengi sutunu otomatik bulunur (skin_tone_class / skin_tone / ...).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- sabitler
CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
CLASS_NAMES = {
    "AKIEC": "Actinic keratosis / intraepidermal carcinoma",
    "BCC": "Basal cell carcinoma",
    "BEN_OTH": "Other benign proliferations",
    "BKL": "Benign keratinocytic lesion",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory / infectious",
    "MAL_OTH": "Other malignant proliferations",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "SCCKA": "Squamous cell carcinoma / keratoacanthoma",
    "VASC": "Vascular lesions / hemorrhage",
}
TONE_CANDIDATES = ["skin_tone_class", "skin_tone", "skintone", "skin_tone_label", "fitzpatrick"]
SITE_CANDIDATES_TRAIN = ["site", "anatom_site_general"]

# renkler (dataviz referans paleti: tek ton mavi, kategorik 1-2)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#a3a29c"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SEQ = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 20,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})


def style_axis(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)


def find_col(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for c in df.columns:  # gevsek eslesme
        if any(cand in c.lower() for cand in candidates):
            return c
    return None


# ---------------------------------------------------------------- veri
def load(gt_path, meta_path, train_meta_path=None):
    gt = pd.read_csv(gt_path)
    meta = pd.read_csv(meta_path)
    gt["label"] = gt[CLASSES].idxmax(axis=1)

    derm = meta[meta["image_type"] == "dermoscopic"]
    lesion = derm.drop(columns=["isic_id", "image_type", "image_manipulation"]).set_index("lesion_id")
    lesion = lesion.join(gt.set_index("lesion_id")["label"], how="outer")

    tone_col = site_col_train = None
    train = None
    if train_meta_path:
        train = pd.read_csv(train_meta_path)
        tone_col = find_col(train, TONE_CANDIDATES)
        site_col_train = find_col(train, ["site"]) if "site" in [c.lower() for c in train.columns] else None
        per_lesion = train.groupby("lesion_id").first()
        if tone_col:
            lesion["skin_tone"] = per_lesion[tone_col].reindex(lesion.index)
        if site_col_train:
            lesion["site_train"] = per_lesion[site_col_train].reindex(lesion.index)
    return gt, meta, lesion, train, tone_col, site_col_train


def validate(gt, meta):
    per_lesion = meta.groupby("lesion_id")["image_type"].agg(list)
    pair_ok = per_lesion.apply(lambda t: sorted(t) == ["clinical: close-up", "dermoscopic"]).all()
    consistent = {}
    d = meta[meta.image_type == "dermoscopic"].set_index("lesion_id")
    c = meta[meta.image_type != "dermoscopic"].set_index("lesion_id")
    for col in ["age_approx", "sex", "anatom_site_general", "diagnosis_3"]:
        a, b = d[col].reindex(c.index), c[col]
        consistent[col] = int(((a != b) & ~(a.isna() & b.isna())).sum())
    checks = {
        "n_lesions_gt": int(gt["lesion_id"].nunique()),
        "n_lesions_meta": int(meta["lesion_id"].nunique()),
        "n_images": int(len(meta)),
        "n_classes": len(CLASSES),
        "one_label_per_lesion": bool((gt[CLASSES].sum(axis=1) == 1).all()),
        "each_lesion_1_clinical_1_dermoscopic": bool(pair_ok),
        "gt_meta_lesion_ids_match": set(gt.lesion_id) == set(meta.lesion_id),
        "pair_metadata_mismatches": consistent,
    }
    checks["passed"] = (
        checks["n_lesions_gt"] == 5240 and checks["n_images"] == 10480
        and checks["one_label_per_lesion"] and checks["each_lesion_1_clinical_1_dermoscopic"]
        and checks["gt_meta_lesion_ids_match"] and sum(consistent.values()) == 0
    )
    return checks


# ---------------------------------------------------------------- grafikler
def plot_class_dist(lesion, out):
    counts = lesion["label"].value_counts().reindex(CLASSES).sort_values()
    n = counts.sum()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    labels = [f"{k}  {CLASS_NAMES[k]}" for k in counts.index]
    ax.barh(labels, counts.values, color=BLUE, height=0.7)
    for i, v in enumerate(counts.values):
        ax.text(v + n * 0.006, i, f"{v:,}  ({v / n:.1%})", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, counts.max() * 1.22)
    ax.set_xlabel("Lesions")
    ratio = counts.max() / counts.min()
    ax.set_title(f"MILK10k training set: class distribution (n = {n:,} lesions)")
    ax.text(0, 1.01, f"Largest / smallest class = {counts.idxmax()} / {counts.idxmin()} ≈ {ratio:.0f}:1",
            transform=ax.transAxes, color=INK2, fontsize=9)
    style_axis(ax, "x")
    fig.savefig(out / "eda_class_dist.png")
    plt.close(fig)
    return counts.sort_values(ascending=False)


def plot_demographics(lesion, out, site_col):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.6, 0.8, 1.2]})
    n = len(lesion)

    # yas
    ax = axes[0]
    age = lesion["age_approx"]
    vc = age.value_counts().sort_index()
    ax.bar(vc.index, vc.values, width=4, color=BLUE)
    ax.set_xticks(np.arange(5, 90, 10))
    ax.set_xlabel("Approximate age (5-year steps)")
    ax.set_ylabel("Lesions")
    ax.set_title("Age")
    ax.text(0, 1.01, f"median {age.median():.0f}, IQR {age.quantile(.25):.0f}–{age.quantile(.75):.0f}; "
            f"missing {age.isna().sum()}", transform=ax.transAxes, color=INK2, fontsize=9)
    style_axis(ax, "y")

    # cinsiyet
    ax = axes[1]
    sx = lesion["sex"].fillna("missing").value_counts()
    colors = [BLUE if s == "male" else ORANGE if s == "female" else GREY for s in sx.index]
    ax.bar(sx.index, sx.values, color=colors, width=0.6)
    for i, v in enumerate(sx.values):
        ax.text(i, v + n * 0.01, f"{v:,}\n({v / n:.0%})", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, sx.max() * 1.25)
    ax.set_title("Sex")
    style_axis(ax, "y")

    # bolge
    ax = axes[2]
    site = lesion[site_col].fillna("missing").value_counts().sort_values()
    colors = [GREY if s == "missing" else BLUE for s in site.index]
    ax.barh(site.index, site.values, color=colors, height=0.6)
    for i, v in enumerate(site.values):
        ax.text(v + n * 0.01, i, f"{v:,} ({v / n:.0%})", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, site.max() * 1.35)
    ax.set_title("Anatomical site")
    style_axis(ax, "x")

    fig.suptitle(f"Patient and lesion metadata (lesion level, n = {n:,})", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "eda_demographics.png")
    plt.close(fig)


def plot_age_by_class(lesion, out, order):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    data = [lesion.loc[lesion.label == c, "age_approx"].dropna().values for c in order[::-1]]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.55, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, linewidth=1.6),
                    boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE, linewidth=1),
                    whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
                    flierprops=dict(marker="o", markersize=3, markerfacecolor=GREY, markeredgecolor="none"))
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels([f"{c} (n={len(d)})" for c, d in zip(order[::-1], data)])
    ax.set_xlabel("Approximate age")
    ax.set_title("Age by diagnostic class")
    ax.text(0, 1.01, "n = lesions with known age", transform=ax.transAxes, color=INK2, fontsize=9)
    style_axis(ax, "x")
    fig.savefig(out / "eda_age_by_class.png")
    plt.close(fig)


def heatmap(table, out_file, title, subtitle, row_pct=True, xlabel=""):
    """table: satirlar sinif, sutunlar grup; hucrede sayi, renk satir yuzdesi."""
    pct = table.div(table.sum(axis=1), axis=0) if row_pct else table / table.values.max()
    fig, ax = plt.subplots(figsize=(1.25 * table.shape[1] + 3.2, 0.48 * table.shape[0] + 1.8))
    ax.imshow(pct.values, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v, p = table.iat[i, j], pct.iat[i, j]
            txt = f"{v:,}\n{p:.0%}" if row_pct else f"{v:,}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if p > 0.55 else INK)
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=0)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_xlabel(xlabel)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, color=INK2, fontsize=9)
    ax.set_xticks(np.arange(-.5, table.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, table.shape[0]), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.savefig(out_file)
    plt.close(fig)


def plot_missing(missing, out):
    m = missing[missing["missing_lesions"] > 0].sort_values("missing_pct")
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(m) + 1.4))
    colors = [BLUE if r else GREY for r in m["model_input"]]
    ax.barh(m["column"], m["missing_pct"], color=colors, height=0.6)
    for i, (v, k, note) in enumerate(zip(m["missing_pct"], m["missing_lesions"], m["note"])):
        ax.text(v + 1, i, f"{k:,} lesions ({v:.1f}%)" + (f"  – {note}" if note else ""),
                va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Missing (% of lesions)")
    ax.set_title("Missing metadata (lesion level)")
    ax.text(0, 1.01, "Blue = used as model input; grey = descriptive / missing by design",
            transform=ax.transAxes, color=INK2, fontsize=9)
    style_axis(ax, "x")
    fig.savefig(out / "eda_missing.png")
    plt.close(fig)


def plot_skin_tone(lesion, out):
    tone = lesion["skin_tone"]
    vc = tone.value_counts().sort_index()
    n = len(lesion)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([str(int(i)) if pd.notna(i) and float(i).is_integer() else str(i) for i in vc.index],
           vc.values, color=BLUE, width=0.65)
    for i, v in enumerate(vc.values):
        ax.text(i, v + n * 0.005, f"{v:,}\n({v / n:.1%})", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, vc.max() * 1.22)
    ax.set_xlabel("Skin tone class (0 = very dark … 5 = very light)")
    ax.set_ylabel("Lesions")
    ax.set_title(f"Skin tone distribution (n = {n:,}; missing {tone.isna().sum()})")
    style_axis(ax, "y")
    fig.savefig(out / "eda_skin_tone.png")
    plt.close(fig)


def plot_samples(lesion, meta, img_dir, out, seed=42):
    img_dir = Path(img_dir)
    paths = {p.stem: p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if not paths:
        print("  ! --img-dir altinda goruntu bulunamadi, eda_samples.png atlandi")
        return False
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(2.0 * len(CLASSES), 4.6))
    for j, c in enumerate(CLASSES):
        ids = lesion.index[lesion.label == c].tolist()
        ids = [i for i in ids if set(meta.loc[meta.lesion_id == i, "isic_id"]) <= set(paths)]
        lid = ids[int(rng.integers(len(ids)))] if ids else None
        for r, itype in enumerate(["clinical: close-up", "dermoscopic"]):
            ax = axes[r, j]
            ax.axis("off")
            if lid is None:
                continue
            iid = meta.loc[(meta.lesion_id == lid) & (meta.image_type == itype), "isic_id"].iloc[0]
            ax.imshow(plt.imread(paths[iid]))
            if r == 0:
                ax.set_title(c, fontsize=10)
    axes[0, 0].text(-0.15, 0.5, "Clinical", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.15, 0.5, "Dermoscopic", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("One random lesion per class (seed 42)", x=0.01, ha="left", fontweight="bold")
    fig.savefig(out / "eda_samples.png")
    plt.close(fig)
    return True


# ---------------------------------------------------------------- tablolar
def missing_table(lesion, meta, train=None, tone_col=None):
    notes = {
        "anatom_site_special": "only filled for acral / oral-genital sites",
        "melanocytic": "only marked when True",
        "diagnosis_4": "sub-type level, often not applicable",
        "diagnosis_3": "fine-grained diagnosis",
    }
    model_inputs = {"age_approx", "sex", "skin_tone", "site_train"}
    if "site_train" not in lesion.columns:
        model_inputs.add("anatom_site_general")
    rows = []
    for col in lesion.columns:
        if col in {"label", "attribution", "copyright_license"}:
            continue
        k = int(lesion[col].isna().sum())
        rows.append({
            "column": col,
            "missing_lesions": k,
            "missing_pct": round(100 * k / len(lesion), 2),
            "missing_images": int(meta[col].isna().sum()) if col in meta.columns else None,
            "model_input": col in model_inputs,
            "note": notes.get(col, ""),
        })
    display = {"site_train": "site", "skin_tone": "skin_tone_class"}
    for r in rows:
        r["column"] = display.get(r["column"], r["column"])
    df = pd.DataFrame(rows).sort_values(["model_input", "missing_pct"], ascending=[False, False])
    if train is not None:
        extra = [{"column": f"train_meta.{c}", "missing_lesions": None,
                  "missing_pct": round(100 * train[c].isna().mean(), 2),
                  "missing_images": int(train[c].isna().sum()), "model_input": None, "note": "from --train-meta"}
                 for c in train.columns if train[c].isna().any()]
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df


def class_demographics(lesion, site_col):
    g = lesion.groupby("label")
    t = pd.DataFrame({
        "class_name": [CLASS_NAMES[c] for c in g.size().index],
        "n_lesions": g.size(),
        "pct": (100 * g.size() / len(lesion)).round(2),
        "age_median": g["age_approx"].median(),
        "age_q1": g["age_approx"].quantile(.25),
        "age_q3": g["age_approx"].quantile(.75),
        "pct_male": (100 * g["sex"].apply(lambda s: (s == "male").mean())).round(1),
        "pct_site_missing": (100 * g[site_col].apply(lambda s: s.isna().mean())).round(1),
        "pct_histopathology": (100 * g["diagnosis_confirm_type"].apply(
            lambda s: (s == "histopathology").mean())).round(1),
    })
    if "skin_tone" in lesion:
        t["skin_tone_median"] = g["skin_tone"].median()
    return t.sort_values("n_lesions", ascending=False)


# ---------------------------------------------------------------- ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--train-meta", default=None)
    ap.add_argument("--img-dir", default=None)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    out = Path(a.out)
    fig_dir, res_dir = out / "figures", out / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    gt, meta, lesion, train, tone_col, site_col_train = load(a.gt, a.meta, a.train_meta)
    site_col = "site_train" if "site_train" in lesion else "anatom_site_general"

    print("1) Veri dogrulama")
    checks = validate(gt, meta)
    for k, v in checks.items():
        print(f"   {k}: {v}")

    print("2) Grafikler")
    counts = plot_class_dist(lesion, fig_dir)
    order = list(counts.index)
    plot_demographics(lesion, fig_dir, site_col)
    plot_age_by_class(lesion, fig_dir, order)

    site_tab = pd.crosstab(lesion["label"], lesion[site_col].fillna("missing")).reindex(order)
    cols = [c for c in ["head_neck_face", "trunk", "upper_extremity", "hand", "lower_extremity", "foot", "genital",
                        "head/neck", "upper extremity", "lower extremity", "anterior torso", "posterior torso",
                        "trunk", "oral/genital", "palms/soles"] if c in site_tab.columns]
    cols += [c for c in site_tab.columns if c not in cols and c != "missing"] + ["missing"]
    site_tab = site_tab[[c for c in cols if c in site_tab.columns]]
    heatmap(site_tab, fig_dir / "eda_site_x_class.png", "Anatomical site × class",
            "Cell = lesions and row % (share of that class); colour = row %")

    missing = missing_table(lesion, meta, train, tone_col)
    plot_missing(missing[missing["model_input"].notna()], fig_dir)

    tone_info = {"skin_tone_column": tone_col}
    if "skin_tone" in lesion and lesion["skin_tone"].notna().any():
        plot_skin_tone(lesion, fig_dir)
        tone_tab = pd.crosstab(lesion["label"], lesion["skin_tone"]).reindex(order).fillna(0).astype(int)
        heatmap(tone_tab, fig_dir / "eda_tone_x_class.png", "Skin tone × class",
                "Cell = lesions and row %; small cells limit per-tone fairness analysis",
                xlabel="Skin tone class (0 = very dark … 5 = very light)")
        tone_tab.to_csv(res_dir / "eda_tone_x_class.csv")
        tone_info["tone_counts"] = {str(k): int(v) for k, v in lesion["skin_tone"].value_counts().sort_index().items()}
        tone_info["cells_under_10"] = int((tone_tab < 10).sum().sum())
    else:
        print("   ! Ten rengi sutunu yok -> eda_skin_tone.png ve eda_tone_x_class.png uretilmedi.")
        print("     Challenge paketindeki MILK10k_Training_Metadata.csv dosyasini --train-meta ile ver.")

    if a.img_dir:
        plot_samples(lesion, meta, a.img_dir, fig_dir)

    print("3) Tablolar")
    cc = pd.DataFrame({"class": counts.index, "class_name": [CLASS_NAMES[c] for c in counts.index],
                       "n_lesions": counts.values, "n_images": counts.values * 2,
                       "pct": (100 * counts.values / counts.sum()).round(2)})
    cc.to_csv(res_dir / "eda_class_counts.csv", index=False)
    missing.to_csv(res_dir / "eda_missing.csv", index=False)
    class_demographics(lesion, site_col).to_csv(res_dir / "eda_class_demographics.csv")
    site_tab.to_csv(res_dir / "eda_site_x_class.csv")
    dx = (lesion.groupby(["label", "diagnosis_3"]).size().rename("n_lesions").reset_index()
          .sort_values(["label", "n_lesions"], ascending=[True, False]))
    dx.to_csv(res_dir / "eda_dx3_to_class.csv", index=False)

    summary = {
        "validation": checks,
        "n_lesions": int(len(lesion)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "largest_class": counts.idxmax(),
        "smallest_class": counts.idxmin(),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 1),
        "malignant_share_pct": round(100 * float((lesion["diagnosis_1"] == "Malignant").mean()), 1),
        "age_median": float(lesion["age_approx"].median()),
        "age_missing_lesions": int(lesion["age_approx"].isna().sum()),
        "pct_male": round(100 * float((lesion["sex"] == "male").mean()), 1),
        "site_column": site_col,
        "site_missing_lesions": int(lesion[site_col].isna().sum()),
        "histopathology_confirmed_pct": round(100 * float((lesion["diagnosis_confirm_type"] == "histopathology").mean()), 1),
        "altered_images": meta.groupby("image_type")["image_manipulation"].apply(
            lambda s: int((s == "altered").sum())).to_dict(),
        **tone_info,
    }
    with open(res_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "validation"}, ensure_ascii=False, indent=2))
    print("Bitti ->", fig_dir.resolve(), res_dir.resolve())


if __name__ == "__main__":
    main()
