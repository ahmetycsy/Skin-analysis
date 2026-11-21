import hashlib
import hmac
import html
import io
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat

import i18n
from about import ADVISOR, AGE_RANGE, APP_VERSION, LOW_DATA_SKIN_TONES, SKIN_TONE_COUNTS, TEAM, render_about, render_guide
from i18n import t
from predictor import (
    AGE_BANDS,
    BENIGN,
    CLASSES,
    GROUP_ORDER,
    GROUPS,
    MALIGN,
    PREMALIGN,
    REFER_MALIGN_OR_PRE,
    REFER_URGENT_MALIGN,
    SITES,
    UNCERTAIN_GAP,
    UNCERTAIN_TOP,
    Meta,
    age_band_label,
    load_predictor,
)
from stats_page import render_stats
from eval_page import render_eval
from db import get_db

try:  # kırpma eklentisi yoksa uygulama yine çalışsın
    from streamlit_cropper import st_cropper
except Exception:  # pragma: no cover
    st_cropper = None
try:  # PDF'teki QR kod için (yoksa QR'sız rapor üretilir)
    import segno
except Exception:  # pragma: no cover
    segno = None
try:
    from report import build_pdf
except Exception:  # pragma: no cover
    build_pdf = None

TZ = ZoneInfo("Europe/Istanbul")


def now_tr() -> datetime:
    """Türkiye saati (sunucu UTC'de çalışır)."""
    return datetime.now(TZ)


st.set_page_config(page_title="Skin Lesion Decision Support", page_icon="🩺", layout="wide",
                   initial_sidebar_state="auto")  # bilgisayarda açık, telefonda kapalı başlar

# ---------------------------------------------------------------- renkler (sakin, düşük kontrastlı yüzeyler)
INK = "#1F3A44"
MUTED = "#5F747C"
TEAL = "#0F766E"
BG = "#F6F8F8"
LINE = "#DDE6E5"
C_MALIGN = "#A63D57"
C_PREMALIGN = "#B0701A"
C_BENIGN = "#2F7F78"
GROUP_COLORS = {MALIGN: C_MALIGN, PREMALIGN: C_PREMALIGN, BENIGN: C_BENIGN}
SKIN_SWATCH = {0: "#4A2C20", 1: "#6F4531", 2: "#94623F", 3: "#B98558", 4: "#D9AE86", 5: "#F0D3B8"}
STATUSES = {"new": ("stat_new", "#5F747C"), "reviewed": ("stat_reviewed", "#2F7F78"),
            "awaiting": ("stat_awaiting", "#B0701A"), "closed": ("stat_closed", "#1F3A44")}
LEVEL_COLORS = {"urgent": C_MALIGN, "refer": C_PREMALIGN, "benign": C_BENIGN}
LEVEL_RANK = {"urgent": 0, "refer": 1, "benign": 2}
METHODS = {"histo": "m_histo", "clinical": "m_clinical", "consensus": "m_consensus"}
APP_URL = "https://bau-skin-cds.streamlit.app"  # QR koddaki bağlantı (Streamlit Cloud adresi)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] {{ font-family:'IBM Plex Sans', system-ui, -apple-system, sans-serif; color:{INK}; }}
.stApp {{ background:{BG}; }}
.block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1080px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
h1, h2, h3 {{ font-family:'IBM Plex Sans', system-ui, sans-serif; color:{INK}; letter-spacing:-0.01em; }}

/* yan menü */
section[data-testid="stSidebar"] {{ background:#fff; border-right:1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div {{ padding-top:1.2rem; }}
.brand {{ display:flex; align-items:center; gap:.65rem; margin-bottom:1.1rem; }}
.brand-mark {{ width:2.2rem; height:2.2rem; border-radius:9px; background:{TEAL}; display:flex; align-items:center; justify-content:center; flex:none; }}
.brand-name {{ font-weight:600; font-size:.98rem; line-height:1.2; }}
.brand-sub {{ font-size:.72rem; color:{MUTED}; }}
section[data-testid="stSidebar"] div.stButton > button {{ justify-content:flex-start; border:0; background:transparent; color:{INK};
  font-weight:500; padding:.45rem .7rem; border-radius:8px; }}
section[data-testid="stSidebar"] div.stButton > button:hover {{ background:{BG}; color:{INK}; }}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{ background:#E7F2F0; color:{TEAL}; font-weight:600; }}
.side-sep {{ border-top:1px solid {LINE}; margin:1rem 0 .8rem; }}
.side-note {{ font-size:.76rem; color:{MUTED}; line-height:1.5; }}
.chip {{ display:inline-block; font-size:.72rem; font-weight:500; border-radius:999px; padding:.15rem .55rem; border:1px solid {LINE};
  background:{BG}; color:{MUTED}; white-space:nowrap; margin:0 .25rem .3rem 0; }}
.chip.demo {{ background:#FBF3E4; border-color:#EBD6AE; color:#875A12; }}
.chip.live {{ background:#E7F2F0; border-color:#C5DFDA; color:{TEAL}; }}

/* sayfa başlığı */
.page-title {{ font-weight:600; font-size:1.55rem; line-height:1.2; margin:0 0 .2rem; }}
.page-sub {{ color:{MUTED}; font-size:.95rem; line-height:1.5; margin:0 0 1rem; }}
.banner {{ background:#fff; border:1px solid {LINE}; border-left:3px solid {C_PREMALIGN}; border-radius:8px; padding:.5rem .8rem;
  font-size:.86rem; line-height:1.5; margin:0 0 1.1rem; color:{INK}; }}

/* adım göstergesi */
.stepper {{ display:flex; align-items:center; gap:.5rem; margin:.2rem 0 1.2rem; flex-wrap:wrap; }}
.stp {{ display:flex; align-items:center; gap:.45rem; font-size:.9rem; color:{MUTED}; }}
.stp i {{ font-style:normal; width:1.6rem; height:1.6rem; border-radius:50%; display:inline-flex; align-items:center; justify-content:center;
  font-size:.8rem; font-weight:600; border:1.5px solid #C9D8D6; color:{MUTED}; background:#fff; }}
.stp.active {{ color:{INK}; font-weight:600; }} .stp.active i {{ background:{TEAL}; border-color:{TEAL}; color:#fff; }}
.stp.done i {{ background:#E7F2F0; border-color:#C5DFDA; color:{TEAL}; }}
.stp-line {{ flex:0 0 2.2rem; height:1.5px; background:#C9D8D6; }}

/* kartlar */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border:1px solid {LINE} !important; border-radius:12px; }}
.card-title {{ font-weight:600; font-size:1rem; margin:0 0 .2rem; }}
.card-text {{ color:{MUTED}; font-size:.86rem; line-height:1.5; margin:0 0 .6rem; }}
.pill {{ display:inline-block; font-size:.72rem; font-weight:500; color:{TEAL}; background:#E7F2F0; border-radius:999px; padding:.1rem .55rem; }}
[data-testid="stFileUploaderDropzone"] {{ background:{BG}; border:1.5px dashed #B5CFCB; border-radius:10px; }}

/* düğmeler */
div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {{ border-radius:9px; font-weight:500; }}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{ background:{TEAL}; border:0; color:#fff; }}
div.stButton > button[kind="primary"]:hover {{ background:#0B5F58; }}
div.stButton > button[kind="primary"]:disabled {{ background:#BCD2CF; color:#fff; }}

/* ten rengi örnekleri */
.swatches {{ display:flex; gap:.4rem; margin:.1rem 0 .9rem; }}
.sw {{ flex:1; text-align:center; font-size:.72rem; color:{MUTED}; }}
.sw i {{ display:block; height:1.5rem; border-radius:6px; border:2px solid transparent; margin-bottom:.2rem; }}
.sw.on i {{ border-color:{INK}; box-shadow:0 0 0 2px #fff inset; }}
.sw.on {{ color:{INK}; font-weight:600; }}

/* kalite etiketleri */
.qrow {{ display:flex; flex-wrap:wrap; gap:.3rem; margin:.35rem 0 .1rem; }}
.q {{ font-size:.72rem; border-radius:999px; padding:.08rem .5rem; border:1px solid; white-space:nowrap; }}
.q.ok {{ color:#2B6F69; background:#EDF6F4; border-color:#CFE5E1; }}
.q.bad {{ color:#875A12; background:#FBF3E4; border-color:#EBD6AE; }}

/* vaka başlığı */
.case-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; flex-wrap:wrap; }}
.case-label {{ font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}; }}
.case-no {{ font-weight:600; font-size:1.45rem; letter-spacing:.03em; font-variant-numeric:tabular-nums; }}
.case-meta {{ font-size:.82rem; color:{MUTED}; margin-top:.15rem; }}
.spill {{ display:inline-block; font-size:.74rem; font-weight:600; border-radius:999px; padding:.15rem .6rem; color:#fff; white-space:nowrap; }}
.case-hint {{ font-size:.8rem; color:{MUTED}; margin-top:.5rem; }}

/* sonuç */
.rec {{ border-radius:10px; padding:.85rem 1rem; margin:0 0 .8rem; border:1px solid; }}
.rec-title {{ font-weight:600; font-size:1.06rem; margin-bottom:.2rem; }}
.rec-text {{ font-size:.9rem; line-height:1.55; }}
.rec.urgent {{ background:#FAF0F2; border-color:#EBCDD5; }} .rec.urgent .rec-title {{ color:{C_MALIGN}; }}
.rec.refer {{ background:#FBF5EA; border-color:#EBD9B8; }} .rec.refer .rec-title {{ color:#8C5A12; }}
.rec.benign {{ background:#EDF6F4; border-color:#CFE5E1; }} .rec.benign .rec-title {{ color:#25685F; }}
.unsure {{ background:#FCF8EE; border:1px solid #EFE2BF; border-radius:8px; padding:.6rem .85rem; font-size:.87rem; line-height:1.5; margin:0 0 .8rem; }}
.sec-label {{ font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}; margin:1rem 0 .45rem; }}
.stack {{ display:flex; height:.6rem; border-radius:999px; overflow:hidden; margin:.2rem 0 .45rem; background:{LINE}; }}
.legend {{ display:flex; flex-wrap:wrap; gap:.3rem 1.1rem; font-size:.84rem; color:{MUTED}; }}
.legend b {{ color:{INK}; font-weight:600; font-variant-numeric:tabular-nums; }}
.dot {{ display:inline-block; width:.55rem; height:.55rem; border-radius:50%; margin-right:.35rem; }}
.cand {{ display:grid; grid-template-columns:1fr auto; gap:.15rem .8rem; padding:.6rem 0; border-top:1px solid {LINE}; }}
.cand:first-of-type {{ border-top:0; }}
.cand-name {{ font-weight:600; font-size:.96rem; }}
.cand-pct {{ font-weight:600; font-size:.96rem; font-variant-numeric:tabular-nums; }}
.cand-bar {{ grid-column:1 / -1; height:.3rem; border-radius:999px; background:{LINE}; overflow:hidden; }}
.cand-bar i {{ display:block; height:100%; }}
.cand-desc {{ grid-column:1 / -1; color:{MUTED}; font-size:.85rem; line-height:1.5; }}
.cand-meta {{ grid-column:1 / -1; color:#7A8D94; font-size:.75rem; }}
.kv {{ display:grid; grid-template-columns:auto 1fr; gap:.3rem .9rem; font-size:.88rem; }}
.kv span {{ color:{MUTED}; }}
.tl {{ font-size:.84rem; line-height:1.5; }}
.tl div {{ display:flex; gap:.6rem; padding:.2rem 0; }}
.tl b {{ font-weight:500; color:{MUTED}; font-variant-numeric:tabular-nums; min-width:3rem; }}
.fb-done {{ font-size:.86rem; color:#25685F; background:#EDF6F4; border:1px solid #CFE5E1; border-radius:8px; padding:.5rem .75rem; margin-bottom:.6rem; }}
.cam-empty {{ color:{MUTED}; font-size:.85rem; line-height:1.55; background:{BG}; border:1px dashed #C9D8D6; border-radius:8px; padding:.65rem .8rem; }}
.fu {{ display:flex; justify-content:space-between; gap:.8rem; font-size:.86rem; padding:.35rem 0; border-top:1px solid {LINE}; }}
.fu:first-of-type {{ border-top:0; }}
.fu b {{ font-weight:600; font-variant-numeric:tabular-nums; }}
.fu .late {{ color:{C_MALIGN}; font-weight:600; }} .fu .soon {{ color:#8C5A12; font-weight:600; }} .fu .later {{ color:{MUTED}; }}
.row-urgent {{ border-left:3px solid {C_MALIGN}; padding-left:.6rem; }}
.final {{ font-size:.86rem; background:{BG}; border:1px solid {LINE}; border-radius:8px; padding:.55rem .75rem; margin-bottom:.5rem; line-height:1.5; }}
.cmp-sum {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.8rem 1rem; font-size:.9rem; line-height:1.7; margin:.2rem 0 1rem; }}
.arrow-up {{ color:{C_MALIGN}; font-weight:600; }} .arrow-down {{ color:{C_BENIGN}; font-weight:600; }} .arrow-flat {{ color:{MUTED}; font-weight:600; }}
.ps-row {{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.1rem 0 .3rem; }}
.gpill-ps {{ font-size:.82rem; font-weight:600; border:1.5px solid; border-radius:999px; padding:.15rem .65rem; background:#fff; }}
.ps-note {{ font-size:.8rem; color:{MUTED}; margin-bottom:.2rem; }}
.empty {{ color:{MUTED}; font-size:.94rem; line-height:1.6; padding:1rem .2rem; }}

/* vaka listesi */
.row-no {{ font-weight:600; font-variant-numeric:tabular-nums; letter-spacing:.02em; }}
.row-sub {{ font-size:.8rem; color:{MUTED}; }}

/* model kartı ve kılavuz */
.lead {{ color:{MUTED}; font-size:.97rem; line-height:1.55; max-width:50rem; margin:.2rem 0 1rem; }}
.kpi {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.75rem .95rem; margin-bottom:.4rem; }}
.kpi-v {{ font-weight:600; font-size:1.45rem; color:{INK}; font-variant-numeric:tabular-nums; }}
.kpi-l {{ font-size:.8rem; color:{MUTED}; }}
.guide {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.95rem; height:100%; }}
.guide .step-no {{ font-weight:600; font-size:.8rem; color:#fff; background:{TEAL}; border-radius:50%; width:1.5rem; height:1.5rem;
  display:inline-flex; align-items:center; justify-content:center; margin-bottom:.55rem; }}
.legend-big div {{ margin:.3rem 0; font-size:.9rem; line-height:1.5; }}
.notice {{ background:#fff; border:1px solid {LINE}; border-left:3px solid {C_PREMALIGN}; border-radius:8px; padding:.55rem .8rem; margin:.4rem 0; font-size:.88rem; }}
.notice.demo {{ border-left-color:{TEAL}; }}
@media (max-width:640px) {{ .page-title {{ font-size:1.3rem; }} .case-no {{ font-size:1.2rem; }} .stp-line {{ flex-basis:.8rem; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- oturum durumu
ss = st.session_state
ss.setdefault("lang", "tr")
ss.setdefault("page", "new")      # new | track | stats | about | guide
ss.setdefault("step", 1)          # yeni analiz adımı: 1 görüntüler, 2 hasta bilgileri, 3 sonuç
ss.setdefault("nonce", 0)         # artırılınca giriş alanları sıfırlanır
ss.setdefault("imgs", {"clin": None, "derm": None})   # {"bytes", "id", "name"} — yalnızca bellekte
ss.setdefault("boxes", {"clin": None, "derm": None})  # kırpma kutuları
ss.setdefault("cases", {})        # vaka no -> kayıt (yalnızca bellekte; veritabanı sonra)
ss.setdefault("current", None)    # yeni analizde gösterilen vaka
ss.setdefault("open_case", None)  # vaka takibinde açık vaka
ss.setdefault("consent", False)   # kullanım koşulları onaylandı mı
ss.setdefault("doctor", None)     # hekim modu: baş harfler (None = giriş yok)
ss.setdefault("pin_fails", 0)
ss.setdefault("audit", [])        # denetim kaydı: yalnızca eklenir, değiştirilmez
ss.setdefault("dirty", [])        # veritabanına yazılacak vakalar
ss.setdefault("pending_audit", [])  # veritabanına yazılacak denetim satırları
ss.setdefault("db_synced", 0.0)   # vakaların veritabanından en son çekildiği an
INPUTS = ("age", "sex", "skin", "site", "crop_clin", "crop_derm", "patient")


def go(page: str):
    ss.page = page
    if page == "track":
        ss.open_case = None


def new_analysis():
    ss.nonce += 1
    ss.step = 1
    ss.current = None
    ss.imgs = {"clin": None, "derm": None}
    ss.boxes = {"clin": None, "derm": None}
    for k in INPUTS:
        ss.pop(f"v_{k}", None)
    ss.page = "new"


def wkey(name: str) -> str:
    """Widget anahtarı. Dil değişince Streamlit kutuyu yeni sanar; değer ss['v_<ad>'] içinde saklanıp geri yüklenir."""
    return f"{name}_{ss.nonce}_{ss.lang}"


def remembered_index(name: str, options: list):
    v = ss.get(f"v_{name}")
    return options.index(v) if v in options else None


def secrets_to_env():
    """Streamlit Secrets'taki HF_TOKEN / MODEL_REPO'yu ortam değişkenine aktarır (predictor.py ve huggingface_hub oradan okur)."""
    import os
    for key in ("HF_TOKEN", "MODEL_REPO", "SUPABASE_URL", "SUPABASE_KEY"):
        try:
            if key in st.secrets and not os.environ.get(key):
                os.environ[key] = str(st.secrets[key])
        except Exception:  # secrets dosyası yoksa sorun değil
            pass


secrets_to_env()


@st.cache_resource(show_spinner="Model yükleniyor, ilk açılışta 1–2 dakika sürebilir… · Loading the model, the first start may take 1–2 minutes…")
def get_predictor(model_repo_setting):
    """Model bir kez yüklenip hafızada tutulur. MODEL_REPO ayarı değişirse (ör. Secrets sonradan girildi) yeniden yüklenir;
    böylece uygulamayı elle yeniden başlatmak gerekmez."""
    return load_predictor()


# ---------------------------------------------------------------- görüntü yardımcıları
MAX_SIDE = 1600  # model 300–384 piksel görür; büyük telefon fotoğrafları belleği gereksiz şişirmesin


def open_bytes(data: bytes):
    """Yüklenen dosyayı açar: telefonun kaydettiği yöne çevirir (EXIF) ve uzun kenarı en çok MAX_SIDE piksele küçültür."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        return None
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def looks_dermoscopic(img: Image.Image) -> bool:
    """Kaba bir tahmin: dermoskopik görüntülerde köşeler genelde koyu (yuvarlak mercek çerçevesi) olur."""
    g = img.convert("L").resize((128, 128))
    k = 16
    corners = [g.crop(b) for b in [(0, 0, k, k), (128 - k, 0, 128, k), (0, 128 - k, k, 128), (128 - k, 128 - k, 128, 128)]]
    corner_mean = sum(ImageStat.Stat(c).mean[0] for c in corners) / 4
    center_mean = ImageStat.Stat(g.crop((32, 32, 96, 96))).mean[0]
    return corner_mean < 60 and corner_mean < 0.45 * center_mean


def square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    return img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def lens_preview(img: Image.Image, size: int = 320) -> Image.Image:
    """Dermoskopik görüntüyü yuvarlak mercek çerçevesinde gösterir (yalnızca önizleme)."""
    img = square(img).resize((size, size))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((3, 3, size - 3, size - 3), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img.convert("RGBA"), (0, 0), mask)
    ImageDraw.Draw(out).ellipse((3, 3, size - 3, size - 3), outline=(31, 58, 68, 255), width=4)
    return out


def jpeg_thumb(img: Image.Image, size: int = 360) -> bytes:
    buf = io.BytesIO()
    square(img.convert("RGB")).resize((size, size)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def get_image(which: str):
    """Kullanılacak görüntü: yüklenen görüntü, kırpma açıksa kırpılmış hali."""
    rec = ss.imgs[which]
    if rec is None:
        return None
    img = open_bytes(rec["bytes"])
    box = ss.boxes.get(which)
    if img is not None and ss.get(f"v_crop_{which}") and box:
        img = img.crop(box)
    return img


_LAPLACE = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def quality_checks(img: Image.Image):
    """Modelden bağımsız basit görüntü ön kontrolü. Netlik eşiği MILK10k görüntüleriyle kabaca ayarlandı:
    gerçek görüntülerin %95'inden fazlası geçer, belirgin bulanık görüntüler takılır."""
    w, h = img.size
    g = img.convert("L")
    g.thumbnail((512, 512))
    sharp = ImageStat.Stat(g.filter(_LAPLACE)).stddev[0]
    gw, gh = g.size
    bright = ImageStat.Stat(g.crop((gw // 4, gh // 4, 3 * gw // 4, 3 * gh // 4))).mean[0]
    return [
        (t("q_res"), min(w, h) >= 300, f"{w}×{h} px" + ("" if min(w, h) >= 300 else t("q_res_low"))),
        (t("q_sharp"), sharp >= 3.5, t("q_sharp_ok") if sharp >= 3.5 else t("q_sharp_bad")),
        (t("q_light"), 50 <= bright <= 225, t("q_light_ok") if 50 <= bright <= 225 else (t("q_dark") if bright < 50 else t("q_bright"))),
    ]


def skin_like(img: Image.Image) -> bool:
    """Görüntü cilt fotoğrafına benziyor mu? (modelden bağımsız kaba kontrol)
    YCbCr renk uzayında cilt tonu aralığındaki piksel oranı ve tek renge yığılma ölçülür. MILK10k'ten 300 görüntü ile
    ekran görüntüsü / gökyüzü / bitki / metin görüntüleri üzerinde ayarlandı: eşik 0,25 gerçek görüntülerin ~%1-2'sini
    yanlışlıkla işaretler, cilt olmayanların %98'inden fazlasını yakalar. Düz renk ağırlıklı görüntüler (ekran görüntüsü,
    belge) ayrıca yakalanır (gerçek görüntülerde en baskın rengin payı en çok ~%4)."""
    im = img.convert("RGB")
    im.thumbnail((256, 256))
    ycc = np.asarray(im.convert("YCbCr")).astype(int)
    y, cb, cr = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    skin = ((cr >= 133) & (cr <= 180) & (cb >= 70) & (cb <= 135) & (y >= 40)).mean()
    q = np.asarray(im.quantize(64))
    flat = np.bincount(q.ravel()).max() / q.size
    return skin >= 0.25 and flat <= 0.30


def show_quality(img: Image.Image, warn: bool = True):
    checks = quality_checks(img)
    ok_skin = skin_like(img)
    checks = [(t("q_skin"), ok_skin, t("q_skin_ok") if ok_skin else t("q_skin_bad"))] + checks
    chips = "".join(
        f'<span class="q {"ok" if ok else "bad"}" title="{html.escape(why)}">{"✓" if ok else "!"} {name}</span>'
        for name, ok, why in checks
    )
    st.markdown(f'<div class="qrow">{chips}</div>', unsafe_allow_html=True)
    bad = [f"**{name}:** {why}" for name, ok, why in checks[1:] if not ok]  # cilt uyarısı ayrıca gösterilir
    if bad and warn:
        st.warning(t("q_warn") + " " + " · ".join(bad))


def crop_tool(img: Image.Image, which: str):
    """İsteğe bağlı kırpma. Seçilen kutu ss.boxes içinde saklanır."""
    if st_cropper is None:
        return
    on = st.toggle(t("crop_toggle"), value=ss.get(f"v_crop_{which}", False), key=wkey(f"crop_on_{which}"),
                   help=t("crop_help"))
    ss[f"v_crop_{which}"] = on
    if not on:
        return
    st.caption(t("crop_hint"))
    # Kırpma ekranı karta sığsın diye küçük bir kopya gösterilir; seçilen kutu orijinal boyuta geri çevrilir.
    disp = img.copy()
    disp.thumbnail((360, 360))
    scale = img.width / disp.width
    box = st_cropper(disp, realtime_update=True, box_color=TEAL, aspect_ratio=None, return_type="box",
                     key=f"cropper_{which}_{ss.imgs[which]['id']}", stroke_width=2, should_resize_image=False)
    left, top = max(0, int(box["left"] * scale)), max(0, int(box["top"] * scale))
    right = min(img.width, int((box["left"] + box["width"]) * scale))
    bottom = min(img.height, int((box["top"] + box["height"]) * scale))
    ss.boxes[which] = (left, top, right, bottom) if right - left >= 32 and bottom - top >= 32 else None


# ---------------------------------------------------------------- sonuç yardımcıları
def new_case_id(seed: str) -> str:
    """Okunması kolay vaka numarası: LZ-yymmdd-XXXXX (karışan harfler 0/O, 1/I kullanılmaz). Kişisel bilgi içermez."""
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    while True:
        h = int(hashlib.sha1(f"{seed}{time.time_ns()}".encode()).hexdigest(), 16)
        code = "".join(alphabet[(h >> (5 * i)) & 31] for i in range(5))
        cid = f"LZ-{now_tr():%y%m%d}-{code}"
        if cid not in ss.cases:
            return cid


def assess(probs: dict) -> dict:
    """Olasılıklardan genel değerlendirme, yönlendirme metni ve kararsızlık durumunu çıkarır."""
    pct = i18n.pct
    totals = {g: sum(p for c, p in probs.items() if GROUPS[c] == g) for g in GROUP_ORDER}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_p, second_p = ranked[0][1], ranked[1][1]
    uncertain = top_p < UNCERTAIN_TOP or (top_p - second_p) < UNCERTAIN_GAP
    if totals[MALIGN] >= REFER_URGENT_MALIGN:
        level, title, text = "urgent", t("r_urgent_t"), t("r_urgent", p=pct(totals[MALIGN]))
    elif totals[MALIGN] + totals[PREMALIGN] >= REFER_MALIGN_OR_PRE:
        level, title, text = "refer", t("r_refer_t"), t("r_refer", p=pct(totals[MALIGN] + totals[PREMALIGN]))
    elif uncertain:
        level, title, text = "refer", t("r_unsure_t"), t("r_unsure")
    else:
        level, title, text = "benign", t("r_benign_t"), t("r_benign")
    return {"totals": totals, "ranked": ranked, "uncertain": uncertain, "level": level, "title": title, "text": text}


AGREE = {"yes": "fb_yes", "partial": "fb_partial", "no": "fb_no"}
ACTIONS = {"follow": "act_follow", "biopsy": "act_biopsy", "excision": "act_excision", "treat": "act_treat", "none": "act_none"}
LEVEL_KEYS = {"urgent": "lvl_urgent", "refer": "lvl_refer", "benign": "lvl_benign"}


def dx_label(k: str) -> str:
    return t("fb_dx_unknown") if k == "unknown" else f"{i18n.plain(k)} ({k})"


def fb_summary(fb: dict) -> str:
    return t("fb_summary", dx=dx_label(fb["dx"]), agree=t(AGREE[fb["agree"]]), action=t(ACTIONS[fb["action"]]))


def fmt_date(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d.%m.%Y")


def physician_lines(rec: dict):
    """PDF için hekim bölümü satırları. Değerlendirme yoksa None (PDF'te boş, elle doldurulacak alan çıkar)."""
    fb, fin = rec.get("feedback"), rec.get("final")
    if not fb and not fin:
        return None
    rows = []
    if fb:
        rows += [(t("fb_dx"), dx_label(fb["dx"])), (t("fb_agree"), t(AGREE[fb["agree"]])), (t("fb_action"), t(ACTIONS[fb["action"]]))]
        if fb.get("followup"):
            rows.append((t("fu_label"), fmt_date(fb["followup"])))
        if fb["note"]:
            rows.append((t("fb_note"), fb["note"]))
    if fin:
        rows.append((t("final_dx"), f"{dx_label(fin['dx'])} · {t(METHODS[fin['method']])}"))
    return rows


def qr_png(cid: str):
    if segno is None:
        return None
    buf = io.BytesIO()
    segno.make(f"{APP_URL}/?case={cid}", error="m").save(buf, kind="png", scale=8, border=1)
    return buf.getvalue()


def priority_key(rec: dict):
    """İş listesi sırası: açık vakalar önce; içinde gecikmiş kontrol, sonra öncelik seviyesi, sonra en yeni."""
    a = assess(rec["probs"])
    fu = (rec.get("feedback") or {}).get("followup")
    overdue = bool(fu) and fu < now_tr().date().isoformat() and rec["status"] != "closed"
    return (rec["status"] == "closed", not overdue, LEVEL_RANK[a["level"]], -rec["ts"])


def export_csv(cases: dict) -> bytes:
    """Bütün vakalar tek tablo. Türkçede Excel'in beklediği noktalı virgül ve ondalık virgül kullanılır."""
    rows = []
    for r in cases.values():
        a = assess(r["probs"])
        fb, fin = r.get("feedback") or {}, r.get("final") or {}
        row = {
            "case_no": r["case"], "date": r["created"],
            "age": age_band_label(r["meta"]["age"]), "sex": r["meta"]["sex"], "skin_tone": r["meta"]["skin_tone"],
            "site": r["meta"]["site"], "assessment": a["level"], "top1": a["ranked"][0][0],
            "top1_prob": round(a["ranked"][0][1], 4), "malignant_total": round(a["totals"][MALIGN], 4),
        }
        row.update({f"p_{c}": round(r["probs"][c], 4) for c in CLASSES})
        row.update({
            "status": r["status"], "physician_dx": fb.get("dx", ""), "agreement": fb.get("agree", ""),
            "action": fb.get("action", ""), "followup": fb.get("followup", "") or "",
            "final_dx": fin.get("dx", ""), "final_method": fin.get("method", ""), "model": r["model_name"],
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    if ss.lang == "tr":
        return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    return df.to_csv(index=False).encode("utf-8-sig")


def pdf_extra_lines(rec: dict):
    rows = [(t("mode_label"), t("mode_hasta") if rec_mode(rec) == "hasta" else t("mode_hekim"))]
    ps = pred_set(rec)
    if ps:
        rows.append((t("ps_title"), ", ".join(i18n.plain(c) for c in ps)))
    return rows


def patient_mode() -> bool:
    return bool(ss.get("v_patient"))


def rec_mode(rec: dict) -> str:
    return rec.get("mode") or ("hasta" if "hasta modu" in (rec.get("model_name") or "") else "hekim")


def pred_set(rec: dict):
    """%90 güvenli olası tanılar listesi (conformal). Yalnızca gerçek modelde; sahte modelde None."""
    if predictor.is_mock or not hasattr(predictor, "prediction_set"):
        return None
    try:
        return predictor.prediction_set(rec["probs"], rec_mode(rec) == "hekim")
    except Exception:
        return None


def meta_lines(m: dict):
    return [
        (t("age_label"), t("age_fmt", b=age_band_label(m["age"]))),
        (t("sex_label"), i18n.sex(m["sex"])),
        (t("skin_label"), f"{m['skin_tone']} · {i18n.tone(m['skin_tone'])}"),
        (t("site_label"), i18n.site(m["site"])),
    ]


def patient_short(m: dict) -> str:
    return f"{age_band_label(m['age'])} · {i18n.sex(m['sex'])} · {t('tone_short')} {m['skin_tone']} · {i18n.site(m['site'])}"


def status_pill(status: str) -> str:
    key, color = STATUSES[status]
    return f'<span class="spill" style="background:{color}">{t(key)}</span>'


def log(kind: str, case=None, **extra) -> dict:
    """Denetim kaydına bir satır ekler (kim, ne zaman, ne yaptı). Kayıtlar yalnızca eklenir."""
    now = now_tr()
    ev = {"ts": now.strftime("%d.%m.%Y %H:%M:%S"), "time": now.strftime("%d.%m %H:%M"), "kind": kind, "case": case,
          "actor": ss.get("doctor"), **extra}
    ss.audit.append(ev)
    if DB is not None:  # vaka kaydına izin verilmediyse o vakanın satırları gönderilmez (flush_db'de elenir)
        ss.pending_audit.append({"iso": now.isoformat(), "case_no": case, "actor": ev["actor"], "kind": kind, "extra": extra})
    return ev


def add_event(rec: dict, kind: str, **extra):
    rec["events"].append(log(kind, rec["case"], **extra))
    if rec.get("saved") and rec["case"] not in ss.dirty:
        ss.dirty.append(rec["case"])


# ---------------------------------------------------------------- veritabanı (Supabase) — anonim kayıt
def case_row(rec: dict) -> dict:
    """Veritabanına giden satır: görüntü ve kişisel bilgi YOK."""
    ranked = sorted(rec["probs"].items(), key=lambda kv: kv[1], reverse=True)
    level = assess(rec["probs"])["level"]
    return {
        "case_no": rec["case"], "created_at": rec["created_iso"], "updated_at": now_tr().isoformat(),
        "app_version": APP_VERSION, "model_name": rec["model_name"],
        "age_band": rec["meta"]["age"], "sex": rec["meta"]["sex"], "skin_tone": rec["meta"]["skin_tone"],
        "site": rec["meta"]["site"], "probs": rec["probs"], "level": level, "top1": ranked[0][0],
        "top1_prob": round(ranked[0][1], 6), "malignant_total": round(sum(p for c, p in rec["probs"].items() if GROUPS[c] == MALIGN), 6),
        "elapsed_s": round(rec["elapsed"], 3), "skin_override": any(ev["kind"] == "skin_override" for ev in rec["events"]),
        "status": rec["status"], "feedback": rec["feedback"], "final": rec.get("final"),
    }


def rec_from_row(row: dict) -> dict:
    """Veritabanındaki satırdan vaka kaydı (görüntüler veritabanında olmadığı için boş)."""
    created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).astimezone(TZ)
    return {
        "case": row["case_no"], "created": created.strftime("%d.%m.%Y %H:%M"), "time": created.strftime("%H:%M"),
        "created_iso": created.isoformat(), "ts": created.timestamp(), "model_name": row.get("model_name") or "",
        "probs": row["probs"], "elapsed": row.get("elapsed_s") or 0.0,
        "meta": {"age": row["age_band"], "sex": row["sex"], "skin_tone": row["skin_tone"], "site": row["site"]},
        "clin_jpg": None, "derm_jpg": None, "cam_img": None, "cam_class": None, "cam_jpg": None,
        "feedback": row.get("feedback"), "final": row.get("final"), "status": row.get("status") or "new",
        "events": [], "saved": True, "from_db": True, "events_loaded": False,
    }


def flush_db():
    """Bekleyen vaka güncellemelerini, sonra denetim satırlarını veritabanına yazar (vaka önce: denetim satırı vakaya bağlı)."""
    if DB is None:
        return
    for cid in list(ss.dirty):
        rec = ss.cases.get(cid)
        if rec is None or not rec.get("saved") or DB.upsert_case(case_row(rec)):
            ss.dirty.remove(cid)
    keep = []
    for row in ss.pending_audit:
        cid = row["case_no"]
        if cid is not None and not (ss.cases.get(cid) or {}).get("saved"):
            continue  # kayıt izni olmayan vakanın işlemleri veritabanına gitmez
        if cid in ss.dirty:  # vaka satırı henüz yazılamadı, sonra tekrar dene
            keep.append(row)
            continue
        if not DB.add_audit({"ts": row["iso"], "case_no": cid, "actor": row["actor"], "kind": row["kind"],
                             "extra": row["extra"] or None}):
            keep.append(row)
    ss.pending_audit = keep


def sync_from_db(force: bool = False):
    """Başka oturumlarda/cihazlarda kaydedilen vakaları çeker (en çok 30 sn'de bir)."""
    if DB is None or (not force and time.time() - ss.db_synced < 30):
        return
    rows = DB.list_cases()
    if DB.last_error is None:
        ss.db_synced = time.time()
    for row in rows:
        if row["case_no"] not in ss.cases:
            ss.cases[row["case_no"]] = rec_from_row(row)


def load_events(rec: dict):
    """Veritabanından gelen vakanın geçmişini (denetim satırları) bir kez yükler."""
    if DB is None or not rec.get("from_db") or rec.get("events_loaded"):
        return
    old = rec["events"]
    evs = []
    for row in reversed(DB.list_audit(rec["case"])):
        ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).astimezone(TZ)
        evs.append({"ts": ts.strftime("%d.%m.%Y %H:%M:%S"), "time": ts.strftime("%d.%m %H:%M"), "kind": row["kind"],
                    "case": row["case_no"], "actor": row.get("actor"), **(row.get("extra") or {})})
    if DB.last_error is None:
        rec["events"] = evs + [e for e in old if e not in evs]
        rec["events_loaded"] = True


def actor_label(ev: dict) -> str:
    return t("actor_doc", who=ev["actor"]) if ev.get("actor") else t("actor_user")


def event_text(ev: dict) -> str:
    if ev["kind"] == "status":
        if ev.get("old"):
            return t("ev_status_from", a=t(STATUSES[ev["old"]][0]), b=t(STATUSES[ev["status"]][0]))
        return t("ev_status", s=t(STATUSES[ev["status"]][0]))
    simple = {"pdf": "ev_pdf", "login": "ev_login", "logout": "ev_logout", "login_fail": "ev_login_fail",
              "consent": "ev_consent", "skin_override": "ev_skin_override"}
    if ev["kind"] in simple:
        return t(simple[ev["kind"]])
    if ev["kind"] == "closed":
        return t("ev_closed", dx=dx_label(ev["dx"]))
    if ev["kind"] == "reopened":
        return t("ev_reopened")
    return t({"created": "ev_created", "feedback": "ev_feedback", "feedback_upd": "ev_feedback_upd"}[ev["kind"]])


def doctor_pin() -> str:
    """Hekim PIN'i Streamlit Secrets'tan (DOCTOR_PIN) okunur; tanımlı değilse demo PIN kullanılır."""
    try:
        return str(st.secrets["DOCTOR_PIN"])
    except Exception:
        return "2026"


def doctor_logout():
    log("logout")
    ss.doctor = None


def step_header(step: int):
    names = [t("s1_title"), t("s2_title"), t("s3_title")]
    parts = []
    for i, name in enumerate(names, start=1):
        cls = "active" if i == step else ("done" if i < step else "")
        mark = "✓" if i < step else str(i)
        parts.append(f'<div class="stp {cls}"><i>{mark}</i>{name}</div>')
    st.markdown(f'<div class="stepper">{"<div class=stp-line></div>".join(parts)}</div>', unsafe_allow_html=True)


def page_header(title: str, sub: str = ""):
    st.markdown(f'<div class="page-title">{title}</div>' + (f'<div class="page-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


# ---------------------------------------------------------------- yan menü
i18n.set_lang(ss.lang)
predictor = get_predictor(os.environ.get("MODEL_REPO", ""))
DB = get_db()
flush_db()  # önceki çalıştırmadan kalan (ör. st.rerun ile yarıda kalan) yazmalar

qp_case = st.query_params.get("case")
if qp_case:  # PDF'teki QR kod bu bağlantıyla gelir
    qp_case = qp_case.strip().upper()
    st.query_params.clear()
    ss.page = "track"
    if qp_case not in ss.cases and DB is not None:
        row = DB.get_case(qp_case)
        if row:
            ss.cases[qp_case] = rec_from_row(row)
    if qp_case in ss.cases:
        ss.open_case = qp_case
    else:
        ss.open_case = None
        ss.qr_missing = qp_case

if not ss.consent:
    @st.dialog(t("consent_title"), dismissible=False, width="medium")
    def consent_dialog():
        def _dlg_lang():
            if ss.get("lang_dlg") and ss.lang_dlg != ss.lang:
                ss.lang = ss.lang_dlg
                ss.lang_picker = ss.lang_dlg
                ss._dlg_lang_changed = True

        if ss.pop("_dlg_lang_changed", False):  # pencere ayrı bir parça olarak çizilir; dil değişince tüm sayfa yenilensin
            st.rerun()
        st.segmented_control("Language", options=list(i18n.LANGS), format_func=i18n.LANGS.get, default=ss.lang,
                             key="lang_dlg", label_visibility="collapsed", on_change=_dlg_lang)
        st.markdown(t("consent_body"))
        ok = st.checkbox(t("consent_check"), key=f"consent_chk_{ss.lang}")
        if st.button(t("consent_btn"), type="primary", disabled=not ok, width="stretch"):
            ss.consent = True
            log("consent")
            st.rerun()

    consent_dialog()

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round">'
        '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.2" fill="#fff" stroke="none"/>'
        '<path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/></svg></div>'
        f'<div><div class="brand-name">{t("app_name")}</div><div class="brand-sub">{t("app_sub")}</div></div></div>',
        unsafe_allow_html=True,
    )
    n_cases = len(ss.cases)
    nav = [
        ("new", t("nav_new"), ":material/add_circle:"),
        ("track", t("nav_track") + (f"  ·  {n_cases}" if n_cases else ""), ":material/folder_open:"),
        ("stats", t("tab_stats"), ":material/bar_chart:"),
        ("eval", t("nav_eval"), ":material/analytics:"),
        ("about", t("tab_about"), ":material/info:"),
        ("guide", t("tab_guide"), ":material/menu_book:"),
    ]
    for key, label, icon in nav:
        st.button(label, key=f"nav_{key}", icon=icon, width="stretch",
                  type="primary" if ss.page == key else "secondary", on_click=go, args=(key,))

    st.markdown('<div class="side-sep"></div>', unsafe_allow_html=True)

    if ss.doctor:
        st.markdown(f'<span class="chip live">● {t("doc_on", who=html.escape(ss.doctor))}</span>', unsafe_allow_html=True)
        st.button(t("doc_logout"), icon=":material/logout:", on_click=doctor_logout, key="doc_logout")
    else:
        with st.expander(t("doc_login"), icon=":material/stethoscope:"):
            if ss.pin_fails >= 5:
                st.error(t("doc_locked"))
            else:
                with st.form(f"doc_form_{ss.lang}", border=False, clear_on_submit=True):
                    pin = st.text_input(t("doc_pin"), type="password")
                    ini = st.text_input(t("doc_initials"), max_chars=10)
                    sub = st.form_submit_button(t("doc_login_btn"), type="primary", width="stretch")
                if sub:
                    if not ini.strip():
                        st.warning(t("doc_need_initials"))
                    elif hmac.compare_digest(pin.encode(), doctor_pin().encode()):
                        ss.doctor = ini.strip().upper()[:10]
                        ss.pin_fails = 0
                        log("login")
                        st.rerun()
                    else:
                        ss.pin_fails += 1
                        log("login_fail")
                        st.warning(t("doc_wrong", n=5 - ss.pin_fails))
            st.caption(t("doc_note"))
    st.markdown('<div class="side-sep"></div>', unsafe_allow_html=True)

    def _set_lang():
        if ss.get("lang_picker"):
            ss.lang = ss.lang_picker
        else:  # aynı düğmeye tekrar basılınca seçim boşalmasın
            ss.lang_picker = ss.lang

    ss.setdefault("lang_picker", ss.lang)
    st.segmented_control("Language", options=list(i18n.LANGS), format_func=i18n.LANGS.get,
                         key="lang_picker", label_visibility="collapsed", on_change=_set_lang)
    if getattr(predictor, "load_error", None):
        st.error(t("model_load_error", e=predictor.load_error[:300]))
    if DB is not None and DB.last_error:
        st.warning(t("db_error_detail", e=DB.last_error[:300]))
    status = (f'<span class="chip demo">● {t("chip_demo")}</span>' if predictor.is_mock
              else f'<span class="chip live">● {html.escape(predictor.name)}</span>')
    if DB is not None:
        status += (f'<span class="chip live">● {t("db_on")}</span>' if DB.last_error is None
                   else f'<span class="chip demo" title="{html.escape(DB.last_error)}">● {t("db_err")}</span>')
    st.markdown(f'<div style="margin-top:.6rem">{status}<span class="chip">{t("chip_version", v=APP_VERSION)}</span></div>'
                f'<div class="side-note" style="margin-top:.5rem">{t("side_nodx")}</div>'
                '<div class="side-sep"></div>'
                f'<div class="side-note">{t("foot_left")}<br><br>{TEAM}<br>'
                f'{"Danışman: Dr. Öğr. Üyesi" if ss.lang == "tr" else "Advisor: Asst. Prof. Dr."} {ADVISOR}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- vaka görünümü (yeni analiz ve vaka takibinde ortak)
def render_case(cid: str, where: str):
    rec = ss.cases[cid]
    load_events(rec)
    a = assess(rec["probs"])

    with st.container(border=True):
        h1, h2 = st.columns([1.6, 1], vertical_alignment="center")
        with h1:
            st.markdown(
                f'<div class="case-label">{t("case_no")}</div><div class="case-no">{cid}</div>'
                f'<div class="case-meta">{rec["created"]} · {html.escape(rec["model_name"])} &nbsp;{status_pill(rec["status"])}</div>',
                unsafe_allow_html=True,
            )
        with h2:
            if build_pdf is not None:
                try:
                    pdf_bytes = build_pdf(rec, a, predictor.is_mock, version=APP_VERSION, group_colors=GROUP_COLORS,
                                          groups_of=GROUPS, group_order=GROUP_ORDER, meta_lines=meta_lines(rec["meta"]) + pdf_extra_lines(rec),
                                          physician=physician_lines(rec), qr=qr_png(cid))
                    st.download_button(t("dl_pdf"), data=pdf_bytes, file_name=f"{cid}.pdf", mime="application/pdf",
                                       icon=":material/download:", width="stretch", type="primary", help=t("dl_pdf_help"),
                                       key=f"pdf_{cid}_{where}",
                                       on_click=lambda c=cid: add_event(ss.cases[c], "pdf"))
                except Exception as exc:  # rapor üretilemezse sonuç ekranı yine çalışsın
                    st.caption(f"PDF: {exc}")
            if where == "new":
                b1, b2 = st.columns(2)
                b1.button(t("btn_edit"), icon=":material/edit:", width="stretch", on_click=lambda: ss.update(step=2),
                          key="edit_inputs")
                b2.button(t("btn_new"), icon=":material/add:", width="stretch", on_click=new_analysis, key="new_from_result")
            else:
                c1, c2 = st.columns(2)
                c1.button(t("btn_back_list"), icon=":material/arrow_back:", width="stretch",
                          on_click=lambda: ss.update(open_case=None, track_q=""), key="back_list")
                c2.button(t("cmp_from_case"), icon=":material/compare:", width="stretch", key="cmp_from_case",
                          on_click=lambda c=cid: ss.update(open_case=None, track_q="", cmp_a=c, cmp_open=True),
                          disabled=len(ss.cases) < 2)
        if where == "new":
            st.markdown(f'<div class="case-hint">{t("case_hint")}</div>', unsafe_allow_html=True)
        if predictor.is_mock:
            st.caption(t("demo_caption"))
        if rec_mode(rec) == "hasta":
            st.markdown(f'<div class="banner" style="margin:.6rem 0 0">{t("pm_note")}</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.45], gap="large")
    with left:
        st.markdown(f'<div class="sec-label">{t("pdf_images")}</div>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        if rec.get("clin_jpg"):
            i1.image(rec["clin_jpg"], caption=t("clin_title"), width="stretch")
            if rec.get("derm_jpg"):
                i2.image(lens_preview(Image.open(io.BytesIO(rec["derm_jpg"]))), caption=t("derm_title"), width="stretch")
            else:
                i2.caption(t("mode_hasta"))
        else:
            st.markdown(f'<div class="cam-empty">{t("no_images")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">{t("s2_title")}</div>', unsafe_allow_html=True)
        kv = "".join(f"<span>{k}</span><div>{v}</div>" for k, v in meta_lines(rec["meta"]))
        st.markdown(f'<div class="kv">{kv}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">{t("timeline")}</div>', unsafe_allow_html=True)
        tl = "".join(f'<div><b>{ev["time"]}</b><span>{event_text(ev)} <span style="color:{MUTED}">· {actor_label(ev)}</span></span></div>'
                     for ev in rec["events"])
        st.markdown(f'<div class="tl">{tl}</div>', unsafe_allow_html=True)

    with right:
        st.markdown(f'<div class="rec {a["level"]}"><div class="rec-title">{a["title"]}</div>'
                    f'<div class="rec-text">{a["text"]}</div></div>', unsafe_allow_html=True)
        if a["uncertain"]:
            st.markdown(f'<div class="unsure">{t("unsure_box")}</div>', unsafe_allow_html=True)
        bar = "".join(f'<div style="width:{a["totals"][g] * 100:.2f}%;background:{GROUP_COLORS[g]}"></div>' for g in GROUP_ORDER)
        legend = "".join(f'<span><span class="dot" style="background:{GROUP_COLORS[g]}"></span>{i18n.group(g)} '
                         f'<b>{i18n.pct(a["totals"][g])}</b></span>' for g in GROUP_ORDER)
        st.markdown(f'<div class="stack">{bar}</div><div class="legend">{legend}</div>', unsafe_allow_html=True)

        ps = pred_set(rec)
        if ps:
            st.markdown(f'<div class="sec-label">{t("ps_title")}</div>', unsafe_allow_html=True)
            chips = "".join(f'<span class="gpill-ps" style="border-color:{GROUP_COLORS[GROUPS[c]]};color:{GROUP_COLORS[GROUPS[c]]}">'
                            f'{i18n.plain(c)} · {i18n.pct(rec["probs"][c])}</span>' for c in ps)
            note = t("ps_one") if len(ps) == 1 else t("ps_many", n=len(ps))
            st.markdown(f'<div class="ps-row">{chips}</div><div class="ps-note">{note}</div>', unsafe_allow_html=True,
                        help=t("ps_help"))
        st.markdown(f'<div class="sec-label">{t("top3")}</div>', unsafe_allow_html=True)
        cards = ""
        for c, p in a["ranked"][:3]:
            col = GROUP_COLORS[GROUPS[c]]
            cards += (
                f'<div class="cand"><div class="cand-name">{i18n.plain(c)}</div>'
                f'<div class="cand-pct" style="color:{col}">{i18n.pct(p)}</div>'
                f'<div class="cand-bar"><i style="width:{p * 100:.2f}%;background:{col}"></i></div>'
                f'<div class="cand-desc">{i18n.desc(c)}</div>'
                f'<div class="cand-meta">{i18n.group(GROUPS[c])} · {t("medical_name")}: {i18n.cname(c)} ({c})</div></div>'
            )
        st.markdown(f"<div>{cards}</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="sec-label">{t("cam_title")}</div>', unsafe_allow_html=True)
        if rec.get("cam_img") is not None and (rec.get("derm_jpg") or rec.get("clin_jpg")):
            g1, g2 = st.columns(2)
            if rec.get("derm_jpg"):
                g1.image(lens_preview(Image.open(io.BytesIO(rec["derm_jpg"]))), caption=t("cam_orig"), width="stretch")
            else:
                g1.image(rec["clin_jpg"], caption=t("clin_title"), width="stretch")
            g2.image(lens_preview(rec["cam_img"]), caption=t("cam_heat", c=i18n.plain(rec["cam_class"])), width="stretch")
            st.caption(t("cam_note"))
        else:
            st.markdown(f'<div class="cam-empty">{t("cam_demo")}</div>', unsafe_allow_html=True)

        st.write("")
        df = pd.DataFrame({
            t("col_plain"): [i18n.plain(c) for c, _ in a["ranked"]],
            t("col_medical"): [f"{i18n.cname(c)} ({c})" for c, _ in a["ranked"]],
            t("col_group"): [i18n.group(GROUPS[c]) for c, _ in a["ranked"]],
            t("col_prob"): [p for _, p in a["ranked"]],
        })
        with st.expander(t("all_results")):
            pc, gc, nc = t("col_prob"), t("col_group"), t("col_plain")
            chart = (
                alt.Chart(df).mark_bar(cornerRadiusEnd=3, size=12)
                .encode(
                    x=alt.X(f"{pc}:Q", axis=alt.Axis(format="%", title=None, grid=False),
                            scale=alt.Scale(domain=[0, max(0.05, float(df[pc].max()) * 1.1)])),
                    y=alt.Y(f"{nc}:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=12, labelLimit=260, domain=False, ticks=False)),
                    color=alt.Color(f"{gc}:N", legend=None, scale=alt.Scale(
                        domain=[i18n.group(g) for g in GROUP_ORDER], range=[GROUP_COLORS[g] for g in GROUP_ORDER])),
                    tooltip=[nc, t("col_medical"), gc, alt.Tooltip(f"{pc}:Q", format=".1%")],
                )
                .properties(height=22 * len(df)).configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df.assign(**{pc: df[pc].map(i18n.pct)}), hide_index=True, width="stretch")
        with st.expander(t("tech")):
            st.caption(t("tech_line", m=rec["model_name"], s=rec["elapsed"], c=rec["created"],
                         ok=t("tech_ok") if rec["elapsed"] < 10 else t("tech_slow")))
            st.caption(t("tech_thr", u=f"{REFER_URGENT_MALIGN * 100:.0f}", r=f"{REFER_MALIGN_OR_PRE * 100:.0f}",
                         t=f"{UNCERTAIN_TOP * 100:.0f}", g=f"{UNCERTAIN_GAP * 100:.0f}"))

    # Hekim değerlendirmesi ve vaka durumu
    st.markdown(f'<div class="sec-label" style="margin-top:1.4rem">{t("fb_section")}</div>', unsafe_allow_html=True)
    if not ss.doctor:
        with st.container(border=True):
            fb, fin = rec["feedback"], rec.get("final")
            st.markdown(status_pill(rec["status"]), unsafe_allow_html=True)
            if fb:
                extra = f" · {t('fu_label')}: {fmt_date(fb['followup'])}" if fb.get("followup") else ""
                st.markdown(f'<div class="fb-done" style="margin-top:.6rem">✓ {html.escape(fb_summary(fb))}{extra}</div>',
                            unsafe_allow_html=True)
            if fin:
                st.markdown(f'<div class="final">{t("closed_info", dx=dx_label(fin["dx"]), m=t(METHODS[fin["method"]]))}</div>',
                            unsafe_allow_html=True)
            st.markdown(f'<div class="banner" style="margin:.6rem 0 0;border-left-color:{TEAL}">{t("doc_needed")}</div>',
                        unsafe_allow_html=True)
        return

    with st.container(border=True):
        fb, fin = rec["feedback"], rec.get("final")
        s1, s2 = st.columns([1, 2.2], gap="large")
        with s1:
            closed = rec["status"] == "closed"
            if closed:
                st.markdown(status_pill("closed"), unsafe_allow_html=True)
            else:
                stat_opts = ["new", "reviewed", "awaiting"]

                def _status_changed(cid=cid, key=f"status_{cid}_{where}_{ss.lang}"):
                    new = ss[key]
                    old = ss.cases[cid]["status"]
                    if new and new != old:
                        ss.cases[cid]["status"] = new
                        add_event(ss.cases[cid], "status", status=new, old=old)

                st.selectbox(t("status_label"), stat_opts, index=stat_opts.index(rec["status"]),
                             format_func=lambda k: t(STATUSES[k][0]), key=f"status_{cid}_{where}_{ss.lang}",
                             on_change=_status_changed)
            # kesin tanı ve kapatma
            if fin:
                top3 = [c for c, _ in assess(rec["probs"])["ranked"][:3]]
                verdict = ("model_right" if fin["dx"] == top3[0] else "model_top3" if fin["dx"] in top3 else "model_wrong")
                st.markdown(f'<div class="final">{t("closed_info", dx=dx_label(fin["dx"]), m=t(METHODS[fin["method"]]))}'
                            f'<br><span style="color:{MUTED}">{t(verdict)}</span></div>', unsafe_allow_html=True)

                def _reopen(cid=cid):
                    r = ss.cases[cid]
                    r["final"] = None
                    r["status"] = "reviewed" if r["feedback"] else "new"
                    add_event(r, "reopened")

                st.button(t("reopen"), key=f"reopen_{cid}_{where}", icon=":material/lock_open:", on_click=_reopen)
            else:
                with st.expander(t("close_title")):
                    st.caption(t("close_intro"))
                    with st.form(key=f"close_form_{cid}_{where}_{ss.lang}", border=False):
                        fdx = st.selectbox(t("final_dx"), CLASSES, index=None, format_func=dx_label, placeholder="—")
                        meth = st.radio(t("final_method"), list(METHODS), index=None, format_func=lambda k: t(METHODS[k]))
                        if st.form_submit_button(t("close_btn"), type="primary"):
                            if fdx is None or meth is None:
                                st.warning(t("missing", items=", ".join(
                                    lab for lab, v in [(t("final_dx"), fdx), (t("final_method"), meth)] if v is None)))
                            else:
                                r = ss.cases[cid]
                                r["final"] = {"dx": fdx, "method": meth, "time": now_tr().strftime("%d.%m.%Y %H:%M")}
                                r["status"] = "closed"
                                add_event(r, "closed", dx=fdx)
                                st.rerun()
            st.caption(t("track_note_db" if DB is not None else "track_note"))
        with s2:
            if fb:
                extra = f" · {t('fu_label')}: {fmt_date(fb['followup'])}" if fb.get("followup") else ""
                st.markdown(f'<div class="fb-done">✓ {html.escape(fb_summary(fb))}{extra}</div>', unsafe_allow_html=True)
            with st.expander(t("fb_edit") if fb else t("fb_open"), expanded=not fb and where == "track"):
                st.caption(t("fb_intro"))
                with st.form(key=f"fb_form_{cid}_{where}_{ss.lang}", border=False):
                    dx_opts = ["unknown"] + CLASSES
                    dx = st.selectbox(t("fb_dx"), dx_opts, index=dx_opts.index(fb["dx"]) if fb else None,
                                      format_func=dx_label, placeholder="—")
                    agree = st.radio(t("fb_agree"), list(AGREE), horizontal=True,
                                     index=list(AGREE).index(fb["agree"]) if fb else None, format_func=lambda k: t(AGREE[k]))
                    action = st.selectbox(t("fb_action"), list(ACTIONS), format_func=lambda k: t(ACTIONS[k]),
                                          index=list(ACTIONS).index(fb["action"]) if fb else None, placeholder="—")
                    fu_val = datetime.fromisoformat(fb["followup"]).date() if fb and fb.get("followup") else None
                    fu = st.date_input(t("follow_date"), value=fu_val, format="DD.MM.YYYY", help=t("follow_help"))
                    note = st.text_area(t("fb_note"), value=fb["note"] if fb else "", max_chars=300,
                                        placeholder=t("fb_note_ph"), height=80)
                    if st.form_submit_button(t("fb_save"), type="primary"):
                        if dx is None or agree is None or action is None:
                            st.warning(t("missing", items=", ".join(
                                lab for lab, v in [(t("fb_dx"), dx), (t("fb_agree"), agree), (t("fb_action"), action)] if v is None)))
                        else:
                            r = ss.cases[cid]
                            add_event(r, "feedback_upd" if r["feedback"] else "feedback")
                            r["feedback"] = {"dx": dx, "agree": agree, "action": action, "note": note.strip(),
                                             "followup": fu.isoformat() if fu else None,
                                             "time": now_tr().strftime("%d.%m.%Y %H:%M")}
                            new_status = "awaiting" if dx == "unknown" or action == "biopsy" else "reviewed"
                            if r["status"] != "closed" and r["status"] != new_status:
                                old = r["status"]
                                r["status"] = new_status
                                add_event(r, "status", status=new_status, old=old)
                            st.rerun()


# ---------------------------------------------------------------- sayfa: yeni analiz
def page_new():
    page_header(t("new_title"), t("new_sub"))
    st.markdown(f'<div class="banner">{t("banner")}{t("banner_demo") if predictor.is_mock else ""}</div>',
                unsafe_allow_html=True)
    step = ss.step if not (ss.step == 3 and ss.current not in ss.cases) else 1
    step_header(step)
    if step == 1:
        step_images()
    elif step == 2:
        step_patient()
    else:
        render_case(ss.current, "new")


def step_images():
    pm = st.toggle(t("pm_toggle"), value=patient_mode(), key=wkey("pm"), help=t("pm_help"))
    ss.v_patient = pm
    used = ("clin",) if pm else ("clin", "derm")
    cards = {"clin": ("clin_pill", "clin_title", "clin_text"), "derm": ("derm_pill", "derm_title", "derm_text")}
    cols = st.columns(2, gap="medium")
    if pm:
        with cols[1], st.container(border=True):
            st.markdown(f'<span class="pill">{t("pm_card_title")}</span><div class="card-title" style="margin-top:.45rem">'
                        f'{t("derm_title")}</div><p class="card-text">{t("pm_card_text")}</p>', unsafe_allow_html=True)
    for col, which in zip(cols, used):
        pill, title, text = cards[which]
        with col, st.container(border=True):
            st.markdown(f'<span class="pill">{t(pill)}</span><div class="card-title" style="margin-top:.45rem">{t(title)}</div>'
                        f'<p class="card-text">{t(text)}</p>', unsafe_allow_html=True)
            rec = ss.imgs[which]
            if rec is None:
                f = st.file_uploader(which, type=["jpg", "jpeg", "png"], key=f"up_{which}_{ss.nonce}",
                                     label_visibility="collapsed")
                if f is not None:
                    data = f.getvalue()
                    if open_bytes(data) is None:
                        st.error(t("read_error"))
                    else:
                        ss.imgs[which] = {"bytes": data, "id": hashlib.md5(data).hexdigest(), "name": f.name}
                        st.rerun()
            else:
                img = open_bytes(rec["bytes"])
                crop_tool(img, which)
                if not ss.get(f"v_crop_{which}"):
                    st.image(img if which == "clin" else lens_preview(img), width=300,
                             caption=None if which == "clin" else t("lens_caption"))
                show_quality(get_image(which))
                if not skin_like(get_image(which)):
                    st.warning(t("skin_warn"))
                    ck = st.checkbox(t("skin_confirm"), value=ss.get(f"v_skinok_{rec['id']}", False),
                                     key=f"skinok_{rec['id']}_{ss.lang}")
                    ss[f"v_skinok_{rec['id']}"] = ck
                st.caption(html.escape(rec["name"]))

                def _clear(w=which):
                    ss.imgs[w] = None
                    ss.boxes[w] = None
                    ss.pop(f"v_crop_{w}", None)

                st.button(t("btn_change"), key=f"chg_{which}", icon=":material/refresh:", on_click=_clear)

    ready = all(ss.imgs[w] is not None for w in used)
    blocked = ready and any(not skin_like(get_image(w)) and not ss.get(f"v_skinok_{ss.imgs[w]['id']}") for w in used)
    both = not pm and ss.imgs["clin"] is not None and ss.imgs["derm"] is not None
    if both:
        if looks_dermoscopic(open_bytes(ss.imgs["clin"]["bytes"])) and not looks_dermoscopic(open_bytes(ss.imgs["derm"]["bytes"])):
            st.warning(t("reversed_warn"))

        def _swap():
            ss.imgs = {"clin": ss.imgs["derm"], "derm": ss.imgs["clin"]}
            ss.boxes = {"clin": ss.boxes["derm"], "derm": ss.boxes["clin"]}
            a, b = ss.get("v_crop_clin"), ss.get("v_crop_derm")
            ss.v_crop_clin, ss.v_crop_derm = b, a

        st.button(t("swap_btn"), on_click=_swap, icon=":material/swap_horiz:")

    with st.expander(t("img_help_title")):
        st.markdown(t("img_help_body"))

    st.write("")
    _, nb = st.columns([2, 1])
    with nb:
        st.button(t("btn_next"), type="primary", width="stretch", disabled=not ready or blocked, icon=":material/arrow_forward:",
                  on_click=lambda: ss.update(step=2), key="to_step2")
        if not ready:
            st.caption(t("need_clin") if pm else t("need_both"))
        elif blocked:
            st.caption(t("skin_block"))


def step_patient():
    pm = patient_mode()
    clin_img, derm_img = get_image("clin"), (None if pm else get_image("derm"))
    if clin_img is None or (derm_img is None and not pm):
        ss.step = 1
        st.rerun()

    left, right = st.columns([1.5, 1], gap="large")
    with left, st.container(border=True):
        age_opts, sex_opts, skin_opts, site_opts = [lo for lo, _ in AGE_BANDS], ["female", "male"], list(SKIN_SWATCH), list(SITES)
        age = st.selectbox(t("age_label"), age_opts, index=remembered_index("age", age_opts), placeholder=t("age_ph"),
                           format_func=lambda lo: t("age_fmt", b=age_band_label(lo)), help=t("age_help"), key=wkey("age"))
        sex = st.radio(t("sex_label"), sex_opts, index=remembered_index("sex", sex_opts), format_func=i18n.sex,
                       horizontal=True, key=wkey("sex"))
        skin_tone = st.radio(t("skin_label"), skin_opts, index=remembered_index("skin", skin_opts),
                             format_func=lambda i: f"{i} · {i18n.tone(i)}", horizontal=True, help=t("skin_help"), key=wkey("skin"))
        sw = "".join(f'<div class="sw{" on" if i == skin_tone else ""}"><i style="background:{c}"></i>{i}</div>'
                     for i, c in SKIN_SWATCH.items())
        st.markdown(f'<div class="swatches">{sw}</div>', unsafe_allow_html=True)
        site = st.selectbox(t("site_label"), site_opts, index=remembered_index("site", site_opts), placeholder=t("site_ph"),
                            format_func=i18n.site, key=wkey("site"))
        ss.v_age, ss.v_sex, ss.v_skin, ss.v_site = age, sex, skin_tone, site
        if skin_tone in LOW_DATA_SKIN_TONES:
            st.info(t("low_tone", n=SKIN_TONE_COUNTS[skin_tone]))
        if age is not None and (age < AGE_RANGE[0] or age > AGE_RANGE[1]):
            st.info(t("age_out", a=AGE_RANGE[0], b=AGE_RANGE[1]))

    with right, st.container(border=True):
        st.markdown(f'<div class="card-title">{t("s2_images")}</div>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        i1.image(jpeg_thumb(clin_img, 240), caption=t("clin_title"), width="stretch")
        if derm_img is not None:
            i2.image(lens_preview(derm_img, 240), caption=t("derm_title"), width="stretch")
        else:
            i2.caption(t("mode_hasta"))

    missing = [t(k) for k, v in [("m_age", age), ("m_sex", sex), ("m_skin", skin_tone), ("m_site", site)] if v is None]
    if DB is not None and not predictor.is_mock:  # sahte (demo) sonuçlar araştırma verisine karışmasın
        save = st.checkbox(t("save_consent"), value=ss.get("v_save", False), key=wkey("save"), help=t("save_help"))
        ss.v_save = save
    elif DB is not None:
        st.caption(t("save_demo_off"))
    st.write("")
    b1, _, b3 = st.columns([1, 1.2, 1])
    b1.button(t("btn_back"), icon=":material/arrow_back:", width="stretch", on_click=lambda: ss.update(step=1), key="back_1")
    with b3:
        run = st.button(t("analyze"), type="primary", width="stretch", disabled=bool(missing), icon=":material/play_arrow:",
                        key="run")
        if missing:
            st.caption(t("missing", items=", ".join(missing)))

    if run and not missing:
        # Adım adım ilerleme kutusu. Kısa beklemeler yalnızca adımların okunabilmesi içindir;
        # "hesaplama süresi" yalnızca modelin gerçek çalışma süresini ölçer.
        with st.status(t("prog_title"), expanded=True) as status:
            st.write("① " + t("prog_1"))
            time.sleep(0.3)
            st.write("② " + t("prog_2"))
            t0 = time.perf_counter()
            meta = Meta(age=age, sex=sex, skin_tone=skin_tone, site=site)
            probs = predictor.predict(clin_img, derm_img, meta)
            cam = predictor.explain(clin_img, derm_img, meta) if hasattr(predictor, "explain") else None
            elapsed = time.perf_counter() - t0
            time.sleep(0.3)
            st.write("③ " + t("prog_3"))
            time.sleep(0.3)
            st.write("④ " + t("prog_4"))
            time.sleep(0.2)
            status.update(label=t("prog_done"), state="complete", expanded=False)
        cid = new_case_id(ss.imgs["clin"]["id"] + ((ss.imgs["derm"] or {}).get("id", "") if not pm else "hasta"))
        rec = {
            "case": cid,
            "created": now_tr().strftime("%d.%m.%Y %H:%M"),
            "time": now_tr().strftime("%H:%M"),
            "model_name": predictor.name + (" – hasta modu" if pm else ""),
            "mode": "hasta" if pm else "hekim",
            "probs": probs,
            "elapsed": elapsed,
            "meta": {"age": age, "sex": sex, "skin_tone": skin_tone, "site": site},
            "clin_jpg": jpeg_thumb(clin_img),
            "derm_jpg": jpeg_thumb(derm_img) if derm_img is not None else None,
            "cam_img": cam[0] if cam else None,
            "cam_class": cam[1] if cam else None,
            "cam_jpg": jpeg_thumb(cam[0]) if cam else None,
            "feedback": None,
            "final": None,
            "ts": time.time(),
            "created_iso": now_tr().isoformat(),
            "saved": DB is not None and not predictor.is_mock and bool(ss.get("v_save")),  # izin + gerçek model şart
            "status": "new",
            "events": [],
        }
        add_event(rec, "created")
        if any(not skin_like(get_image(w)) for w in (("clin",) if pm else ("clin", "derm"))):
            add_event(rec, "skin_override")
        ss.cases = {cid: rec, **ss.cases}  # en yeni vaka en üstte
        ss.current = cid
        ss.step = 3
        st.rerun()


# ---------------------------------------------------------------- sayfa: vaka takibi
def followup_panel():
    """Yaklaşan ve geciken kontroller (kapatılmamış vakalar)."""
    today = now_tr().date()
    items = []
    for r in ss.cases.values():
        fu = (r.get("feedback") or {}).get("followup")
        if fu and r["status"] != "closed":
            items.append(((datetime.fromisoformat(fu).date() - today).days, r))
    if not items:
        return
    items.sort(key=lambda x: x[0])
    rows = ""
    for days, r in items:
        if days < 0:
            cls, label, when = "late", t("fu_overdue"), t("fu_late", n=-days)
        elif days <= 14:
            cls, label, when = "soon", t("fu_upcoming"), t("fu_today") if days == 0 else t("fu_left", n=days)
        else:
            cls, label, when = "later", t("fu_later"), t("fu_left", n=days)
        rows += (f'<div class="fu"><span><b>{r["case"]}</b> · {fmt_date(r["feedback"]["followup"])}</span>'
                 f'<span class="{cls}">{label} · {when}</span></div>')
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{t("fu_title")}</div>{rows}', unsafe_allow_html=True)


def case_column(rec: dict):
    a = assess(rec["probs"])
    st.markdown(f'<div class="row-no">{rec["case"]}</div><div class="row-sub">{rec["created"]} · {patient_short(rec["meta"])}</div>',
                unsafe_allow_html=True)
    if rec.get("clin_jpg"):
        i1, i2 = st.columns(2)
        i1.image(rec["clin_jpg"], width="stretch")
        if rec.get("derm_jpg"):
            i2.image(lens_preview(Image.open(io.BytesIO(rec["derm_jpg"]))), width="stretch")
    else:
        st.markdown(f'<div class="cam-empty">{t("no_images")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="rec {a["level"]}" style="padding:.6rem .8rem"><div class="rec-title" style="font-size:.95rem">'
                f'{a["title"]}</div></div>', unsafe_allow_html=True)
    bar = "".join(f'<div style="width:{a["totals"][g] * 100:.2f}%;background:{GROUP_COLORS[g]}"></div>' for g in GROUP_ORDER)
    legend = "".join(f'<span><span class="dot" style="background:{GROUP_COLORS[g]}"></span>{i18n.group(g)} '
                     f'<b>{i18n.pct(a["totals"][g])}</b></span>' for g in GROUP_ORDER)
    st.markdown(f'<div class="stack">{bar}</div><div class="legend">{legend}</div>', unsafe_allow_html=True)
    rows = "".join(f'<div class="cand"><div class="cand-name">{i18n.plain(c)}</div>'
                   f'<div class="cand-pct" style="color:{GROUP_COLORS[GROUPS[c]]}">{i18n.pct(p)}</div></div>'
                   for c, p in a["ranked"][:3])
    st.markdown(f'<div style="margin-top:.6rem">{rows}</div>', unsafe_allow_html=True)


def render_compare(a_id: str, b_id: str):
    ra, rb = ss.cases[a_id], ss.cases[b_id]
    aa, ab = assess(ra["probs"]), assess(rb["probs"])
    d = ab["totals"][MALIGN] - aa["totals"][MALIGN]
    if abs(d) < 0.05:
        arrow = f'<span class="arrow-flat">→ {t("cmp_flat")}</span>'
    elif d > 0:
        arrow = f'<span class="arrow-up">↑ {t("cmp_up")} (+{abs(d) * 100:.1f})</span>'
    else:
        arrow = f'<span class="arrow-down">↓ {t("cmp_down")} (−{abs(d) * 100:.1f})</span>'
    ta, tb = aa["ranked"][0][0], ab["ranked"][0][0]
    top_line = (t("cmp_top_same", a=i18n.plain(ta)) if ta == tb
                else t("cmp_top_changed", a=i18n.plain(ta), b=i18n.plain(tb)))
    lvl_line = ("" if aa["level"] == ab["level"] else
                "<br>" + t("cmp_level_changed", a=t(LEVEL_KEYS[aa["level"]]), b=t(LEVEL_KEYS[ab["level"]])))
    st.markdown(f'<div class="cmp-sum"><b>{t("cmp_change")}</b><br>{t("cmp_malign")}: '
                f'{i18n.pct(aa["totals"][MALIGN])} → {i18n.pct(ab["totals"][MALIGN])} {arrow}<br>{top_line}{lvl_line}'
                f'<div class="row-sub" style="margin-top:.3rem">{t("cmp_note")}</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(border=True):
        st.markdown(f'<div class="case-label">{t("cmp_a")}</div>', unsafe_allow_html=True)
        case_column(ra)
    with c2, st.container(border=True):
        st.markdown(f'<div class="case-label">{t("cmp_b")}</div>', unsafe_allow_html=True)
        case_column(rb)


def audit_panel():
    """Denetim kaydı: oturumdaki bütün işlemler (en yeni üstte) ve CSV indirme."""
    with st.expander(t("audit_title"), icon=":material/history:"):
        st.caption(t("audit_intro"))
        events = list(reversed(ss.audit))
        if DB is not None:
            flush_db()
            rows = DB.list_audit(limit=500)
            if DB.last_error is None:
                events = []
                for row in rows:
                    ts = datetime.fromisoformat(row["ts"].replace("Z", "+00:00")).astimezone(TZ)
                    events.append({"ts": ts.strftime("%d.%m.%Y %H:%M:%S"), "case": row["case_no"], "actor": row.get("actor"),
                                   "kind": row["kind"], **(row.get("extra") or {})})
        if not events:
            st.caption(t("audit_empty"))
            return
        df = pd.DataFrame([{t("audit_time"): ev["ts"], t("audit_case"): ev.get("case") or "–",
                            t("audit_actor"): actor_label(ev), t("audit_action"): event_text(ev)} for ev in events])
        st.dataframe(df, hide_index=True, width="stretch")
        sep, enc = (";", "utf-8-sig")
        st.download_button(t("audit_export"), data=df.to_csv(index=False, sep=sep).encode(enc),
                           file_name=f"denetim_kaydi_{now_tr():%Y%m%d_%H%M}.csv", mime="text/csv",
                           icon=":material/download:", key="audit_dl")


def page_track():
    sync_from_db()
    if ss.open_case in ss.cases:
        render_case(ss.open_case, "track")
        return
    page_header(t("track_title"), t("track_lead"))
    missing_case = ss.pop("qr_missing", None)
    if missing_case:
        st.markdown(f'<div class="banner">{t("case_not_here", c=html.escape(missing_case))}</div>', unsafe_allow_html=True)
    if not ss.cases:
        st.markdown(f'<div class="empty">{t("track_empty")}</div>', unsafe_allow_html=True)
        st.caption(t("track_note_db" if DB is not None else "track_note"))
        audit_panel()
        return

    followup_panel()

    # karşılaştırma
    ids = list(ss.cases)
    with st.expander(t("cmp_title"), expanded=bool(ss.get("cmp_open")), icon=":material/compare:"):
        if len(ids) < 2:
            st.caption(t("cmp_need2"))
        else:
            st.caption(t("cmp_intro"))
            older_first = sorted(ids, key=lambda c: ss.cases[c]["ts"])
            label = lambda c: f'{c} · {ss.cases[c]["created"]}'
            pre_a = ss.get("cmp_a")
            ia = older_first.index(pre_a) if pre_a in older_first else 0
            ib = len(older_first) - 1 if ia != len(older_first) - 1 else 0
            x1, x2 = st.columns(2)
            ca = x1.selectbox(t("cmp_a"), older_first, index=ia, format_func=label, key=f"cmp_sel_a_{pre_a}")
            cb = x2.selectbox(t("cmp_b"), older_first, index=ib, format_func=label, key=f"cmp_sel_b_{pre_a}")
            if ca == cb:
                st.info(t("cmp_same"))
            else:
                render_compare(ca, cb)
                st.button(t("cmp_close"), key="cmp_close", on_click=lambda: ss.update(cmp_open=False, cmp_a=None))

    c1, c2, c3 = st.columns([2, 1.1, 1.1])
    query = c1.text_input(t("track_search"), placeholder=t("track_ph"), key="track_q").strip().upper()
    stat_opts = ["all"] + list(STATUSES)
    flt = c2.selectbox(t("track_filter"), stat_opts, format_func=lambda k: t("track_all") if k == "all" else t(STATUSES[k][0]),
                       key=f"track_f_{ss.lang}")
    sort = c3.selectbox(t("sort_label"), ["priority", "date"], format_func=lambda k: t(f"sort_{k}"), key=f"track_s_{ss.lang}")
    if query in ss.cases:  # tam numara yazıldıysa vakayı doğrudan aç
        ss.open_case = query
        st.rerun()

    rows = [r for cid, r in ss.cases.items() if (not query or query in cid) and (flt == "all" or r["status"] == flt)]
    rows.sort(key=priority_key if sort == "priority" else (lambda r: -r["ts"]))
    h1, h2 = st.columns([3, 1.2], vertical_alignment="center")
    h1.caption(t("track_count", n=len(rows)))
    h2.download_button(t("export_btn"), data=export_csv(ss.cases), file_name=f"vakalar_{now_tr():%Y%m%d_%H%M}.csv",
                       mime="text/csv", icon=":material/table_view:", width="stretch", help=t("export_help"))
    if not rows:
        st.info(t("track_none"))
    today = now_tr().date().isoformat()
    for r in rows:
        a = assess(r["probs"])
        top_c, top_p = a["ranked"][0]
        fu = (r.get("feedback") or {}).get("followup")
        fu_txt = ""
        if fu and r["status"] != "closed":
            late = fu < today
            fu_txt = (f' · <span style="color:{C_MALIGN if late else MUTED};font-weight:{600 if late else 400}">'
                      f'{t("fu_label")}: {fmt_date(fu)}</span>')
        urgent = a["level"] == "urgent" and r["status"] != "closed"
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns([1.5, 2.3, 1.2, .7], vertical_alignment="center")
            k1.markdown(f'<div class="{"row-urgent" if urgent else ""}"><div class="row-no">{r["case"]}</div>'
                        f'<div class="row-sub">{r["created"]}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div><span class="dot" style="background:{LEVEL_COLORS[a["level"]]}"></span><b style="font-weight:'
                        f'{600 if urgent else 400}">{t(LEVEL_KEYS[a["level"]])}</b> · {i18n.plain(top_c)} ({i18n.pct(top_p)})</div>'
                        f'<div class="row-sub">{patient_short(r["meta"])}{fu_txt}</div>', unsafe_allow_html=True)
            k3.markdown(status_pill(r["status"]), unsafe_allow_html=True)
            k4.button(t("btn_open"), key=f"open_{r['case']}", on_click=lambda c=r["case"]: ss.update(open_case=c), width="stretch")
    st.caption(t("track_note_db" if DB is not None else "track_note"))
    audit_panel()


# ---------------------------------------------------------------- yönlendirme
if ss.page == "new":
    page_new()
elif ss.page == "track":
    page_track()
elif ss.page == "stats":
    sync_from_db()
    page_header(t("tab_stats"))
    history = [{"case": r["case"], "time": r["time"], "meta": r["meta"], "top": assess(r["probs"])["ranked"][0],
                "level": assess(r["probs"])["level"], "elapsed": r["elapsed"]} for r in ss.cases.values()]
    feedback = {cid: r["feedback"] for cid, r in ss.cases.items() if r["feedback"]}
    finals = [{"case": cid, "final": r["final"]["dx"], "ranked": [c for c, _ in assess(r["probs"])["ranked"]],
               "level": assess(r["probs"])["level"]} for cid, r in ss.cases.items() if r.get("final")]
    render_stats(history, feedback, predictor.is_mock, GROUP_COLORS, finals, db_on=DB is not None)
elif ss.page == "eval":
    page_header(t("nav_eval"), t("eval_sub"))
    render_eval()
elif ss.page == "about":
    page_header(t("tab_about"))
    render_about()
else:
    page_header(t("tab_guide"))
    render_guide()

flush_db()  # bu çalıştırmada biriken kayıtları veritabanına yaz
