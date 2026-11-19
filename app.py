import hashlib
import html
import io
import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageStat

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

try:  # kırpma eklentisi yoksa uygulama yine çalışsın
    from streamlit_cropper import st_cropper
except Exception:  # pragma: no cover
    st_cropper = None
try:
    from report import build_pdf
except Exception:  # pragma: no cover
    build_pdf = None

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
INPUTS = ("age", "sex", "skin", "site", "crop_clin", "crop_derm")


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


@st.cache_resource
def get_predictor():
    return load_predictor()


# ---------------------------------------------------------------- görüntü yardımcıları
def open_bytes(data: bytes):
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


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


def show_quality(img: Image.Image, warn: bool = True):
    checks = quality_checks(img)
    chips = "".join(
        f'<span class="q {"ok" if ok else "bad"}" title="{html.escape(why)}">{"✓" if ok else "!"} {name}</span>'
        for name, ok, why in checks
    )
    st.markdown(f'<div class="qrow">{chips}</div>', unsafe_allow_html=True)
    bad = [f"**{name}:** {why}" for name, ok, why in checks if not ok]
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
        cid = f"LZ-{datetime.now():%y%m%d}-{code}"
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


def physician_lines(fb):
    """PDF için hekim bölümü satırları. Değerlendirme yoksa None (PDF'te boş, elle doldurulacak alan çıkar)."""
    if not fb:
        return None
    rows = [(t("fb_dx"), dx_label(fb["dx"])), (t("fb_agree"), t(AGREE[fb["agree"]])), (t("fb_action"), t(ACTIONS[fb["action"]]))]
    if fb["note"]:
        rows.append((t("fb_note"), fb["note"]))
    return rows


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


def add_event(rec: dict, kind: str, **extra):
    rec["events"].append({"time": datetime.now().strftime("%d.%m %H:%M"), "kind": kind, **extra})


def event_text(ev: dict) -> str:
    if ev["kind"] == "status":
        return t("ev_status", s=t(STATUSES[ev["status"]][0]))
    return t({"created": "ev_created", "feedback": "ev_feedback", "feedback_upd": "ev_feedback_upd"}[ev["kind"]])


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
predictor = get_predictor()

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
        ("about", t("tab_about"), ":material/info:"),
        ("guide", t("tab_guide"), ":material/menu_book:"),
    ]
    for key, label, icon in nav:
        st.button(label, key=f"nav_{key}", icon=icon, width="stretch",
                  type="primary" if ss.page == key else "secondary", on_click=go, args=(key,))

    st.markdown('<div class="side-sep"></div>', unsafe_allow_html=True)

    def _set_lang():
        if ss.get("lang_picker"):
            ss.lang = ss.lang_picker
        else:  # aynı düğmeye tekrar basılınca seçim boşalmasın
            ss.lang_picker = ss.lang

    ss.setdefault("lang_picker", ss.lang)
    st.segmented_control("Language", options=list(i18n.LANGS), format_func=i18n.LANGS.get,
                         key="lang_picker", label_visibility="collapsed", on_change=_set_lang)
    status = (f'<span class="chip demo">● {t("chip_demo")}</span>' if predictor.is_mock
              else f'<span class="chip live">● {html.escape(predictor.name)}</span>')
    st.markdown(f'<div style="margin-top:.6rem">{status}<span class="chip">{t("chip_version", v=APP_VERSION)}</span></div>'
                f'<div class="side-note" style="margin-top:.5rem">{t("side_nodx")}</div>'
                '<div class="side-sep"></div>'
                f'<div class="side-note">{t("foot_left")}<br><br>{TEAM}<br>'
                f'{"Danışman: Dr. Öğr. Üyesi" if ss.lang == "tr" else "Advisor: Asst. Prof. Dr."} {ADVISOR}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------- vaka görünümü (yeni analiz ve vaka takibinde ortak)
def render_case(cid: str, where: str):
    rec = ss.cases[cid]
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
                                          groups_of=GROUPS, group_order=GROUP_ORDER, meta_lines=meta_lines(rec["meta"]),
                                          physician=physician_lines(rec["feedback"]))
                    st.download_button(t("dl_pdf"), data=pdf_bytes, file_name=f"{cid}.pdf", mime="application/pdf",
                                       icon=":material/download:", width="stretch", type="primary", help=t("dl_pdf_help"),
                                       key=f"pdf_{cid}_{where}")
                except Exception as exc:  # rapor üretilemezse sonuç ekranı yine çalışsın
                    st.caption(f"PDF: {exc}")
            if where == "new":
                b1, b2 = st.columns(2)
                b1.button(t("btn_edit"), icon=":material/edit:", width="stretch", on_click=lambda: ss.update(step=2),
                          key="edit_inputs")
                b2.button(t("btn_new"), icon=":material/add:", width="stretch", on_click=new_analysis, key="new_from_result")
            else:
                st.button(t("btn_back_list"), icon=":material/arrow_back:", width="stretch",
                          on_click=lambda: ss.update(open_case=None, track_q=""), key="back_list")
        if where == "new":
            st.markdown(f'<div class="case-hint">{t("case_hint")}</div>', unsafe_allow_html=True)
        if predictor.is_mock:
            st.caption(t("demo_caption"))

    left, right = st.columns([1, 1.45], gap="large")
    with left:
        st.markdown(f'<div class="sec-label">{t("pdf_images")}</div>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        i1.image(rec["clin_jpg"], caption=t("clin_title"), width="stretch")
        i2.image(lens_preview(Image.open(io.BytesIO(rec["derm_jpg"]))), caption=t("derm_title"), width="stretch")
        st.markdown(f'<div class="sec-label">{t("s2_title")}</div>', unsafe_allow_html=True)
        kv = "".join(f"<span>{k}</span><div>{v}</div>" for k, v in meta_lines(rec["meta"]))
        st.markdown(f'<div class="kv">{kv}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">{t("timeline")}</div>', unsafe_allow_html=True)
        tl = "".join(f'<div><b>{ev["time"]}</b>{event_text(ev)}</div>' for ev in rec["events"])
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
        if rec.get("cam_img") is not None:
            g1, g2 = st.columns(2)
            g1.image(lens_preview(Image.open(io.BytesIO(rec["derm_jpg"]))), caption=t("cam_orig"), width="stretch")
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
    with st.container(border=True):
        fb = rec["feedback"]
        s1, s2 = st.columns([1, 2.2], gap="large")
        with s1:
            stat_opts = list(STATUSES)

            def _status_changed(cid=cid, key=f"status_{cid}_{where}_{ss.lang}"):
                new = ss[key]
                if new and new != ss.cases[cid]["status"]:
                    ss.cases[cid]["status"] = new
                    add_event(ss.cases[cid], "status", status=new)

            st.selectbox(t("status_label"), stat_opts, index=stat_opts.index(rec["status"]),
                         format_func=lambda k: t(STATUSES[k][0]), key=f"status_{cid}_{where}_{ss.lang}",
                         on_change=_status_changed)
            st.caption(t("track_note"))
        with s2:
            if fb:
                st.markdown(f'<div class="fb-done">✓ {html.escape(fb_summary(fb))}</div>', unsafe_allow_html=True)
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
                                             "time": datetime.now().strftime("%d.%m.%Y %H:%M")}
                            new_status = "awaiting" if dx == "unknown" or action == "biopsy" else "reviewed"
                            if r["status"] != "closed" and r["status"] != new_status:
                                r["status"] = new_status
                                add_event(r, "status", status=new_status)
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
    cards = {"clin": ("clin_pill", "clin_title", "clin_text"), "derm": ("derm_pill", "derm_title", "derm_text")}
    cols = st.columns(2, gap="medium")
    for col, which in zip(cols, ("clin", "derm")):
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
                st.caption(html.escape(rec["name"]))

                def _clear(w=which):
                    ss.imgs[w] = None
                    ss.boxes[w] = None
                    ss.pop(f"v_crop_{w}", None)

                st.button(t("btn_change"), key=f"chg_{which}", icon=":material/refresh:", on_click=_clear)

    both = ss.imgs["clin"] is not None and ss.imgs["derm"] is not None
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
        st.button(t("btn_next"), type="primary", width="stretch", disabled=not both, icon=":material/arrow_forward:",
                  on_click=lambda: ss.update(step=2), key="to_step2")
        if not both:
            st.caption(t("need_both"))


def step_patient():
    clin_img, derm_img = get_image("clin"), get_image("derm")
    if clin_img is None or derm_img is None:
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
        i2.image(lens_preview(derm_img, 240), caption=t("derm_title"), width="stretch")

    missing = [t(k) for k, v in [("m_age", age), ("m_sex", sex), ("m_skin", skin_tone), ("m_site", site)] if v is None]
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
        cid = new_case_id(ss.imgs["clin"]["id"] + ss.imgs["derm"]["id"])
        rec = {
            "case": cid,
            "created": datetime.now().strftime("%d.%m.%Y %H:%M"),
            "time": datetime.now().strftime("%H:%M"),
            "model_name": predictor.name,
            "probs": probs,
            "elapsed": elapsed,
            "meta": {"age": age, "sex": sex, "skin_tone": skin_tone, "site": site},
            "clin_jpg": jpeg_thumb(clin_img),
            "derm_jpg": jpeg_thumb(derm_img),
            "cam_img": cam[0] if cam else None,
            "cam_class": cam[1] if cam else None,
            "cam_jpg": jpeg_thumb(cam[0]) if cam else None,
            "feedback": None,
            "status": "new",
            "events": [],
        }
        add_event(rec, "created")
        ss.cases = {cid: rec, **ss.cases}  # en yeni vaka en üstte
        ss.current = cid
        ss.step = 3
        st.rerun()


# ---------------------------------------------------------------- sayfa: vaka takibi
def page_track():
    if ss.open_case in ss.cases:
        render_case(ss.open_case, "track")
        return
    page_header(t("track_title"), t("track_lead"))
    if not ss.cases:
        st.markdown(f'<div class="empty">{t("track_empty")}</div>', unsafe_allow_html=True)
        st.caption(t("track_note"))
        return

    c1, c2 = st.columns([2, 1.2])
    query = c1.text_input(t("track_search"), placeholder=t("track_ph"), key="track_q").strip().upper()
    stat_opts = ["all"] + list(STATUSES)
    flt = c2.selectbox(t("track_filter"), stat_opts, format_func=lambda k: t("track_all") if k == "all" else t(STATUSES[k][0]),
                       key=f"track_f_{ss.lang}")
    if query in ss.cases:  # tam numara yazıldıysa vakayı doğrudan aç
        ss.open_case = query
        st.rerun()

    rows = [r for cid, r in ss.cases.items()
            if (not query or query in cid) and (flt == "all" or r["status"] == flt)]
    st.caption(t("track_count", n=len(rows)))
    if not rows:
        st.info(t("track_none"))
    for r in rows:
        a = assess(r["probs"])
        top_c, top_p = a["ranked"][0]
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns([1.5, 2.2, 1.3, .7], vertical_alignment="center")
            k1.markdown(f'<div class="row-no">{r["case"]}</div><div class="row-sub">{r["created"]}</div>', unsafe_allow_html=True)
            k2.markdown(f'<div><span class="dot" style="background:{LEVEL_COLORS[a["level"]]}"></span>{t(LEVEL_KEYS[a["level"]])} · '
                        f'{i18n.plain(top_c)} ({i18n.pct(top_p)})</div><div class="row-sub">{patient_short(r["meta"])}</div>',
                        unsafe_allow_html=True)
            k3.markdown(status_pill(r["status"]), unsafe_allow_html=True)
            k4.button(t("btn_open"), key=f"open_{r['case']}", on_click=lambda c=r["case"]: ss.update(open_case=c), width="stretch")
    st.caption(t("track_note"))


# ---------------------------------------------------------------- yönlendirme
if ss.page == "new":
    page_new()
elif ss.page == "track":
    page_track()
elif ss.page == "stats":
    page_header(t("tab_stats"))
    history = [{"case": r["case"], "time": r["time"], "meta": r["meta"], "top": assess(r["probs"])["ranked"][0],
                "level": assess(r["probs"])["level"], "elapsed": r["elapsed"]} for r in ss.cases.values()]
    feedback = {cid: r["feedback"] for cid, r in ss.cases.items() if r["feedback"]}
    render_stats(history, feedback, predictor.is_mock, GROUP_COLORS)
elif ss.page == "about":
    page_header(t("tab_about"))
    render_about()
else:
    page_header(t("tab_guide"))
    render_guide()
