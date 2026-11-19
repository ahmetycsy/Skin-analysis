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
from stats_page import render_stats
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

try:  # kırpma eklentisi yoksa uygulama yine çalışsın
    from streamlit_cropper import st_cropper
except Exception:  # pragma: no cover
    st_cropper = None
try:
    from report import build_pdf
except Exception:  # pragma: no cover
    build_pdf = None

st.set_page_config(page_title="Skin Lesion Decision Support", page_icon="🩺", layout="wide")

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

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] {{ font-family:'IBM Plex Sans', system-ui, -apple-system, sans-serif; color:{INK}; }}
.stApp {{ background:{BG}; }}
.block-container {{ padding-top:1rem; padding-bottom:3rem; max-width:1140px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
h1, h2, h3 {{ font-family:'IBM Plex Sans', system-ui, sans-serif; color:{INK}; letter-spacing:-0.01em; }}

/* üst bar */
.st-key-appbar {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:.55rem .9rem; margin-bottom:.4rem; }}
.brand {{ display:flex; align-items:center; gap:.7rem; }}
.brand-mark {{ width:2.2rem; height:2.2rem; border-radius:9px; background:{TEAL}; display:flex; align-items:center; justify-content:center; flex:none; }}
.brand-name {{ font-weight:600; font-size:1.02rem; line-height:1.2; }}
.brand-sub {{ font-size:.76rem; color:{MUTED}; }}
.chips {{ display:flex; gap:.4rem; flex-wrap:wrap; justify-content:flex-end; }}
.chip {{ font-size:.74rem; font-weight:500; border-radius:999px; padding:.18rem .6rem; border:1px solid {LINE}; background:{BG}; color:{MUTED}; white-space:nowrap; }}
.chip.demo {{ background:#FBF3E4; border-color:#EBD6AE; color:#875A12; }}
.chip.live {{ background:#E7F2F0; border-color:#C5DFDA; color:{TEAL}; }}

/* sekmeler */
.stTabs [data-baseweb="tab-list"] {{ gap:.2rem; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:500; padding:.5rem .85rem; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:{INK}; font-weight:600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{TEAL}; }}

/* başlık ve uyarı */
.hero-title, h1.hero-title {{ font-weight:600 !important; font-size:1.65rem !important; line-height:1.2 !important; margin:.9rem 0 .25rem !important; padding:0 !important; }}
.hero-sub {{ color:{MUTED}; font-size:.97rem; line-height:1.55; max-width:48rem; margin:0 0 .9rem; }}
.notice {{ background:#fff; border:1px solid {LINE}; border-left:3px solid {C_PREMALIGN}; border-radius:8px;
  padding:.6rem .85rem; margin:.4rem 0; font-size:.9rem; line-height:1.5; }}
.notice.demo {{ border-left-color:{TEAL}; }}

/* adım başlıkları */
.step-head {{ display:flex; align-items:center; gap:.6rem; margin:1.5rem 0 .2rem; }}
.step-no {{ font-weight:600; font-size:.85rem; color:#fff; background:{TEAL}; border-radius:50%; width:1.55rem; height:1.55rem;
  display:inline-flex; align-items:center; justify-content:center; flex:none; }}
.step-title {{ font-weight:600; font-size:1.18rem; }}
.step-desc {{ color:{MUTED}; font-size:.92rem; margin:0 0 .7rem 2.15rem; max-width:50rem; }}

/* kartlar */
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border:1px solid {LINE} !important; border-radius:12px; }}
.card-title {{ font-weight:600; font-size:1.02rem; margin:0 0 .2rem; }}
.card-text {{ color:{MUTED}; font-size:.88rem; line-height:1.5; margin:0 0 .6rem; }}
.pill {{ display:inline-block; font-size:.72rem; font-weight:500; color:{TEAL}; background:#E7F2F0; border-radius:999px; padding:.1rem .55rem; }}
[data-testid="stFileUploaderDropzone"] {{ background:{BG}; border:1.5px dashed #B5CFCB; border-radius:10px; }}

/* düğmeler */
div.stButton > button, div.stDownloadButton > button {{ border-radius:9px; font-weight:500; }}
div.stButton > button[kind="primary"] {{ background:{TEAL}; border:0; color:#fff; padding:.6rem 1.3rem; }}
div.stButton > button[kind="primary"]:hover {{ background:#0B5F58; }}
div.stButton > button[kind="primary"]:disabled {{ background:#BCD2CF; color:#fff; }}
div.stDownloadButton > button {{ background:{TEAL}; color:#fff; border:0; }}
div.stDownloadButton > button:hover {{ background:#0B5F58; color:#fff; }}

/* ten rengi örnekleri */
.swatches {{ display:flex; gap:.4rem; margin:.1rem 0 .9rem; }}
.sw {{ flex:1; text-align:center; font-size:.72rem; color:{MUTED}; }}
.sw i {{ display:block; height:1.6rem; border-radius:6px; border:2px solid transparent; margin-bottom:.2rem; }}
.sw.on i {{ border-color:{INK}; box-shadow:0 0 0 2px #fff inset; }}
.sw.on {{ color:{INK}; font-weight:600; }}

/* kalite etiketleri */
.qrow {{ display:flex; flex-wrap:wrap; gap:.3rem; margin:.35rem 0 .1rem; }}
.q {{ font-size:.72rem; border-radius:999px; padding:.08rem .5rem; border:1px solid; white-space:nowrap; }}
.q.ok {{ color:#2B6F69; background:#EDF6F4; border-color:#CFE5E1; }}
.q.bad {{ color:#875A12; background:#FBF3E4; border-color:#EBD6AE; }}

/* sonuç */
.case {{ display:flex; justify-content:space-between; gap:.6rem; flex-wrap:wrap; font-size:.8rem; color:{MUTED}; margin-bottom:.6rem; }}
.case b {{ color:{INK}; font-weight:600; font-variant-numeric:tabular-nums; }}
.rec {{ border-radius:10px; padding:.85rem 1rem; margin:0 0 .8rem; border:1px solid; }}
.rec-title {{ font-weight:600; font-size:1.08rem; margin-bottom:.2rem; }}
.rec-text {{ font-size:.9rem; line-height:1.55; }}
.rec.urgent {{ background:#FAF0F2; border-color:#EBCDD5; }} .rec.urgent .rec-title {{ color:{C_MALIGN}; }}
.rec.refer {{ background:#FBF5EA; border-color:#EBD9B8; }} .rec.refer .rec-title {{ color:#8C5A12; }}
.rec.benign {{ background:#EDF6F4; border-color:#CFE5E1; }} .rec.benign .rec-title {{ color:#25685F; }}
.unsure {{ background:#FCF8EE; border:1px solid #EFE2BF; border-radius:8px; padding:.6rem .85rem; font-size:.88rem; line-height:1.5; margin:0 0 .8rem; }}
.sec-label {{ font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:{MUTED}; margin:1rem 0 .45rem; }}
.stack {{ display:flex; height:.6rem; border-radius:999px; overflow:hidden; margin:.2rem 0 .45rem; background:{LINE}; }}
.legend {{ display:flex; flex-wrap:wrap; gap:.3rem 1.1rem; font-size:.84rem; color:{MUTED}; }}
.legend b {{ color:{INK}; font-weight:600; font-variant-numeric:tabular-nums; }}
.dot {{ display:inline-block; width:.55rem; height:.55rem; border-radius:50%; margin-right:.35rem; }}
.cand {{ display:grid; grid-template-columns:1fr auto; gap:.15rem .8rem; padding:.65rem 0; border-top:1px solid {LINE}; }}
.cand:first-of-type {{ border-top:0; }}
.cand-name {{ font-weight:600; font-size:.98rem; }}
.cand-pct {{ font-weight:600; font-size:.98rem; font-variant-numeric:tabular-nums; }}
.cand-bar {{ grid-column:1 / -1; height:.3rem; border-radius:999px; background:{LINE}; overflow:hidden; }}
.cand-bar i {{ display:block; height:100%; }}
.cand-desc {{ grid-column:1 / -1; color:{MUTED}; font-size:.86rem; line-height:1.5; }}
.cand-meta {{ grid-column:1 / -1; color:#7A8D94; font-size:.76rem; }}
.empty {{ color:{MUTED}; font-size:.94rem; line-height:1.6; padding:1rem .2rem; }}
.fb-done {{ font-size:.86rem; color:#25685F; background:#EDF6F4; border:1px solid #CFE5E1; border-radius:8px; padding:.5rem .75rem; margin-bottom:.5rem; }}
.cam-empty {{ color:{MUTED}; font-size:.86rem; line-height:1.55; background:{BG}; border:1px dashed #C9D8D6; border-radius:8px; padding:.7rem .85rem; }}

/* model kartı ve kılavuz */
.lead {{ color:{MUTED}; font-size:.97rem; line-height:1.55; max-width:50rem; margin:.9rem 0 1rem; }}
.kpi {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.75rem .95rem; margin-bottom:.4rem; }}
.kpi-v {{ font-weight:600; font-size:1.45rem; color:{INK}; font-variant-numeric:tabular-nums; }}
.kpi-l {{ font-size:.8rem; color:{MUTED}; }}
.guide {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.95rem; height:100%; }}
.guide .step-no {{ margin-bottom:.55rem; }}
.legend-big div {{ margin:.3rem 0; font-size:.9rem; line-height:1.5; }}

.foot-grid {{ display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; border-top:1px solid {LINE};
  padding-top:1rem; margin-top:2.4rem; color:#7A8D94; font-size:.78rem; line-height:1.6; }}
@media (max-width:640px) {{ .step-desc {{ margin-left:0; }} .chips {{ justify-content:flex-start; }} h1.hero-title {{ font-size:1.4rem !important; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- oturum durumu
ss = st.session_state
ss.setdefault("nonce", 0)      # artırılınca tüm giriş alanları sıfırlanır (Yeni hasta)
ss.setdefault("swap", False)   # iki görüntünün yeri değiştirildi mi
ss.setdefault("result", None)  # son analiz sonucu
ss.setdefault("history", [])   # bu oturumdaki analizler (yalnızca bellekte, kaydedilmez)
ss.setdefault("lang", "tr")
ss.setdefault("feedback", {})  # vaka no -> hekim değerlendirmesi (yalnızca bellekte)


INPUTS = ("age", "sex", "skin", "site", "crop_clin", "crop_derm")


def reset_all():
    ss.nonce += 1
    ss.swap = False
    ss.result = None
    for k in INPUTS:
        ss.pop(f"v_{k}", None)


def wkey(name: str) -> str:
    """Widget anahtarı. Dil değişince Streamlit kutuyu yeni sanar; değer ss['v_<ad>'] içinde saklanıp geri yüklenir."""
    return f"{name}_{ss.nonce}_{ss.lang}"


def remembered_index(name: str, options: list):
    v = ss.get(f"v_{name}")
    return options.index(v) if v in options else None


def toggle_swap():
    ss.swap = not ss.swap


@st.cache_resource
def get_predictor():
    return load_predictor()


# ---------------------------------------------------------------- yardımcılar
def load_image(file):
    """Yüklenen dosyayı açar. (görüntü, parmak izi) döner; okunamazsa (None, None)."""
    if file is None:
        return None, None
    data = file.getvalue()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None, None
    return img, hashlib.md5(data).hexdigest()


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


def show_quality(img: Image.Image):
    checks = quality_checks(img)
    chips = "".join(
        f'<span class="q {"ok" if ok else "bad"}" title="{html.escape(why)}">{"✓" if ok else "!"} {name}</span>'
        for name, ok, why in checks
    )
    st.markdown(f'<div class="qrow">{chips}</div>', unsafe_allow_html=True)
    bad = [f"**{name}:** {why}" for name, ok, why in checks if not ok]
    if bad:
        st.warning(t("q_warn") + " " + " · ".join(bad))


def crop_tool(img: Image.Image, which: str, key: str):
    """İsteğe bağlı kırpma. (kullanılacak görüntü, kırpma kutusu ya da None) döner."""
    if st_cropper is None or img is None:
        return img, None
    on = st.toggle(t("crop_toggle"), value=ss.get(f"v_crop_{which}", False), key=wkey(f"crop_on_{which}_{key}"),
                   help=t("crop_help"))
    ss[f"v_crop_{which}"] = on
    if not on:
        return img, None
    st.caption(t("crop_hint"))
    # Kırpma ekranı karta sığsın diye küçük bir kopya gösterilir; seçilen kutu orijinal boyuta geri çevrilir.
    disp = img.copy()
    disp.thumbnail((360, 360))
    scale = img.width / disp.width
    box = st_cropper(disp, realtime_update=True, box_color=TEAL, aspect_ratio=None, return_type="box",
                     key=f"cropper_{key}", stroke_width=2, should_resize_image=False)
    left, top = max(0, int(box["left"] * scale)), max(0, int(box["top"] * scale))
    right = min(img.width, int((box["left"] + box["width"]) * scale))
    bottom = min(img.height, int((box["top"] + box["height"]) * scale))
    if right - left < 32 or bottom - top < 32:
        return img, None
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def case_id(sig: tuple) -> str:
    """Aynı görüntü + bilgi kombinasyonu için aynı kalan kısa vaka numarası (kişisel bilgi içermez)."""
    digest = hashlib.sha1(repr(sig).encode()).hexdigest()[:6].upper()
    return f"LZ-{datetime.now():%y%m%d}-{digest}"


def step(no: int, title: str, desc: str):
    st.markdown(
        f'<div class="step-head"><span class="step-no">{no}</span><span class="step-title">{title}</span></div>'
        f'<div class="step-desc">{desc}</div>',
        unsafe_allow_html=True,
    )


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


# ---------------------------------------------------------------- üst bar (dil seçimi dahil)
i18n.set_lang(ss.lang)
predictor = get_predictor()
n = ss.nonce

with st.container(key="appbar"):
    c_brand, c_chips, c_lang = st.columns([3.2, 2.3, 0.9], vertical_alignment="center")
    with c_brand:
        st.markdown(
            '<div class="brand"><div class="brand-mark">'
            '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round">'
            '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.2" fill="#fff" stroke="none"/>'
            '<path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/></svg></div>'
            f'<div><div class="brand-name">{t("app_name")}</div><div class="brand-sub">{t("app_sub")}</div></div></div>',
            unsafe_allow_html=True,
        )
    with c_chips:
        status = (
            f'<span class="chip demo">● {t("chip_demo")}</span>'
            if predictor.is_mock
            else f'<span class="chip live">● {html.escape(predictor.name)}</span>'
        )
        st.markdown(
            f'<div class="chips">{status}<span class="chip">{t("chip_version", v=APP_VERSION)}</span></div>',
            unsafe_allow_html=True,
        )
    with c_lang:
        # Dil, sayfa çizilmeden önce çalışan geri çağırmayla değişir. (st.rerun kullanılmaz: yarıda kesilen çalıştırma
        # yüklenen dosyaları sildiriyordu.)
        def _set_lang():
            if ss.get("lang_picker"):
                ss.lang = ss.lang_picker
            else:  # aynı düğmeye tekrar basılınca seçim boşalmasın
                ss.lang_picker = ss.lang

        ss.setdefault("lang_picker", ss.lang)
        st.segmented_control(
            "Language", options=list(i18n.LANGS), format_func=i18n.LANGS.get,
            key="lang_picker", label_visibility="collapsed", on_change=_set_lang,
        )

tab_analyze, tab_stats, tab_about, tab_guide = st.tabs([t("tab_analyze"), t("tab_stats"), t("tab_about"), t("tab_guide")])
with tab_about:
    render_about()
with tab_guide:
    render_guide()

# ---------------------------------------------------------------- analiz sekmesi
with tab_analyze:
    st.markdown(f'<h1 class="hero-title">{t("hero_title")}</h1><p class="hero-sub">{t("hero_sub")}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="notice">{t("notice_nodx")}</div>', unsafe_allow_html=True)
    if predictor.is_mock:
        st.markdown(f'<div class="notice demo">{t("notice_demo")}</div>', unsafe_allow_html=True)

    # 1. görüntüler
    step(1, t("s1_title"), t("s1_desc"))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown(f'<span class="pill">{t("clin_pill")}</span><div class="card-title" style="margin-top:.45rem">'
                        f'{t("clin_title")}</div><p class="card-text">{t("clin_text")}</p>', unsafe_allow_html=True)
            file_a = st.file_uploader("clinical", type=["jpg", "jpeg", "png"], key=f"clin_{n}", label_visibility="collapsed")
            prev_a = st.container()
    with c2:
        with st.container(border=True):
            st.markdown(f'<span class="pill">{t("derm_pill")}</span><div class="card-title" style="margin-top:.45rem">'
                        f'{t("derm_title")}</div><p class="card-text">{t("derm_text")}</p>', unsafe_allow_html=True)
            file_b = st.file_uploader("dermoscopic", type=["jpg", "jpeg", "png"], key=f"derm_{n}", label_visibility="collapsed")
            prev_b = st.container()

    img_a, id_a = load_image(file_a)
    img_b, id_b = load_image(file_b)
    if ss.swap:
        clin_raw, clin_id, derm_raw, derm_id = img_b, id_b, img_a, id_a
    else:
        clin_raw, clin_id, derm_raw, derm_id = img_a, id_a, img_b, id_b

    with prev_a:
        if file_a is not None and img_a is None:
            st.error(t("read_error"))
        clin_img, clin_box = crop_tool(clin_raw, "clin", str(clin_id))
        if clin_img is not None:
            if clin_box is None:
                st.image(clin_img, width=320, caption=t("swap_note") if ss.swap else None)
            show_quality(clin_img)
    with prev_b:
        if file_b is not None and img_b is None:
            st.error(t("read_error"))
        derm_img, derm_box = crop_tool(derm_raw, "derm", str(derm_id))
        if derm_img is not None:
            if derm_box is None:
                st.image(lens_preview(derm_img), width=320, caption=t("swap_note") if ss.swap else t("lens_caption"))
            show_quality(derm_img)

    if clin_raw is not None and derm_raw is not None:
        if looks_dermoscopic(clin_raw) and not looks_dermoscopic(derm_raw):
            st.warning(t("reversed_warn"))
        st.button(t("unswap_btn") if ss.swap else t("swap_btn"), on_click=toggle_swap, icon=":material/swap_horiz:")

    with st.expander(t("img_help_title")):
        st.markdown(t("img_help_body"))

    # 2. hasta bilgileri
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        step(2, t("s2_title"), t("s2_desc"))
        with st.container(border=True):
            age_opts, sex_opts, skin_opts, site_opts = [lo for lo, _ in AGE_BANDS], ["female", "male"], list(SKIN_SWATCH), list(SITES)
            age = st.selectbox(
                t("age_label"), options=age_opts, index=remembered_index("age", age_opts), placeholder=t("age_ph"),
                format_func=lambda lo: t("age_fmt", b=age_band_label(lo)), help=t("age_help"), key=wkey("age"),
            )
            sex = st.radio(t("sex_label"), sex_opts, index=remembered_index("sex", sex_opts), format_func=i18n.sex,
                           horizontal=True, key=wkey("sex"))
            skin_tone = st.radio(
                t("skin_label"), skin_opts, index=remembered_index("skin", skin_opts), format_func=lambda i: f"{i} · {i18n.tone(i)}",
                horizontal=True, help=t("skin_help"), key=wkey("skin"),
            )
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

        missing = [t(k) for k, v in [("m_clin", clin_img), ("m_derm", derm_img), ("m_age", age), ("m_sex", sex),
                                      ("m_skin", skin_tone), ("m_site", site)] if v is None]
        ready = not missing
        current_sig = (clin_id, clin_box, derm_id, derm_box, age, sex, skin_tone, site)

        b1, b2 = st.columns([1.4, 1])
        with b1:
            run = st.button(t("analyze"), type="primary", disabled=not ready, width="stretch")
        with b2:
            st.button(t("new_patient"), on_click=reset_all, width="stretch", icon=":material/restart_alt:",
                      help=t("new_patient_help"))
        if missing:
            st.caption(t("missing", items=", ".join(missing)))

        if run and ready:
            # Adım adım ilerleme kutusu. Kısa beklemeler yalnızca adımların okunabilmesi içindir;
            # "hesaplama süresi" yalnızca modelin gerçek çalışma süresini ölçer.
            with st.status(t("prog_title"), expanded=True) as status:
                st.write("① " + t("prog_1"))
                time.sleep(0.3)
                st.write("② " + t("prog_2"))
                t0 = time.perf_counter()
                meta = Meta(age=age, sex=sex, skin_tone=skin_tone, site=site)
                probs = predictor.predict(clin_img, derm_img, meta)
                cam = None
                if hasattr(predictor, "explain"):
                    cam = predictor.explain(clin_img, derm_img, meta)  # (ısı haritası görüntüsü, sınıf kodu) ya da None
                elapsed = time.perf_counter() - t0
                time.sleep(0.3)
                st.write("③ " + t("prog_3"))
                time.sleep(0.3)
                st.write("④ " + t("prog_4"))
                time.sleep(0.2)
                status.update(label=t("prog_done"), state="complete", expanded=False)
            ss.result = {
                "probs": probs,
                "elapsed": elapsed,
                "sig": current_sig,
                "case": case_id(current_sig),
                "created": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "model_name": predictor.name,
                "meta": {"age": age, "sex": sex, "skin_tone": skin_tone, "site": site},
                "clin_jpg": jpeg_thumb(clin_img),
                "derm_jpg": jpeg_thumb(derm_img),
                "cam_img": cam[0] if cam else None,
                "cam_class": cam[1] if cam else None,
                "cam_jpg": jpeg_thumb(cam[0]) if cam else None,
            }
            a0 = assess(probs)
            ss.history = [h for h in ss.history if h["case"] != ss.result["case"]]
            ss.history.insert(0, {"case": ss.result["case"], "time": datetime.now().strftime("%H:%M"),
                                  "meta": ss.result["meta"], "top": a0["ranked"][0], "level": a0["level"],
                                  "elapsed": elapsed})

    # 3. sonuç
    with right:
        step(3, t("s3_title"), t("s3_desc"))
        with st.container(border=True):
            res = ss.result
            if res is None:
                st.markdown(f'<div class="empty">{t("empty")}</div>', unsafe_allow_html=True)
            else:
                if res["sig"] != current_sig:
                    st.warning(t("stale"))
                a = assess(res["probs"])
                demo_note = f' · {t("demo_caption")}' if predictor.is_mock else ""
                st.markdown(f'<div class="case"><span>{t("case_no")} <b>{res["case"]}</b>{demo_note}</span>'
                            f'<span>{res["created"]}</span></div>', unsafe_allow_html=True)
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

                # Grad-CAM alanı
                st.markdown(f'<div class="sec-label">{t("cam_title")}</div>', unsafe_allow_html=True)
                if res.get("cam_img") is not None:
                    g1, g2 = st.columns(2)
                    g1.image(lens_preview(Image.open(io.BytesIO(res["derm_jpg"]))), caption=t("cam_orig"), width="stretch")
                    g2.image(lens_preview(res["cam_img"]), caption=t("cam_heat", c=i18n.plain(res["cam_class"])), width="stretch")
                    st.caption(t("cam_note"))
                else:
                    st.markdown(f'<div class="cam-empty">{t("cam_demo")}</div>', unsafe_allow_html=True)

                df = pd.DataFrame({
                    t("col_plain"): [i18n.plain(c) for c, _ in a["ranked"]],
                    t("col_medical"): [f"{i18n.cname(c)} ({c})" for c, _ in a["ranked"]],
                    t("col_group"): [i18n.group(GROUPS[c]) for c, _ in a["ranked"]],
                    t("col_prob"): [p for _, p in a["ranked"]],
                })
                st.write("")
                with st.expander(t("all_results")):
                    pc, gc, nc = t("col_prob"), t("col_group"), t("col_plain")
                    chart = (
                        alt.Chart(df).mark_bar(cornerRadiusEnd=3, size=12)
                        .encode(
                            x=alt.X(f"{pc}:Q", axis=alt.Axis(format="%", title=None, grid=False),
                                    scale=alt.Scale(domain=[0, max(0.05, float(df[pc].max()) * 1.1)])),
                            y=alt.Y(f"{nc}:N", sort="-x", title=None,
                                    axis=alt.Axis(labelFontSize=12, labelLimit=260, domain=False, ticks=False)),
                            color=alt.Color(f"{gc}:N", legend=None, scale=alt.Scale(
                                domain=[i18n.group(g) for g in GROUP_ORDER], range=[GROUP_COLORS[g] for g in GROUP_ORDER])),
                            tooltip=[nc, t("col_medical"), gc, alt.Tooltip(f"{pc}:Q", format=".1%")],
                        )
                        .properties(height=22 * len(df)).configure_view(strokeWidth=0)
                    )
                    st.altair_chart(chart, use_container_width=True)
                    st.dataframe(df.assign(**{pc: df[pc].map(i18n.pct)}), hide_index=True, width="stretch")

                # Hekim değerlendirmesi
                st.markdown(f'<div class="sec-label">{t("fb_title")}</div>', unsafe_allow_html=True)
                fb = ss.feedback.get(res["case"])
                if fb:
                    st.markdown(f'<div class="fb-done">✓ {html.escape(fb_summary(fb))}</div>', unsafe_allow_html=True)
                with st.expander(t("fb_edit") if fb else t("fb_open")):
                    st.caption(t("fb_intro"))
                    with st.form(key=f"fb_form_{res['case']}_{ss.lang}", border=False):
                        dx_opts = ["unknown"] + CLASSES
                        dx = st.selectbox(t("fb_dx"), dx_opts,
                                          index=dx_opts.index(fb["dx"]) if fb else None,
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
                                ss.feedback[res["case"]] = {"dx": dx, "agree": agree, "action": action, "note": note.strip(),
                                                            "time": datetime.now().strftime("%d.%m.%Y %H:%M")}
                                st.rerun()
                    st.caption(t("fb_session_note"))

                if build_pdf is not None:
                    try:
                        pdf_bytes = build_pdf(
                            res, a, predictor.is_mock, version=APP_VERSION, group_colors=GROUP_COLORS, groups_of=GROUPS,
                            group_order=GROUP_ORDER, meta_lines=meta_lines(res["meta"]),
                            physician=physician_lines(fb),
                        )
                        st.download_button(t("dl_pdf"), data=pdf_bytes, file_name=f"{res['case']}.pdf", mime="application/pdf",
                                           icon=":material/download:", width="stretch", help=t("dl_pdf_help"))
                    except Exception as exc:  # rapor üretilemezse sonuç ekranı yine çalışsın
                        st.caption(f"PDF: {exc}")

                with st.expander(t("tech")):
                    st.caption(t("tech_line", m=res["model_name"], s=res["elapsed"], c=res["created"],
                                 ok=t("tech_ok") if res["elapsed"] < 10 else t("tech_slow")))
                    st.caption(t("tech_thr", u=f"{REFER_URGENT_MALIGN * 100:.0f}", r=f"{REFER_MALIGN_OR_PRE * 100:.0f}",
                                 t=f"{UNCERTAIN_TOP * 100:.0f}", g=f"{UNCERTAIN_GAP * 100:.0f}"))

        if ss.history:
            with st.expander(t("history", n=len(ss.history))):
                lvl = {"urgent": t("lvl_urgent"), "refer": t("lvl_refer"), "benign": t("lvl_benign")}
                rows = [{
                    t("h_case"): h["case"], t("h_time"): h["time"],
                    t("h_patient"): f"{age_band_label(h['meta']['age'])} · {i18n.sex(h['meta']['sex'])} · "
                                    f"{t('tone_short')} {h['meta']['skin_tone']}",
                    t("h_top"): f"{i18n.plain(h['top'][0])} ({i18n.pct(h['top'][1])})",
                    t("h_level"): lvl[h["level"]],
                    t("h_physician"): t(AGREE[ss.feedback[h["case"]]["agree"]]) if h["case"] in ss.feedback else "–",
                } for h in ss.history]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(t("history_note"))

with tab_stats:
    render_stats(ss.history, ss.feedback, predictor.is_mock, GROUP_COLORS)

st.markdown(
    f'<div class="foot-grid"><div>{t("foot_left")}</div>'
    f'<div>{t("foot_right", v=APP_VERSION, team=TEAM, adv=ADVISOR)}</div></div>',
    unsafe_allow_html=True,
)
