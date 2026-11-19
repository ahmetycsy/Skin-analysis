import hashlib
import html
import io
import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageStat

from about import ADVISOR, AGE_RANGE, APP_VERSION, LOW_DATA_SKIN_TONES, SKIN_TONE_COUNTS, TEAM, render_about, render_guide

from predictor import (
    AGE_BANDS,
    BENIGN,
    CLASS_NAMES,
    CLASSES,
    DESCRIPTIONS,
    GROUP_ORDER,
    GROUPS,
    MALIGN,
    PLAIN_NAMES,
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

st.set_page_config(page_title="Cilt Lezyonu Karar Destek Sistemi", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------- renkler
INK = "#12303A"
TEAL = "#0F766E"
MIST = "#F2F7F6"
LINE = "#D3E2DF"
C_MALIGN = "#B4234A"
C_PREMALIGN = "#C27A12"
C_BENIGN = "#2A8C84"
GROUP_COLORS = {MALIGN: C_MALIGN, PREMALIGN: C_PREMALIGN, BENIGN: C_BENIGN}

SKIN_TONES = {
    0: ("#4A2C20", "Çok koyu"),
    1: ("#6F4531", "Koyu"),
    2: ("#94623F", "Orta koyu"),
    3: ("#B98558", "Orta"),
    4: ("#D9AE86", "Açık"),
    5: ("#F0D3B8", "Çok açık"),
}
SEX_LABELS = {"female": "Kadın", "male": "Erkek"}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; color: {INK}; }}
.stApp {{ background: {MIST}; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

h1, h2, h3, .display {{ font-family: 'Bricolage Grotesque', 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; letter-spacing: -0.01em; color: {INK}; }}
.hero-title {{ font-family: 'Bricolage Grotesque', 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif; font-weight: 700; font-size: 2.35rem; line-height: 1.1; margin: 0; }}
.hero-sub {{ color: #4B6670; font-size: 1.02rem; margin-top: .55rem; max-width: 46rem; line-height: 1.55; }}

.notice {{ display:flex; gap:.7rem; align-items:flex-start; background:#FFFFFF; border:1px solid {LINE};
  border-left:4px solid {C_PREMALIGN}; border-radius:10px; padding:.8rem 1rem; margin:1.1rem 0 .6rem; font-size:.93rem; line-height:1.5; }}
.notice.demo {{ border-left-color:{TEAL}; }}

.step-head {{ display:flex; align-items:baseline; gap:.7rem; margin: 1.6rem 0 .3rem; }}
.step-no {{ font-family:'Bricolage Grotesque','IBM Plex Sans','Segoe UI',system-ui,sans-serif; font-weight:700; font-size:1.05rem; color:#fff; background:{TEAL};
  border-radius:50%; width:1.9rem; height:1.9rem; display:inline-flex; align-items:center; justify-content:center; flex:none; }}
.step-title {{ font-family:'Bricolage Grotesque','IBM Plex Sans','Segoe UI',system-ui,sans-serif; font-weight:700; font-size:1.4rem; }}
.step-desc {{ color:#4B6670; font-size:.95rem; margin: 0 0 .8rem 2.6rem; max-width: 52rem; }}

.card-title {{ font-family:'Bricolage Grotesque','IBM Plex Sans','Segoe UI',system-ui,sans-serif; font-weight:700; font-size:1.12rem; margin:0 0 .25rem; }}
.card-text {{ color:#3F5963; font-size:.92rem; line-height:1.55; margin:0 0 .7rem; }}
.pill {{ display:inline-block; font-size:.78rem; font-weight:600; color:{TEAL}; background:#E2F1EF; border-radius:999px; padding:.15rem .6rem; margin-right:.35rem; }}

div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#FFFFFF; border:1px solid {LINE} !important; border-radius:14px; }}
[data-testid="stFileUploaderDropzone"] {{ background:{MIST}; border:1.5px dashed #9CC3BE; border-radius:12px; }}
[data-testid="stFileUploaderDropzone"] button {{ border-radius:8px; }}

div.stButton > button[kind="primary"] {{ background:{TEAL}; border:0; color:#fff; font-weight:600; border-radius:10px; padding:.65rem 1.4rem; font-size:1rem; }}
div.stButton > button[kind="primary"]:hover {{ background:#0B5F58; }}
div.stButton > button[kind="primary"]:disabled {{ background:#B9CFCC; color:#fff; }}
div.stButton > button[kind="secondary"], div.stDownloadButton > button {{ border-radius:10px; padding:.65rem 1.1rem; }}
div.stButton > button:focus-visible, div.stDownloadButton > button:focus-visible {{ outline:3px solid #7CC4BC; outline-offset:2px; }}

[data-testid="stSlider"], [data-testid="stRadio"] {{ filter: hue-rotate(178deg) saturate(.85) brightness(.82); }}
.swatches {{ display:flex; gap:.45rem; margin:.15rem 0 1rem; }}
.sw {{ flex:1; text-align:center; font-size:.74rem; color:#4B6670; }}
.sw i {{ display:block; height:2.1rem; border-radius:8px; border:2px solid transparent; margin-bottom:.25rem; }}
.sw.on i {{ border-color:{INK}; box-shadow:0 0 0 2px #fff inset; }}
.sw.on {{ color:{INK}; font-weight:600; }}

.rec {{ border-radius:12px; padding:.9rem 1rem; margin:.2rem 0 1rem; border:1px solid; }}
.rec-title {{ font-family:'Bricolage Grotesque','IBM Plex Sans','Segoe UI',system-ui,sans-serif; font-weight:700; font-size:1.25rem; margin-bottom:.25rem; }}
.rec-text {{ font-size:.94rem; line-height:1.55; }}
.rec.urgent {{ background:#FBEFF2; border-color:#EBC3CE; }}
.rec.urgent .rec-title {{ color:{C_MALIGN}; }}
.rec.refer {{ background:#FCF5EA; border-color:#EED7B0; }}
.rec.refer .rec-title {{ color:#97600E; }}
.rec.benign {{ background:#EAF5F3; border-color:#BFDDD8; }}
.rec.benign .rec-title {{ color:#1E6E67; }}

.unsure {{ background:#FFF8E6; border:1px solid #F0DCA4; border-radius:10px; padding:.7rem .9rem; font-size:.92rem; line-height:1.5; margin:0 0 1rem; }}

.sec-label {{ font-size:.8rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; color:#5B7480; margin:1.1rem 0 .5rem; }}
.stack {{ display:flex; height:.9rem; border-radius:999px; overflow:hidden; margin:.4rem 0 .5rem; background:{LINE}; }}
.legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.2rem; font-size:.88rem; margin-bottom:.4rem; }}
.legend b {{ font-variant-numeric: tabular-nums; }}
.dot {{ display:inline-block; width:.65rem; height:.65rem; border-radius:50%; margin-right:.35rem; }}

.cand {{ border:1px solid {LINE}; border-radius:12px; padding:.8rem .95rem; margin-bottom:.6rem; background:#FFFFFF; }}
.cand.first {{ border-width:2px; }}
.cand-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:.8rem; }}
.cand-name {{ font-family:'Bricolage Grotesque','IBM Plex Sans','Segoe UI',system-ui,sans-serif; font-weight:700; font-size:1.08rem; line-height:1.25; }}
.cand.first .cand-name {{ font-size:1.3rem; }}
.cand-pct {{ font-family:'Bricolage Grotesque','IBM Plex Sans','Segoe UI',system-ui,sans-serif; font-weight:700; font-size:1.15rem; font-variant-numeric: tabular-nums; white-space:nowrap; }}
.cand-bar {{ height:.4rem; border-radius:999px; background:{LINE}; margin:.45rem 0 .5rem; overflow:hidden; }}
.cand-bar i {{ display:block; height:100%; border-radius:999px; }}
.cand-desc {{ color:#3F5963; font-size:.9rem; line-height:1.5; }}
.cand-meta {{ color:#6A8089; font-size:.8rem; margin-top:.35rem; }}
.gpill {{ display:inline-block; font-size:.74rem; font-weight:600; border-radius:999px; padding:.05rem .5rem; margin-right:.4rem; color:#fff; }}

.empty {{ color:#4B6670; font-size:.98rem; line-height:1.6; padding:1.2rem .2rem; }}
.foot {{ color:#6A8089; font-size:.8rem; margin-top:2.2rem; line-height:1.55; }}
.appbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;
  background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; padding:.7rem 1rem; margin-bottom:1.1rem; }}
.brand {{ display:flex; align-items:center; gap:.7rem; }}
.brand-mark {{ width:2.4rem; height:2.4rem; border-radius:10px; background:{TEAL}; display:flex; align-items:center; justify-content:center; flex:none; }}
.brand-name {{ font-family:'Bricolage Grotesque','IBM Plex Sans',system-ui,sans-serif; font-weight:700; font-size:1.08rem; line-height:1.15; }}
.brand-sub {{ font-size:.78rem; color:#5B7480; }}
.status {{ display:flex; gap:.45rem; flex-wrap:wrap; }}
.chip {{ font-size:.76rem; font-weight:600; border-radius:999px; padding:.22rem .65rem; border:1px solid {LINE}; background:{MIST}; color:#3F5963; white-space:nowrap; }}
.chip.demo {{ background:#FFF4DD; border-color:#F0D49A; color:#8A5A0B; }}
.chip.live {{ background:#E2F1EF; border-color:#BFDDD8; color:{TEAL}; }}
.chip .led {{ display:inline-block; width:.45rem; height:.45rem; border-radius:50%; margin-right:.35rem; vertical-align:middle; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.3rem; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:.55rem .9rem; }}
.stTabs [aria-selected="true"] {{ color:{TEAL}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{TEAL}; }}
.lead {{ color:#4B6670; font-size:1rem; line-height:1.55; max-width:50rem; margin:.6rem 0 1rem; }}
.kpi {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; padding:.8rem 1rem; margin-bottom:.4rem; }}
.kpi-v {{ font-family:'Bricolage Grotesque','IBM Plex Sans',system-ui,sans-serif; font-weight:700; font-size:1.7rem; color:{TEAL}; font-variant-numeric:tabular-nums; }}
.kpi-l {{ font-size:.82rem; color:#5B7480; }}
.guide {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:12px; padding:1rem; height:100%; }}
.guide .step-no {{ margin-bottom:.6rem; }}
.legend-big div {{ margin:.35rem 0; font-size:.93rem; line-height:1.5; }}
.qrow {{ display:flex; flex-wrap:wrap; gap:.35rem; margin:.3rem 0 .2rem; }}
.q {{ font-size:.76rem; border-radius:999px; padding:.12rem .55rem; border:1px solid; white-space:nowrap; }}
.q.ok {{ color:#1E6E67; background:#EAF5F3; border-color:#BFDDD8; }}
.q.bad {{ color:#8A5A0B; background:#FFF4DD; border-color:#F0D49A; }}
.case {{ display:flex; justify-content:space-between; align-items:center; gap:.6rem; flex-wrap:wrap;
  font-size:.82rem; color:#5B7480; margin-bottom:.7rem; }}
.case b {{ color:{INK}; font-variant-numeric:tabular-nums; letter-spacing:.02em; }}
.foot-grid {{ display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; border-top:1px solid {LINE};
  padding-top:1rem; margin-top:2.4rem; color:#6A8089; font-size:.8rem; line-height:1.6; }}
@media (max-width: 640px) {{ .hero-title {{ font-size:1.7rem; }} .step-desc {{ margin-left:0; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- oturum durumu
ss = st.session_state
ss.setdefault("nonce", 0)      # artırılınca tüm giriş alanları sıfırlanır (Yeni hasta)
ss.setdefault("swap", False)   # iki görüntünün yeri değiştirildi mi
ss.setdefault("result", None)  # son analiz sonucu (sayfa yenilense de ekranda kalsın diye)
ss.setdefault("history", [])   # bu oturumdaki analizler (yalnızca bellekte, kaydedilmez)


def reset_all():
    ss.nonce += 1
    ss.swap = False
    ss.result = None


def toggle_swap():
    ss.swap = not ss.swap


@st.cache_resource
def get_predictor():
    return load_predictor()


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


def lens_preview(img: Image.Image, size: int = 340) -> Image.Image:
    """Dermoskopik görüntüyü dermatoskop merceğini andıran yuvarlak çerçevede gösterir (yalnızca önizleme)."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((4, 4, size - 4, size - 4), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img.convert("RGBA"), (0, 0), mask)
    ImageDraw.Draw(out).ellipse((4, 4, size - 4, size - 4), outline=(18, 48, 58, 255), width=5)
    return out


_LAPLACE = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128)


def quality_checks(img: Image.Image) -> list[tuple[str, bool, str]]:
    """Modelden bağımsız basit görüntü ön kontrolü. (ad, geçti mi, açıklama) listesi döner.
    Netlik eşiği MILK10k görüntüleriyle kabaca ayarlandı: gerçek görüntülerin %95'inden fazlası geçer,
    belirgin bulanık görüntüler takılır."""
    w, h = img.size
    g = img.convert("L")
    g.thumbnail((512, 512))
    sharp = ImageStat.Stat(g.filter(_LAPLACE)).stddev[0]
    gw, gh = g.size
    bright = ImageStat.Stat(g.crop((gw // 4, gh // 4, 3 * gw // 4, 3 * gh // 4))).mean[0]
    return [
        ("Çözünürlük", min(w, h) >= 300, f"{w}×{h} piksel" + ("" if min(w, h) >= 300 else "; en az 300 piksel önerilir")),
        ("Netlik", sharp >= 3.5, "net" if sharp >= 3.5 else "bulanık olabilir; odak kontrol edilmeli"),
        ("Işık", 50 <= bright <= 225, "uygun" if 50 <= bright <= 225 else ("çok karanlık" if bright < 50 else "aşırı parlak")),
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
        st.warning("Görüntü kalitesi sonucu etkileyebilir. " + " · ".join(bad))


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


def pct(p: float) -> str:
    return f"%{p * 100:.1f}"


def assess(probs: dict) -> dict:
    """Olasılıklardan genel değerlendirme, yönlendirme metni ve kararsızlık durumunu çıkarır."""
    totals = {g: sum(p for c, p in probs.items() if GROUPS[c] == g) for g in GROUP_ORDER}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_p, second_p = ranked[0][1], ranked[1][1]
    uncertain = top_p < UNCERTAIN_TOP or (top_p - second_p) < UNCERTAIN_GAP

    if totals[MALIGN] >= REFER_URGENT_MALIGN:
        level = "urgent"
        title = "Öncelikli dermatolog değerlendirmesi önerilir"
        text = (
            f"Model bu görüntülerde kötü huylu oluşumlara benzeyen özellikler görüyor "
            f"(kötü huylu türlerin toplam olasılığı {pct(totals[MALIGN])}). "
            "Lezyonun kısa süre içinde bir dermatolog tarafından incelenmesi önerilir."
        )
    elif totals[MALIGN] + totals[PREMALIGN] >= REFER_MALIGN_OR_PRE:
        level = "refer"
        title = "Dermatolog değerlendirmesi önerilir"
        text = (
            "Model bu görüntülerde kansere dönüşebilen veya kötü huylu oluşumlara benzer özellikler görüyor "
            f"(toplam olasılık {pct(totals[MALIGN] + totals[PREMALIGN])}). "
            "Lezyonun bir dermatolog tarafından incelenmesi önerilir."
        )
    elif uncertain:
        level = "refer"
        title = "Model kararsız: hekim değerlendirmesi önerilir"
        text = (
            "Model belirgin bir kötü huylu özellik görmüyor, ancak sonuçtan emin değil. "
            "Bu nedenle lezyonun bir hekim tarafından değerlendirilmesi önerilir."
        )
    else:
        level = "benign"
        title = "İyi huylu bir oluşuma benziyor"
        text = (
            "Model bu görüntülerde belirgin bir kötü huylu özellik görmüyor. Yine de lezyon büyürse; "
            "rengi veya şekli değişirse; kanar, kaşınır ya da yara açarsa bir hekime başvurulmalıdır."
        )
    return {"totals": totals, "ranked": ranked, "uncertain": uncertain, "level": level, "title": title, "text": text}


def build_report(res: dict, a: dict, is_mock: bool) -> str:
    """İndirilebilir, yazdırılabilir tek sayfalık HTML özet."""
    e = html.escape
    m = res["meta_labels"]
    rows = "".join(
        f"<tr><td>{e(PLAIN_NAMES[c])}</td><td>{e(CLASS_NAMES[c])} ({c})</td><td>{e(GROUPS[c])}</td><td style='text-align:right'>{pct(p)}</td></tr>"
        for c, p in a["ranked"]
    )
    top3 = "".join(
        f"<li><b>{e(PLAIN_NAMES[c])}</b> ({pct(p)}): {e(DESCRIPTIONS[c])}</li>" for c, p in a["ranked"][:3]
    )
    demo = (
        "<p class='warn'><b>DEMO MODU:</b> Bu sonuç sahte (rastgele) bir modelden üretilmiştir ve hiçbir tıbbi anlam taşımaz.</p>"
        if is_mock
        else ""
    )
    unsure = (
        "<p class='warn'><b>Model kararsız:</b> En olası sonuçlar birbirine yakın veya düşük olasılıklı. Bu sonuca güvenilmemelidir.</p>"
        if a["uncertain"]
        else ""
    )
    totals = " · ".join(f"{e(g)}: {pct(a['totals'][g])}" for g in GROUP_ORDER)
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>Cilt lezyonu analiz özeti</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#12303A;max-width:760px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
h1{{font-size:1.5rem;margin-bottom:.2rem}} h2{{font-size:1.1rem;margin-top:1.6rem}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} td,th{{border-bottom:1px solid #D3E2DF;padding:.35rem .4rem;text-align:left}}
.muted{{color:#5B7480;font-size:.9rem}} .warn{{background:#FFF8E6;border:1px solid #F0DCA4;padding:.6rem .8rem;border-radius:8px}}
.rec{{background:#F2F7F6;border:1px solid #D3E2DF;padding:.7rem .9rem;border-radius:8px}}
</style></head><body>
<h1>Cilt lezyonu analiz özeti</h1>
<p class="muted">Vaka no: <b>{e(res['case'])}</b> · Oluşturulma: {e(res['created'])} · Model: {e(res['model_name'])} · Sürüm {APP_VERSION}</p>
<p class="warn"><b>Bu belge tanı değildir.</b> Araştırma ve eğitim amaçlı bir prototipin çıktısıdır; hekim değerlendirmesinin yerine geçmez.</p>
{demo}
<h2>Hasta bilgileri</h2>
<table><tr><td>Yaş aralığı</td><td>{e(m['age'])}</td></tr><tr><td>Cinsiyet</td><td>{e(m['sex'])}</td></tr>
<tr><td>Ten rengi</td><td>{e(m['skin'])}</td></tr><tr><td>Bölge</td><td>{e(m['site'])}</td></tr></table>
<h2>Genel değerlendirme</h2>
<div class="rec"><b>{e(a['title'])}</b><br>{e(a['text'])}</div>
{unsure}
<p class="muted">{totals}</p>
<h2>En olası üç tür</h2><ol>{top3}</ol>
<h2>Tüm türler</h2>
<table><tr><th>Sade ad</th><th>Tıbbi adı</th><th>Grup</th><th style="text-align:right">Olasılık</th></tr>{rows}</table>
<p class="muted" style="margin-top:2rem">Veri seti: MILK10k (ISIC), CC-BY-NC 4.0. Bu uygulama bir tıbbi cihaz değildir.
Yazdırmak veya PDF olarak kaydetmek için tarayıcınızın Yazdır menüsünü kullanın.</p>
</body></html>"""


predictor = get_predictor()
n = ss.nonce

# ---------------------------------------------------------------- üst bar ve sekmeler
status_chip = (
    '<span class="chip demo"><span class="led" style="background:#C27A12"></span>Demo modu · sahte model</span>'
    if predictor.is_mock
    else f'<span class="chip live"><span class="led" style="background:#0F766E"></span>{html.escape(predictor.name)}</span>'
)
st.markdown(
    '<div class="appbar"><div class="brand"><div class="brand-mark">'
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round">'
    '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="3.2" fill="#fff" stroke="none"/>'
    '<path d="M12 1.5v3M12 19.5v3M1.5 12h3M19.5 12h3"/></svg></div>'
    '<div><div class="brand-name">Cilt Lezyonu Karar Destek Sistemi</div>'
    '<div class="brand-sub">BAU Endüstri Mühendisliği · Bitirme Projesi</div></div></div>'
    f'<div class="status">{status_chip}<span class="chip">Sürüm {APP_VERSION}</span><span class="chip">Veri: MILK10k</span></div></div>',
    unsafe_allow_html=True,
)

tab_analyze, tab_about, tab_guide = st.tabs(["Analiz", "Model hakkında", "Kullanım kılavuzu"])
with tab_about:
    render_about()
with tab_guide:
    render_guide()

with tab_analyze:
    st.markdown(
        '<h1 class="hero-title">Lezyon analizi</h1>'
        '<p class="hero-sub">Aynı lezyona ait klinik fotoğrafı, dermoskopik görüntüyü ve temel hasta bilgilerini birlikte '
        "değerlendirerek 11 lezyon türü için olasılık hesaplar. MILK10k veri setiyle geliştirilen bir araştırma çalışmasıdır.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice"><div><b>Bu araç tanı koymaz.</b> Araştırma ve eğitim amaçlı bir prototiptir; '
        "çıktıları bir hekimin değerlendirmesinin yerine geçmez. Şüpheli bir lezyon için mutlaka bir hekime başvurun.</div></div>",
        unsafe_allow_html=True,
    )
    if predictor.is_mock:
        st.markdown(
            '<div class="notice demo"><div><b>Demo modu.</b> Şu an gerçek model bağlı değil; '
            "görünen olasılıklar rastgele üretilir ve hiçbir anlam taşımaz. Arayüz denemesi içindir.</div></div>",
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------- 1. görüntüler
    step(1, "Görüntüler", "Aynı lezyona ait iki görüntüyü de yükleyin. İkisi birbirini tamamlar; model ikisini birlikte kullanır.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown(
                '<span class="pill">Fotoğraf</span><div class="card-title" style="margin-top:.5rem">Klinik yakın çekim</div>'
                '<p class="card-text">Lezyonun normal bir kamerayla veya telefonla, yaklaşık 10–30 cm mesafeden çekilmiş yakın fotoğrafı. '
                "Lezyonun rengini, şeklini, kenarlarını ve çevresindeki deriyi gösterir.</p>",
                unsafe_allow_html=True,
            )
            file_a = st.file_uploader("Klinik yakın çekim", type=["jpg", "jpeg", "png"], key=f"clin_{n}", label_visibility="collapsed")
            prev_a = st.container()
    with c2:
        with st.container(border=True):
            st.markdown(
                '<span class="pill">Büyütmeli görüntü</span><div class="card-title" style="margin-top:.5rem">Dermoskopik görüntü</div>'
                '<p class="card-text">Dermatoskop adı verilen, ışıklı bir büyüteçle çekilen görüntü. Çıplak gözle görünmeyen renk desenlerini, '
                "damarları ve deri yüzeyinin hemen altındaki ayrıntıları gösterir; genellikle yuvarlak bir çerçeveye sahiptir.</p>",
                unsafe_allow_html=True,
            )
            file_b = st.file_uploader("Dermoskopik görüntü", type=["jpg", "jpeg", "png"], key=f"derm_{n}", label_visibility="collapsed")
            prev_b = st.container()

    img_a, id_a = load_image(file_a)
    img_b, id_b = load_image(file_b)
    if ss.swap:
        clin_img, clin_id, derm_img, derm_id = img_b, id_b, img_a, id_a
    else:
        clin_img, clin_id, derm_img, derm_id = img_a, id_a, img_b, id_b

    swap_note = "Yer değiştirildi: diğer kutuya yüklenen görüntü burada kullanılıyor."
    with prev_a:
        if file_a is not None and img_a is None:
            st.error("Bu dosya okunamadı. Lütfen geçerli bir JPG veya PNG yükleyin.")
        if clin_img is not None:
            st.image(clin_img, width=340, caption=swap_note if ss.swap else None)
            show_quality(clin_img)
    with prev_b:
        if file_b is not None and img_b is None:
            st.error("Bu dosya okunamadı. Lütfen geçerli bir JPG veya PNG yükleyin.")
        if derm_img is not None:
            cap = swap_note if ss.swap else "Önizleme (mercek çerçevesi yalnızca gösterim içindir)"
            st.image(lens_preview(derm_img), caption=cap, width=340)
            show_quality(derm_img)

    if clin_img is not None and derm_img is not None:
        if looks_dermoscopic(clin_img) and not looks_dermoscopic(derm_img):
            st.warning(
                "**Görüntüler ters yüklenmiş olabilir.** Klinik yakın çekim kutusundaki görüntü dermoskopik bir görüntüye "
                "benziyor (koyu, yuvarlak çerçeve). Kontrol edin; gerekiyorsa aşağıdaki düğmeyle yerlerini değiştirin."
            )
        st.button(
            "Görüntülerin yerini geri al" if ss.swap else "Görüntülerin yerini değiştir",
            on_click=toggle_swap,
            icon=":material/swap_horiz:",
        )

    with st.expander("İki görüntü arasındaki fark nedir, nasıl çekilmeli?"):
        st.markdown(
            """
    | | Klinik yakın çekim | Dermoskopik görüntü |
    |---|---|---|
    | Nasıl çekilir | Telefon veya kamera, 10–30 cm | Dermatoskop (ışıklı büyüteç), deriye temas ederek veya yakından |
    | Ne gösterir | Renk, şekil, kenar, lezyonun konumu ve çevre deri | Renk desenleri, damarlar, yüzeyin hemen altındaki ayrıntılar |
    | Kimler çeker | Herkes | Genellikle hekim veya eğitimli sağlık çalışanı |

    **İyi bir görüntü için:** Lezyon net ve odakta olmalı, ışık eşit olmalı (gölge ve parlama olmamalı), lezyon kadrajın ortasında yer almalı.
    Filtre, kırpma veya renk düzeltmesi uygulanmış görüntüleri kullanmayın.
            """
        )

    # ---------------------------------------------------------------- 2. hasta bilgileri
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        step(2, "Hasta bilgileri", "Model bu dört bilgiyi görüntülerle birlikte kullanır. Hepsini hastaya göre seçin.")
        with st.container(border=True):
            band_labels = {lo: age_band_label(lo) for lo, _ in AGE_BANDS}
            age = st.selectbox(
                "Yaş aralığı",
                options=[lo for lo, _ in AGE_BANDS],
                index=None,
                placeholder="Yaş aralığını seçin",
                format_func=lambda lo: f"{band_labels[lo]} yaş",
                help="Veri seti yaşı 5 yıllık aralıklarla tutar.",
                key=f"age_{n}",
            )
            sex = st.radio("Cinsiyet", list(SEX_LABELS), index=None, format_func=SEX_LABELS.get, horizontal=True, key=f"sex_{n}")
            skin_tone = st.radio(
                "Ten rengi",
                list(SKIN_TONES),
                index=None,
                format_func=lambda i: f"{i} · {SKIN_TONES[i][1]}",
                horizontal=True,
                help="0 en koyu, 5 en açık ten rengidir. Aşağıdaki renk örnekleri yalnızca fikir vermek içindir.",
                key=f"skin_{n}",
            )
            sw = "".join(
                f'<div class="sw{" on" if i == skin_tone else ""}"><i style="background:{c}"></i>{i}</div>'
                for i, (c, _) in SKIN_TONES.items()
            )
            st.markdown(f'<div class="swatches">{sw}</div>', unsafe_allow_html=True)

            site = st.selectbox(
                "Lezyonun bulunduğu bölge",
                list(SITES),
                index=None,
                placeholder="Vücut bölgesini seçin",
                format_func=lambda k: SITES[k],
                key=f"site_{n}",
            )

        if skin_tone in LOW_DATA_SKIN_TONES:
            st.info(
                f"Eğitim verisinde bu ten rengi grubundan yalnızca **{SKIN_TONE_COUNTS[skin_tone]} lezyon** var. "
                "Bu grupta sonuçlar daha az güvenilir olabilir; hekim değerlendirmesine öncelik verin."
            )
        if age is not None and (age < AGE_RANGE[0] or age > AGE_RANGE[1]):
            st.info(f"Eğitim verisinde yaş {AGE_RANGE[0]}–{AGE_RANGE[1]} arasındadır. Bu yaş aralığında sonuçlar daha belirsizdir.")

        missing = [
            name
            for name, val in [
                ("klinik yakın çekim", clin_img),
                ("dermoskopik görüntü", derm_img),
                ("yaş aralığı", age),
                ("cinsiyet", sex),
                ("ten rengi", skin_tone),
                ("bölge", site),
            ]
            if val is None
        ]
        ready = not missing
        current_sig = (clin_id, derm_id, age, sex, skin_tone, site)

        b1, b2 = st.columns([1.4, 1])
        with b1:
            run = st.button("Analiz et", type="primary", disabled=not ready, width="stretch")
        with b2:
            st.button("Yeni hasta", on_click=reset_all, width="stretch", icon=":material/restart_alt:",
                      help="Tüm görüntüleri, bilgileri ve sonucu temizler.")
        if missing:
            st.caption("Analiz için eksik: " + ", ".join(missing) + ".")

        if run and ready:
            t0 = time.perf_counter()
            probs = predictor.predict(clin_img, derm_img, Meta(age=age, sex=sex, skin_tone=skin_tone, site=site))
            ss.result = {
                "probs": probs,
                "elapsed": time.perf_counter() - t0,
                "sig": current_sig,
                "case": case_id(current_sig),
                "created": datetime.now().strftime("%d.%m.%Y %H:%M"),
                "model_name": predictor.name,
                "meta_labels": {
                    "age": f"{band_labels[age]} yaş",
                    "sex": SEX_LABELS[sex],
                    "skin": f"{skin_tone} · {SKIN_TONES[skin_tone][1]}",
                    "site": SITES[site],
                },
            }
            _a = assess(probs)
            top_c, top_p = _a["ranked"][0]
            ss.history = [h for h in ss.history if h["Vaka no"] != ss.result["case"]]
            ss.history.insert(0, {
                "Vaka no": ss.result["case"],
                "Saat": datetime.now().strftime("%H:%M"),
                "Hasta": f"{band_labels[age]} · {SEX_LABELS[sex]} · ten {skin_tone}",
                "En olası tür": f"{PLAIN_NAMES[top_c]} ({pct(top_p)})",
                "Değerlendirme": {"urgent": "Öncelikli", "refer": "Değerlendirme", "benign": "İyi huylu görünüm"}[_a["level"]],
            })

    # ---------------------------------------------------------------- 3. sonuç
    with right:
        step(3, "Sonuç", "Model 11 lezyon türünün her biri için bir olasılık hesaplar. Önce genel değerlendirme, sonra en olası üç tür gösterilir.")
        with st.container(border=True):
            res = ss.result
            if res is None:
                st.markdown(
                    '<div class="empty">İki görüntüyü yükleyin, hasta bilgilerini seçin ve <b>Analiz et</b> düğmesine basın. '
                    "Sonuç burada görünür.</div>",
                    unsafe_allow_html=True,
                )
            else:
                if res["sig"] != current_sig:
                    st.warning(
                        "**Görüntüler veya hasta bilgileri değişti.** Aşağıdaki sonuç önceki bilgilere aittir. "
                        "Güncel sonuç için yeniden **Analiz et** düğmesine basın."
                    )
                if predictor.is_mock:
                    st.caption("Demo modu: aşağıdaki sonuçlar rastgeledir.")

                a = assess(res["probs"])
                st.markdown(
                    f'<div class="case"><span>Vaka no <b>{res["case"]}</b></span><span>{res["created"]}</span></div>',
                    unsafe_allow_html=True,
                )

                # Genel değerlendirme
                st.markdown(
                    f'<div class="rec {a["level"]}"><div class="rec-title">{a["title"]}</div>'
                    f'<div class="rec-text">{a["text"]}</div></div>',
                    unsafe_allow_html=True,
                )
                if a["uncertain"]:
                    st.markdown(
                        '<div class="unsure"><b>Model bu görüntülerde kararsız.</b> En olası sonuçlar birbirine çok yakın '
                        "veya olasılıkları düşük. Aşağıdaki sıralamaya güvenmeyin; görüntü kalitesini kontrol edin ve "
                        "hekim değerlendirmesine öncelik verin.</div>",
                        unsafe_allow_html=True,
                    )

                bar = "".join(
                    f'<div style="width:{a["totals"][g] * 100:.2f}%;background:{GROUP_COLORS[g]}"></div>' for g in GROUP_ORDER
                )
                legend = "".join(
                    f'<span><span class="dot" style="background:{GROUP_COLORS[g]}"></span>{g} <b>{pct(a["totals"][g])}</b></span>'
                    for g in GROUP_ORDER
                )
                st.markdown(f'<div class="stack">{bar}</div><div class="legend">{legend}</div>', unsafe_allow_html=True)

                # En olası üç tür
                st.markdown('<div class="sec-label">En olası üç tür</div>', unsafe_allow_html=True)
                cards = ""
                for i, (c, p) in enumerate(a["ranked"][:3]):
                    col = GROUP_COLORS[GROUPS[c]]
                    cards += (
                        f'<div class="cand{" first" if i == 0 else ""}" style="{"border-color:" + col if i == 0 else ""}">'
                        f'<div class="cand-head"><div class="cand-name">{i + 1}. {PLAIN_NAMES[c]}</div>'
                        f'<div class="cand-pct" style="color:{col}">{pct(p)}</div></div>'
                        f'<div class="cand-bar"><i style="width:{p * 100:.2f}%;background:{col}"></i></div>'
                        f'<div class="cand-desc">{DESCRIPTIONS[c]}</div>'
                        f'<div class="cand-meta"><span class="gpill" style="background:{col}">{GROUPS[c]}</span>'
                        f"Tıbbi adı: {CLASS_NAMES[c]} ({c})</div></div>"
                    )
                st.markdown(cards, unsafe_allow_html=True)

                # Tüm türler
                df = pd.DataFrame(
                    {
                        "Sade ad": [PLAIN_NAMES[c] for c, _ in a["ranked"]],
                        "Tıbbi adı": [f"{CLASS_NAMES[c]} ({c})" for c, _ in a["ranked"]],
                        "Grup": [GROUPS[c] for c, _ in a["ranked"]],
                        "Olasılık": [p for _, p in a["ranked"]],
                    }
                )
                with st.expander("Tüm sonuçlar (11 tür)"):
                    chart = (
                        alt.Chart(df)
                        .mark_bar(cornerRadiusEnd=4, size=14)
                        .encode(
                            x=alt.X("Olasılık:Q", axis=alt.Axis(format="%", title=None, grid=False),
                                    scale=alt.Scale(domain=[0, max(0.05, float(df["Olasılık"].max()) * 1.1)])),
                            y=alt.Y("Sade ad:N", sort="-x", title=None,
                                    axis=alt.Axis(labelFontSize=12, labelLimit=260, domain=False, ticks=False)),
                            color=alt.Color("Grup:N", scale=alt.Scale(domain=GROUP_ORDER, range=[GROUP_COLORS[g] for g in GROUP_ORDER]), legend=None),
                            tooltip=["Sade ad", "Tıbbi adı", "Grup", alt.Tooltip("Olasılık:Q", format=".1%")],
                        )
                        .properties(height=24 * len(df))
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(chart, use_container_width=True)
                    st.dataframe(df.assign(Olasılık=df["Olasılık"].map(pct)), hide_index=True, width="stretch")

                st.download_button(
                    "Sonucu indir (yazdırılabilir özet)",
                    data=build_report(res, a, predictor.is_mock),
                    file_name=f"lezyon_analiz_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html",
                    icon=":material/download:",
                    width="stretch",
                    help="Tarayıcıda açılan bu özeti Yazdır menüsünden yazdırabilir veya PDF olarak kaydedebilirsiniz.",
                )

                with st.expander("Teknik ayrıntılar"):
                    ok = res["elapsed"] < 10
                    st.caption(
                        f"Model: {res['model_name']} · Hesaplama süresi {res['elapsed']:.2f} sn "
                        f"({'10 sn hedefinin altında' if ok else '10 sn hedefinin üzerinde'}) · Analiz zamanı {res['created']}"
                    )
                    st.caption(
                        f"Yönlendirme eşikleri (geçici): kötü huylu toplamı ≥ %{REFER_URGENT_MALIGN * 100:.0f} → öncelikli; "
                        f"kötü huylu + kansere dönüşebilir ≥ %{REFER_MALIGN_OR_PRE * 100:.0f} → değerlendirme. "
                        f"Kararsızlık: en yüksek olasılık < %{UNCERTAIN_TOP * 100:.0f} veya ilk iki fark < %{UNCERTAIN_GAP * 100:.0f}."
                    )

        if ss.history:
            with st.expander(f"Bu oturumdaki analizler ({len(ss.history)})"):
                st.dataframe(pd.DataFrame(ss.history), hide_index=True, width="stretch")
                st.caption("Yalnızca bu sayfa açıkken tutulur; hiçbir yere kaydedilmez. Görüntüler listede saklanmaz.")

st.markdown(
    '<div class="foot-grid"><div>Veri seti: MILK10k (ISIC Archive), CC-BY-NC 4.0.<br>'
    "Bu uygulama bir tıbbi cihaz değildir ve klinik kullanım için onaylanmamıştır.<br>"
    "Yüklenen görüntüler yalnızca analiz sırasında kullanılır ve kaydedilmez.</div>"
    f"<div>BAU Endüstri Mühendisliği · Bitirme Projesi · Sürüm {APP_VERSION}<br>{TEAM}<br>Danışman: {ADVISOR}</div></div>",
    unsafe_allow_html=True,
)
