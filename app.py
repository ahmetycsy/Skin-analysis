import hashlib
import hmac
import html
import io
import os
import time
import zipfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
    MEL_SAFETY,
    PRIORITY_RULE,
    PRIORITY_TYPE_T,
    PRIORITY_TYPES,
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
from survey_page import render_survey
from patient_help import (appointment_html, calm_card_html, calm_meaning_html, details_warning, ics_bytes, reliability_html, render_symptom_form,
                          symptom_advice, symptom_lines)
from health_page import render_photo_tips, render_share, render_skin_health, render_versions
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

# ---------------------------------------------------------------- tema (açık / koyu)
# Koyu tema Streamlit'in hazır "Dark" temasıdır; yan menüdeki anahtar tarayıcıya kaydeder ve sayfayı ?theme=dark ile yeniler.
# İlk açılışta st.context.theme.type gecikmeli gelebildiği için adres satırındaki ?theme önceliklidir.
_qp_theme = st.query_params.get("theme")
if _qp_theme in ("dark", "light"):
    st.session_state["dark"] = _qp_theme == "dark"
elif st.context.theme.type in ("dark", "light"):
    st.session_state["dark"] = st.context.theme.type == "dark"
DARK = bool(st.session_state.get("dark"))

# ---------------------------------------------------------------- renkler (sakin, düşük kontrastlı yüzeyler)
if DARK:
    INK, MUTED, TEAL, BG, LINE = "#E4ECEF", "#9AA8AF", "#0F766E", "#0E1117", "#2B3340"
    SURF, TEAL_TXT, SOFT, SOFT_B = "#161B22", "#3CC2B2", "#10302C", "#1E4A44"
else:
    INK, MUTED, TEAL, BG, LINE = "#1C2B31", "#55686F", "#0E5E57", "#F7F9F9", "#DCE4E3"
    SURF, TEAL_TXT, SOFT, SOFT_B = "#FFFFFF", "#0E5E57", "#E6F0EE", "#C3D9D5"
C_MALIGN = "#A63D57"
C_PREMALIGN = "#B0701A"
C_BENIGN = "#2F7F78"
GROUP_COLORS = {MALIGN: C_MALIGN, PREMALIGN: C_PREMALIGN, BENIGN: C_BENIGN}
SKIN_SWATCH = {0: "#4A2C20", 1: "#6F4531", 2: "#94623F", 3: "#B98558", 4: "#D9AE86", 5: "#F0D3B8"}
STATUSES = {"new": ("stat_new", "#5F747C"), "reviewed": ("stat_reviewed", "#2F7F78"),
            "awaiting": ("stat_awaiting", "#B0701A"), "closed": ("stat_closed", "#56707A" if DARK else "#1F3A44")}
LEVEL_COLORS = {"urgent": C_MALIGN, "refer": C_PREMALIGN, "benign": C_BENIGN}
LEVEL_RANK = {"urgent": 0, "refer": 1, "benign": 2}
ABCDE = ["A", "B", "C", "D", "E"]  # asimetri, sınır, renk, çap > 6 mm, değişim (hekim işaretler; modele girmez)
METHODS = {"histo": "m_histo", "clinical": "m_clinical", "consensus": "m_consensus"}
APP_URL = "https://bau-skin-cds.streamlit.app"  # QR koddaki bağlantı (Streamlit Cloud adresi)

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] {{ font-family:'IBM Plex Sans', system-ui, -apple-system, sans-serif; color:{INK}; font-variant-numeric:tabular-nums; }}
.stApp {{ background:{BG}; }}
.block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1080px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
[data-testid="stToolbarActions"] {{ display:none !important; }}  /* Fork / GitHub gizli; menü düğmesi (stExpandSidebarButton) aynı araç çubuğunda, görünür kalmalı */
h1, h2, h3 {{ font-family:'IBM Plex Sans', system-ui, sans-serif; color:{INK}; letter-spacing:-0.01em; }}

/* yan menü */
section[data-testid="stSidebar"] {{ background:#fff; border-right:1px solid {LINE}; }}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div {{ padding-top:1.2rem; }}
.brand {{ display:flex; align-items:center; gap:.65rem; margin-bottom:1.1rem; }}
.brand-mark {{ width:2.2rem; height:2.2rem; border-radius:9px; background:{TEAL}; display:flex; align-items:center; justify-content:center; flex:none; }}
.brand-name {{ font-weight:600; font-size:.98rem; line-height:1.2; }}
.brand-sub {{ font-size:.76rem; color:{MUTED}; line-height:1.35; }}
section[data-testid="stSidebar"] div.stButton > button {{ justify-content:flex-start; border:0; background:transparent; color:{INK};
  font-weight:500; padding:.45rem .7rem; border-radius:8px; }}
section[data-testid="stSidebar"] div.stButton > button:hover {{ background:{BG}; color:{INK}; }}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{ background:{SOFT}; color:{TEAL}; font-weight:600; }}
.side-sep {{ border-top:1px solid {LINE}; margin:1rem 0 .8rem; }}
.side-note {{ font-size:.8rem; color:{MUTED}; line-height:1.5; }}
.chip {{ display:inline-block; font-size:.76rem; font-weight:500; border-radius:999px; padding:.15rem .55rem; border:1px solid {LINE};
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
.card-text {{ color:{MUTED}; font-size:.92rem; line-height:1.55; margin:0 0 .6rem; }}
.pill {{ display:inline-block; font-size:.72rem; font-weight:500; color:{TEAL}; background:#E7F2F0; border-radius:999px; padding:.1rem .55rem; }}
[data-testid="stFileUploaderDropzone"] {{ background:{BG}; border:1.5px dashed #B5CFCB; border-radius:10px; }}

/* düğmeler */
div.stButton > button, div.stDownloadButton > button, div.stFormSubmitButton > button {{ border-radius:9px; font-weight:500; }}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{ background:{TEAL}; border:0; color:#fff; }}
div.stButton > button[kind="primary"]:hover {{ background:#0A4B45; }}
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
.case-label {{ font-size:.88rem; font-weight:500; color:{MUTED}; }}
.case-no {{ font-weight:600; font-size:1.45rem; letter-spacing:.03em; font-variant-numeric:tabular-nums; }}
.case-meta {{ font-size:.88rem; color:{MUTED}; margin-top:.15rem; }}
.spill {{ display:inline-block; font-size:.8rem; font-weight:600; border-radius:999px; padding:.12rem .6rem; white-space:nowrap; border:1px solid; }}
.case-hint {{ font-size:.8rem; color:{MUTED}; margin-top:.5rem; }}

/* sonuç */
.rec {{ border-radius:10px; padding:.85rem 1rem; margin:0 0 .8rem; border:1px solid; }}
.rec-ic {{ display:inline-block; font-weight:700; margin-right:.45rem; }}
.dc-rows div {{ padding:.15rem 0; font-size:.92rem; }}
.rec-title {{ font-weight:600; font-size:1.06rem; margin-bottom:.2rem; }}
.rec-text {{ font-size:.9rem; line-height:1.55; }}
.rec.urgent {{ background:#FAF0F2; border-color:#EBCDD5; }} .rec.urgent .rec-title {{ color:{C_MALIGN}; }}
.rec.refer {{ background:#FBF5EA; border-color:#EBD9B8; }} .rec.refer .rec-title {{ color:#8C5A12; }}
.rec.benign {{ background:#EDF6F4; border-color:#CFE5E1; }} .rec.benign .rec-title {{ color:#25685F; }}
.unsure {{ background:#FCF8EE; border:1px solid #EFE2BF; border-radius:8px; padding:.6rem .85rem; font-size:.87rem; line-height:1.5; margin:0 0 .8rem; }}
.sym-warn {{ background:#FBEFF2; border:1px solid #EBC9D2; border-radius:8px; padding:.6rem .85rem; font-size:.87rem; line-height:1.5; margin:0 0 .8rem; }}
.sym-info {{ background:var(--soft,#E7F2F0); border:1px solid var(--softb,#C5DFDA); border-radius:8px; padding:.6rem .85rem; font-size:.87rem; line-height:1.5; margin:0 0 .8rem; }}
.rely {{ border:1px solid var(--line,#DDE6E5); border-radius:8px; padding:.65rem .85rem; margin:0 0 .8rem; font-size:.87rem; line-height:1.5; }}
.rely-t {{ font-weight:600; margin-bottom:.2rem; }} .rely-h {{ color:var(--muted,#5F747C); font-size:.8rem; margin-top:.3rem; }}
.sec-label {{ font-size:.95rem; font-weight:600; color:{INK}; margin:1.1rem 0 .45rem; }}
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
.cand-meta {{ grid-column:1 / -1; color:{MUTED}; font-size:.8rem; }}
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
.mod-row {{ margin:.1rem 0 .55rem; }} .mod-name {{ font-size:.86rem; font-weight:600; }}
.mod-sub {{ font-size:.8rem; color:{MUTED}; }}
.tour-dots {{ display:flex; gap:.35rem; align-items:center; }} .tour-dots i {{ width:.5rem; height:.5rem; border-radius:50%; background:{LINE}; }}
.tour-dots i.on {{ background:{TEAL}; width:1.3rem; border-radius:999px; }} .tour-dots span {{ font-size:.75rem; color:{MUTED}; margin-left:.4rem; }}
.ps-note {{ font-size:.85rem; color:{MUTED}; margin-bottom:.2rem; }}
.empty {{ color:{MUTED}; font-size:.94rem; line-height:1.6; padding:1rem .2rem; }}

/* vaka listesi */
.row-no {{ font-weight:600; font-variant-numeric:tabular-nums; letter-spacing:.02em; }}
.row-sub {{ font-size:.85rem; color:{MUTED}; }}

/* model kartı ve kılavuz */
.lead {{ color:{MUTED}; font-size:.97rem; line-height:1.55; max-width:50rem; margin:.2rem 0 1rem; }}
.kpi {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.75rem .95rem; margin-bottom:.4rem; }}
.kpi-v {{ font-weight:600; font-size:1.45rem; color:{INK}; font-variant-numeric:tabular-nums; }}
.kpi-l {{ font-size:.8rem; color:{MUTED}; }}
.guide {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:.95rem; height:100%; }}
[class*="st-key-peek_"] button {{ padding-left:.5rem; padding-right:.5rem; }}
[class*="st-key-peek_"] button p {{ white-space:nowrap; }}
.peek-line {{ border-top:1px dashed {LINE}; margin:.2rem 0 .3rem; }}
.peek-h {{ font-size:.82rem; color:{MUTED}; margin-bottom:.2rem; }}
.peek-kv {{ display:flex; justify-content:space-between; gap:.8rem; font-size:.9rem; padding:.12rem 0; }}
.peek-kv b {{ font-weight:600; font-variant-numeric:tabular-nums; }}
.peek-kv2 {{ font-size:.86rem; margin-bottom:.35rem; line-height:1.4; }}
.peek-kv2 span {{ display:block; color:{MUTED}; font-size:.78rem; }}
.tips-grid {{ display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); grid-auto-rows:1fr; gap:.75rem; margin:.3rem 0 .6rem; }}
.tips-grid .guide {{ height:auto; }}
@media (max-width:900px) {{ .tips-grid {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }} }}
@media (max-width:480px) {{ .tips-grid {{ grid-template-columns:1fr; grid-auto-rows:auto; }} }}
.mtable-wrap {{ overflow-x:auto; border:1px solid {LINE}; border-radius:10px; margin:.3rem 0 .5rem; }}
.mtable {{ width:100%; border-collapse:collapse; font-size:.86rem; line-height:1.45; }}
.mtable th {{ text-align:left; font-weight:600; color:{MUTED}; padding:.55rem .7rem; border-bottom:1px solid {LINE}; white-space:nowrap; }}
.mtable td {{ padding:.55rem .7rem; border-bottom:1px solid {LINE}; vertical-align:top; color:{INK}; }}
.mtable tr:last-child td {{ border-bottom:none; }}
.mtable .mt-name {{ font-weight:500; min-width:12rem; }}
.mtable .mt-nw {{ white-space:nowrap; }}
.mtable .mt-num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.mtable .mt-note {{ color:{MUTED}; min-width:16rem; }}
.guide .step-no {{ font-weight:600; font-size:.8rem; color:#fff; background:{TEAL}; border-radius:50%; width:1.5rem; height:1.5rem;
  display:inline-flex; align-items:center; justify-content:center; margin-bottom:.55rem; }}
.legend-big div {{ margin:.3rem 0; font-size:.9rem; line-height:1.5; }}
.notice {{ background:#fff; border:1px solid {LINE}; border-left:3px solid {C_PREMALIGN}; border-radius:8px; padding:.55rem .8rem; margin:.4rem 0; font-size:.88rem; }}
.notice.demo {{ border-left-color:{TEAL}; }}
@media (max-width:640px) {{ .page-title {{ font-size:1.3rem; }} .case-no {{ font-size:1.2rem; }}
  .stepper {{ flex-wrap:nowrap; gap:.3rem; }} .stp {{ font-size:.8rem; gap:.3rem; white-space:nowrap; }} .stp i {{ width:1.35rem; height:1.35rem; }}
  .stp-line {{ flex-basis:.6rem; }}
  [data-testid="stHorizontalBlock"]:has(.kpi), [data-testid="stHorizontalBlock"]:has(.guide) {{ flex-wrap:wrap !important; gap:.5rem !important; }}
  [data-testid="stHorizontalBlock"]:has(.kpi) > [data-testid="stColumn"], [data-testid="stHorizontalBlock"]:has(.guide) > [data-testid="stColumn"]
    {{ flex:1 1 calc(50% - .5rem) !important; width:calc(50% - .5rem) !important; min-width:calc(50% - .5rem) !important; }}
  .kpi {{ padding:.6rem .7rem; }} .role-card {{ padding:1rem; }} }}
.chip {{ white-space:normal; max-width:100%; overflow-wrap:anywhere; }}
.role-card {{ background:var(--surf,#fff); border:1px solid {LINE}; border-radius:12px; padding:1.2rem 1.2rem .9rem; margin-bottom:.5rem; }}
.role-card .ic {{ color:{TEAL_TXT}; line-height:1; }} .role-card .rt {{ font-weight:600; font-size:1.05rem; margin:.5rem 0 .2rem; }}
.role-card .rd {{ color:{MUTED}; font-size:.9rem; line-height:1.45; }}
/* yönlendirme cetveli (imza öğe) */
.ruler {{ margin:.35rem 0 .2rem; }}
.ruler-bar {{ position:relative; display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; height:.8rem; }}
.ruler-bar i {{ border-radius:3px; }}
.ruler-mk {{ position:absolute; top:-.4rem; width:4px; height:1.6rem; background:{INK}; border-radius:2px; transform:translateX(-50%); box-shadow:0 0 0 2px {SURF}; }}
.ruler-lb {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:4px; font-size:.84rem; color:{MUTED}; margin-top:.45rem; line-height:1.3; }}
.ruler-lb span:nth-child(2) {{ text-align:center; }} .ruler-lb span:nth-child(3) {{ text-align:right; }}
.ruler-lb span.on {{ color:{INK}; font-weight:600; }}
.ruler-note {{ font-size:.8rem; color:{MUTED}; margin:.25rem 0 0; }}
.appt {{ margin:.2rem 0 0 1.1rem; padding:0; font-size:.9rem; line-height:1.55; }}
.appt li {{ margin:.15rem 0; }}
.border-note {{ border-left:3px solid {C_PREMALIGN}; background:{C_PREMALIGN}14; color:{INK}; padding:.55rem .8rem;
  border-radius:6px; font-size:.9rem; line-height:1.5; margin:0 0 .8rem; }}
.crop-guide {{ font-size:.88rem; color:{INK}; line-height:1.5; margin:.1rem 0 .45rem; }}
.mini-ruler {{ display:inline-grid; grid-template-columns:repeat(3,.9rem); gap:2px; vertical-align:middle; margin-right:.45rem; }}
.mini-ruler i {{ height:.42rem; border-radius:2px; }}
.hero {{ background:{SURF}; border:1px solid {LINE}; border-radius:14px; padding:1.1rem 1.2rem .9rem; margin:.4rem 0 .9rem; }}
.hero .rec {{ border:0; background:transparent; padding:0; margin:0 0 .7rem; }}
.hero .rec-title {{ font-size:1.32rem; line-height:1.3; }}
.hero .rec-text {{ font-size:.98rem; max-width:46rem; }}
.case-top {{ display:flex; justify-content:space-between; align-items:baseline; gap:.4rem 1rem; flex-wrap:wrap; margin:.1rem 0 .2rem; }}
.mode-line {{ font-size:.95rem; color:{INK}; padding:.35rem 0; }}
.mode-line b {{ font-weight:600; }}
.how {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.8rem; margin:1.4rem 0 .6rem; }}
.how div {{ border-top:2px solid {LINE}; padding-top:.55rem; font-size:.92rem; color:{MUTED}; line-height:1.45; }}
.how b {{ display:block; color:{INK}; font-weight:600; margin-bottom:.15rem; }}
.privacy {{ font-size:.88rem; color:{MUTED}; line-height:1.5; max-width:46rem; }}
.role-card svg {{ display:block; }}
.summary-line {{ font-size:1rem; line-height:1.6; background:{SURF}; border:1px solid {LINE}; border-radius:10px; padding:.75rem 1rem; margin:.2rem 0 1rem; }}
.summary-line b {{ font-weight:600; }}
@media (max-width:640px) {{ .how {{ grid-template-columns:1fr; }} .hero .rec-title {{ font-size:1.15rem; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
</style>
""",
    unsafe_allow_html=True,
)

_DARK_CSS = f"""
<style>
:root {{ --surf:{SURF}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --teal:{TEAL_TXT}; --soft:{SOFT}; --softb:{SOFT_B}; --bg:{BG}; }}
section[data-testid="stSidebar"] {{ background:{SURF}; }}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {{ background:{SOFT}; color:{TEAL_TXT}; }}
.chip.demo {{ background:#2E2412; border-color:#5A4520; color:#E0B86A; }}
.chip.live, .pill {{ background:{SOFT}; border-color:{SOFT_B}; color:{TEAL_TXT}; }}
.banner, .cmp-sum, .kpi, .guide, .notice, .gpill-ps {{ background:{SURF}; }}
.stp i {{ border-color:#3A4452; background:{SURF}; }} .stp-line {{ background:#3A4452; }}
.stp.done i {{ background:{SOFT}; border-color:{SOFT_B}; color:{TEAL_TXT}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{SURF}; }}
[data-testid="stFileUploaderDropzone"] {{ border-color:#35505A; }}
div.stButton > button[kind="primary"]:hover {{ background:#13897F; }}
div.stButton > button[kind="primary"]:disabled {{ background:#24403D; color:#8FA9A5; }}
.sw.on i {{ box-shadow:0 0 0 2px {SURF} inset; }}
.q.ok, .fb-done {{ color:#7FD1C6; background:#10302C; border-color:#1E4A44; }}
.q.bad {{ color:#E0B86A; background:#2E2412; border-color:#5A4520; }}
.rec.urgent {{ background:#2C1820; border-color:#5A2A38; }} .rec.urgent .rec-title {{ color:#E58AA0; }}
.rec.refer {{ background:#2B2216; border-color:#5A4520; }} .rec.refer .rec-title {{ color:#E0B86A; }}
.rec.benign {{ background:#10302C; border-color:#1E4A44; }} .rec.benign .rec-title {{ color:#6FCABD; }}
.unsure {{ background:#2B2516; border-color:#5A4C24; }}
.sym-warn {{ background:#2E1A20; border-color:#5C2F3B; }}
.cam-empty {{ border-color:#3A4452; }} .cand-meta {{ color:#7F8C93; }}
.fu .soon {{ color:#E0B86A; }} .fu .late, .arrow-up {{ color:#E58AA0; }} .arrow-down {{ color:#6FCABD; }}
.ev-best {{ background:{SOFT} !important; border-color:{SOFT_B} !important; color:{INK} !important; }}
.ev-best-t {{ color:{TEAL_TXT} !important; }} .ev-note {{ color:{MUTED} !important; }}
a {{ color:{TEAL_TXT}; }}
.hero, .summary-line {{ background:{SURF}; border-color:{LINE}; }}
/* hazır koyu temanın kırmızı vurgusu yerine turkuaz */
button[kind^="primary"] {{ background-color:{TEAL} !important; border-color:{TEAL} !important; color:#fff !important; }}
button[kind^="primary"]:hover {{ background-color:#13897F !important; border-color:#13897F !important; }}
button[kind^="primary"]:disabled {{ background-color:#24403D !important; border-color:#24403D !important; color:#8FA9A5 !important; }}
[data-baseweb="tab-highlight"] {{ background-color:{TEAL_TXT} !important; }}
button[data-baseweb="tab"][aria-selected="true"] p {{ color:{TEAL_TXT} !important; }}
[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {{ color:{TEAL_TXT} !important; border-color:{TEAL_TXT} !important; background:{SOFT} !important; }}
label[data-baseweb="checkbox"]:has(input:checked) > span {{ background-color:{TEAL} !important; border-color:{TEAL} !important; }}
label[data-baseweb="checkbox"]:has(input:checked) > div:first-child {{ background-color:{TEAL} !important; }}
label[data-baseweb="radio"]:has(input:checked) > div:first-child {{ background-color:{TEAL} !important; border-color:{TEAL} !important; }}
[data-baseweb="slider"] [role="slider"] {{ background-color:{TEAL_TXT} !important; }}
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color:{TEAL_TXT} !important; }}
[data-baseweb="tag"] {{ background-color:{SOFT} !important; }} [data-baseweb="tag"] span {{ color:{INK} !important; }}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within, [data-baseweb="textarea"]:focus-within {{ border-color:{TEAL_TXT} !important; }}
</style>
"""
st.markdown(_DARK_CSS if DARK else f"<style>:root {{ --surf:{SURF}; --ink:{INK}; --muted:{MUTED}; --line:{LINE}; --teal:{TEAL_TXT}; "
            f"--soft:{SOFT}; --softb:{SOFT_B}; --bg:{BG}; }}</style>", unsafe_allow_html=True)

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
ss.setdefault("peek", set())  # vaka listesinde ayrıntısı açılan satırlar
ss.setdefault("consent", False)   # kullanım koşulları onaylandı mı
ss.setdefault("doctor", None)     # hekim modu: baş harfler (None = giriş yok)
ss.setdefault("pin_fails", 0)
ss.setdefault("audit", [])        # denetim kaydı: yalnızca eklenir, değiştirilmez
ss.setdefault("dirty", [])        # veritabanına yazılacak vakalar
ss.setdefault("pending_audit", [])  # veritabanına yazılacak denetim satırları
ss.setdefault("db_synced", 0.0)   # vakaların veritabanından en son çekildiği an
INPUTS = ("age", "sex", "skin", "site", "crop_clin", "crop_derm", "patient", "prev", "symp", "kind", "kindok")


def go(page: str):
    ss.page = page
    ss.nav_close = True  # telefonda menü seçimden sonra kapansın
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


def crop_tool(img: Image.Image, which: str, required: bool = False):
    """Kare kırpma (model görüntünün ortasından kare keser, bu yüzden kutu karedir). Kutu ss.boxes içinde saklanır.
    required=True (hasta modu): kırpma her zaman açık ve kullanıcı kutuyu onaylamadan ileri geçilmez.
    Neden: Tuğalp'in PAD testinde lezyon karede küçük ve arkada kumaş varken tahmin vakaların %47–67'sinde değişti."""
    if st_cropper is None:
        return
    if required:
        on = True
    else:
        on = st.toggle(t("crop_toggle"), value=ss.get(f"v_crop_{which}", False), key=wkey(f"crop_on_{which}"),
                       help=t("crop_help"))
    ss[f"v_crop_{which}"] = on
    if not on:
        return
    st.markdown(f'<div class="crop-guide">{t("crop_guide_pm" if required else "crop_hint")}</div>', unsafe_allow_html=True)
    # Kırpma ekranı karta sığsın diye küçük bir kopya gösterilir; seçilen kutu orijinal boyuta geri çevrilir.
    disp = img.copy()
    disp.thumbnail((360, 360))
    scale = img.width / disp.width
    side = int(min(disp.width, disp.height) * 0.6)  # başlangıç: ortada, kısa kenarın %60'ı kadar kare
    l0, t0 = (disp.width - side) // 2, (disp.height - side) // 2
    box = st_cropper(disp, realtime_update=True, box_color=TEAL, aspect_ratio=(1, 1), return_type="box",
                     default_coords=(l0, l0 + side, t0, t0 + side),
                     key=f"cropper_{which}_{ss.imgs[which]['id']}", stroke_width=2, should_resize_image=False)
    left, top = max(0, int(box["left"] * scale)), max(0, int(box["top"] * scale))
    right = min(img.width, int((box["left"] + box["width"]) * scale))
    bottom = min(img.height, int((box["top"] + box["height"]) * scale))
    ss.boxes[which] = (left, top, right, bottom) if right - left >= 32 and bottom - top >= 32 else None
    if required and ss.boxes[which] is not None:
        pc, tc = st.columns([1, 2.2], vertical_alignment="center")
        pc.image(img.crop(ss.boxes[which]), width=110)
        key = f"v_cropok_{ss.imgs[which]['id']}"
        ok = tc.checkbox(t("crop_confirm"), value=bool(ss.get(key)), key=f"cropok_{which}_{ss.imgs[which]['id']}_{ss.lang}")
        ss[key] = ok


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


BORDER_MARGIN = 0.05  # Tuğalp'in önerisi: eşiğe bu kadar yakın sonuçta "sınırda" uyarısı (yönlendirmeyi değiştirmez)


def assess(probs: dict, mode: str = "hekim") -> dict:
    """Olasılıklardan genel değerlendirme, yönlendirme metni ve kararsızlık durumunu çıkarır.
    mode: "hekim" ya da "hasta" (melanom güvenlik eşiği moda göre farklıdır)."""
    pct = i18n.pct
    totals = {g: sum(p for c, p in probs.items() if GROUPS[c] == g) for g in GROUP_ORDER}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_p, second_p = ranked[0][1], ranked[1][1]
    uncertain = top_p < UNCERTAIN_TOP or (top_p - second_p) < UNCERTAIN_GAP
    mel_rule = False
    risk = totals[MALIGN] + totals[PREMALIGN]
    mel_hit = probs.get("MEL", 0.0) >= MEL_SAFETY[mode if mode in MEL_SAFETY else "hekim"]
    fast = sum(probs.get(c, 0.0) for c in PRIORITY_TYPES)  # hızlı ilerleyebilen türler (melanom, SCC, diğer kötü huylu)
    if PRIORITY_RULE == "type":
        urgent = fast >= PRIORITY_TYPE_T and (risk >= REFER_MALIGN_OR_PRE or uncertain or mel_hit)
    else:
        urgent = totals[MALIGN] >= REFER_URGENT_MALIGN
    if urgent:
        if PRIORITY_RULE == "type":
            level, title, text = "urgent", t("r_urgent_t"), t("r_urgent_type", p=pct(fast))
        else:
            level, title, text = "urgent", t("r_urgent_t"), t("r_urgent", p=pct(totals[MALIGN]))
    elif risk >= REFER_MALIGN_OR_PRE:
        level, title, text = "refer", t("r_refer_t"), t("r_refer", p=pct(risk))
    elif mel_hit:
        # melanom güvenlik kuralı: kaçırılan melanomların çoğu "ben" sanılıyordu (bkz. predictor.MEL_SAFETY)
        level, title, text, mel_rule = "refer", t("r_mel_t"), t("r_mel", p=pct(probs["MEL"])), True
    elif uncertain:
        level, title, text = "refer", t("r_unsure_t"), t("r_unsure")
    else:
        level, title, text = "benign", t("r_benign_t"), t("r_benign")
    # sınırda sonuç: belirleyici toplam eşiğe BORDER_MARGIN'den yakın (doğrulamada hekim %7,6, hasta %9,9 vaka; sezgisel)
    near_u = abs(fast - PRIORITY_TYPE_T) if PRIORITY_RULE == "type" else abs(totals[MALIGN] - REFER_URGENT_MALIGN)
    borderline = abs(risk - REFER_MALIGN_OR_PRE) < BORDER_MARGIN or near_u < BORDER_MARGIN
    return {"totals": totals, "ranked": ranked, "uncertain": uncertain, "level": level, "title": title, "text": text,
            "mel_rule": mel_rule, "borderline": borderline, "fast": fast}


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
        if fb.get("abcde") is not None:
            rows.append((t("abcde_short"), abcde_text(fb["abcde"])))
        if fb.get("followup"):
            rows.append((t("fu_label"), fmt_date(fb["followup"])))
        if fb["note"]:
            rows.append((t("fb_note"), fb["note"]))
    if fin:
        rows.append((t("final_dx"), f"{dx_label(fin['dx'])} · {t(METHODS[fin['method']])}"))
    return rows


def abcde_text(ks: list) -> str:
    ks = [k for k in ABCDE if k in (ks or [])]
    if not ks:
        return t("abcde_none")
    return f"{len(ks)}/5 · " + ", ".join(t(f"abcde_{k}") for k in ks)


def abcde_conflict(rec: dict) -> bool:
    """Hekim ABCDE'de 3+ madde işaretledi ama model iyi huylu görünüm dedi (ya da tersine hiç işaret yok ama öncelikli)."""
    fb = rec.get("feedback") or {}
    if fb.get("abcde") is None:
        return False
    lvl = assess(rec["probs"], rec_mode(rec))["level"]
    return (len(fb["abcde"]) >= 3 and lvl == "benign") or (len(fb["abcde"]) == 0 and lvl == "urgent" and
                                                             assess(rec["probs"], rec_mode(rec))["ranked"][0][0] == "MEL")


def qr_png(cid: str):
    if segno is None:
        return None
    buf = io.BytesIO()
    segno.make(f"{APP_URL}/?case={cid}", error="m").save(buf, kind="png", scale=8, border=1)
    return buf.getvalue()


def priority_key(rec: dict):
    """İş listesi sırası: açık vakalar önce; içinde gecikmiş kontrol, sonra öncelik seviyesi, sonra en yeni."""
    a = assess(rec["probs"], rec_mode(rec))
    fu = (rec.get("feedback") or {}).get("followup")
    overdue = bool(fu) and fu < now_tr().date().isoformat() and rec["status"] != "closed"
    return (rec["status"] == "closed", not overdue, LEVEL_RANK[a["level"]], -rec["ts"])


def export_csv(cases: dict) -> bytes:
    """Bütün vakalar tek tablo. Türkçede Excel'in beklediği noktalı virgül ve ondalık virgül kullanılır."""
    rows = []
    for r in cases.values():
        a = assess(r["probs"], rec_mode(r))
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
            "abcde": "".join(fb.get("abcde") or []) if fb.get("abcde") is not None else "",
            "final_dx": fin.get("dx", ""), "final_method": fin.get("method", ""), "model": r["model_name"],
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    if ss.lang == "tr":
        return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    return df.to_csv(index=False).encode("utf-8-sig")


def model_label(name: str) -> str:
    """Kayıttaki model adı Türkçe saklanır (DB ile uyum için değiştirilmez); İngilizce arayüzde çevrilerek gösterilir."""
    if ss.get("lang") != "en" or not name:
        return name
    for a, b in (("Gerçek model", "Real model"), ("Sahte model (demo)", "Mock model (demo)"), ("– hasta modu", "– patient mode"),
                 ("klinik:", "clinical:"), ("dermoskopik:", "dermoscopic:"), ("4 model,", "4 models,")):
        name = name.replace(a, b)
    return name


def make_pdf(rec: dict) -> bytes:
    a = assess(rec["probs"], rec_mode(rec))
    a["pos"] = ruler_pos(a)  # PDF'te de aynı yönlendirme cetveli
    return build_pdf({**rec, "model_name": model_label(rec["model_name"])}, a, predictor.is_mock, version=APP_VERSION, group_colors=GROUP_COLORS,
                     groups_of=GROUPS, group_order=GROUP_ORDER, meta_lines=meta_lines(rec["meta"]) + pdf_extra_lines(rec),
                     physician=physician_lines(rec), qr=qr_png(rec["case"]))


def make_zip(case_ids: list) -> bytes:
    """Seçilen vakaların PDF raporları + özet CSV tek ZIP dosyasında."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for cid in case_ids:
            try:
                z.writestr(f"{cid}.pdf", make_pdf(ss.cases[cid]))
            except Exception as exc:  # tek rapor bozulursa diğerleri yine insin
                z.writestr(f"{cid}_HATA.txt", str(exc))
        z.writestr("vakalar.csv", export_csv({c: ss.cases[c] for c in case_ids}))
    return buf.getvalue()


def pdf_extra_lines(rec: dict):
    rows = [(t("mode_label"), t("mode_hasta") if rec_mode(rec) == "hasta" else t("mode_hekim"))]
    ps = pred_set(rec)
    if ps:
        rows.append((t("ps_title"), ", ".join(i18n.plain(c) for c in ps)))
    if modality_disagree(rec.get("parts")):
        rows.append((t("mod_pdf"), t("mod_pdf_v")))
    if rec.get("kind"):
        rows.append((t("kind_q"), t("kind_" + rec["kind"])))
    _a = assess(rec["probs"], rec_mode(rec))
    if _a.get("mel_rule"):
        rows.append((t("mel_pdf"), t("mel_pdf_v", p=i18n.pct(rec["probs"]["MEL"]))))
    if _a.get("borderline"):
        rows.append((t("border_pdf"), t("border_pdf_v")))
    if _a["level"] in ("urgent", "refer"):
        rows.append((t("appt_pdf"), t("appt_pdf_v", w=t("appt_w_" + _a["level"]))))
    rows += symptom_lines(rec.get("symptoms"))
    return rows


def patient_mode() -> bool:
    v = ss.get("v_patient")
    return bool(v) if v is not None else ss.get("role") == "patient"


def choose_role(role: str):
    ss.role = role
    ss.v_patient = role == "patient"


ICON_PHONE = ('<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
              'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6.5" y="2.5" width="11" height="19" rx="2.2"/>'
              '<circle cx="12" cy="11" r="3"/><path d="M10.5 18.5h3"/></svg>')
ICON_LENS = ('<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
             'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="10" cy="10" r="6.5"/><circle cx="10" cy="10" r="3.2"/>'
             '<path d="M14.8 14.8l5.7 5.7"/></svg>')


def role_screen():
    """Yeni analizin ilk ekranı (oturumda bir kez): kim kullanıyor? Seçime göre doğru akış açılır."""
    st.markdown(f'<p class="lead">{t("role_q")}</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    for col, role, ic in ((c1, "patient", ICON_PHONE), (c2, "doctor", ICON_LENS)):
        with col:
            st.markdown(f'<div class="role-card"><div class="ic">{ic}</div><div class="rt">{t("role_" + role)}</div>'
                        f'<div class="rd">{t("role_" + role + "_d")}</div></div>', unsafe_allow_html=True)
            st.button(t("role_" + role + "_btn"), key=f"role_{role}", type="primary" if role == "patient" else "secondary",
                      width="stretch", on_click=choose_role, args=(role,))
    steps = "".join(f'<div><b>{i}. {t("how_" + str(i) + "_t")}</b>{t("how_" + str(i))}</div>' for i in (1, 2, 3))
    st.markdown(f'<div class="how">{steps}</div><div class="privacy">{t("role_privacy")}</div>', unsafe_allow_html=True)
    st.caption(t("role_note"))


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
    return f'<span class="spill" style="background:{color}1A;color:{color};border-color:{color}55">{t(key)}</span>'


ZONE = {"benign": (C_BENIGN, "#CFE3DF"), "refer": (C_PREMALIGN, "#EEDDBC"), "urgent": (C_MALIGN, "#EDD0D8")}
if DARK:
    ZONE = {"benign": (C_BENIGN, "#1E3A37"), "refer": (C_PREMALIGN, "#3A2E1A"), "urgent": (C_MALIGN, "#3A2029")}
LEVELS = ("benign", "refer", "urgent")


def ruler_pos(a: dict) -> float:
    """Cetvel üzerindeki işaretin yeri (%). Bölge = yönlendirme seviyesi; bölge içindeki yer olasılıklardan yaklaşık hesaplanır."""
    mal, risk = a["totals"][MALIGN], a["totals"][MALIGN] + a["totals"][PREMALIGN]
    if PRIORITY_RULE == "type":
        val, cut = a.get("fast", 0.0), PRIORITY_TYPE_T
    else:
        val, cut = mal, REFER_URGENT_MALIGN
    if a["level"] == "benign":
        x = min(risk / REFER_MALIGN_OR_PRE, 1) * 33.3
    elif a["level"] == "refer":
        x = 33.3 + min(val / cut, 1) * 33.3
    else:
        x = 66.7 + min((val - cut) / (1 - cut), 1) * 33.3
    return max(3.0, min(97.0, x))


def ruler_html(a: dict, simple: bool) -> str:
    """Yönlendirme cetveli: üç bölge (rutin / değerlendirme / öncelikli) ve bu vakanın yeri."""
    lab = [t(("ru_pm_" if simple else "ru_") + k) for k in LEVELS]
    bar = "".join(f'<i style="background:{ZONE[k][0] if k == a["level"] else ZONE[k][1]}"></i>' for k in LEVELS)
    lbs = "".join(f'<span class="{"on" if k == a["level"] else ""}">{lab[i]}</span>' for i, k in enumerate(LEVELS))
    aria = t("ru_aria", z=lab[LEVELS.index(a["level"])])
    return (f'<div class="ruler" role="img" aria-label="{html.escape(aria)}"><div class="ruler-bar">{bar}'
            f'<span class="ruler-mk" style="left:{ruler_pos(a):.1f}%"></span></div><div class="ruler-lb">{lbs}</div></div>')


def mini_ruler(level: str) -> str:
    return '<span class="mini-ruler">' + "".join(
        f'<i style="background:{ZONE[k][0] if k == level else ZONE[k][1]}"></i>' for k in LEVELS) + "</span>"


def log(kind: str, case=None, **extra) -> dict:
    """Denetim kaydına bir satır ekler (kim, ne zaman, ne yaptı). Kayıtlar yalnızca eklenir."""
    now = now_tr()
    ev = {"ts": now.strftime("%d.%m.%Y %H:%M:%S"), "time": now.strftime("%d.%m %H:%M"), "kind": kind, "case": case,
          "actor": ss.get("doctor"), **extra}
    ss.audit.append(ev)
    if DB is not None:  # vaka kaydına izin verilmediyse o vakanın satırları gönderilmez (flush_db'de elenir)
        ss.pending_audit.append({"iso": now.isoformat(), "case_no": case, "actor": ev["actor"], "kind": kind, "extra": extra})
    return ev


TOASTS = {"pdf": ("toast_pdf", ":material/picture_as_pdf:"), "reminder": ("toast_reminder", ":material/event:"),
          "feedback": ("toast_fb", ":material/check_circle:"), "feedback_upd": ("toast_fb", ":material/check_circle:"),
          "status": ("toast_status", ":material/sync:"), "closed": ("toast_closed", ":material/lock:"),
          "reopened": ("toast_reopened", ":material/lock_open:")}


def add_event(rec: dict, kind: str, **extra):
    rec["events"].append(log(kind, rec["case"], **extra))
    if rec.get("saved") and rec["case"] not in ss.dirty:
        ss.dirty.append(rec["case"])
    if kind in TOASTS:  # kısa onay mesajı (işlemin gerçekleştiğini gösterir)
        key, icon = TOASTS[kind]  # sonraki çalıştırmanın başında gösterilir (st.rerun öncesi toast kaybolabiliyor)
        ss.setdefault("toast_q", []).append((t(key) + (" " + t("toast_saved_db") if rec.get("saved") and DB is not None else ""), icon))


# ---------------------------------------------------------------- veritabanı (Supabase) — anonim kayıt
def case_row(rec: dict) -> dict:
    """Veritabanına giden satır: görüntü ve kişisel bilgi YOK."""
    ranked = sorted(rec["probs"].items(), key=lambda kv: kv[1], reverse=True)
    level = assess(rec["probs"], rec_mode(rec))["level"]
    return {
        "case_no": rec["case"], "created_at": rec["created_iso"], "updated_at": now_tr().isoformat(),
        "app_version": APP_VERSION, "model_name": rec["model_name"],
        "age_band": rec["meta"]["age"], "sex": rec["meta"]["sex"], "skin_tone": rec["meta"]["skin_tone"],
        "site": rec["meta"]["site"], "probs": rec["probs"], "level": level, "top1": ranked[0][0],
        "top1_prob": round(ranked[0][1], 6), "malignant_total": round(sum(p for c, p in rec["probs"].items() if GROUPS[c] == MALIGN), 6),
        "elapsed_s": round(rec["elapsed"], 3), "skin_override": any(ev["kind"] == "skin_override" for ev in rec["events"]),
        "status": rec["status"], "feedback": rec["feedback"], "final": rec.get("final"),
        **({"prev_case": rec["prev"]} if rec.get("prev") else {}),  # sütun yoksa (supabase_ek.sql) yalnızca bağlı vakalar etkilenir
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
        "events": [], "saved": True, "from_db": True, "events_loaded": False, "prev": row.get("prev_case"),
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
    if ev["kind"] == "zip":
        return t("ev_zip", n=ev.get("n", 0))
    if ev["kind"] == "sus":
        return t("ev_sus", s=ev.get("score", ""))
    simple = {"pdf": "ev_pdf", "login": "ev_login", "logout": "ev_logout", "login_fail": "ev_login_fail",
              "consent": "ev_consent", "skin_override": "ev_skin_override", "reminder": "ev_reminder",
              "scope_override": "ev_scope_override"}
    if ev["kind"] in simple:
        return t(simple[ev["kind"]])
    if ev["kind"] == "closed":
        return t("ev_closed", dx=dx_label(ev["dx"]))
    if ev["kind"] == "reopened":
        return t("ev_reopened")
    if ev["kind"] == "linked":
        return t("ev_linked", c=ev.get("prev", ""))
    return t({"created": "ev_created", "feedback": "ev_feedback", "feedback_upd": "ev_feedback_upd"}[ev["kind"]])


def modality_disagree(parts: dict):
    """İki görüntü belirgin biçimde farklı sonuç veriyor mu? En olası tür farklı VE kötü huylu toplam farkı ≥ 25 puan
    ya da kötü huylu toplam farkı ≥ 40 puan."""
    if not parts or "derm" not in parts or "clin" not in parts:
        return False
    mal = {k: sum(p for c, p in v.items() if GROUPS[c] == MALIGN) for k, v in parts.items()}
    top = {k: max(v, key=v.get) for k, v in parts.items()}
    d = abs(mal["derm"] - mal["clin"])
    return (top["derm"] != top["clin"] and d >= 0.25) or d >= 0.40


def modality_panel(rec: dict):
    parts = rec.get("parts")
    if not parts or len(parts) < 2:
        return
    st.markdown(f'<div class="sec-label">{t("mod_title")}</div>', unsafe_allow_html=True)
    rows = ""
    for kind, label in (("derm", t("derm_title")), ("clin", t("clin_title"))):
        p = parts[kind]
        tot = {g: sum(v for c, v in p.items() if GROUPS[c] == g) for g in GROUP_ORDER}
        top = max(p, key=p.get)
        bar = "".join(f'<div style="width:{tot[g] * 100:.2f}%;background:{GROUP_COLORS[g]}"></div>' for g in GROUP_ORDER)
        rows += (f'<div class="mod-row"><div class="mod-name">{label}</div><div class="stack" style="margin:.15rem 0">{bar}</div>'
                 f'<div class="mod-sub">{t("mod_top", c=i18n.plain(top), p=i18n.pct(p[top]))} · '
                 f'{i18n.group(MALIGN)} {i18n.pct(tot[MALIGN])}</div></div>')
    st.markdown(f'<div class="mod">{rows}</div>', unsafe_allow_html=True, help=t("mod_help"))
    if modality_disagree(parts):
        st.markdown(f'<div class="unsure">{t("mod_disagree")}</div>', unsafe_allow_html=True)


def find_case(cid: str):
    """Vaka numarası bu oturumda ya da veritabanında varsa numarayı döndürür (veritabanındakini belleğe alır)."""
    if cid in ss.cases:
        return cid
    if DB is not None and cid.startswith("LZ-"):
        row = DB.get_case(cid)
        if row:
            ss.cases[cid] = rec_from_row(row)
            return cid
    return None


def lesion_chain(cid: str) -> list:
    """Aynı lezyonun birbirine "önceki vaka" ile bağlı tüm vakaları, eskiden yeniye."""
    links = {}
    for c, r in ss.cases.items():
        p = r.get("prev")
        if p:
            links.setdefault(c, set()).add(p)
            links.setdefault(p, set()).add(c)
    seen, todo = {cid}, [cid]
    while todo:
        for n in links.get(todo.pop(), ()):
            if n not in seen and n in ss.cases:
                seen.add(n)
                todo.append(n)
    return sorted(seen, key=lambda c: ss.cases[c]["ts"])


def lesion_timeline(cid: str):
    chain = lesion_chain(cid)
    if len(chain) < 2:
        return
    st.markdown(f'<div class="sec-label">{t("lt_title")}</div>', unsafe_allow_html=True)
    recs = []
    for c in chain:
        r = ss.cases[c]
        a = assess(r["probs"], rec_mode(r))
        when = datetime.fromtimestamp(r["ts"], TZ).replace(tzinfo=None)
        for g, key in ((MALIGN, "lt_mal"), ("risk", "lt_risk")):
            v = a["totals"][MALIGN] + (a["totals"][PREMALIGN] if g == "risk" else 0)
            recs.append({"case": c, "when": when, "date": r["created"], "v": v, "s": t(key),
                         "top": i18n.plain(a["ranked"][0][0]), "level": t(LEVEL_KEYS[a["level"]]), "me": c == cid})
    df = pd.DataFrame(recs)
    dom = [t("lt_mal"), t("lt_risk")]
    base = alt.Chart(df).encode(
        x=alt.X("when:T", title=None, axis=alt.Axis(format="%d.%m.%y", labelAngle=0, tickCount=min(6, len(chain)))),
        y=alt.Y("v:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("s:N", title=None, scale=alt.Scale(domain=dom, range=[C_MALIGN, C_PREMALIGN]),
                        legend=alt.Legend(orient="bottom", labelLimit=320)),
        tooltip=[alt.Tooltip("case:N", title=t("case_no")), alt.Tooltip("date:N", title=t("lt_date")),
                 alt.Tooltip("v:Q", title=t("lt_value"), format=".1%"), alt.Tooltip("top:N", title=t("h_top")),
                 alt.Tooltip("level:N", title=t("h_level"))])
    line = base.mark_line(strokeWidth=2.2) + base.mark_point(filled=True, size=70)
    me = base.transform_filter(alt.datum.me).mark_point(size=230, filled=False, strokeWidth=2)
    st.altair_chart((line + me).properties(height=280).configure_view(strokeWidth=0), use_container_width=True)
    first, last = ss.cases[chain[0]], ss.cases[chain[-1]]
    d = assess(last["probs"], rec_mode(last))["totals"][MALIGN] - assess(first["probs"], rec_mode(first))["totals"][MALIGN]
    trend = ("lt_up" if d > 0.05 else "lt_down" if d < -0.05 else "lt_flat")
    st.caption(t("lt_note", n=len(chain), a=first["created"][:10], b=last["created"][:10], trend=t(trend),
                 d=f"{d * 100:+.1f}".replace(".", "," if ss.lang == "tr" else ".")))
    others = [c for c in chain if c != cid]
    cols = st.columns(min(4, len(others)))
    for i, c in enumerate(others[:4]):
        cols[i].button(c, key=f"lt_open_{cid}_{c}", icon=":material/open_in_new:", width="stretch",
                       on_click=lambda c=c: ss.update(page="track", open_case=c))


def start_followup(cid: str):
    """Aynı lezyonun yeni kontrolü: yeni analiz açılır, "önceki vaka" alanı doldurulur."""
    new_analysis()
    ss.v_prev = cid


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
for _msg, _icon in ss.pop("toast_q", []):
    st.toast(_msg, icon=_icon)
i18n.set_pm(False)  # hasta modunda sade dil; yeni analiz ve hasta vakası ekranında açılır
st.markdown(
    "<style>"
    '[data-testid="stFileUploaderDropzoneInstructions"] > div { display:none; }'
    f'[data-testid="stFileUploaderDropzoneInstructions"]::after {{ content:"{t("up_hint")}"; font-size:.85rem; color:{MUTED}; }}'
    '[data-testid="stFileUploaderDropzone"] button { font-size:0 !important; }'
    f'[data-testid="stFileUploaderDropzone"] button::after {{ content:"{t("up_btn")}"; font-size:.95rem; }}'
    "</style>", unsafe_allow_html=True)
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
        st.markdown(f'<div class="banner" style="border-left-color:{TEAL}">{t("consent_how")}</div>', unsafe_allow_html=True)
        ok = st.checkbox(t("consent_check"), key=f"consent_chk_{ss.lang}")
        if st.button(t("consent_btn"), type="primary", disabled=not ok, width="stretch"):
            ss.consent = True
            log("consent")
            st.rerun()

    consent_dialog()
elif not ss.get("tour_done") and ss.get("tour_force"):
    # Hızlı tur yalnızca istenince açılır (Yardım → Kılavuz). Girişte tek pencere: onay + kısa kullanım bilgisi.
    @st.dialog(t("tour_title"), width="medium", on_dismiss=lambda: ss.update(tour_done=True, tour_force=False))
    def tour_dialog():
        steps = [("tour1_t", "tour1"), ("tour2_t", "tour2"), ("tour3_t", "tour3"), ("tour4_t", "tour4")]
        i = min(ss.get("tour_step", 0), len(steps) - 1)
        dots = "".join(f'<i class="{"on" if k == i else ""}"></i>' for k in range(len(steps)))
        st.markdown(f'<div class="tour-dots">{dots}<span>{i + 1}/{len(steps)}</span></div>'
                    f'<div class="card-title" style="margin-top:.4rem">{t(steps[i][0])}</div>', unsafe_allow_html=True)
        st.markdown(t(steps[i][1]))
        st.checkbox(t("tour_off"), key="tour_off_chk")
        b1, b2, b3 = st.columns([1, 1, 1.2])
        last = i == len(steps) - 1

        def _close():
            ss.update(tour_done=True, tour_force=False, tour_step=0, tour_cookie=bool(ss.get("tour_off_chk")))

        if b1.button(t("tour_skip"), width="stretch", key="tour_skip"):
            _close()
            st.rerun()
        if i > 0:
            b2.button(t("btn_back"), width="stretch", key="tour_back", on_click=lambda: ss.update(tour_step=i - 1))
        if last:
            if b3.button(t("tour_done"), type="primary", width="stretch", key="tour_next"):
                _close()
                st.rerun()
        else:
            b3.button(t("tour_next"), type="primary", width="stretch", key="tour_next",
                      on_click=lambda: ss.update(tour_step=i + 1))

    tour_dialog()
if ss.pop("tour_cookie", False):
    components.html("<script>window.parent.document.cookie = 'cds_tour_off=1; max-age=31536000; path=/; SameSite=Lax';</script>",
                    height=0)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">'
        # dermatoskop merceği + içinde üç bölgeli yönlendirme cetveli (sitenin imza öğesi)
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" aria-hidden="true">'
        '<circle cx="12" cy="12" r="9"/><path d="M6.5 13.5h3M10.5 13.5h3M14.5 13.5h3" stroke-width="2.4"/>'
        '<path d="M12 9.2v5.8" stroke-width="1.6"/></svg></div>'
        f'<div><div class="brand-name">{t("app_name")}</div><div class="brand-sub">{t("app_sub")}</div></div></div>',
        unsafe_allow_html=True,
    )
    n_cases = len(ss.cases)
    nav = [
        ("new", t("nav_new"), ":material/add_circle:"),
        ("track", t("nav_track") + (f"  ·  {n_cases}" if n_cases else ""), ":material/folder_open:"),
        ("health", t("nav_health"), ":material/health_and_safety:"),
        ("model", t("nav_model"), ":material/insights:"),
        ("help", t("nav_help"), ":material/help:"),
    ]
    for key, label, icon in nav:
        st.button(label, key=f"nav_{key}", icon=icon, width="stretch",
                  type="primary" if ss.page == key else "secondary", on_click=go, args=(key,))
    if not ss.get("sb_opened"):  # bilgisayarda menü açık başlasın (telefonda kapalı kalır)
        ss.sb_opened = True
        components.html(
            "<script>(function(){const w = window.parent; if (w.innerWidth < 1024) return;"
            "const b = w.document.querySelector('[data-testid=\"stExpandSidebarButton\"]'); if (b) b.click();})();</script>",
            height=0)
    if ss.pop("nav_close", False):
        components.html(
            "<script>(function(){const w = window.parent; if (w.innerWidth >= 768) return;"
            "const b = w.document.querySelector('[data-testid=\"stSidebarCollapseButton\"] button'); if (b) b.click();})();</script>",
            height=0)

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
    st.toggle(t("dark_toggle"), value=DARK, key=f"dark_tg_{DARK}", help=t("dark_help"),
              on_change=lambda: ss.update(theme_req="light" if DARK else "dark"))
    if ss.get("theme_req"):
        # Tercih tarayıcıda (Streamlit'in kendi tema kaydı) saklanır, sayfa yeni temayla yeniden açılır.
        mode = ss.pop("theme_req")
        act = ('w.localStorage.setItem(k, JSON.stringify({name: "Dark"}));' if mode == "dark"
               else "w.localStorage.removeItem(k);")
        components.html(
            "<script>(function(){const w = window.parent; const k = `stActiveTheme-${w.location.pathname}-v1`;"
            "try {" + act + "} catch (e) {}"
            "const u = new URL(w.location.href); u.searchParams.set('theme', '" + mode + "');"
            # yenileme komutu ana sayfanın kendi zamanlayıcısında çalışır (gömülü çerçevenin gezinme kısıtına takılmaz)
            "w.setTimeout('window.location.replace(' + JSON.stringify(u.toString()) + ')', 30);"
            "})();</script>", height=0)
    if getattr(predictor, "load_error", None):
        st.error(t("model_load_error", e=predictor.load_error[:300]))
    if DB is not None and DB.last_error:
        st.warning(t("db_error_detail", e=DB.last_error[:300]))
    status = (f'<span class="chip demo">● {t("chip_demo")}</span>' if predictor.is_mock
              else f'<span class="chip live" title="{html.escape(model_label(predictor.name))}">● {t("chip_real")}</span>')
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
    i18n.set_pm(rec_mode(rec) == "hasta")
    a = assess(rec["probs"], rec_mode(rec))

    simple = rec_mode(rec) == "hasta"
    # 1) kayıt satırı: numara, tarih, durum (model adı aşağıda, vaka bilgilerinde)
    st.markdown(f'<div class="case-top"><div><span class="case-label">{t("case_no")}</span> '
                f'<span class="case-no">{cid}</span></div>'
                f'<div class="case-meta">{rec["created"]} &nbsp;{status_pill(rec["status"])}</div></div>', unsafe_allow_html=True)
    # 2) sonuç: sayfanın ilk okunan şeyi (telefonda da)
    if rec.get("kind") == "rash":
        st.markdown(f'<div class="sym-warn">{t("kind_rash_result")}</div>', unsafe_allow_html=True)
    card = (calm_card_html(a["level"], a.get("mel_rule", False)) if simple else
            f'<div class="rec {a["level"]}"><div class="rec-title">{a["title"]}</div><div class="rec-text">{a["text"]}</div></div>')
    st.markdown(f'<div class="hero">{card}{ruler_html(a, simple)}<div class="ruler-note">{t("ru_note")}</div></div>',
                unsafe_allow_html=True)
    if a.get("borderline"):
        st.markdown(f'<div class="border-note">{t("border_pm" if simple else "border_dr")}</div>', unsafe_allow_html=True)
    if simple:
        st.markdown(f'<div class="banner" style="margin:0 0 .8rem">{t("pm_note")}</div>', unsafe_allow_html=True)
    if predictor.is_mock:
        st.caption(t("demo_caption"))
    # 3) işlemler: tek satır, ikincil olanlar "Diğer işlemler" menüsünde
    _case_actions(rec, a, cid, where, simple)
    if where == "new":
        st.markdown(f'<div class="case-hint">{t("case_hint")}</div>', unsafe_allow_html=True)
    if rec.get("prev"):
        st.markdown(f'<div class="case-hint">{t("prev_info", c=rec["prev"])}</div>', unsafe_allow_html=True)

    # 4) ayrıntı (geniş sütun, telefonda önce gelir) + vaka bilgileri
    main, side = st.columns([1.45, 1], gap="large")
    with side:
        st.markdown(f'<div class="sec-label">{t("case_info")}</div>', unsafe_allow_html=True)
        i1, i2 = st.columns(2)
        if rec.get("clin_jpg"):
            i1.image(rec["clin_jpg"], caption=t("clin_title"), width="stretch")
            if rec.get("derm_jpg"):
                i2.image(lens_preview(Image.open(io.BytesIO(rec["derm_jpg"]))), caption=t("derm_title"), width="stretch")
        else:
            st.markdown(f'<div class="cam-empty">{t("no_images")}</div>', unsafe_allow_html=True)
        kv = "".join(f"<span>{k}</span><div>{v}</div>" for k, v in meta_lines(rec["meta"]))
        kv += f'<span>{t("mode_label")}</span><div>{t("mode_hasta") if simple else t("mode_hekim")}</div>'
        st.markdown(f'<div class="kv" style="margin-top:.4rem">{kv}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sec-label">{t("timeline")}</div>', unsafe_allow_html=True)
        tl = "".join(f'<div><b>{ev["time"]}</b><span>{event_text(ev)} <span style="color:{MUTED}">· {actor_label(ev)}</span></span></div>'
                     for ev in rec["events"])
        st.markdown(f'<div class="tl">{tl}</div>', unsafe_allow_html=True)
        st.caption(t("model_used", m=html.escape(model_label(rec["model_name"]))))

    with main:
        if simple:  # hasta modu: belirti notları ve sonucun anlamı; yüzdeler ayrıntılarda (sağlık kaygısı)
            for kind, txt in symptom_advice(rec.get("symptoms"), a["level"], a["ranked"][0][0]):
                st.markdown(f'<div class="sym-{kind}">{txt}</div>', unsafe_allow_html=True)
            st.markdown(appointment_html(a["level"]), unsafe_allow_html=True)
            st.markdown(calm_meaning_html(a["level"]), unsafe_allow_html=True)
        detail = st.expander(t("details_pm"), icon=":material/tune:") if simple else st.container()
        with detail:
            if simple:
                st.caption(details_warning())
            _result_body(rec, a, simple, cid, where)

    _case_footer(rec, cid, where)
    if ss.get("show_doc") == cid:
        doctor_card(cid)


def _case_actions(rec, a, cid, where, simple):
    """Sonucun altındaki işlem satırı: PDF (ana), doktora göster, gezinme ve 'Diğer işlemler'."""
    cols = st.columns([1.25, 1.1, 1, 1] if simple else [1.25, 1, 1], gap="small")
    with cols[0]:
        if build_pdf is not None:
            try:
                st.download_button(t("dl_pdf"), data=make_pdf(rec), file_name=f"{cid}.pdf", mime="application/pdf",
                                   icon=":material/download:", width="stretch", type="primary", help=t("dl_pdf_help"),
                                   key=f"pdf_{cid}_{where}", on_click=lambda c=cid: add_event(ss.cases[c], "pdf"))
            except Exception as exc:  # rapor üretilemezse sonuç ekranı yine çalışsın
                st.caption(f"PDF: {exc}")
    k = 1
    if simple:
        cols[1].button(t("show_doc"), icon=":material/smartphone:", width="stretch", key=f"showdoc_{cid}_{where}",
                       on_click=lambda c=cid: ss.update(show_doc=c))
        k = 2
    if where == "new":
        cols[k].button(t("btn_new"), icon=":material/add:", width="stretch", on_click=new_analysis, key="new_from_result")
    else:
        cols[k].button(t("btn_back_short"), icon=":material/arrow_back:", width="stretch",
                       on_click=lambda: ss.update(open_case=None, track_q=""), key="back_list")
    with cols[k + 1].popover(t("more_actions"), icon=":material/more_horiz:", width="stretch"):
        if where == "new":
            st.button(t("btn_edit"), icon=":material/edit:", width="stretch", on_click=lambda: ss.update(step=2),
                      key="edit_inputs")
        else:
            st.button(t("cmp_from_case"), icon=":material/compare:", width="stretch", key="cmp_from_case",
                      on_click=lambda c=cid: ss.update(open_case=None, track_q="", cmp_a=c, cmp_open=True),
                      disabled=len(ss.cases) < 2)
        st.button(t("btn_followup"), icon=":material/history:", width="stretch", key=f"fu_start_{cid}_{where}",
                  on_click=start_followup, args=(cid,), help=t("btn_followup_help"))
        st.markdown(f'<div class="sec-label" style="margin-top:.6rem">{t("remind_btn")}</div>', unsafe_allow_html=True)
        st.caption(t("remind_help"))
        months = st.segmented_control(t("remind_when"), [1, 3, 6], default=1 if a["level"] != "benign" else 3,
                                      format_func=lambda m: t("remind_m", m=m), key=f"rm_{cid}_{where}")
        months = months or 3
        st.download_button(t("remind_dl"), ics_bytes(cid, months, APP_URL), file_name=f"{cid}-kontrol.ics",
                           mime="text/calendar", icon=":material/event:", width="stretch", key=f"ics_{cid}_{where}",
                           on_click=lambda c=cid: add_event(ss.cases[c], "reminder"))


@st.dialog(" ", width="large", on_dismiss=lambda: ss.update(show_doc=None))
def doctor_card(cid: str):
    """Hastanın telefonu doktora uzatabileceği sade özet (tanı değildir). Doktor okuyacağı için tıbbi adlar da gösterilir."""
    rec = ss.cases[cid]
    i18n.set_pm(False)
    a = assess(rec["probs"], rec_mode(rec))
    st.markdown(f'<div class="case-label">{t("dc_title")}</div><div class="case-no">{cid}</div>'
                f'<div class="case-meta">{rec["created"]} · {t("mode_hasta") if rec_mode(rec) == "hasta" else t("mode_hekim")}</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.4], gap="medium")
    with c1:
        if rec.get("clin_jpg"):
            st.image(rec["clin_jpg"], width="stretch")
        kv = "".join(f"<span>{k}</span><div>{v}</div>" for k, v in meta_lines(rec["meta"]))
        st.markdown(f'<div class="kv">{kv}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="rec {a["level"]}"><div class="rec-title">{a["title"]}</div></div>{ruler_html(a, False)}',
                    unsafe_allow_html=True)
        rows = "".join(f'<div><b>{i18n.pct(p)}</b> · {i18n.cname(c)} ({c})</div>' for c, p in a["ranked"][:3])
        tot = " · ".join(f"{i18n.group(g)} {i18n.pct(a['totals'][g])}" for g in GROUP_ORDER)
        st.markdown(f'<div class="sec-label">{t("dc_model")}</div><div class="dc-rows">{rows}</div>'
                    f'<div class="case-meta" style="margin-top:.3rem">{tot}</div>', unsafe_allow_html=True)
        sym = symptom_lines(rec.get("symptoms"))
        if sym:
            items = "".join(f"<span>{html.escape(q)}</span><div>{html.escape(v)}</div>" for q, v in sym)
            st.markdown(f'<div class="sec-label">{t("dc_symptoms")}</div><div class="kv">{items}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="banner" style="margin-top:.6rem">{t("dc_note")}</div>', unsafe_allow_html=True)
    st.caption(t("dc_shot"))


def _result_body(rec, a, simple, cid, where):
    if True:
        if a["uncertain"]:
            st.markdown(f'<div class="unsure">{t("unsure_box")}</div>', unsafe_allow_html=True)
        if not predictor.is_mock:
            rel = reliability_html(a["ranked"][0][0], rec_mode(rec), a["ranked"][0][1])
            if rel:
                st.markdown(rel, unsafe_allow_html=True)
        bar = "".join(f'<div style="width:{a["totals"][g] * 100:.2f}%;background:{GROUP_COLORS[g]}"></div>' for g in GROUP_ORDER)
        legend = "".join(f'<span><span class="dot" style="background:{GROUP_COLORS[g]}"></span>{i18n.group(g)} '
                         f'<b>{i18n.pct(a["totals"][g])}</b></span>' for g in GROUP_ORDER)
        st.markdown(f'<div class="stack">{bar}</div><div class="legend">{legend}</div>', unsafe_allow_html=True)

        if True:
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

            modality_panel(rec)

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
            with (st.container() if simple else st.expander(t("all_results"))):
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
            with (st.container() if simple else st.expander(t("tech"))):
                st.caption(t("tech_line", m=model_label(rec["model_name"]), s=rec["elapsed"], c=rec["created"],
                             ok=t("tech_ok") if rec["elapsed"] < 10 else t("tech_slow")))
                st.caption(t("tech_thr_type" if PRIORITY_RULE == "type" else "tech_thr", ft=f"{PRIORITY_TYPE_T * 100:.0f}", u=f"{REFER_URGENT_MALIGN * 100:.0f}", r=f"{REFER_MALIGN_OR_PRE * 100:.0f}",
                             t=f"{UNCERTAIN_TOP * 100:.0f}", g=f"{UNCERTAIN_GAP * 100:.0f}"))


def _case_footer(rec, cid, where):
    # Aynı lezyonun önceki/sonraki kontrolleri
    if len(lesion_chain(cid)) > 1:
        with st.container(border=True):
            lesion_timeline(cid)

    # Hekim değerlendirmesi ve vaka durumu
    st.markdown(f'<div class="sec-label" style="margin-top:1.4rem">{t("fb_section")}</div>', unsafe_allow_html=True)
    if not ss.doctor:
        with st.container(border=True):
            fb, fin = rec["feedback"], rec.get("final")
            st.markdown(status_pill(rec["status"]), unsafe_allow_html=True)
            if fb:
                extra = f" · {t('fu_label')}: {fmt_date(fb['followup'])}" if fb.get("followup") else ""
                extra += f" · ABCDE {abcde_text(fb['abcde'])}" if fb.get("abcde") is not None else ""
                st.markdown(f'<div class="fb-done" style="margin-top:.6rem">✓ {html.escape(fb_summary(fb))}{extra}</div>',
                            unsafe_allow_html=True)
                if abcde_conflict(rec):
                    st.markdown(f'<div class="unsure">{t("abcde_conflict")}</div>', unsafe_allow_html=True)
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
                top3 = [c for c, _ in assess(rec["probs"], rec_mode(rec))["ranked"][:3]]
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
                extra += f" · ABCDE {abcde_text(fb['abcde'])}" if fb.get("abcde") is not None else ""
                st.markdown(f'<div class="fb-done">✓ {html.escape(fb_summary(fb))}{extra}</div>', unsafe_allow_html=True)
                if abcde_conflict(rec):
                    st.markdown(f'<div class="unsure">{t("abcde_conflict")}</div>', unsafe_allow_html=True)
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
                    abcde = st.pills(t("abcde_label"), ABCDE, selection_mode="multi", default=(fb or {}).get("abcde") or [],
                                     format_func=lambda k: t(f"abcde_{k}"), help=t("abcde_help"))
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
                                             "abcde": [k for k in ABCDE if k in (abcde or [])],
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
    if ss.get("role") is None and ss.get("v_patient") is None and ss.step == 1:
        page_header(t("new_title"))
        role_screen()
        return
    i18n.set_pm(patient_mode())
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
    pm = patient_mode()
    ss.v_patient = pm
    ml, mb = st.columns([3.2, 1], vertical_alignment="center")
    ml.markdown(f'<div class="mode-line">{t("mode_line_pm" if pm else "mode_line_dr")}</div>', unsafe_allow_html=True)
    mb.button(t("mode_switch_dr" if pm else "mode_switch_pm"), key=wkey("pm_switch"), icon=":material/swap_horiz:",
              width="stretch", help=t("pm_help"), on_click=lambda: ss.update(v_patient=not patient_mode()))
    kind_block = False
    if pm:  # kapsam sorusu: araç benler/lekeler için; yaygın döküntü ve egzama için uygun değil
        kinds = ["spot", "rash", "unsure"]
        kind = st.radio(t("kind_q"), kinds, index=kinds.index(ss.v_kind) if ss.get("v_kind") in kinds else None,
                        format_func=lambda k: t("kind_" + k), horizontal=True, key=wkey("kind"))
        ss.v_kind = kind
        if kind == "rash":
            st.markdown(f'<div class="sym-warn">{t("kind_rash_warn")}</div>', unsafe_allow_html=True)
            ok = st.checkbox(t("kind_rash_ok"), value=bool(ss.get("v_kindok")), key=wkey("kindok"))
            ss.v_kindok = ok
            kind_block = not ok
    if not pm:
        with st.expander(t("photo_tips"), icon=":material/photo_camera:"):
            render_photo_tips(compact=True)
    used = ("clin",) if pm else ("clin", "derm")
    cards = {"clin": ("clin_pill", "clin_title", "clin_text"), "derm": ("derm_pill", "derm_title", "derm_text")}
    cols = st.columns([1.15, 1] if pm else [1, 1], gap="medium")
    if pm:  # hasta modunda dermoskopi kartı yok; yerinde fotoğraf ipuçları
        with cols[1]:
            st.markdown(f'<div class="card-title" style="margin-top:.2rem">{t("photo_tips")}</div>', unsafe_allow_html=True)
            render_photo_tips(compact=True)
    for col, which in zip(cols, used):
        pill, title, text = cards[which]
        with col, st.container(border=True):
            st.markdown(f'<div class="card-title">{t(title)}</div>'
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
                crop_tool(img, which, required=pm)
                if not ss.get(f"v_crop_{which}"):
                    st.image(img if which == "clin" else lens_preview(img), width=300,
                             caption=None if which == "clin" else t("lens_caption"))
                show_quality(get_image(which))
                if not skin_like(get_image(which)):
                    st.warning(t("skin_warn"))
                    ck = st.checkbox(t("skin_confirm"), value=ss.get(f"v_skinok_{rec['id']}", False),
                                     key=f"skinok_{which}_{rec['id']}_{ss.lang}")
                    ss[f"v_skinok_{rec['id']}"] = ck
                st.caption(html.escape(rec["name"]))

                def _clear(w=which):
                    ss.imgs[w] = None
                    ss.boxes[w] = None
                    ss.pop(f"v_crop_{w}", None)

                def _rotate(w=which):
                    im = open_bytes(ss.imgs[w]["bytes"])
                    if im is None:
                        return
                    buf = io.BytesIO()
                    im.rotate(-90, expand=True).save(buf, format="PNG")  # saat yönünde 90°, kayıpsız
                    data = buf.getvalue()
                    ss.imgs[w] = {**ss.imgs[w], "bytes": data, "id": hashlib.md5(data).hexdigest()}
                    ss.boxes[w] = None  # kırpma kutusu eski yöne göreydi

                r1, r2 = st.columns(2)
                r1.button(t("btn_rotate"), key=f"rot_{which}", icon=":material/rotate_right:", on_click=_rotate,
                          width="stretch", help=t("rotate_help"))
                r2.button(t("btn_change"), key=f"chg_{which}", icon=":material/refresh:", on_click=_clear, width="stretch")

    ready = all(ss.imgs[w] is not None for w in used)
    blocked = ready and any(not skin_like(get_image(w)) and not ss.get(f"v_skinok_{ss.imgs[w]['id']}") for w in used)
    blocked = blocked or kind_block
    crop_block = pm and ready and st_cropper is not None and not ss.get(f"v_cropok_{ss.imgs['clin']['id']}")
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

    if not pm:  # iki görüntü arasındaki fark yalnızca hekim modunda anlamlı
        with st.expander(t("img_help_title")):
            st.markdown(t("img_help_body"))

    st.write("")
    _, nb = st.columns([2, 1])
    with nb:
        st.button(t("btn_next"), type="primary", width="stretch", disabled=not ready or blocked or crop_block, icon=":material/arrow_forward:",
                  on_click=lambda: ss.update(step=2), key="to_step2")
        if not ready:
            st.caption(t("need_clin") if pm else t("need_both"))
        elif kind_block:
            st.caption(t("kind_block"))
        elif blocked:
            st.caption(t("skin_block"))
        elif crop_block:
            st.caption(t("crop_block"))


def step_patient():
    pm = patient_mode()
    clin_img, derm_img = get_image("clin"), (None if pm else get_image("derm"))
    if clin_img is None or (derm_img is None and not pm):
        ss.step = 1
        st.rerun()

    left, right = st.columns([1.5, 1], gap="large")
    with left, st.container(border=True):
        age_opts, sex_opts, skin_opts, site_opts = [lo for lo, _ in AGE_BANDS], ["female", "male"], list(SKIN_SWATCH), list(SITES)
        age = st.pills(t("age_label"), age_opts, default=ss.get("v_age") if ss.get("v_age") in age_opts else None,
                       format_func=age_band_label, help=t("age_help"), key=wkey("age"))
        sex = st.radio(t("sex_label"), sex_opts, index=remembered_index("sex", sex_opts), format_func=i18n.sex,
                       horizontal=True, key=wkey("sex"))
        skin_tone = st.radio(t("skin_label"), skin_opts, index=remembered_index("skin", skin_opts),
                             format_func=lambda i: f"{i} · {i18n.tone(i)}", horizontal=True, help=t("skin_help"), key=wkey("skin"))
        sw = "".join(f'<div class="sw{" on" if i == skin_tone else ""}"><i style="background:{c}"></i>{i}</div>'
                     for i, c in SKIN_SWATCH.items())
        st.markdown(f'<div class="swatches">{sw}</div>', unsafe_allow_html=True)
        site = st.pills(t("site_label"), site_opts, default=ss.get("v_site") if ss.get("v_site") in site_opts else None,
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

    if pm:
        with st.container(border=True):
            ss.v_symp = render_symptom_form(ss.get("v_symp") or {}, wkey)

    prev_txt = st.text_input(t("prev_label"), value=ss.get("v_prev") or "", key=wkey("prev"), placeholder="LZ-…",
                             help=t("prev_help")).strip().upper()
    ss.v_prev = prev_txt
    prev_ok = None
    if prev_txt:
        prev_ok = find_case(prev_txt)
        if prev_ok:
            pa = assess(ss.cases[prev_ok]["probs"], rec_mode(ss.cases[prev_ok]))
            st.caption(t("prev_found", d=ss.cases[prev_ok]["created"], top=i18n.plain(pa["ranked"][0][0])))
        else:
            st.warning(t("prev_missing", c=html.escape(prev_txt)))

    missing = [t(k) for k, v in [("m_age", age), ("m_sex", sex), ("m_skin", skin_tone), ("m_site", site)] if v is None]
    if prev_txt and not prev_ok:
        missing.append(t("m_prev"))
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
            if hasattr(predictor, "predict_detail"):
                probs, parts = predictor.predict_detail(clin_img, derm_img, meta)
            else:
                probs, parts = predictor.predict(clin_img, derm_img, meta), None
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
            "parts": parts,  # görüntü başına ham olasılıklar (yalnızca bellekte; veritabanına gitmez)
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
            "prev": prev_ok,
            "symptoms": (ss.get("v_symp") or None) if pm else None,  # yalnızca bellekte ve PDF'te
            "kind": ss.get("v_kind") if pm else None,  # kullanıcının beyanı: spot / rash / unsure
        }
        add_event(rec, "created")
        if prev_ok:
            add_event(rec, "linked", prev=prev_ok)
        if any(not skin_like(get_image(w)) for w in (("clin",) if pm else ("clin", "derm"))):
            add_event(rec, "skin_override")
        if pm and ss.get("v_kind") == "rash":
            add_event(rec, "scope_override")
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
    a = assess(rec["probs"], rec_mode(rec))
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
    aa, ab = assess(ra["probs"], rec_mode(ra)), assess(rb["probs"], rec_mode(rb))
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


def case_peek(r: dict, a: dict):
    """Vaka listesinde satırın altında açılan kısa özet (fotoğrafsız); ayrıntı için vakanın kendisi açılır."""
    st.markdown('<div class="peek-line"></div>', unsafe_allow_html=True)
    # gizlilik: listede fotoğraf gösterilmez (ekranda birden çok vaka açık olabilir); fotoğraf yalnız vaka ekranında
    p2, p3 = st.columns(2, gap="large")
    tot = a["totals"]
    top3 = "".join(f'<div class="peek-kv"><span>{i18n.plain(c)}</span><b>{i18n.pct(p)}</b></div>' for c, p in a["ranked"][:3])
    flags = []
    if a.get("mel_rule"):
        flags.append(t("r_mel_t"))
    if a.get("borderline"):
        flags.append(t("border_pdf"))
    p2.markdown(f'<div class="peek-h">{t("top3")}</div>{top3}'
                f'<div class="row-sub" style="margin-top:.35rem">{i18n.group(MALIGN)} {i18n.pct(tot[MALIGN])} · '
                f'{i18n.group(PREMALIGN)} {i18n.pct(tot[PREMALIGN])}</div>'
                + (f'<div class="row-sub" style="color:{C_PREMALIGN}">{" · ".join(flags)}</div>' if flags else ""),
                unsafe_allow_html=True)
    fb = r.get("feedback")
    rows = [(t("mode_label"), t("mode_hasta") if rec_mode(r) == "hasta" else t("mode_hekim")),
            (t("peek_fb"), fb_summary(fb) if fb and fb.get("dx") else t("peek_fb_none"))]
    if fb and fb.get("followup"):
        rows.append((t("fu_label"), fmt_date(fb["followup"])))
    if r.get("prev"):
        rows.append((t("peek_prev"), r["prev"]))
    if r.get("events"):
        ev = r["events"][-1]
        rows.append((t("peek_last"), f'{event_text(ev)} ({ev.get("time", "")})'))
    p3.markdown("".join(f'<div class="peek-kv2"><span>{html.escape(k)}</span><div>{html.escape(str(v))}</div></div>' for k, v in rows),
                unsafe_allow_html=True)
    _, ob = st.columns([3, 1])
    ob.button(t("peek_open"), key=f"open_{r['case']}", type="primary", icon=":material/open_in_new:", width="stretch",
              on_click=lambda c=r["case"]: ss.update(open_case=c))


def page_track():
    sync_from_db()
    if ss.open_case in ss.cases:
        render_case(ss.open_case, "track")
        return
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
    h1, h2, h3 = st.columns([2.2, 1.2, 1.2], vertical_alignment="center")
    h1.caption(t("track_count", n=len(rows)))
    h2.download_button(t("export_btn"), data=export_csv(ss.cases), file_name=f"vakalar_{now_tr():%Y%m%d_%H%M}.csv",
                       mime="text/csv", icon=":material/table_view:", width="stretch", help=t("export_help"))
    with h3.popover(t("zip_btn"), icon=":material/folder_zip:", width="stretch", help=t("zip_help")):
        all_ids = [r["case"] for r in rows]
        pick = st.multiselect(t("zip_pick"), list(ss.cases), default=all_ids[:20], key=f"zip_pick_{ss.lang}",
                              placeholder=t("zip_ph"))
        st.caption(t("zip_note"))
        if st.button(t("zip_make", n=len(pick)), type="primary", disabled=not pick or build_pdf is None,
                     width="stretch", key="zip_make"):
            with st.spinner(t("zip_wait")):
                ss.zip_data = (tuple(pick), make_zip(pick))
            log("zip", None, n=len(pick))
        zd = ss.get("zip_data")
        if zd and list(zd[0]) == pick:
            st.download_button(t("zip_dl", n=len(pick)), data=zd[1], file_name=f"raporlar_{now_tr():%Y%m%d_%H%M}.zip",
                               mime="application/zip", icon=":material/download:", width="stretch", key="zip_dl")
    if not rows:
        st.info(t("track_none"))
    today = now_tr().date().isoformat()
    for r in rows:
        a = assess(r["probs"], rec_mode(r))
        top_c, top_p = a["ranked"][0]
        fu = (r.get("feedback") or {}).get("followup")
        fu_txt = ""
        if fu and r["status"] != "closed":
            late = fu < today
            fu_txt = (f' · <span style="color:{C_MALIGN if late else MUTED};font-weight:{600 if late else 400}">'
                      f'{t("fu_label")}: {fmt_date(fu)}</span>')
        urgent = a["level"] == "urgent" and r["status"] != "closed"
        with st.container(border=True):
            k1, k2, k3, k4 = st.columns([1.45, 2.25, 1.1, .95], vertical_alignment="center")
            k1.markdown(f'<div class="{"row-urgent" if urgent else ""}"><div class="row-no">{r["case"]}</div>'
                        f'<div class="row-sub">{r["created"]}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div>{mini_ruler(a["level"])}<b style="font-weight:'
                        f'{600 if urgent else 400}">{t(LEVEL_KEYS[a["level"]])}</b> · {i18n.plain(top_c)} ({i18n.pct(top_p)})</div>'
                        f'<div class="row-sub">{patient_short(r["meta"])}{fu_txt}</div>', unsafe_allow_html=True)
            k3.markdown(status_pill(r["status"]), unsafe_allow_html=True)
            shown = r["case"] in ss.peek
            k4.button(t("btn_peek_hide" if shown else "btn_peek"), key=f"peek_{r['case']}", width="stretch",
                      icon=":material/expand_less:" if shown else ":material/expand_more:",
                      on_click=lambda c=r["case"]: ss.peek.symmetric_difference_update({c}))
            if shown:
                case_peek(r, a)
    st.caption(t("track_note_db" if DB is not None else "track_note"))
    audit_panel()


# ---------------------------------------------------------------- yönlendirme
def sub_nav(name: str, options: dict) -> str:
    """Sayfa içi sade alt menü (sekme yerine; seçim oturumda kalır)."""
    key = f"sub_{name}"
    if ss.get(key) not in options:
        ss[key] = next(iter(options))
    st.segmented_control(name, list(options), format_func=options.get, key=key, label_visibility="collapsed",
                         on_change=lambda: ss[key] or ss.update({key: next(iter(options))}))
    return ss[key] or next(iter(options))


OLD_PAGES = {"stats": ("track", "stats"), "eval": ("model", "details_eval"), "survey": ("help", "survey"),
             "about": ("model", "overview"), "guide": ("help", "guide")}
if ss.page in OLD_PAGES:  # eski sayfa adları (açık oturumlar)
    _pg, _sub = OLD_PAGES[ss.page]
    ss.page, ss[f"sub_{_pg}"] = _pg, _sub

if ss.page == "new":
    page_new()
elif ss.page == "track":
    if ss.open_case not in ss.cases:
        page_header(t("track_title"), t("track_lead"))
    view = "cases" if ss.open_case in ss.cases else sub_nav("track", {"cases": t("sub_cases"), "stats": t("tab_stats")})
    if view == "cases":
        page_track()
    else:
        sync_from_db()
        history = [{"case": r["case"], "time": r["time"], "meta": r["meta"], "top": assess(r["probs"], rec_mode(r))["ranked"][0],
                    "level": assess(r["probs"], rec_mode(r))["level"], "elapsed": r["elapsed"]} for r in ss.cases.values()]
        feedback = {cid: r["feedback"] for cid, r in ss.cases.items() if r["feedback"]}
        finals = [{"case": cid, "final": r["final"]["dx"], "ranked": [c for c, _ in assess(r["probs"], rec_mode(r))["ranked"]],
                   "level": assess(r["probs"], rec_mode(r))["level"]} for cid, r in ss.cases.items() if r.get("final")]
        render_stats(history, feedback, predictor.is_mock, GROUP_COLORS, finals, db_on=DB is not None)
elif ss.page == "health":
    page_header(t("nav_health"), t("health_sub"))
    render_skin_health()
elif ss.page == "model":
    page_header(t("nav_model"))
    view = sub_nav("model", {"overview": t("sub_overview"), "versions": t("sub_versions"),
                             "details_eval": t("nav_eval"), "details": t("sub_details")})
    if view == "overview":
        render_about()
    elif view == "versions":
        render_versions()
    elif view == "details_eval":
        st.caption(t("eval_sub"))
        render_eval()
    else:
        render_about("details")
else:
    page_header(t("nav_help"))
    view = sub_nav("help", {"guide": t("tab_guide"), "survey": t("nav_survey"), "share": t("sub_share")})
    if view == "guide":
        st.button(t("tour_again"), icon=":material/tour:", key="tour_again",
                  on_click=lambda: ss.update(tour_done=False, tour_force=True, tour_step=0))
        render_guide()
    elif view == "survey":
        render_survey(DB, APP_VERSION, log)
    else:
        render_share(APP_URL)

flush_db()  # bu çalıştırmada biriken kayıtları veritabanına yaz
