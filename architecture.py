"""'Model hakkında' sayfasındaki sistem mimarisi diyagramı (satır içi SVG, açık/koyu temaya uyar, TR/EN)."""
from __future__ import annotations

import html

import streamlit as st

import i18n


def L(tr: str, en: str) -> str:
    return tr if i18n.get_lang() == "tr" else en


def _box(x, y, w, h, title, lines, kind="box", num=None):
    out = [f'<rect class="a-{kind}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>']
    tx = x + 14
    if num is not None:
        out.append(f'<circle class="a-num" cx="{x + 22}" cy="{y + 22}" r="11"/>'
                   f'<text class="a-numt" x="{x + 22}" y="{y + 26}" text-anchor="middle">{num}</text>')
        tx = x + 42
    out.append(f'<text class="a-title" x="{tx}" y="{y + 27}">{html.escape(title)}</text>')
    for i, line in enumerate(lines):
        out.append(f'<text class="a-sub" x="{x + 14}" y="{y + 48 + i * 17}">{html.escape(line)}</text>')
    return "".join(out)


def _arrow(d, dashed=False, both=False):
    cls = "a-arrow dashed" if dashed else "a-arrow"
    start = ' marker-start="url(#ah-s)"' if both else ""
    return f'<path class="{cls}" d="{d}" marker-end="url(#ah)"{start}/>'


def _label(x, y, text, anchor="middle"):
    return f'<text class="a-lab" x="{x}" y="{y}" text-anchor="{anchor}">{html.escape(text)}</text>'


def svg() -> str:
    W, H = 980, 640
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="{html.escape(L("Sistem mimarisi diyagramı", "System architecture diagram"))}">',
             """<style>
             .a-box{fill:var(--surf,#fff);stroke:var(--line,#DDE6E5);stroke-width:1.2}
             .a-app{fill:none;stroke:var(--teal,#0F766E);stroke-width:1.4;stroke-dasharray:6 5}
             .a-ext{fill:var(--soft,#E7F2F0);stroke:var(--softb,#C5DFDA);stroke-width:1.2}
             .a-off{fill:var(--bg,#F6F8F8);stroke:var(--line,#DDE6E5);stroke-width:1.2;stroke-dasharray:3 3}
             .a-title{font:600 14px 'IBM Plex Sans',system-ui,sans-serif;fill:var(--ink,#1F3A44)}
             .a-sub{font:400 12px 'IBM Plex Sans',system-ui,sans-serif;fill:var(--muted,#5F747C)}
             .a-lab{font:500 11px 'IBM Plex Sans',system-ui,sans-serif;fill:var(--muted,#5F747C)}
             .a-head{font:600 11px 'IBM Plex Sans',system-ui,sans-serif;fill:var(--teal,#0F766E);letter-spacing:.06em}
             .a-num{fill:var(--teal,#0F766E)} .a-numt{font:600 12px 'IBM Plex Sans',sans-serif;fill:#fff}
             .a-arrow{fill:none;stroke:var(--muted,#5F747C);stroke-width:1.5}
             .a-arrow.dashed{stroke-dasharray:5 4}
             .a-tip{fill:var(--muted,#5F747C)}
             </style>""",
             '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path class="a-tip" d="M0,0 L10,5 L0,10 z"/></marker>'
             '<marker id="ah-s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path class="a-tip" d="M0,0 L10,5 L0,10 z"/></marker></defs>']

    # kullanıcı
    parts.append(_box(20, 215, 160, 96, L("Kullanıcı", "User"),
                      [L("Hekim ya da hasta", "Physician or patient"), L("tarayıcı / telefon", "browser / phone")]))

    # uygulama kutusu
    parts.append('<rect class="a-app" x="215" y="18" width="500" height="452" rx="14"/>')
    parts.append(f'<text class="a-head" x="233" y="40">{html.escape(L("STREAMLIT UYGULAMASI · STREAMLIT COMMUNITY CLOUD", "STREAMLIT APP · STREAMLIT COMMUNITY CLOUD"))}</text>')
    steps = [
        (L("Giriş ve ön kontrol", "Input and pre-checks"),
         [L("Klinik + dermoskopik görüntü (hasta modunda yalnız klinik)", "Clinical + dermoscopic image (clinical only in patient mode)"),
          L("Kırpma, döndürme · netlik, ışık, cilt kontrolü", "Crop, rotate · sharpness, light, skin checks")]),
        (L("Tahmin motoru", "Prediction engine"),
         [L("4 EfficientNet (B3 + V2-S; klinik ve dermoskopik)", "4 EfficientNets (B3 + V2-S; clinical and dermoscopic)"),
          L("Çevirme ile test artırımı · sıcaklık kalibrasyonu", "Flip test-time augmentation · temperature scaling")]),
        (L("Karar katmanı", "Decision layer"),
         [L("Sevk eşikleri (öncelikli %30, değerlendirme %20)", "Referral thresholds (priority 30%, review 20%)"),
          L("Kararsızlık uyarısı · %90 olası tanılar (conformal)", "Uncertainty flag · 90% possible diagnoses (conformal)")]),
        (L("Sonuç, rapor ve takip", "Result, report and follow-up"),
         [L("Yönlendirme, Grad-CAM, PDF + QR, toplu ZIP", "Routing, Grad-CAM, PDF + QR, bulk ZIP"),
          L("Vaka takibi, hekim değerlendirmesi, zaman çizelgesi", "Case tracking, physician review, timeline")]),
    ]
    y = 56
    for i, (title, lines) in enumerate(steps, 1):
        parts.append(_box(240, y, 450, 86, title, lines, num=i))
        if i < len(steps):
            parts.append(_arrow(f"M465,{y + 86} L465,{y + 102}"))
        y += 102

    # dış servisler
    parts.append(_box(752, 120, 210, 112, L("Hugging Face", "Hugging Face"),
                      [L("Gizli model deposu", "Private model repository"), L("4 model ağırlığı + ayarlar", "4 model weights + config"),
                       L("Açılışta bir kez iner", "Downloaded once at start-up")], kind="ext"))
    parts.append(_box(752, 300, 210, 130, "Supabase (PostgreSQL)",
                      [L("Anonim vaka sonuçları", "Anonymous case results"), L("Denetim kaydı", "Audit log"),
                       L("Kullanılabilirlik anketi", "Usability survey"), L("Görüntü / kişisel bilgi yok", "No images / personal data")],
                      kind="ext"))

    # çevrim dışı eğitim şeridi
    parts.append(f'<text class="a-head" x="20" y="512">{html.escape(L("ÇEVRİM DIŞI (BİR KEZ)", "OFFLINE (ONCE)"))}</text>')
    parts.append(_box(20, 524, 200, 96, L("MILK10k + ek veri", "MILK10k + extra data"), [L("5.240 lezyon, 11 tür", "5,240 lesions, 11 classes"),
                                                                 L("+ PAD-UFES-20, ISIC 2019", "+ PAD-UFES-20, ISIC 2019")], kind="off"))
    parts.append(_box(262, 524, 240, 96, L("Eğitim (Kaggle GPU)", "Training (Kaggle GPU)"),
                      [L("v3 · kontrol kümesi Macro F1 0,594", "v3 · validation Macro F1 0.594"),
                       L("Klinik + PAD-UFES-20 · dermoskopik + ISIC 2019", "Clinical + PAD-UFES-20 · dermoscopic + ISIC 2019")],
                      kind="off"))
    parts.append(_box(544, 524, 418, 96, L("Değerlendirme", "Evaluation"),
                      [L("1.048 lezyonluk doğrulama → Model analizi sayfası", "1,048-lesion validation → Model analysis page"),
                       L("ISIC gizli test seti (v3): Macro F1 0,483", "ISIC hidden test set (v3): Macro F1 0.483")], kind="off"))

    # oklar
    parts.append(_arrow("M180,263 L238,99", both=True))
    parts.append(_label(196, 160, L("görüntü", "images"), "start"))
    parts.append(_arrow("M752,176 L692,194"))
    parts.append(_label(722, 168, L("model", "model")))
    parts.append(_arrow("M692,404 L752,375", both=True))
    parts.append(_label(722, 420, L("izinle", "consent")))
    parts.append(_arrow("M180,290 C200,420 214,400 238,404", dashed=True))
    parts.append(_label(150, 360, L("sonuç, PDF", "result, PDF")))
    parts.append(_arrow("M220,572 L260,572"))
    parts.append(_arrow("M502,560 C620,500 760,280 857,234", dashed=True))
    parts.append(_label(612, 497, L(".pt dosyaları yüklenir", ".pt files uploaded")))
    parts.append(_arrow("M502,590 L542,590"))
    parts.append("</svg>")
    return "".join(parts)


def render_architecture():
    st.markdown(f'<div style="width:100%;overflow-x:auto"><div style="min-width:620px">{svg()}</div></div>',
                unsafe_allow_html=True)
    st.caption(L("Kesikli kutular yalnızca bir kez yapılan çevrim dışı işlerdir; site çalışırken MILK10k verisine erişmez. "
                 "Görüntüler yalnızca analiz sırasında bellekte tutulur, veritabanına gitmez.",
                 "Dashed boxes are one-off offline steps; the running site never accesses MILK10k data. Images stay in memory "
                 "only during analysis and never reach the database."))
