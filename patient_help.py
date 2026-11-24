"""Hasta deneyimi yardımcıları: belirti soruları, belirtiye göre öneri, "bu sonuç ne kadar güvenilir?" kutusu,
kontrol hatırlatıcısı (.ics takvim dosyası).

Belirti soruları modelin olasılıklarını DEĞİŞTİRMEZ; yalnızca fotoğrafın gösteremediği bilgiye göre öneriyi netleştirir.
Model "öncelikli" dediyse belirtiler bu öneriyi asla hafifletmez.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import streamlit as st

import i18n

ROOT = os.path.dirname(os.path.abspath(__file__))

# anahtar, Türkçe soru, İngilizce soru
SYMPTOMS = [
    ("changed", "Son aylarda büyüdü, şekli ya da rengi değişti mi?", "Has it grown or changed shape or colour in recent months?"),
    ("bleed", "Kendiliğinden kanıyor, kabuklanıyor ya da iyileşmeyen bir yara mı?", "Does it bleed or crust on its own, or is it a sore that does not heal?"),
    ("rough", "Dokununca pürüzlü, zımpara gibi mi?", "Does it feel rough, like sandpaper?"),
    ("years", "Yıllardır aynı mı duruyor?", "Has it looked the same for years?"),
    ("similar", "Vücutta buna çok benzeyen başka lekeler de var mı?", "Are there other spots on the body that look very similar?"),
]
ANS = ("yes", "no", "unk")


def L(tr: str, en: str) -> str:
    return tr if i18n.get_lang() == "tr" else en


def ans_label(k: str) -> str:
    return {"yes": L("Evet", "Yes"), "no": L("Hayır", "No"), "unk": L("Bilmiyorum", "Not sure")}[k]


def render_symptom_form(stored: dict, key_fn) -> dict:
    """Hasta modunda isteğe bağlı belirti soruları. Döndürür: {anahtar: yes/no/unk} (yanıtlananlar)."""
    st.markdown(f'<div class="card-title">{L("Fotoğrafın göstermediği bilgiler (isteğe bağlı)", "What the photo cannot show (optional)")}</div>'
                f'<p class="card-text">{L("Cevaplar modelin sonucunu değiştirmez; yalnızca öneriyi netleştirir ve PDF rapora eklenir.", "Answers do not change the model result; they only refine the advice and are added to the PDF report.")}</p>',
                unsafe_allow_html=True)
    out = {}
    for k, tr, en in SYMPTOMS:
        v = st.radio(L(tr, en), ANS, index=ANS.index(stored[k]) if stored.get(k) in ANS else None, horizontal=True,
                     format_func=ans_label, key=key_fn(f"sy_{k}"))
        if v:
            out[k] = v
    return out


def symptom_lines(sym: dict):
    """PDF / özet için (soru, cevap) satırları."""
    return [(L(tr, en), ans_label(sym[k])) for k, tr, en in SYMPTOMS if k in (sym or {})]


def symptom_advice(sym: dict, level: str, top: str):
    """Belirtilere göre ek öneri. Döndürür: (tür, metin) listesi; tür 'warn' (kırmızı) ya da 'info'."""
    if not sym:
        return []
    out = []
    red = [k for k in ("changed", "bleed") if sym.get(k) == "yes"]
    if red:
        out.append(("warn", L(
            "Belirttiğiniz değişim ya da kanama/kabuklanma, modelin sonucundan bağımsız olarak <b>beklemeden</b> bir "
            "dermatoloğa görünmeyi gerektiren uyarı işaretleridir.",
            "The change or bleeding/crusting you reported is a warning sign that needs a dermatologist <b>without "
            "waiting</b>, whatever the model shows.")))
    calm = sym.get("years") == "yes" and sym.get("changed") != "yes" and sym.get("bleed") != "yes"
    if calm and level == "urgent":
        out.append(("info", L(
            "Lekenin yıllardır değişmemesi rahatlatıcıdır, ancak model öncelikli değerlendirme önerdiği için yine de "
            "kısa sürede doktora gidin.",
            "The spot staying the same for years is reassuring, but the model suggests priority review, so please see a "
            "doctor soon anyway.")))
    elif calm:
        extra = L(" Vücuttaki benzer lekelerle aynı görünmesi de iyiye işarettir.", " Looking like your other spots is also a good sign.") \
            if sym.get("similar") == "yes" else ""
        out.append(("info", L(
            "Lekenin yıllardır değişmemesi rahatlatıcı bir işarettir." + extra + " Acele etmeden, ilk fırsatta rutin bir "
            "dermatolog kontrolünde gösterin; değişirse daha erken gidin.",
            "The spot staying the same for years is a reassuring sign." + extra + " Show it at a routine dermatology "
            "check when convenient; go sooner if it changes.")))
    if top == "AKIEC" and sym.get("rough") == "no":
        out.append(("info", L(
            "Model en çok aktinik keratoza benzetti, ama bu lekeler genelde dokununca pürüzlüdür. Pürüzsüz, düz kahverengi "
            "lekelerin çoğu iyi huylu güneş lentigosudur. Telefon fotoğrafı dokuyu gösteremediği için ayrımı doktor yapar.",
            "The model's top match is actinic keratosis, but these usually feel rough. Most smooth, flat brown spots are "
            "benign solar lentigines. A phone photo cannot show texture, so a doctor should tell them apart.")))
    elif top == "AKIEC" and sym.get("rough") == "yes":
        out.append(("info", L("Pürüzlü olması aktinik keratozla uyumludur; tedavi edilebilir bir durumdur, doktora gösterin.",
                              "Feeling rough fits actinic keratosis, which is treatable; please show it to a doctor.")))
    return out


# ---------------------------------------------------------------- güvenilirlik
@st.cache_data(show_spinner=False)
def _eval_summary():
    try:
        with open(os.path.join(ROOT, "assets", "eval_summary.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def reliability(top: str, mode: str):
    """Model bu türü dediğinde doğrulama kümesinde ne sıklıkla doğru çıktı? (doğru, toplam, [(diğer tür, sayı), ...])"""
    d = _eval_summary()
    if not d:
        return None
    m = d["modes"].get("p" if mode == "hasta" else "h")
    if not m or top not in d["classes"]:
        return None
    cm = np.array(m["cm"])
    i = d["classes"].index(top)
    col = cm[:, i]
    n = int(col.sum())
    if n == 0:
        return None
    others = sorted(((d["classes"][j], int(col[j])) for j in range(len(col)) if j != i and col[j] > 0), key=lambda x: -x[1])
    return int(col[i]), n, others[:2]


def reliability_html(top: str, mode: str, top_p: float):
    r = reliability(top, mode)
    if r is None:
        return None
    ok, n, others = r
    name = i18n.plain(top)
    if n < 10:
        body = L(f"Model <b>{name}</b> sonucunu doğrulama lezyonlarında yalnızca {n} kez verdi; bu tür için güvenilirlik "
                 "ölçülemeyecek kadar az örnek var.",
                 f"The model gave <b>{name}</b> only {n} times on the validation lesions; too few to judge its reliability.")
    else:
        per100 = round(100 * ok / n)
        oth_tr = ", ".join(f"{i18n.plain(c)} (100'de {round(100 * k / n)})" for c, k in others)
        oth_en = ", ".join(f"{i18n.plain(c)} ({round(100 * k / n)} in 100)" for c, k in others)
        body = L(f"Model <b>{name}</b> dediğinde, sonucu bilinen {n} test lezyonunda doğru çıkma oranı <b>100'de {per100}</b>."
                 + (f" Yanlış çıktığında gerçek tür çoğunlukla: {oth_tr}." if others else ""),
                 f"When the model said <b>{name}</b>, it was right <b>{per100} times in 100</b> on {n} test lesions with a "
                 f"known answer." + (f" When wrong, the true type was mostly: {oth_en}." if others else ""))
    pct = i18n.pct(top_p)
    hint = L(f"Ekrandaki {pct}, modelin bu fotoğraf için ne kadar emin olduğudur; %50 civarı “emin değilim” demektir.",
             f"The {pct} on screen is how sure the model is for this photo; around 50% means “not sure”.")
    title = L("Bu sonuç ne kadar güvenilir?", "How reliable is this result?")
    return (f'<div class="rely"><div class="rely-t">{title}</div><div class="rely-b">{body}</div>'
            f'<div class="rely-h">{hint}</div></div>')


# ---------------------------------------------------------------- takvim hatırlatıcısı
def ics_bytes(case_no: str, months: int, url: str) -> bytes:
    day = date.today() + timedelta(days=round(30.4 * months))
    nxt = day + timedelta(days=1)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = L(f"Cilt lekesi kontrolü ({case_no})", f"Skin spot check ({case_no})")
    desc = L("Aynı ışıkta yeni bir fotoğraf çekip siteye yükleyin (Önceki vaka no: {c}). Leke büyüdüyse, rengi ya da şekli "
             "değiştiyse veya kanıyorsa dermatoloğa başvurun. {u}",
             "Take a new photo in the same light and upload it to the site (previous case no: {c}). See a dermatologist if "
             "the spot has grown, changed colour or shape, or bleeds. {u}").format(c=case_no, u=f"{url}/?case={case_no}")
    desc = desc.replace(",", "\\,").replace(";", "\\;")
    summary = summary.replace(",", "\\,")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//BAU Skin CDS//TR", "CALSCALE:GREGORIAN", "BEGIN:VEVENT",
             f"UID:{case_no}-{months}m@skin-cds", f"DTSTAMP:{stamp}", f"DTSTART;VALUE=DATE:{day:%Y%m%d}",
             f"DTEND;VALUE=DATE:{nxt:%Y%m%d}", f"SUMMARY:{summary}", f"DESCRIPTION:{desc}",
             "BEGIN:VALARM", "TRIGGER:-PT0M", "ACTION:DISPLAY", f"DESCRIPTION:{summary}", "END:VALARM",
             "END:VEVENT", "END:VCALENDAR"]
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


# ---------------------------------------------------------------- sağlık kaygısı: hasta modunda sakin sonuç
# Sayılar 1.048 lezyonluk doğrulama kümesinden, hasta modu ve şu anki eşiklerle (analysis/thresholds.py, 22.11.2026):
#   iyi huylu lekelerin %44'ü "öncelikli", %14'ü "değerlendirme" sonucu alıyor (toplam %58 sevk);
#   poliklinik karışımı varsayımıyla (%8 kötü huylu, %10 kansere dönüşebilir) "öncelikli" sonuç alanların ~%15'i kötü huylu.
# Eşikler değişirse bu sayılar da güncellenmeli.
CALM = {
    "urgent": ("Lekenizi kısa süre içinde (1–2 hafta) bir dermatoloğa gösterin",
               "Show the spot to a dermatologist soon (within 1–2 weeks)",
               "Model bu fotoğrafta doktorun bakması gereken bazı özellikler gördü. Bu bir tanı değildir; yalnızca lekenin "
               "bir uzman tarafından görülmesi gerektiği anlamına gelir.",
               "The model saw some features in this photo that a doctor should look at. This is not a diagnosis; it only "
               "means the spot should be seen by a specialist."),
    "refer": ("Lekenizi bir dermatoloğa gösterin (acil değil)", "Show the spot to a dermatologist (not urgent)",
              "Model fotoğraftan kesin bir şey söyleyemedi ya da hafif bazı özellikler gördü. İlk uygun randevuda "
              "bir dermatoloğa göstermeniz yeterli.",
              "The model could not tell from the photo, or saw some mild features. Showing it to a dermatologist at the "
              "next convenient appointment is enough."),
    "benign": ("Belirgin bir endişe işareti görülmedi", "No clear warning sign was seen",
               "Model fotoğrafta belirgin bir uyarı işareti görmedi. Leke büyürse, rengi ya da şekli değişirse, kanar ya da "
               "yara açarsa bir doktora gidin.",
               "The model saw no clear warning sign in the photo. See a doctor if the spot grows, changes colour or shape, "
               "bleeds or becomes a sore."),
}


def calm_card_html(level: str, mel_rule: bool = False) -> str:
    tr_t, en_t, tr_b, en_b = CALM[level]
    if mel_rule:  # melanom güvenlik kuralıyla yönlendirildi
        tr_b = ("Leke büyük ihtimalle zararsız bir bene benziyor; ancak model melanomu (ben kanseri) tamamen dışlayamadı. "
                "Güvenlik için ilk uygun randevuda bir dermatoloğa gösterin.")
        en_b = ("The spot most likely looks like a harmless mole, but the model could not fully rule out melanoma. To be safe, "
                "show it to a dermatologist at the next convenient appointment.")
    cls = "benign" if level == "benign" else "refer"  # hasta modunda kırmızı yerine daha sakin renk
    ic = "✓" if level == "benign" else "→"
    return (f'<div class="rec {cls}"><div class="rec-title"><span class="rec-ic">{ic}</span>{L(tr_t, en_t)}</div>'
            f'<div class="rec-text">{L(tr_b, en_b)}</div></div>')


def calm_meaning_html(level: str) -> str:
    """Yönlendirilen hastaya sonucun anlamını dürüst sayılarla anlatır."""
    if level == "benign":
        body = L("Bu araç kanseri kaçırmamak için temkinli ayarlandı; yine de hiçbir yapay zekâ aracı her lekeyi doğru "
                 "değerlendiremez. Sizi endişelendiren bir değişiklik olursa sonuç ne olursa olsun doktora gidin.",
                 "This tool is set to be cautious so as not to miss cancer; still, no AI tool judges every spot correctly. "
                 "If a change worries you, see a doctor whatever the result.")
    else:
        body = L("<b>Doktora yönlendirilmek kanser olduğunuz anlamına gelmez.</b> Bu araç kanseri kaçırmamak için bilerek "
                 "temkinli ayarlandı; bu yüzden testlerde <b>zararsız lekelerin yarısından fazlasını</b> da doktora "
                 "yönlendirdi. Tipik bir poliklinik hastası grubunda, “kısa süre içinde” sonucu alan her 100 kişiden "
                 "<b>yaklaşık 85'inin lekesi kanser çıkmaz</b> (tahmini). Doktor çoğu lekeye bakarak birkaç dakikada karar verir; "
                 "cilt kanserlerinin büyük çoğunluğu da erken fark edildiğinde tamamen tedavi edilir.",
                 "<b>Being referred does not mean you have cancer.</b> This tool is deliberately cautious so as not to miss "
                 "cancer, so in testing it also referred <b>more than half of harmless spots</b>. In a typical clinic "
                 "population, of every 100 people who get the “soon” result, <b>about 85 do not have skin cancer</b> "
                 "(estimate). A doctor can usually decide within minutes by looking, and most skin cancers are fully "
                 "treatable when found early.")
    tail = L("Aynı lekeyi tekrar tekrar analiz etmek sonucu daha doğru yapmaz; en iyi adım bir doktorla konuşmaktır.",
             "Analysing the same spot again and again does not make the result more accurate; the best step is to talk "
             "to a doctor.")
    title = L("Bu sonuç ne anlama geliyor?", "What does this result mean?")
    return (f'<div class="rely"><div class="rely-t">{title}</div><div class="rely-b">{body}</div>'
            f'<div class="rely-h">{tail}</div></div>')


def details_warning() -> str:
    return L("Aşağıdaki yüzdeler modelin bu fotoğraftaki benzerlik tahminleridir; tek başına anlamlı değildir ve tanı yerine "
             "geçmez. Özellikle tek bir fotoğrafla model sık yanılır.",
             "The percentages below are the model's similarity estimates for this photo; on their own they mean little and "
             "are not a diagnosis. With a single photo the model is often wrong.")
