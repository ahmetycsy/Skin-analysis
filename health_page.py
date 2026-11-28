"""Sade bilgi sayfaları: cilt sağlığı (kendi kendine muayene), fotoğraf çekme rehberi, model sürüm geçmişi, siteyi paylaş (QR)."""
from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st

import i18n

try:
    import segno
except ImportError:  # pragma: no cover
    segno = None


def L(tr: str, en: str) -> str:
    return tr if i18n.get_lang() == "tr" else en


def _h(text: str):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- fotoğraf çekme rehberi
PHOTO_TIPS = [
    ("Lezyon ortada ve net olsun", "Keep the lesion centred and sharp",
     "Telefonu sabit tutun, ekranda lezyona dokunup odaklayın.", "Hold the phone steady and tap the lesion on screen to focus."),
    ("Gün ışığı, flaş yok", "Daylight, no flash",
     "Pencere önünde dolaylı gün ışığı en iyisidir; flaş parlama ve renk bozulması yapar.",
     "Indirect daylight near a window works best; flash causes glare and colour shifts."),
    ("10–15 cm uzaklık", "10–15 cm away",
     "Lezyon ve çevresindeki sağlam cilt birlikte görünsün; aşırı yaklaşınca görüntü bulanıklaşır.",
     "Show the lesion with some normal skin around it; getting too close blurs the image."),
    ("Dik açı, gölgesiz", "Straight on, no shadow",
     "Telefonu cilde paralel tutun; elinizin ya da telefonun gölgesi düşmesin.",
     "Hold the phone parallel to the skin; avoid shadows from your hand or the phone."),
    ("Sade arka plan", "Plain background",
     "Kumaş, giysi ya da takı kareye girmesin; uzak çekimde model arka plana bakabilir.",
     "No fabric, clothing or jewellery in the frame; in distant shots the model may look at the background."),
    ("Temiz cilt", "Clean skin",
     "Krem, makyaj, kıl ya da bant lezyonu örtmesin. Yakınlaştırma (zoom) yerine yaklaşın.",
     "No cream, make-up, hair or plaster over the lesion. Move closer instead of zooming."),
]


def render_photo_tips(compact: bool = False):
    """Fotoğraf çekme ipuçları. compact=True: yeni analiz ekranındaki açılır kutu için kısa liste."""
    if compact:
        st.markdown("\n".join(f"- **{L(a, b)}** — {L(c, d)}" for a, b, c, d in PHOTO_TIPS))
        return
    # tek ızgara: bütün kartlar aynı genişlik ve aynı yükseklik (bilgisayarda 3×2, telefonda 2 ya da 1 sütun)
    cards = "".join(f'<div class="guide"><span class="step-no">{i}</span><div class="card-title">{L(a, b)}</div>'
                    f'<div class="card-text">{L(c, d)}</div></div>' for i, (a, b, c, d) in enumerate(PHOTO_TIPS, 1))
    st.markdown(f'<div class="tips-grid">{cards}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- cilt sağlığı sayfası
ABCDE_INFO = [
    ("A", "Asimetri", "Asymmetry", "Ortadan ikiye bölünse iki yarısı birbirine benzemez.", "The two halves do not match."),
    ("B", "Kenar", "Border", "Kenarları düzensiz, girintili ya da bulanıktır.", "Edges are irregular, notched or blurred."),
    ("C", "Renk", "Colour", "Birden çok renk (kahve, siyah, kırmızı, beyaz, mavi) içerir.",
     "Several colours (brown, black, red, white, blue)."),
    ("D", "Çap", "Diameter", "6 mm'den (kurşun kalem silgisi) büyüktür.", "Larger than 6 mm (a pencil eraser)."),
    ("E", "Değişim", "Evolving", "Boyutu, şekli ya da rengi değişiyor; kaşınıyor ya da kanıyor.",
     "Changing in size, shape or colour; itching or bleeding."),
]


def render_skin_health():
    lead = L("Cilt kanserlerinin çoğu erken fark edilirse kolayca tedavi edilir. Ayda bir kez cildinizi kendiniz kontrol "
             "etmek, değişen bir beni erken yakalamanın en kolay yoludur.",
             "Most skin cancers are easily treated when found early. Checking your own skin once a month is the easiest "
             "way to catch a changing mole early.")
    st.markdown(f'<p class="lead">{lead}</p>', unsafe_allow_html=True)

    _h(L("Bir beni ABCDE kuralıyla kontrol edin", "Check a mole with the ABCDE rule"))
    cols = st.columns(5, gap="small")
    for col, (k, tr, en, dtr, den) in zip(cols, ABCDE_INFO):
        col.markdown(f'<div class="guide"><span class="step-no">{k}</span><div class="card-title">{L(tr, en)}</div>'
                     f'<p class="card-text">{L(dtr, den)}</p></div>', unsafe_allow_html=True)
    st.caption(L("“Çirkin ördek yavrusu” işareti: vücuttaki diğer benlerinizden belirgin şekilde farklı görünen bir ben de "
                 "kontrol ettirilmelidir.", "“Ugly duckling” sign: a mole that looks clearly different from your other moles "
                 "should also be checked."))

    a, b = st.columns(2, gap="large")
    with a:
        _h(L("Ne zaman doktora gitmeli?", "When to see a doctor"))
        st.markdown(L(
            "- ABCDE işaretlerinden **biri ya da birkaçı** varsa\n"
            "- Yeni çıkan ve **büyüyen** bir ben ya da leke\n"
            "- **İyileşmeyen** yara, kabuklanan ya da kendiliğinden **kanayan** bir lezyon\n"
            "- Parlak, inci gibi kabarık ya da pullanan, kızarık bir alan\n"
            "- Bu sitenin sonucundan bağımsız olarak, sizi **endişelendiren** her değişiklik",
            "- **One or more** ABCDE signs\n"
            "- A new mole or spot that is **growing**\n"
            "- A sore that **does not heal**, crusts or **bleeds** on its own\n"
            "- A shiny, pearly bump or a scaly, red patch\n"
            "- **Any change that worries you**, whatever this site shows"))
    with b:
        _h(L("Kendi kendine muayene (ayda bir)", "Self-examination (monthly)"))
        st.markdown(L(
            "1. İyi ışıklı bir odada, boy aynası ve el aynasıyla\n"
            "2. Yüz, kulak arkası, boyun ve saçlı deri (tarakla ayırarak)\n"
            "3. Kollar, el ayası, tırnak altları\n"
            "4. Göğüs, karın, sırt (el aynasıyla)\n"
            "5. Bacaklar, ayak tabanı ve parmak araları\n"
            "6. Şüpheli benin fotoğrafını çekip bir ay sonra karşılaştırın",
            "1. In a well-lit room, with a full-length and a hand mirror\n"
            "2. Face, behind the ears, neck and scalp (part the hair)\n"
            "3. Arms, palms, under the nails\n"
            "4. Chest, abdomen, back (use the hand mirror)\n"
            "5. Legs, soles and between the toes\n"
            "6. Photograph a suspicious mole and compare after a month"))

    _h(L("Güneşten korunma", "Sun protection"))
    st.markdown(L(
        "- Güneşin en güçlü olduğu öğle saatlerinde (yaklaşık 10:00–14:00; gölgeniz boyunuzdan kısaysa) gölge arayın\n"
        "- En az **SPF 30**, geniş spektrumlu güneş kremi; 2 saatte bir ve yüzme sonrası yenileyin\n"
        "- Şapka, güneş gözlüğü ve uzun kollu giysi\n"
        "- Solaryum kullanmayın",
        "- Seek shade around midday, when the sun is strongest (about 10 a.m.–2 p.m.; if your shadow is shorter than you)\n"
        "- Broad-spectrum sunscreen of at least **SPF 30**; reapply every 2 hours and after swimming\n"
        "- Hat, sunglasses and long sleeves\n"
        "- Do not use tanning beds"))

    _h(L("Telefonla iyi fotoğraf çekmek", "Taking a good phone photo"))
    render_photo_tips()
    note = L("Bu sayfa genel bilgi içindir; tanı ya da tedavi önerisi değildir. Şüpheli bir değişiklikte dermatoloğa başvurun. "
             "Sitedeki analiz aracı benler ve lekeler içindir; egzama, sedef, mantar ya da döküntüler için uygun değildir.",
             "This page is general information, not a diagnosis or treatment advice. See a dermatologist about any "
             "suspicious change. The analysis tool on this site is for moles and spots; it is not suitable for eczema, "
             "psoriasis, fungal infections or rashes.")
    st.markdown(f'<div class="banner" style="margin-top:1rem">{note}</div>', unsafe_allow_html=True)

    _h(L("Kaynaklar", "Sources"))
    st.markdown("\n".join(f"- [{L(a, b)}]({u})" for a, b, u in HEALTH_SOURCES))
    st.caption(L("Bilgiler bu kaynaklardan sadeleştirilerek alınmıştır (erişim: Kasım 2026).",
                 "Information on this page is simplified from these sources (accessed November 2026)."))


HEALTH_SOURCES = [
    ("AAD – Melanomun ABCDE işaretleri", "AAD – What to look for: ABCDEs of melanoma",
     "https://www.aad.org/public/diseases/skin-cancer/find/at-risk/abcdes"),
    ("AAD – Cildinizi kendiniz nasıl kontrol edersiniz", "AAD – How to perform a skin self-exam",
     "https://www.aad.org/public/diseases/skin-cancer/find/check-skin"),
    ("AAD – Cilt kanserinden korunma", "AAD – How to prevent skin cancer",
     "https://www.aad.org/public/diseases/skin-cancer/prevent/how"),
    ("WHO – Ultraviyole radyasyon (bilgi notu)", "WHO – Ultraviolet radiation (fact sheet)",
     "https://www.who.int/news-room/fact-sheets/detail/ultraviolet-radiation"),
]


# ---------------------------------------------------------------- model sürüm geçmişi
VERSIONS = [
    {"v": "v1", "date": "21.11.2026", "tr": "Yalnızca MILK10k", "en": "MILK10k only",
     "h_f1": 0.546, "h_acc": 0.768, "h_miss": 19, "p_f1": 0.484, "p_acc": 0.703, "p_miss": 24, "phone": 0.399, "isic": 0.485},
    {"v": "v2", "date": "22.11.2026", "tr": "+ PAD-UFES-20 (klinik, telefon fotoğrafı)", "en": "+ PAD-UFES-20 (clinical, phone photos)",
     "h_f1": 0.535, "h_acc": 0.771, "h_miss": 18, "p_f1": 0.496, "p_acc": 0.724, "p_miss": 20, "phone": 0.668, "isic": None},
    {"v": "v3", "date": "22.11.2026", "tr": "+ ISIC 2019 (dermoskopik)", "en": "+ ISIC 2019 (dermoscopic)",
     "h_f1": 0.594, "h_acc": 0.801, "h_miss": 14, "p_f1": 0.496, "p_acc": 0.724, "p_miss": 20, "phone": 0.668, "isic": 0.483},
]


def render_versions():
    fmt = (lambda x: "–" if x is None else f"{x:.3f}".replace(".", ",") if i18n.get_lang() == "tr" else f"{x:.3f}")
    pct = (lambda x: i18n.pct(x))
    st.markdown(L("Model aynı 1.048 lezyonluk MILK10k kontrol kümesinde ölçüldü; her sürümde yalnızca eğitim verisi "
                  "değişti (aynı ayarlar). En son sürüm sitede kullanılan sürümdür.",
                  "Every version was measured on the same 1,048-lesion MILK10k validation set; only the training data "
                  "changed (same settings). The latest version is the one in use on this site."))
    rows = [{
        L("Sürüm", "Version"): f'{r["v"]} · {L(r["tr"], r["en"])}',
        L("Hekim F1", "Physician F1"): fmt(r["h_f1"]), L("Hekim doğruluk", "Physician accuracy"): pct(r["h_acc"]),
        L("Kaçırılan kötü huylu (hekim)", "Missed malignant (physician)"): f'{r["h_miss"]}/691',
        L("Hasta F1", "Patient F1"): fmt(r["p_f1"]), L("Hasta doğruluk", "Patient accuracy"): pct(r["p_acc"]),
        L("Telefon testi F1", "Phone test F1"): fmt(r["phone"]),
        L("ISIC gizli test", "ISIC hidden test"): fmt(r["isic"]),
    } for r in VERSIONS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    long = []
    for r in VERSIONS:
        for key, tr, en in (("h_f1", "Hekim modu", "Physician mode"), ("p_f1", "Hasta modu", "Patient mode"),
                            ("phone", "Telefon fotoğrafı testi", "Phone photo test")):
            long.append({"v": r["v"], "s": L(tr, en), "f1": r[key]})
    df = pd.DataFrame(long)
    chart = (alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2.5)
             .encode(x=alt.X("v:N", title=None, axis=alt.Axis(labelAngle=0)),
                     y=alt.Y("f1:Q", title="Macro F1", scale=alt.Scale(domain=[0.3, 0.75])),
                     color=alt.Color("s:N", title=None, legend=alt.Legend(orient="bottom"),
                                     scale=alt.Scale(range=["#0F766E", "#B0701A", "#5B6EAE"])),
                     tooltip=["v", "s", alt.Tooltip("f1:Q", format=".3f")])
             .properties(height=260))
    st.altair_chart(chart, use_container_width=True)
    st.caption(L(
        "Telefon testi: PAD-UFES-20'nin eğitimde hiç kullanılmamış, hasta bazında ayrılmış 433 fotoğrafı (6 tür). "
        "ISIC gizli test: ISIC'in cevap anahtarını paylaşmadığı 479 lezyon; v1 ve v3 gönderildi, v3'teki kontrol kümesi artışı bu testte görülmedi (0,485 → 0,483). "
        "v2'de hekim modundaki küçük düşüş istatistiksel olarak anlamlı değildir. Ek veriler (PAD-UFES-20, ISIC 2019) "
        "MILK10k'in yerine değil, yanına eklendi; MILK10k ile aynı ya da çok benzer 33 ISIC 2019 görüntüsü çıkarıldı.",
        "Phone test: 433 patient-grouped PAD-UFES-20 photos never used in training (6 types). ISIC hidden test: 479 "
        "lesions whose labels ISIC does not share; v1 and v3 were submitted, and the v3 validation gain did not carry over to this test (0.485 → 0.483). The small physician-mode drop in v2 is not "
        "statistically significant. Extra data (PAD-UFES-20, ISIC 2019) was added alongside MILK10k, not instead of it; "
        "33 ISIC 2019 images identical or near-identical to MILK10k were removed."))


# ---------------------------------------------------------------- siteyi paylaş
def render_share(url: str):
    a, b = st.columns([1, 2], gap="large")
    with a:
        if segno is not None:
            buf = io.BytesIO()
            segno.make(url, error="m").save(buf, kind="png", scale=10, border=2, dark="#1F3A44", light="#FFFFFF")
            st.image(buf.getvalue(), width=240)
            st.download_button(L("QR kodu indir (PNG)", "Download QR code (PNG)"), buf.getvalue(), "skin-cds-qr.png",
                               "image/png", icon=":material/download:")
    with b:
        st.markdown(L("Telefonunuzun kamerasıyla kodu okutarak siteyi açabilirsiniz. Sunumda slayta koymak için "
                      "PNG olarak indirebilirsiniz.",
                      "Scan the code with your phone camera to open the site. Download it as PNG to put it on a slide."))
        st.code(url, language=None)


# ---------------------------------------------------------------- sürüm notları (Model → Sürüm notları)
# En yeni en üstte. Etiket: model / site / safety / eval. Yeni değişiklikte en üste bir giriş ekleyin.
RELEASE_NOTES = [
    ("29.11.2026", [
        ("model", "Öncelikli sevk artık kanser türüne göre: melanom, SCC veya diğer kötü huylu şüphesi (toplam ≥ %15) öncelikli; "
                  "yavaş seyirli BCC tek başına rutin sevke düşer. Doğrulamada kaçırılan kanser değişmedi, öncelikli yük yaklaşık %25 azaldı.",
                  "Priority referral now depends on the cancer type: suspected melanoma, SCC or other malignant growth (total ≥ 15%) is "
                  "priority; slow-growing BCC alone gets a routine referral. On validation, missed cancers did not change and the priority "
                  "load fell by about 25%."),
        ("safety", "Sonuçlarda süre: öncelikli 1–2 hafta, rutin 4 hafta. Hasta modunda randevu yönlendirmesi (MHRS, e-Nabız, 182, aile "
                   "hekimi, PDF'i götürme, kötüleşmede beklememe).",
                   "Timeframes in results: priority 1–2 weeks, routine 4 weeks. Appointment guidance in patient mode (MHRS, e-Nabız, 182, "
                   "family doctor, bring the PDF, do not wait if it gets worse)."),
        ("site", "PDF raporda uzun satırların görüntülerin altına taşması düzeltildi; rapora randevu satırı eklendi.",
                 "PDF report: long lines no longer run under the images; an appointment line was added."),
        ("site", "Model sayfasına Sürüm notları sekmesi eklendi.", "A Release notes tab was added to the Model page."),
        ("site", "Hasta modunda \"Örnek fotoğrafla dene\" (PAD-UFES-20, CC BY 4.0; veritabanına kaydedilmez) ve ankete doğrudan bağlantı.",
                 "\"Try a sample photo\" in patient mode (PAD-UFES-20, CC BY 4.0; not stored) and a direct link to the survey."),
        ("eval", "Eşik simülatöründe türe göre öncelik kuralı (sitedeki kural) seçilebiliyor; poliklinik senaryosu da bu kurala göre hesaplanıyor.",
                 "The threshold simulator offers the by-type priority rule used by the site; the clinic scenario uses it too."),
        ("site", "Yan menüdeki simge yerine BAU mührü.", "The BAU seal replaces the icon in the side menu."),
        ("safety", "Hekim girişinde vaka silme: vaka sitede gizlenir, nedeni ve silen hekim denetim kaydında saklanır; Silinen vakalar "
                   "bölümünden geri alınabilir.",
                   "Case deletion for signed-in physicians: the case is hidden on the site, the reason and the physician are kept in "
                   "the audit log, and it can be restored from Deleted cases."),
        ("model", "Isı haritası (Grad-CAM) artık kararı açıklıyor: sevki tetikleyen türler (örneğin melanom ve SCC) için çiziliyor. Daha sakin "
                  "renkler, en yoğun bölgenin beyaz çizgiyle gösterilmesi ve renk göstergesi eklendi.",
                  "The heat map (Grad-CAM) now explains the decision: it is drawn for the types that triggered the referral (e.g. "
                  "melanoma and SCC). Calmer colours, a white outline of the strongest region and a colour legend were added."),
        ("model", "Hekim modunda iki görüntünün de ısı haritası (dermoskopik ve klinik) gösteriliyor; haritalar görüntünün aynalı "
                  "hâliyle ortalanarak daha kararlı hâle getirildi.",
                  "In physician mode both images now have a heat map (dermoscopic and clinical); maps are averaged with the "
                  "mirrored image for stability."),
    ]),
    ("28.11.2026", [
        ("safety", "Hasta modunda kare kırpma zorunlu: lekeyi ortalayıp kutunun yaklaşık üçte birini doldurması isteniyor (kumaş ve "
                   "arka plan modeli yanıltabiliyordu).",
                   "Square cropping is mandatory in patient mode: the spot must be centred and fill about a third of the box (fabric "
                   "and background could mislead the model)."),
        ("safety", "Eşiğe çok yakın sonuçlarda \"sınırda sonuç\" uyarısı.", "\"Borderline result\" note for results very close to a threshold."),
        ("eval", "Danışman geri bildirimi işlendi: tüm sayılar v3'e göre; ana ölçüt gizli test Macro F1 0,483, dengeli doğruluk öne alındı; "
                 "ek veri kararı, model × veri tablosu, kopya kontrolü ve lisanslar Veri ve sınırlılıklar sayfasında.",
                 "Advisor feedback applied: all figures as of v3; the hidden-test Macro F1 0.483 is the main measure and balanced "
                 "accuracy comes first; the extra-data decision, model × data table, duplicate check and licences are on the Data "
                 "and limitations page."),
        ("site", "Cilt sağlığı sayfasına WHO ve AAD kaynakları; İngilizce çeviri bütün sayfalarda kontrol edildi.",
                 "WHO and AAD sources on the Skin health page; the English version was checked on every page."),
        ("site", "PDF raporda BAU logosu ve yeni düzen; vaka listesinde \"Ayrıntı\" önizlemesi; fotoğraf ipuçları ve model tablosu düzeni.",
                 "BAU logo and new layout in the PDF report; \"Details\" preview in the case list; layout of photo tips and the model table."),
    ]),
    ("24.11.2026", [
        ("safety", "Melanom güvenlik kuralı: melanom olasılığı hekim modunda %10, hasta modunda %15'in üstündeyse lezyon yine sevk edilir.",
                   "Melanoma safety rule: a lesion is referred when the melanoma probability exceeds 10% (physician) or 15% (patient)."),
        ("safety", "Kapsam uyarısı: döküntü ve egzama için araç uygun değil; anket için bilgilendirilmiş onam metni.",
                   "Scope warning: the tool is not suitable for rashes or eczema; informed-consent statement for the survey."),
        ("site", "Tasarım yenilemesi: yönlendirme cetveli, yeni sonuç ekranı düzeni, yeni renkler ve logo.",
                 "Visual redesign: referral ruler, new result layout, new colours and logo."),
    ]),
    ("22.11.2026", [
        ("model", "v2: klinik modeller PAD-UFES-20 telefon fotoğraflarıyla yeniden eğitildi (telefon testi Macro F1 0,399 → 0,668). "
                  "v3: dermoskopik modeller ISIC 2019 ile yeniden eğitildi (doğrulama Macro F1 0,594).",
                  "v2: clinical models retrained with PAD-UFES-20 phone photos (phone test Macro F1 0.399 → 0.668). v3: dermoscopic "
                  "models retrained with ISIC 2019 (validation Macro F1 0.594)."),
        ("eval", "ISIC MILK10k gizli test gönderimi (v3): Macro F1 0,483.", "ISIC MILK10k hidden-test submission (v3): Macro F1 0.483."),
        ("site", "Hasta modunda sakin, eylem odaklı sonuç; belirti soruları; \"Doktoruma göster\"; Cilt sağlığı sayfası; sadeleştirilmiş menü.",
                 "Calm, action-first result in patient mode; symptom questions; \"Show my doctor\"; Skin health page; simpler menu."),
    ]),
    ("21.11.2026", [
        ("eval", "İlk ISIC MILK10k gizli test gönderimi (v1, yalnız MILK10k): Macro F1 0,485.",
                 "First ISIC MILK10k hidden-test submission (v1, MILK10k only): Macro F1 0.485."),
        ("model", "Hasta modu (tek telefon fotoğrafı), olası tanılar (%90 güven kümesi), maliyete göre sevk eşikleri.",
                  "Patient mode (single phone photo), possible diagnoses (90% conformal set), cost-based referral thresholds."),
        ("site", "Anonim veritabanı (izinle); Model analizi sayfası (eşik simülatörü, kalibrasyon, eşitlik, karar eğrisi, poliklinik "
                 "senaryosu); koyu tema; kullanılabilirlik anketi; görüntü döndürme; toplu PDF.",
                 "Anonymous database (with consent); Model analysis page (threshold simulator, calibration, equity, decision curve, "
                 "clinic scenario); dark theme; usability survey; image rotation; bulk PDF."),
    ]),
    ("20.11.2026", [
        ("model", "Gerçek modeller bağlandı: klinik ve dermoskopik görüntü için EfficientNet-B3 ve EfficientNetV2-S (4 model, kalibre ortalama).",
                  "Real models connected: EfficientNet-B3 and EfficientNetV2-S for clinical and dermoscopic images (4 models, calibrated average)."),
        ("site", "Türkçe/İngilizce; vaka takibi; PDF rapor ve QR kod; hekim modu ve değerlendirme formu; denetim kaydı; Grad-CAM ısı haritası.",
                 "Turkish/English; case tracking; PDF report and QR code; physician mode and review form; audit log; Grad-CAM heat map."),
    ]),
    ("18.11.2026", [
        ("site", "İlk çevrim içi prototip (demo modelle) ve ortak Kaggle eğitim notebook'u.",
                 "First online prototype (demo model) and shared Kaggle training notebook."),
    ]),
]

RN_TAGS = {"model": ("Model", "Model", "#0E5E57"), "site": ("Site", "Site", "#3A6EA5"),
           "safety": ("Güvenlik", "Safety", "#B0701A"), "eval": ("Değerlendirme", "Evaluation", "#6B4E9B")}


def render_release_notes():
    st.markdown(f'<p class="lead">{L("Sistemde nelerin, ne zaman değiştiği. En yeni en üstte. Model başarı sayıları Sürüm geçmişi sekmesinde.", "What changed in the system and when. Newest first. Model performance figures are on the Version history tab.")}</p>',
                unsafe_allow_html=True)
    legend = " ".join(f'<span class="rn-tag" style="--c:{c}">{L(tr, en)}</span>' for tr, en, c in RN_TAGS.values())
    st.markdown(f'<div class="rn-legend">{legend}</div>', unsafe_allow_html=True)
    for date, items in RELEASE_NOTES:
        rows = "".join(f'<li><span class="rn-tag" style="--c:{RN_TAGS[k][2]}">{L(RN_TAGS[k][0], RN_TAGS[k][1])}</span>'
                       f'<span class="rn-text">{L(tr, en)}</span></li>' for k, tr, en in items)
        st.markdown(f'<div class="rn-day"><div class="rn-date">{date}</div><ul class="rn-list">{rows}</ul></div>',
                    unsafe_allow_html=True)
