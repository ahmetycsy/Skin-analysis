"""Türkçe / İngilizce metinler. Arayüzdeki her yazı buradan gelir.

Kullanım: set_lang("tr" | "en"), sonra t("anahtar", isim=değer).
Yeni bir metin eklerken iki dili de yazın.
"""
from __future__ import annotations

from predictor import BENIGN, CLASS_NAMES, DESCRIPTIONS, MALIGN, PLAIN_NAMES, PREMALIGN, SITES

LANGS = {"tr": "TR", "en": "EN"}
_lang = "tr"


def set_lang(lang: str):
    global _lang
    _lang = lang if lang in LANGS else "tr"


def get_lang() -> str:
    return _lang


def t(key: str, **kw) -> str:
    s = S[key][0 if _lang == "tr" else 1]
    return s.format(**kw) if kw else s


# ------------------------------------------------------------------ sınıf, grup, bölge adları
PLAIN_NAMES_EN = {
    "AKIEC": "Sun-damage spot (can become cancer)",
    "BCC": "Slow-growing skin cancer",
    "BEN_OTH": "Other benign growths",
    "BKL": "Benign spot / wart-like growth",
    "DF": "Small firm bump under the skin",
    "INF": "Inflammation or infection",
    "MAL_OTH": "Other malignant growths",
    "MEL": "Melanoma (serious skin cancer)",
    "NV": "Mole",
    "SCCKA": "Scaly skin cancer",
    "VASC": "Blood-vessel growth",
}
CLASS_NAMES_EN = {
    "AKIEC": "Actinic keratosis / intraepidermal carcinoma",
    "BCC": "Basal cell carcinoma",
    "BEN_OTH": "Other benign proliferations",
    "BKL": "Benign keratinocytic lesion",
    "DF": "Dermatofibroma",
    "INF": "Inflammatory and infectious conditions",
    "MAL_OTH": "Other malignant proliferations",
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "SCCKA": "Squamous cell carcinoma / keratoacanthoma",
    "VASC": "Vascular lesions and hemorrhage",
}
DESCRIPTIONS_EN = {
    "AKIEC": "A rough, scaly patch on long-term sun-exposed skin. It can turn into cancer over time, so it needs follow-up and treatment.",
    "BCC": "The most common skin cancer. It grows slowly and rarely spreads, but it needs treatment.",
    "BEN_OTH": "Harmless skin growths that do not fit the other categories.",
    "BKL": "Harmless brown spots and wart-like bumps that usually appear with age.",
    "DF": "A small, harmless bump that feels firm to the touch, most often on the legs.",
    "INF": "Not cancer. Indicates an inflammatory or infection-related skin problem.",
    "MAL_OTH": "Rarer malignant skin growths that do not fit the other categories.",
    "MEL": "A skin cancer that can develop from a mole or normal skin and has a high risk of spreading. Early detection is very important.",
    "NV": "An ordinary, harmless mole.",
    "SCCKA": "A scaly skin cancer that can grow, usually on sun-exposed areas.",
    "VASC": "A red-purple growth made of blood vessels, usually harmless.",
}
GROUP_LABELS = {
    MALIGN: ("Kötü huylu", "Malignant"),
    PREMALIGN: ("Kansere dönüşebilir", "Precancerous"),
    BENIGN: ("İyi huylu", "Benign"),
}
SITES_EN = {
    "head/neck": "Head and neck",
    "upper extremity": "Arm and hand (incl. shoulder)",
    "lower extremity": "Leg and foot (incl. hip)",
    "anterior torso": "Chest and abdomen",
    "posterior torso": "Back",
    "lateral torso": "Side of the torso",
    "palms/soles": "Palm or sole",
    "oral/genital": "Mouth or genital area",
}
SKIN_TONE_NAMES = {
    0: ("Çok koyu", "Very dark"),
    1: ("Koyu", "Dark"),
    2: ("Orta koyu", "Medium-dark"),
    3: ("Orta", "Medium"),
    4: ("Açık", "Light"),
    5: ("Çok açık", "Very light"),
}
SEX_NAMES = {"female": ("Kadın", "Female"), "male": ("Erkek", "Male")}


def _pick(pair):
    return pair[0] if _lang == "tr" else pair[1]


def plain(c):
    return PLAIN_NAMES[c] if _lang == "tr" else PLAIN_NAMES_EN[c]


def cname(c):
    return CLASS_NAMES[c] if _lang == "tr" else CLASS_NAMES_EN[c]


def desc(c):
    return DESCRIPTIONS[c] if _lang == "tr" else DESCRIPTIONS_EN[c]


def group(g):
    return _pick(GROUP_LABELS[g])


def site(k):
    return SITES[k] if _lang == "tr" else SITES_EN[k]


def tone(i):
    return _pick(SKIN_TONE_NAMES[i])


def sex(k):
    return _pick(SEX_NAMES[k])


def pct(p: float) -> str:
    return f"%{p * 100:.1f}" if _lang == "tr" else f"{p * 100:.1f}%"


# ------------------------------------------------------------------ metinler: anahtar -> (tr, en)
S = {
    # üst bar ve sekmeler
    "app_name": ("Cilt Lezyonu Karar Destek Sistemi", "Skin Lesion Decision-Support System"),
    "app_sub": ("BAU Endüstri Mühendisliği · Bitirme Projesi", "BAU Industrial Engineering · Capstone Project"),
    "chip_demo": ("Demo modu · sahte model", "Demo mode · mock model"),
    "chip_version": ("Sürüm {v}", "Version {v}"),
    "chip_data": ("Veri: MILK10k", "Data: MILK10k"),
    "lang_label": ("Dil", "Language"),
    "tab_analyze": ("Analiz", "Analysis"),
    "tab_about": ("Model hakkında", "About the model"),
    "tab_guide": ("Kullanım kılavuzu", "User guide"),

    # başlık ve uyarılar
    "hero_title": ("Lezyon analizi", "Lesion analysis"),
    "hero_sub": ('Aynı lezyona ait klinik fotoğraf ve dermoskopik görüntüden 11 lezyon türü için olasılık hesaplar. MILK10k veri setiyle geliştirilen bir araştırma çalışmasıdır.',
                'Estimates probabilities for 11 lesion types from a clinical photo and a dermoscopic image of the same lesion. A research project developed with the MILK10k dataset.'),
    "notice_nodx": (
        "<b>Bu araç tanı koymaz.</b> Araştırma ve eğitim amaçlı bir prototiptir; çıktıları bir hekimin değerlendirmesinin "
        "yerine geçmez. Şüpheli bir lezyon için mutlaka bir hekime başvurun.",
        "<b>This tool does not make a diagnosis.</b> It is a research and education prototype; its output does not replace "
        "a physician's assessment. Always consult a physician about a suspicious lesion.",
    ),
    "notice_demo": (
        "<b>Demo modu.</b> Şu an gerçek model bağlı değil; görünen olasılıklar rastgele üretilir ve hiçbir anlam taşımaz. "
        "Arayüz denemesi içindir.",
        "<b>Demo mode.</b> The real model is not connected yet; the probabilities shown are random and carry no meaning. "
        "For interface testing only.",
    ),

    # 1. görüntüler
    "s1_title": ("Görüntüler", "Images"),
    "s1_desc": (
        "Aynı lezyona ait iki görüntüyü de yükleyin. İkisi birbirini tamamlar; model ikisini birlikte kullanır.",
        "Upload both images of the same lesion. They complement each other; the model uses both together.",
    ),
    "clin_pill": ("Fotoğraf", "Photo"),
    "clin_title": ("Klinik yakın çekim", "Clinical close-up"),
    "clin_text": (
        "Lezyonun normal bir kamerayla veya telefonla, yaklaşık 10–30 cm mesafeden çekilmiş yakın fotoğrafı. "
        "Lezyonun rengini, şeklini, kenarlarını ve çevresindeki deriyi gösterir.",
        "A close-up of the lesion taken with a regular camera or phone from about 10–30 cm. "
        "Shows its colour, shape, borders and the surrounding skin.",
    ),
    "derm_pill": ("Büyütmeli görüntü", "Magnified image"),
    "derm_title": ("Dermoskopik görüntü", "Dermoscopic image"),
    "derm_text": (
        "Dermatoskop adı verilen ışıklı bir büyüteçle çekilen görüntü. Çıplak gözle görünmeyen renk desenlerini, "
        "damarları ve deri yüzeyinin hemen altındaki ayrıntıları gösterir; genellikle yuvarlak bir çerçevesi vardır.",
        "An image taken with a dermatoscope, a lighted magnifier. Shows colour patterns, vessels and details just below "
        "the skin surface that are invisible to the naked eye; usually has a round frame.",
    ),
    "read_error": ("Bu dosya okunamadı. Lütfen geçerli bir JPG veya PNG yükleyin.", "This file could not be read. Please upload a valid JPG or PNG."),
    "swap_note": ("Yer değiştirildi: diğer kutuya yüklenen görüntü burada kullanılıyor.", "Swapped: the image uploaded in the other box is used here."),
    "lens_caption": ("Önizleme (mercek çerçevesi yalnızca gösterim içindir)", "Preview (the lens frame is for display only)"),
    "reversed_warn": (
        "**Görüntüler ters yüklenmiş olabilir.** Klinik yakın çekim kutusundaki görüntü dermoskopik bir görüntüye benziyor "
        "(koyu, yuvarlak çerçeve). Kontrol edin; gerekiyorsa aşağıdaki düğmeyle yerlerini değiştirin.",
        "**The images may be in the wrong boxes.** The image in the clinical close-up box looks dermoscopic "
        "(dark, round frame). Please check and swap them with the button below if needed.",
    ),
    "swap_btn": ("Görüntülerin yerini değiştir", "Swap images"),
    "unswap_btn": ("Görüntülerin yerini geri al", "Undo swap"),
    "img_help_title": ("İki görüntü arasındaki fark nedir, nasıl çekilmeli?", "What is the difference between the two images, and how should they be taken?"),
    "img_help_body": (
        """
| | Klinik yakın çekim | Dermoskopik görüntü |
|---|---|---|
| Nasıl çekilir | Telefon veya kamera, 10–30 cm | Dermatoskop (ışıklı büyüteç), deriye temas ederek veya yakından |
| Ne gösterir | Renk, şekil, kenar, lezyonun konumu ve çevre deri | Renk desenleri, damarlar, yüzeyin hemen altındaki ayrıntılar |
| Kimler çeker | Herkes | Genellikle hekim veya eğitimli sağlık çalışanı |

**İyi bir görüntü için:** Lezyon net ve odakta olmalı, ışık eşit olmalı (gölge ve parlama olmamalı), lezyon kadrajın ortasında yer almalı.
Filtre, kırpma veya renk düzeltmesi uygulanmış görüntüleri kullanmayın.
""",
        """
| | Clinical close-up | Dermoscopic image |
|---|---|---|
| How it is taken | Phone or camera, 10–30 cm | Dermatoscope (lighted magnifier), in contact or very close |
| What it shows | Colour, shape, border, location and surrounding skin | Colour patterns, vessels, details just below the surface |
| Who takes it | Anyone | Usually a physician or trained health worker |

**For a good image:** the lesion should be sharp and in focus, lighting should be even (no shadows or glare), and the lesion should be centred.
Do not use filtered, cropped or colour-corrected images.
""",
    ),

    # kırpma
    "crop_toggle": ("Kırp", "Crop"),
    "crop_help": ("Görüntüde yalnızca lezyonun bulunduğu bölgeyi seçmek için açın. Kapatınca görüntünün tamamı kullanılır.",
                  "Turn on to keep only the area around the lesion. Turn off to use the whole image."),
    "crop_hint": ("Kutuyu sürükleyip köşelerinden boyutlandırın. Analizde yalnızca kutunun içi kullanılır.",
                  "Drag the box and resize it by its corners. Only the area inside the box is analysed."),

    # görüntü kalitesi
    "q_res": ("Çözünürlük", "Resolution"),
    "q_sharp": ("Netlik", "Sharpness"),
    "q_light": ("Işık", "Lighting"),
    "q_res_low": ("; en az 300 piksel önerilir", "; at least 300 px recommended"),
    "q_sharp_ok": ("net", "sharp"),
    "q_sharp_bad": ("bulanık olabilir; odak kontrol edilmeli", "may be blurry; check the focus"),
    "q_light_ok": ("uygun", "good"),
    "q_dark": ("çok karanlık", "too dark"),
    "q_bright": ("aşırı parlak", "overexposed"),
    "q_warn": ("Görüntü kalitesi sonucu etkileyebilir.", "Image quality may affect the result."),

    # 2. hasta bilgileri
    "s2_title": ("Hasta bilgileri", "Patient information"),
    "s2_desc": ('Bu bilgiler rapora ve vaka kaydına girer, az veri uyarıları için kullanılır. Mevcut model sonucu yalnızca görüntülerden hesaplar (hasta bilgisini eklemek başarıyı artırmadı).',
                'This information goes into the report and case record and is used for low-data warnings. The current model computes the result from the images only (adding patient information did not improve accuracy).'),
    "age_label": ("Yaş aralığı", "Age range"),
    "age_ph": ("Yaş aralığını seçin", "Select age range"),
    "age_help": ("Veri seti yaşı 5 yıllık aralıklarla tutar.", "The dataset records age in 5-year ranges."),
    "age_fmt": ("{b} yaş", "{b} years"),
    "sex_label": ("Cinsiyet", "Sex"),
    "skin_label": ("Ten rengi", "Skin tone"),
    "skin_help": ("0 en koyu, 5 en açık ten rengidir. Renk örnekleri yalnızca fikir vermek içindir.",
                  "0 is the darkest and 5 the lightest skin tone. The swatches are only a rough guide."),
    "site_label": ("Lezyonun bulunduğu bölge", "Body site of the lesion"),
    "site_ph": ("Vücut bölgesini seçin", "Select body site"),
    "low_tone": (
        "Eğitim verisinde bu ten rengi grubundan yalnızca **{n} lezyon** var. Bu grupta sonuçlar daha az güvenilir olabilir; "
        "hekim değerlendirmesine öncelik verin.",
        "The training data contains only **{n} lesions** from this skin-tone group. Results may be less reliable here; "
        "prioritise a physician's assessment.",
    ),
    "age_out": ("Eğitim verisinde yaş {a}–{b} arasındadır. Bu yaş aralığında sonuçlar daha belirsizdir.",
                "Ages in the training data range from {a} to {b}. Results are less certain outside this range."),
    "missing": ("Analiz için eksik: {items}.", "Missing for analysis: {items}."),
    "m_clin": ("klinik yakın çekim", "clinical close-up"),
    "m_derm": ("dermoskopik görüntü", "dermoscopic image"),
    "m_age": ("yaş aralığı", "age range"),
    "m_sex": ("cinsiyet", "sex"),
    "m_skin": ("ten rengi", "skin tone"),
    "m_site": ("bölge", "body site"),
    "analyze": ("Analiz et", "Analyse"),
    "new_patient": ("Yeni hasta", "New patient"),
    "new_patient_help": ("Tüm görüntüleri, bilgileri ve sonucu temizler.", "Clears all images, inputs and the result."),

    # 3. sonuç
    "s3_title": ("Sonuç", "Result"),
    "s3_desc": ("Model 11 lezyon türünün her biri için bir olasılık hesaplar. Önce genel değerlendirme, sonra en olası üç tür gösterilir.",
                "The model estimates a probability for each of the 11 lesion types. The overall assessment comes first, then the three most likely types."),
    "empty": ("İki görüntüyü yükleyin, hasta bilgilerini seçin ve <b>Analiz et</b> düğmesine basın. Sonuç burada görünür.",
              "Upload both images, fill in the patient information and press <b>Analyse</b>. The result will appear here."),
    "stale": ("**Görüntüler veya hasta bilgileri değişti.** Aşağıdaki sonuç önceki bilgilere aittir. Güncel sonuç için yeniden **Analiz et** düğmesine basın.",
              "**The images or patient information have changed.** The result below belongs to the previous inputs. Press **Analyse** again for an updated result."),
    "demo_caption": ("Demo modu: aşağıdaki sonuçlar rastgeledir.", "Demo mode: the results below are random."),
    "case_no": ("Vaka no", "Case no"),
    "r_urgent_t": ("Öncelikli dermatolog değerlendirmesi önerilir", "Priority dermatology review recommended"),
    "r_urgent": (
        "Model bu görüntülerde kötü huylu oluşumlara benzeyen özellikler görüyor (kötü huylu türlerin toplam olasılığı {p}). "
        "Lezyonun kısa süre içinde bir dermatolog tarafından incelenmesi önerilir.",
        "The model sees features resembling malignant growths in these images (combined probability of malignant types {p}). "
        "A dermatologist should examine the lesion soon.",
    ),
    "r_refer_t": ("Dermatolog değerlendirmesi önerilir", "Dermatology review recommended"),
    "r_refer": (
        "Model bu görüntülerde kansere dönüşebilen veya kötü huylu oluşumlara benzer özellikler görüyor (toplam olasılık {p}). "
        "Lezyonun bir dermatolog tarafından incelenmesi önerilir.",
        "The model sees features resembling precancerous or malignant growths (combined probability {p}). "
        "A dermatologist should examine the lesion.",
    ),
    "r_unsure_t": ("Model kararsız: hekim değerlendirmesi önerilir", "Model uncertain: physician review recommended"),
    "r_unsure": (
        "Model belirgin bir kötü huylu özellik görmüyor, ancak sonuçtan emin değil. Bu nedenle lezyonun bir hekim tarafından "
        "değerlendirilmesi önerilir.",
        "The model does not see clear malignant features but is not confident in its result. A physician should therefore "
        "assess the lesion.",
    ),
    "r_benign_t": ("İyi huylu bir oluşuma benziyor", "Resembles a benign growth"),
    "r_benign": (
        "Model bu görüntülerde belirgin bir kötü huylu özellik görmüyor. Yine de lezyon büyürse; rengi veya şekli değişirse; "
        "kanar, kaşınır ya da yara açarsa bir hekime başvurulmalıdır.",
        "The model does not see clear malignant features. Still, see a physician if the lesion grows, changes colour or shape, "
        "bleeds, itches or ulcerates.",
    ),
    "unsure_box": (
        "<b>Model bu görüntülerde kararsız.</b> En olası sonuçlar birbirine çok yakın veya olasılıkları düşük. Aşağıdaki "
        "sıralamaya güvenmeyin; görüntü kalitesini kontrol edin ve hekim değerlendirmesine öncelik verin.",
        "<b>The model is uncertain about these images.</b> The top results are very close or have low probabilities. Do not "
        "rely on the ranking below; check image quality and prioritise a physician's assessment.",
    ),
    "top3": ("En olası üç tür", "Three most likely types"),
    "medical_name": ("Tıbbi adı", "Medical name"),
    "all_results": ("Tüm sonuçlar (11 tür)", "All results (11 types)"),
    "col_plain": ("Sade ad", "Plain name"),
    "col_medical": ("Tıbbi adı", "Medical name"),
    "col_group": ("Grup", "Group"),
    "col_prob": ("Olasılık", "Probability"),
    "dl_pdf": ("Raporu indir (PDF)", "Download report (PDF)"),
    "dl_pdf_help": ("Vaka numarası, hasta bilgileri, görüntüler ve sonuçları içeren tek sayfalık özet.",
                    "A one-page summary with the case number, patient information, images and results."),
    "tech": ("Teknik ayrıntılar", "Technical details"),
    "tech_line": ("Model: {m} · Hesaplama süresi {s:.2f} sn ({ok}) · Analiz zamanı {c}",
                  "Model: {m} · Inference time {s:.2f} s ({ok}) · Analysed at {c}"),
    "tech_ok": ("10 sn hedefinin altında", "below the 10 s target"),
    "tech_slow": ("10 sn hedefinin üzerinde", "above the 10 s target"),
    "tech_thr": (
        "Yönlendirme eşikleri (geçici): kötü huylu toplamı ≥ %{u} → öncelikli; kötü huylu + kansere dönüşebilir ≥ %{r} → "
        "değerlendirme. Kararsızlık: en yüksek olasılık < %{t} veya ilk iki fark < %{g}.",
        "Referral thresholds (provisional): malignant total ≥ {u}% → priority; malignant + precancerous ≥ {r}% → review. "
        "Uncertain: top probability < {t}% or gap between the top two < {g}%.",
    ),
    "history": ("Bu oturumdaki analizler ({n})", "Analyses in this session ({n})"),
    "history_note": ("Yalnızca bu sayfa açıkken tutulur; hiçbir yere kaydedilmez. Görüntüler listede saklanmaz.",
                     "Kept only while this page is open; nothing is stored. Images are not kept in the list."),
    "h_case": ("Vaka no", "Case no"),
    "h_time": ("Saat", "Time"),
    "h_patient": ("Hasta", "Patient"),
    "h_top": ("En olası tür", "Most likely type"),
    "h_level": ("Değerlendirme", "Assessment"),
    "lvl_urgent": ("Öncelikli", "Priority"),
    "lvl_refer": ("Değerlendirme", "Review"),
    "lvl_benign": ("İyi huylu görünüm", "Benign appearance"),
    "tone_short": ("ten", "tone"),

    # menü ve sayfa düzeni
    "nav_new": ("Yeni analiz", "New analysis"),
    "nav_track": ("Vaka takibi", "Case tracking"),
    "side_nodx": ("Bu araç tanı koymaz; sonuçları hekim değerlendirmesinin yerine geçmez.",
                  "This tool does not diagnose; its results do not replace a physician's assessment."),
    "new_title": ("Yeni analiz", "New analysis"),
    "new_sub": ("Üç adım: görüntüleri yükleyin, hasta bilgilerini girin, sonucu inceleyin.",
                "Three steps: upload the images, enter patient information, review the result."),
    "banner": ("<b>Tanı koymaz.</b> Sonuçlar hekim değerlendirmesinin yerine geçmez.",
               "<b>Not a diagnosis.</b> Results do not replace a physician's assessment."),
    "banner_demo": (" <b>Demo modu:</b> gerçek model henüz bağlı değil, sonuçlar rastgeledir.",
                    " <b>Demo mode:</b> the real model is not connected yet; results are random."),
    "btn_next": ("İleri: Hasta bilgileri", "Next: Patient information"),
    "btn_back": ("Geri", "Back"),
    "btn_change": ("Değiştir", "Replace"),
    "btn_edit": ("Düzenle", "Edit"),
    "btn_new": ("Yeni analiz", "New analysis"),
    "btn_open": ("Aç", "Open"),
    "btn_back_list": ("Vaka listesine dön", "Back to case list"),
    "need_both": ("Devam etmek için iki görüntüyü de yükleyin.", "Upload both images to continue."),
    "s2_images": ("Yüklenen görüntüler", "Uploaded images"),

    # vaka
    "case_hint": ("Bu numarayı not edin. <b>Vaka takibi</b> sayfasından bu numarayla vakayı tekrar açabilirsiniz.",
                  "Note this number. You can reopen the case with it on the <b>Case tracking</b> page."),
    "status_label": ("Vaka durumu", "Case status"),
    "stat_new": ("Analiz edildi", "Analysed"),
    "stat_reviewed": ("Hekim değerlendirdi", "Reviewed by physician"),
    "stat_awaiting": ("Sonuç bekleniyor", "Awaiting result"),
    "stat_closed": ("Kapatıldı", "Closed"),
    "timeline": ("Vaka geçmişi", "Case history"),
    "ev_created": ("Analiz yapıldı", "Analysed"),
    "ev_feedback": ("Hekim değerlendirmesi eklendi", "Physician assessment added"),
    "ev_feedback_upd": ("Hekim değerlendirmesi güncellendi", "Physician assessment updated"),
    "ev_status": ("Durum: {s}", "Status: {s}"),
    "fb_section": ("Hekim değerlendirmesi ve vaka durumu", "Physician assessment and case status"),

    # vaka takibi
    "track_title": ("Vaka takibi", "Case tracking"),
    "track_lead": ("Analiz edilen vakaları vaka numarasıyla bulun, hekim değerlendirmesi ekleyin ve durumunu güncelleyin.",
                   "Find analysed cases by case number, add a physician assessment and update their status."),
    "track_search": ("Vaka numarası ile ara", "Search by case number"),
    "track_ph": ("Örn. LZ-261120-7K4QM", "e.g. LZ-261120-7K4QM"),
    "track_filter": ("Durum", "Status"),
    "track_all": ("Tümü", "All"),
    "track_empty": ("Henüz vaka yok. <b>Yeni analiz</b> sayfasından ilk vakayı oluşturun.",
                    "No cases yet. Create the first one on the <b>New analysis</b> page."),
    "track_none": ("Bu aramayla eşleşen vaka bulunamadı.", "No case matches this search."),
    "track_note": ("Vakalar şimdilik yalnızca bu sayfa açıkken tutulur; veritabanı eklendiğinde kalıcı olacak.",
                   "For now, cases are kept only while this page is open; they will be stored permanently once the database is added."),
    "track_count": ("{n} vaka", "Cases: {n}"),

    # takip, kapatma, sıralama, dışa aktarma, karşılaştırma, QR
    "follow_date": ("Kontrol tarihi (takip planlandıysa)", "Follow-up date (if follow-up is planned)"),
    "follow_help": ("Vaka takibi sayfasında yaklaşan ve geciken kontroller bu tarihe göre listelenir.",
                    "Upcoming and overdue follow-ups on the Case tracking page are listed by this date."),
    "fu_title": ("Kontrol takvimi", "Follow-up schedule"),
    "fu_overdue": ("Gecikmiş", "Overdue"),
    "fu_upcoming": ("Yaklaşıyor", "Upcoming"),
    "fu_later": ("Planlandı", "Scheduled"),
    "fu_late": ("{n} gün gecikti", "{n} days overdue"),
    "fu_left": ("{n} gün kaldı", "in {n} days"),
    "fu_today": ("bugün", "today"),
    "fu_label": ("Kontrol tarihi", "Follow-up date"),
    "close_title": ("Vakayı kapat (kesin tanı)", "Close case (final diagnosis)"),
    "close_intro": ("Biyopsi veya takip sonucunda kesin tanı belli olunca girin. Vaka kapatılır ve modelin başarısı bu tanıya "
                    "göre ölçülür.",
                    "Enter this once the final diagnosis is known from biopsy or follow-up. The case is closed and the "
                    "model is evaluated against this diagnosis."),
    "final_dx": ("Kesin tanı", "Final diagnosis"),
    "final_method": ("Tanı nasıl doğrulandı?", "How was it confirmed?"),
    "m_histo": ("Histopatoloji (biyopsi)", "Histopathology (biopsy)"),
    "m_clinical": ("Klinik takip", "Clinical follow-up"),
    "m_consensus": ("Uzman görüş birliği", "Expert consensus"),
    "close_btn": ("Vakayı kapat", "Close case"),
    "closed_info": ("Kesin tanı: {dx} · {m}", "Final diagnosis: {dx} · {m}"),
    "reopen": ("Vakayı yeniden aç", "Reopen case"),
    "ev_closed": ("Vaka kapatıldı · kesin tanı: {dx}", "Case closed · final diagnosis: {dx}"),
    "ev_reopened": ("Vaka yeniden açıldı", "Case reopened"),
    "model_right": ("Model ilk tahmininde doğru", "Model's top prediction was correct"),
    "model_top3": ("Doğru tanı modelin ilk 3 tahmini içinde", "Correct diagnosis was in the model's top 3"),
    "model_wrong": ("Doğru tanı modelin ilk 3 tahmininde yok", "Correct diagnosis was not in the model's top 3"),
    "export_btn": ("Excel'e aktar (CSV)", "Export to Excel (CSV)"),
    "export_help": ("Bütün vakaları tek bir tabloya indirir. Excel ile açılır; görüntü ve kişisel bilgi içermez.",
                    "Downloads all cases as one table. Opens in Excel; contains no images or personal information."),
    "sort_label": ("Sıralama", "Sort"),
    "sort_priority": ("Önceliğe göre", "By priority"),
    "sort_date": ("En yeni önce", "Newest first"),
    "cmp_title": ("Vakaları karşılaştır", "Compare cases"),
    "cmp_intro": ("Aynı lezyonun farklı zamanlardaki iki analizini yan yana görün. Vakaların aynı lezyona ait olduğuna hekim "
                  "karar verir; sistemde hasta kimliği tutulmaz.",
                  "View two analyses of the same lesion at different times side by side. The physician decides that the "
                  "cases belong to the same lesion; no patient identity is stored."),
    "cmp_a": ("Önceki vaka", "Earlier case"),
    "cmp_b": ("Sonraki vaka", "Later case"),
    "cmp_btn": ("Karşılaştır", "Compare"),
    "cmp_need2": ("Karşılaştırmak için en az iki vaka gerekir.", "At least two cases are needed to compare."),
    "cmp_same": ("İki farklı vaka seçin.", "Select two different cases."),
    "cmp_change": ("Değişim", "Change"),
    "cmp_up": ("arttı", "increased"),
    "cmp_down": ("azaldı", "decreased"),
    "cmp_flat": ("değişmedi", "unchanged"),
    "cmp_malign": ("Kötü huylu türlerin toplam olasılığı", "Combined probability of malignant types"),
    "cmp_top_changed": ("En olası tür değişti: {a} → {b}", "Most likely type changed: {a} → {b}"),
    "cmp_top_same": ("En olası tür aynı kaldı: {a}", "Most likely type unchanged: {a}"),
    "cmp_level_changed": ("Değerlendirme değişti: {a} → {b}", "Assessment changed: {a} → {b}"),
    "cmp_close": ("Karşılaştırmayı kapat", "Close comparison"),
    "cmp_from_case": ("Karşılaştır", "Compare"),
    "cmp_note": ("Fark ±5 puandan küçükse “değişmedi” sayılır.", "Differences smaller than ±5 points count as “unchanged”."),
    "pdf_qr": ("Vakayı açmak için okutun", "Scan to open the case"),
    "case_not_here": ("<b>{c}</b> numaralı vaka bu oturumda bulunamadı. Vakalar şimdilik yalnızca analizin yapıldığı tarayıcıda "
                      "tutulur; veritabanı eklendiğinde her cihazdan açılabilecek.",
                      "Case <b>{c}</b> was not found in this session. For now, cases are kept only in the browser where the "
                      "analysis was done; once the database is added they will open from any device."),
    "st_final_title": ("Kesin tanıya göre model başarısı", "Model performance against final diagnosis"),
    "st_final_none": ("Henüz kapatılmış (kesin tanısı girilmiş) vaka yok.", "No closed cases (with a final diagnosis) yet."),
    "st_closed": ("kapatılan vaka", "closed cases"),
    "st_top1": ("ilk tahmin doğru", "top prediction correct"),
    "st_top3": ("doğru tanı ilk 3 içinde", "correct diagnosis in top 3"),
    "st_sens": ("kötü huylu vakalar sevk edildi", "malignant cases referred"),
    "st_final_note": ("Yalnızca kesin tanısı girilmiş vakalar sayılır. “Kötü huylu vakalar sevk edildi”: kesin tanısı kötü huylu "
                      "olan vakalarda sistemin “öncelikli” veya “değerlendirme” önermesi (kaçırılan kanser olmaması).",
                      "Only cases with a final diagnosis are counted. “Malignant cases referred”: for cases whose final "
                      "diagnosis is malignant, the system suggested “priority” or “review” (no missed cancer)."),

    # kullanım onayı
    "consent_title": ("Başlamadan önce", "Before you start"),
    "consent_body": (
        "Bu sistem, **BAU Endüstri Mühendisliği bitirme projesi** kapsamında geliştirilmiş bir **araştırma prototipidir**.\n\n"
        "- **Tanı koymaz.** Sonuçlar yalnızca karar desteğidir; hekim değerlendirmesinin yerine geçmez.\n"
        "- **Tıbbi cihaz değildir** ve klinik kullanım için onaylanmamıştır.\n"
        "- Yüklenen görüntüler **kaydedilmez**; ad, kimlik numarası gibi kişisel bilgi istenmez ve girilmemelidir. İzin verirseniz yalnızca **anonim sonuçlar** (yaş aralığı, olasılıklar, hekim değerlendirmesi vb.) araştırma amacıyla saklanır.\n"
        "- Model emin olmadığında bunu açıkça belirtir. Şüpheli bir lezyon için mutlaka bir hekime başvurun.",
        "This system is a **research prototype** developed as part of a **BAU Industrial Engineering capstone project**.\n\n"
        "- **It does not make a diagnosis.** Results are decision support only and do not replace a physician's assessment.\n"
        "- **It is not a medical device** and is not approved for clinical use.\n"
        "- Uploaded images are **not stored**; no personal information such as name or ID number is requested, and none should be entered. With your consent, only **anonymous results** (age range, probabilities, physician assessment, etc.) are kept for research.\n"
        "- When the model is not confident, it says so. Always consult a physician about a suspicious lesion.",
    ),
    "consent_check": ("Okudum ve anladım", "I have read and understood this"),
    "consent_btn": ("Devam et", "Continue"),

    # görüntü uygunluk kontrolü
    "q_skin": ("Cilt görüntüsü", "Skin image"),
    "q_skin_ok": ("cilt fotoğrafına benziyor", "looks like a skin photo"),
    "q_skin_bad": ("cilt fotoğrafına benzemiyor", "does not look like a skin photo"),
    "skin_warn": ("**Bu görüntü bir cilt fotoğrafına benzemiyor.** Ekran görüntüsü, belge veya başka bir nesne yüklenmiş "
                  "olabilir; böyle bir görüntünün sonucu anlamsız olur. Doğru dosyayı yüklediğinizden emin olun.",
                  "**This image does not look like a skin photo.** A screenshot, document or another object may have been "
                  "uploaded; the result for such an image would be meaningless. Make sure you uploaded the right file."),
    "skin_confirm": ("Görüntünün doğru olduğundan eminim, yine de devam et", "I am sure the image is correct, continue anyway"),
    "skin_block": ("Devam etmek için şüpheli görüntüyü değiştirin ya da doğru olduğunu onaylayın.",
                   "To continue, replace the suspicious image or confirm that it is correct."),

    # hekim modu
    "doc_login": ("Hekim girişi", "Physician login"),
    "doc_pin": ("PIN", "PIN"),
    "doc_initials": ("Baş harfleriniz (ör. Z.Ö.Y.)", "Your initials (e.g. Z.Ö.Y.)"),
    "doc_login_btn": ("Giriş yap", "Log in"),
    "doc_wrong": ("PIN hatalı. Kalan deneme: {n}", "Wrong PIN. Attempts left: {n}"),
    "doc_locked": ("Çok fazla hatalı deneme. Sayfayı yenileyip tekrar deneyin.", "Too many wrong attempts. Reload the page and try again."),
    "doc_need_initials": ("Baş harflerinizi girin.", "Enter your initials."),
    "doc_on": ("Hekim modu açık · {who}", "Physician mode on · {who}"),
    "doc_logout": ("Çıkış yap", "Log out"),
    "doc_needed": ("Hekim değerlendirmesi, vaka durumu ve vakayı kapatma yalnızca <b>hekim girişi</b> ile yapılabilir. "
                   "Sol menüden giriş yapın.",
                   "Physician assessment, case status and closing a case require <b>physician login</b>. "
                   "Log in from the left menu."),
    "doc_note": ("Rol ayrımını göstermek içindir; gerçek bir kimlik doğrulama sistemi değildir.",
                 "Demonstrates role separation; this is not a real authentication system."),

    # denetim kaydı
    "actor_user": ("Kullanıcı", "User"),
    "actor_doc": ("Hekim {who}", "Physician {who}"),
    "ev_pdf": ("PDF rapor indirildi", "PDF report downloaded"),
    "ev_status_from": ("Durum değişti: {a} → {b}", "Status changed: {a} → {b}"),
    "ev_login": ("Hekim girişi yapıldı", "Physician logged in"),
    "ev_logout": ("Hekim çıkışı yapıldı", "Physician logged out"),
    "ev_login_fail": ("Hatalı PIN denemesi", "Failed PIN attempt"),
    "audit_title": ("Denetim kaydı (tüm işlemler)", "Audit log (all actions)"),
    "audit_intro": ("Bu oturumda kimin, ne zaman, hangi işlemi yaptığını gösterir. Kayıtlar değiştirilemez ve silinemez; "
                    "yalnızca yenileri eklenir.",
                    "Shows who did what and when in this session. Entries cannot be edited or deleted; new ones are only appended."),
    "audit_time": ("Zaman", "Time"),
    "audit_case": ("Vaka", "Case"),
    "audit_actor": ("Kim", "Who"),
    "audit_action": ("İşlem", "Action"),
    "audit_export": ("Denetim kaydını indir (CSV)", "Download audit log (CSV)"),
    "audit_empty": ("Henüz işlem yok.", "No actions yet."),
    "save_consent": ("Bu vakanın anonim sonuçlarının araştırma amacıyla kaydedilmesine izin veriyorum",
                     "I consent to the anonymous results of this case being stored for research"),
    "save_help": ("Kaydedilenler: vaka no, yaş aralığı, cinsiyet, ten rengi, bölge, olasılıklar, hekim değerlendirmesi ve "
                  "vaka geçmişi. Görüntüler ve kişisel bilgiler (ad, kimlik no) KAYDEDİLMEZ. İzin verilmezse vaka yalnızca bu sayfa "
                  "açıkken tutulur.",
                  "Stored: case number, age range, sex, skin tone, body site, probabilities, physician assessment and case "
                  "history. Images and personal details (name, ID number) are NOT stored. Without consent, the case is kept "
                  "only while this page is open."),
    "no_images": ("Görüntüler kaydedilmez; bu vaka veritabanından açıldığı için görüntü gösterilemiyor.",
                  "Images are not stored; this case was opened from the database, so no images can be shown."),
    "db_on": ("Veritabanı bağlı", "Database connected"),
    "save_demo_off": ("Demo modunda sonuçlar veritabanına kaydedilmez (sahte sonuçlar araştırma verisine karışmasın diye).",
                      "In demo mode results are not stored in the database (so that mock results do not mix with research data)."),
    "db_err": ("Veritabanına ulaşılamadı", "Database unreachable"),
    "db_error_detail": ("Veritabanı hatası (kayıtlar bu oturumda tutuluyor, bağlantı gelince tekrar denenecek): {e}",
                        "Database error (records are kept in this session and will be retried once connected): {e}"),
    "track_note_db": ("Kayıt izni verilen vakalar anonim olarak veritabanında saklanır ve her cihazdan vaka numarasıyla açılabilir. "
                      "İzin verilmeyen vakalar yalnızca bu sayfa açıkken tutulur.",
                      "Cases with storage consent are kept anonymously in the database and can be opened by case number from "
                      "any device. Cases without consent are kept only while this page is open."),
    "st_scope_db": ("Veritabanındaki tüm kayıtlı vakaları ve bu oturumdaki vakaları kapsar.",
                    "Covers all stored cases in the database and the cases in this session."),
    "model_load_error": ("Gerçek model yüklenemedi, demo modunda çalışılıyor. Hata: {e}",
                         "The real model could not be loaded; running in demo mode. Error: {e}"),
    "ev_consent": ("Kullanım koşulları onaylandı", "Terms of use accepted"),
    "ev_skin_override": ("Cilt uyarısına rağmen görüntü kullanıldı", "Image used despite the skin warning"),

    # analiz ilerleme göstergesi
    "prog_title": ("Analiz ediliyor…", "Analysing…"),
    "prog_1": ("Görüntüler hazırlanıyor", "Preparing the images"),
    "prog_2": ("Görüntü modelleri çalışıyor", "Running the image models"),
    "prog_3": ('Görüntü modellerinin sonuçları birleştiriliyor',
                "Combining the image models' results"),
    "prog_cam": ("Isı haritası çıkarılıyor", "Generating the heat map"),
    "prog_4": ("Sonuç ve rapor hazırlanıyor", "Preparing the result and report"),
    "prog_done": ("Analiz tamamlandı", "Analysis complete"),

    # hekim değerlendirmesi
    "fb_title": ("Hekim değerlendirmesi", "Physician assessment"),
    "fb_open": ("Hekim değerlendirmesi ekle", "Add physician assessment"),
    "fb_edit": ("Hekim değerlendirmesini düzenle", "Edit physician assessment"),
    "fb_intro": ("Lezyonu değerlendiren hekim kendi görüşünü buraya girer. Bu bilgi rapora eklenir ve modelin gerçek "
                 "kullanımdaki başarısını ölçmek için kullanılır.",
                 "The physician who examined the lesion enters their opinion here. It is added to the report and used to "
                 "measure how well the model performs in practice."),
    "fb_dx": ("Hekimin tanısı / ön tanısı", "Physician's diagnosis / working diagnosis"),
    "fb_dx_unknown": ("Henüz belli değil (biyopsi vb. bekleniyor)", "Not yet known (awaiting biopsy etc.)"),
    "fb_agree": ("Model sonucu hekim görüşüyle uyumlu mu?", "Does the model's result agree with the physician?"),
    "fb_yes": ("Uyumlu", "Agrees"),
    "fb_partial": ("Kısmen", "Partly"),
    "fb_no": ("Uyumsuz", "Disagrees"),
    "fb_action": ("Planlanan işlem", "Planned action"),
    "act_follow": ("Takip (kontrol randevusu)", "Follow-up visit"),
    "act_biopsy": ("Biyopsi", "Biopsy"),
    "act_excision": ("Cerrahi olarak çıkarma", "Surgical excision"),
    "act_treat": ("İlaç veya başka tedavi", "Medication or other treatment"),
    "act_none": ("İşlem gerekmiyor", "No action needed"),
    "fb_note": ("Not (isteğe bağlı)", "Note (optional)"),
    "fb_note_ph": ("Ad, kimlik numarası gibi kişisel bilgi yazmayın.", "Do not enter personal details such as name or ID number."),
    "fb_save": ("Değerlendirmeyi kaydet", "Save assessment"),
    "fb_saved": ("Hekim değerlendirmesi eklendi. Rapora ve istatistiklere yansıdı.",
                 "Physician assessment added. It now appears in the report and statistics."),
    "fb_session_note": ("Şimdilik yalnızca bu sayfa açıkken tutulur; veritabanı eklendiğinde kalıcı olacak.",
                        "For now it is kept only while this page is open; it will be stored permanently once the database is added."),
    "fb_summary": ("Hekim: {dx} · {agree} · {action}", "Physician: {dx} · {agree} · {action}"),
    "h_physician": ("Hekim", "Physician"),
    "pdf_physician": ("Hekim değerlendirmesi", "Physician assessment"),
    "pdf_physician_blank": ("Bu bölüm hekim tarafından doldurulur.", "To be completed by the physician."),
    "pdf_signature": ("İmza / kaşe", "Signature / stamp"),

    # istatistik sekmesi
    "tab_stats": ("İstatistikler", "Statistics"),
    "st_lead": ("Bu oturumda yapılan analizlerin ve hekim değerlendirmelerinin özeti.",
                "A summary of the analyses and physician assessments made in this session."),
    "st_scope": ("Şimdilik yalnızca bu sayfa açıkken yapılan analizleri kapsar. Veritabanı bağlandığında tüm analizleri gösterecek.",
                 "For now it covers only the analyses made while this page is open. Once the database is connected it will cover all analyses."),
    "st_demo": ("Demo modunda sonuçlar rastgele olduğu için bu sayılar yalnızca sayfanın nasıl çalıştığını gösterir.",
                "In demo mode the results are random, so these numbers only show how the page works."),
    "st_empty": ("Henüz analiz yapılmadı. <b>Analiz</b> sekmesinde bir vaka analiz edildiğinde istatistikler burada görünür.",
                 "No analyses yet. Statistics appear here once a case is analysed in the <b>Analysis</b> tab."),
    "st_total": ("analiz", "analyses"),
    "st_time": ("ortalama hesaplama süresi", "average inference time"),
    "st_fb": ("hekim değerlendirmesi", "physician assessments"),
    "st_match": ("model–hekim tanı eşleşmesi", "model–physician diagnosis match"),
    "st_levels": ("Değerlendirme dağılımı", "Assessment distribution"),
    "st_types": ("En olası tür dağılımı", "Most likely type distribution"),
    "st_count": ("Vaka", "Cases"),
    "st_fb_table": ("Hekim değerlendirmeleri", "Physician assessments"),
    "st_model_top": ("Modelin ilk tahmini", "Model's top prediction"),
    "st_match_col": ("Eşleşme", "Match"),
    "st_match_note": ("Eşleşme: modelin en olası türü ile hekimin tanısının aynı olması. Tanısı henüz belli olmayan vakalar "
                      "hesaba katılmaz.",
                      "Match: the model's most likely type is the same as the physician's diagnosis. Cases without a "
                      "diagnosis yet are not counted."),
    "st_no_fb": ("Henüz hekim değerlendirmesi girilmedi.", "No physician assessments yet."),

    # Grad-CAM
    "cam_title": ("Modelin dikkat ettiği bölgeler", "Where the model looked"),
    "cam_demo": (
        "Gerçek model bağlandığında, dermoskopik görüntünün üzerinde modelin karar verirken en çok dikkat ettiği bölgeler "
        "burada bir ısı haritası olarak gösterilecek. Demo modunda gösterilmez, çünkü sahte modelin baktığı bir yer yoktur.",
        "Once the real model is connected, a heat map over the dermoscopic image will show the regions the model relied on "
        "most. It is not shown in demo mode because the mock model does not look at the image.",
    ),
    "cam_orig": ("Dermoskopik görüntü", "Dermoscopic image"),
    "cam_heat": ("Isı haritası: {c}", "Heat map: {c}"),
    "cam_note": (
        "Kırmızı ve sarı alanlar modelin en çok dikkat ettiği yerlerdir. Bu harita lezyonun sınırını veya tanı kanıtını "
        "göstermez; yalnızca modelin kararını anlamaya yardımcı olur (Grad-CAM yöntemi).",
        "Red and yellow areas are where the model focused most. The map does not show the lesion border or diagnostic "
        "evidence; it only helps to understand the model's decision (Grad-CAM method).",
    ),

    # alt bilgi
    "foot_left": ('Veri seti: MILK10k (ISIC Archive), CC-BY-NC 4.0.<br>Bu uygulama bir tıbbi cihaz değildir ve klinik kullanım için onaylanmamıştır.<br>Görüntüler kaydedilmez; izin verilirse yalnızca anonim sonuçlar saklanır.',
                'Dataset: MILK10k (ISIC Archive), CC-BY-NC 4.0.<br>This application is not a medical device and is not approved for clinical use.<br>Images are not stored; with consent, only anonymous results are kept.'),
    "foot_right": ("BAU Endüstri Mühendisliği · Bitirme Projesi · Sürüm {v}<br>{team}<br>Danışman: Dr. Öğr. Üyesi {adv}",
                   "BAU Industrial Engineering · Capstone Project · Version {v}<br>{team}<br>Advisor: Asst. Prof. Dr. {adv}"),

    # PDF rapor
    "pdf_title": ("Cilt lezyonu analiz raporu", "Skin lesion analysis report"),
    "pdf_meta": ("Vaka no: {case}   ·   Tarih: {date}   ·   Model: {model}   ·   Sürüm {v}",
                 "Case no: {case}   ·   Date: {date}   ·   Model: {model}   ·   Version {v}"),
    "pdf_nodx": ("Bu belge tanı değildir. Araştırma ve eğitim amaçlı bir prototipin çıktısıdır; hekim değerlendirmesinin yerine geçmez.",
                 "This document is not a diagnosis. It is the output of a research and education prototype and does not replace a physician's assessment."),
    "pdf_demo": ("DEMO MODU: Bu sonuç sahte (rastgele) bir modelden üretilmiştir ve hiçbir tıbbi anlam taşımaz.",
                 "DEMO MODE: This result was produced by a mock (random) model and has no medical meaning."),
    "pdf_patient": ("Hasta bilgileri", "Patient information"),
    "pdf_images": ("Görüntüler", "Images"),
    "pdf_assess": ("Genel değerlendirme", "Overall assessment"),
    "pdf_unsure": ("Model kararsız: en olası sonuçlar birbirine yakın veya düşük olasılıklı. Bu sıralamaya güvenilmemelidir.",
                   "Model uncertain: the top results are close or have low probability. This ranking should not be relied upon."),
    "pdf_all": ("Tüm türler", "All types"),
    "pdf_footer": ("MILK10k (ISIC Archive), CC-BY-NC 4.0 · Bu uygulama bir tıbbi cihaz değildir.",
                   "MILK10k (ISIC Archive), CC-BY-NC 4.0 · This application is not a medical device."),
}
