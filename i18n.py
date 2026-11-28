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


_pm = False  # hasta modu: bazı metinlerin sade (tıbbi terimsiz) sürümü kullanılır


def set_pm(on: bool):
    global _pm
    _pm = bool(on)


def t(key: str, **kw) -> str:
    pair = PM[key] if _pm and key in PM else S[key]
    s = pair[0 if _lang == "tr" else 1]
    return s.format(**kw) if kw else s


# ------------------------------------------------------------------ sınıf, grup, bölge adları
PLAIN_NAMES_EN = {
    "AKIEC": "Actinic keratosis (rough sun-damage patch, can become cancer)",
    "BCC": "Slow-growing skin cancer",
    "BEN_OTH": "Other benign growths",
    "BKL": "Benign spot (solar lentigo, seborrhoeic keratosis)",
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
    "AKIEC": "A patch on long-term sun-exposed skin that feels rough like sandpaper, often pink-red and scaly. Not the same as a harmless flat brown \"sun spot\". It can turn into cancer over time, so it needs follow-up and treatment.",
    "BCC": "The most common skin cancer. It grows slowly and rarely spreads, but it needs treatment.",
    "BEN_OTH": "Harmless skin growths that do not fit the other categories.",
    "BKL": "Harmless flat brown spots that come with age and sun (commonly called sun or age spots) and wart-like bumps.",
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
    "crop_guide_pm": ("<b>Kutuyu lekenin üstüne getirin:</b> leke kutunun <b>ortasında</b> olsun ve kutunun yaklaşık "
                      "<b>üçte birini</b> doldursun. Arkadaki kumaş, takı ya da başka nesneler kutunun dışında kalsın. "
                      "Analizde yalnızca kutunun içi kullanılır.",
                      "<b>Place the box over the spot:</b> the spot should be in the <b>centre</b> and fill about "
                      "<b>one third</b> of the box. Keep fabric, jewellery and other objects outside the box. Only the "
                      "area inside the box is analysed."),
    "crop_confirm": ("Leke kutunun ortasında ve kutunun yaklaşık üçte birini dolduruyor",
                     "The spot is in the centre and fills about one third of the box"),
    "crop_block": ("İleri geçmek için kutuyu ayarlayıp onaylayın.", "Adjust the box and confirm it to continue."),
    "border_pm": ("<b>Sonuç sınırda.</b> Değerlendirme eşiğe çok yakın çıktı; küçük bir fotoğraf farkı sonucu "
                  "değiştirebilir. Fotoğraf uzaktan çekildiyse ya da arkada kumaş veya başka nesneler varsa, lekeyi "
                  "ortalayıp yakından, gün ışığında <b>bir kez</b> yeni fotoğrafla deneyin. İki sonuç farklı çıkarsa "
                  "<b>daha temkinli olanı</b> esas alın. Endişeniz varsa doktora başvurun.",
                  "<b>Borderline result.</b> The assessment is very close to a threshold, so a small difference in the "
                  "photo could change it. If the photo was taken from far away or shows fabric or other objects, try "
                  "<b>once</b> more with a close, centred photo in daylight. If the two results differ, go by the "
                  "<b>more cautious</b> one. If you are worried, see a doctor."),
    "border_dr": ("<b>Sınırda sonuç:</b> kötü huylu ya da riskli toplam, yönlendirme eşiğine 5 puandan yakın. "
                  "Yönlendirme değişmedi; klinik değerlendirme ve gerekirse ikinci bir görüntü önerilir.",
                  "<b>Borderline result:</b> the malignant or at-risk total is within 5 points of a referral "
                  "threshold. The referral is unchanged; clinical judgement and, if needed, a second image are advised."),
    "border_pdf": ("Sınırda sonuç", "Borderline result"),
    "appt_pdf": ("Randevu", "Appointment"),
    "appt_pdf_v": ("{w} içinde dermatoloji: MHRS uygulaması, e-Nabız veya 182. Bulamazsanız aile hekimine başvurun. "
                   "Kanama, hızlı büyüme veya şekil değişikliğinde beklemeden başvurun.",
                   "Dermatology within {w}: MHRS app, e-Nabız or phone 182 (Türkiye). Otherwise see your family doctor. "
                   "Do not wait if it bleeds, grows quickly or changes shape."),
    "appt_w_urgent": ("1–2 hafta", "1–2 weeks"),
    "appt_w_refer": ("4 hafta", "4 weeks"),
    "border_pdf_v": ("Toplam, yönlendirme eşiğine 5 puandan yakın", "Total within 5 points of a referral threshold"),
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
    "r_urgent_type": (
        "Model bu görüntülerde melanom veya skuamöz hücreli karsinom gibi hızlı ilerleyebilen türlere benzeyen özellikler görüyor "
        "(bu türlerin toplam olasılığı {p}). Lezyonun 1–2 hafta içinde bir dermatolog tarafından incelenmesi önerilir.",
        "The model sees features resembling faster-growing types such as melanoma or squamous cell carcinoma in these images "
        "(combined probability of these types {p}). A dermatologist should examine the lesion within 1–2 weeks.",
    ),
    "r_refer_t": ("Dermatolog değerlendirmesi önerilir", "Dermatology review recommended"),
    "r_refer": (
        "Model bu görüntülerde kansere dönüşebilen veya kötü huylu oluşumlara benzer özellikler görüyor (toplam olasılık {p}). "
        "Lezyonun 4 hafta içinde bir dermatolog tarafından incelenmesi önerilir.",
        "The model sees features resembling precancerous or malignant growths (combined probability {p}). "
        "A dermatologist should examine the lesion within 4 weeks.",
    ),
    "r_mel_t": ("Dermatolog değerlendirmesi önerilir (melanom dışlanamadı)", "Dermatology review recommended (melanoma not ruled out)"),
    "r_mel": ("Model lezyonu en çok iyi huylu bir türe benzetiyor, ancak melanom olasılığı ({p}) güvenlik eşiğinin üstünde. "
              "Kaçırılan melanomların çoğu ben ile karıştırıldığı için lezyonun 4 hafta içinde bir dermatolog tarafından incelenmesi önerilir.",
              "The model's closest match is a benign type, but the melanoma probability ({p}) is above the safety threshold. "
              "Because most missed melanomas were mistaken for moles, review by a dermatologist within 4 weeks is recommended."),
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
    "tech_thr_type": (
        "Yönlendirme kuralı: melanom + SCC + diğer kötü huylu olasılığı ≥ %{ft} → öncelikli (1–2 hafta); kötü huylu + kansere "
        "dönüşebilir ≥ %{r}, kararsızlık ya da melanom güvenlik kuralı → değerlendirme (4 hafta). Kararsızlık: en yüksek olasılık "
        "< %{t} veya ilk iki fark < %{g}.",
        "Referral rule: melanoma + SCC + other malignant ≥ {ft}% → priority (1–2 weeks); malignant + precancerous ≥ {r}%, "
        "uncertainty or the melanoma safety rule → review (4 weeks). Uncertain: top probability < {t}% or gap < {g}%.",
    ),
    "tech_thr": (
        "Yönlendirme eşikleri (1.048 lezyonluk doğrulama verisine göre): kötü huylu toplamı ≥ %{u} → öncelikli; kötü huylu + kansere dönüşebilir ≥ %{r} → "
        "değerlendirme. Kararsızlık: en yüksek olasılık < %{t} veya ilk iki fark < %{g}.",
        "Referral thresholds (set on 1,048 validation lesions): malignant total ≥ {u}% → priority; malignant + precancerous ≥ {r}% → review. "
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
    "btn_peek": ("Ayrıntı", "Details"),
    "btn_peek_hide": ("Gizle", "Hide"),
    "peek_open": ("Vakayı aç", "Open case"),
    "peek_fb": ("Hekim değerlendirmesi", "Physician assessment"),
    "peek_fb_none": ("Henüz yok", "Not yet"),
    "peek_prev": ("Önceki vaka", "Previous case"),
    "peek_last": ("Son işlem", "Last action"),
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
    "pm_toggle": ("Dermoskopik görüntü yok — yalnızca klinik fotoğrafla analiz et (hasta modu)",
                  "No dermoscopic image — analyse with the clinical photo only (patient mode)"),
    "pm_help": ("Dermatoskopu olmayan sağlık çalışanları için. Model yalnızca klinik fotoğrafa bakar; başarısı iki görüntülü "
                "moddan düşüktür (Macro F1 0,496, hekim modunda 0,594). Klinik modeller telefon fotoğraflarıyla da eğitildi.",
                "For health workers without a dermatoscope. The model looks at the clinical photo only; accuracy is lower than "
                "with both images (Macro F1 0.496 vs 0.594). The clinical models were also trained on phone photos."),
    "pm_card_title": ("Hasta modu", "Patient mode"),
    "pm_card_text": ("Yalnızca klinik yakın çekim kullanılacak. Mümkünse dermoskopik görüntü de ekleyin: iki görüntüyle sonuç daha "
                     "güvenilirdir.",
                     "Only the clinical close-up will be used. If possible, add a dermoscopic image too: results are more "
                     "reliable with both images."),
    "need_clin": ("Devam etmek için klinik fotoğrafı yükleyin.", "Upload the clinical photo to continue."),
    "pm_note": ("<b>Hasta modu:</b> sonuç yalnızca klinik fotoğraftan hesaplandı; iki görüntülü moda göre daha az güvenilirdir.",
                "<b>Patient mode:</b> the result was computed from the clinical photo only and is less reliable than with both images."),
    "mode_label": ("Analiz modu", "Analysis mode"),
    "mode_hekim": ("Hekim modu (iki görüntü)", "Physician mode (two images)"),
    "mode_hasta": ("Hasta modu (yalnızca klinik fotoğraf)", "Patient mode (clinical photo only)"),
    "ps_title": ("Olası tanılar (%90 güven)", "Possible diagnoses (90% confidence)"),
    "ps_one": ("Model %90 güvenle tek bir türü işaret ediyor.", "The model points to a single type with 90% confidence."),
    "ps_many": ("Gerçek tanı %90 olasılıkla bu {n} türden biridir. Liste uzadıkça model daha az emindir.",
                "With 90% probability the true diagnosis is one of these {n} types. The longer the list, the less certain the model."),
    "ps_help": ("Bu liste “conformal prediction” yöntemiyle hesaplanır: 5.240 lezyonluk doğrulamada gerçek tanı bu listenin "
                "içinde %90 oranında yer aldı. Ayırıcı tanı için bir başlangıç noktasıdır, tanı değildir.",
                "This list is computed with conformal prediction: in validation on 5,240 lesions the true diagnosis was inside "
                "the list 90% of the time. It is a starting point for differential diagnosis, not a diagnosis."),
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
"nav_eval": ("Ayrıntılı analiz", "Detailed analysis"),
"nav_survey": ("Kullanılabilirlik anketi", "Usability survey"),
"nav_health": ("Cilt sağlığı", "Skin health"),
"nav_model": ("Model", "Model"),
"nav_help": ("Yardım", "Help"),
"sub_cases": ("Vakalar", "Cases"),
"sub_overview": ("Genel bakış", "Overview"),
"sub_versions": ("Sürüm geçmişi", "Version history"),
"sub_details": ("Veri ve sınırlılıklar", "Data and limitations"),
"sub_notes": ("Sürüm notları", "Release notes"),
"sub_share": ("Siteyi paylaş", "Share the site"),
"health_sub": ("Erken fark etmek için kendi kendine muayene, uyarı işaretleri ve korunma", "Self-examination, warning signs and protection for early detection"),
"photo_tips": ("İyi fotoğraf nasıl çekilir?", "How to take a good photo"),
"kap_title": ("Hekim–model uyumu (Cohen kappa)", "Physician–model agreement (Cohen's kappa)"),
"kap_need": ("Kappa için tanısı belirtilmiş en az 5 hekim değerlendirmesi gerekir (şu an {n}).",
             "Kappa needs at least 5 physician reviews with a stated diagnosis (currently {n})."),
"kap_n": ("Karşılaştırılan vaka", "Cases compared"),
"kap_type": ("Tür düzeyinde kappa · {lab}", "Class-level kappa · {lab}"),
"kap_group": ("Grup düzeyinde kappa · {lab}", "Group-level kappa · {lab}"),
"kap_group_agree": ("Grup düzeyinde aynı karar", "Same group decision"),
"kap_note": ("Kappa, şans eseri uyumu düşerek hesaplanır: 0 = şanstan farksız, 1 = tam uyum. Tür düzeyi 11 türün aynısını, "
             "grup düzeyi kötü huylu / kansere dönüşebilir / iyi huylu kararını karşılaştırır. Yorum: Landis ve Koch (1977). "
             "Hekim tanısı kesin tanı değildir; az vakayla değer çok oynaktır.",
             "Kappa corrects for chance agreement: 0 = no better than chance, 1 = perfect. Class level compares the same of "
             "11 classes; group level compares malignant / premalignant / benign. Interpretation: Landis & Koch (1977). "
             "The physician's diagnosis is not a confirmed diagnosis; with few cases the value is very unstable."),
"kap_poor": ("şanstan farksız", "no agreement"),
"kap_slight": ("çok zayıf", "slight"),
"kap_fair": ("zayıf", "fair"),
"kap_moderate": ("orta", "moderate"),
"kap_substantial": ("iyi", "substantial"),
"kap_perfect": ("çok iyi", "almost perfect"),
"tour_title": ("Hızlı tur", "Quick tour"),
"tour1_t": ("1 · Üç adımda analiz", "1 · Analysis in three steps"),
"tour1": ("**Yeni analiz**'de önce iki görüntüyü yükleyin (klinik yakın çekim + dermoskopik). Yalnızca telefon fotoğrafı "
          "varsa **hasta modu**nu açın. Görüntüyü gerekirse **kırpın** ya da **döndürün**; site netlik, ışık ve cilt "
          "kontrolü yapar. Sonra yaş, cinsiyet, ten rengi ve bölgeyi seçip **Analiz et**'e basın.",
          "In **New analysis**, first upload both images (clinical close-up + dermoscopic). If you only have a phone photo, "
          "turn on **patient mode**. **Crop** or **rotate** if needed; the site checks sharpness, light and skin. Then choose "
          "age, sex, skin tone and site and press **Analyse**."),
"tour2_t": ("2 · Sonucu okumak", "2 · Reading the result"),
"tour2": ("Renkli kutu **yönlendirme önerisidir** (öncelikli / dermatolog değerlendirmesi / iyi huylu görünüm). Altında "
          "**%90 güvenli olası tanılar**, **her görüntünün ayrı sonucu** ve modelin baktığı yeri gösteren **ısı haritası** var. "
          "Sistem **tanı koymaz**; son karar hekimindir.",
          "The coloured box is a **routing suggestion** (priority / dermatologist review / benign appearance). Below it are "
          "the **90% possible diagnoses**, **each image's own result** and a **heat map** of where the model looked. The "
          "system **does not diagnose**; the physician decides."),
"tour3_t": ("3 · Rapor ve takip", "3 · Report and follow-up"),
"tour3": ("Her analize bir **vaka numarası** verilir. **PDF raporu** QR kodludur. Aynı lezyon tekrar çekildiğinde "
          "**\"Bu lezyonun kontrolünü başlat\"** ile vakalar bağlanır ve değişim grafikte görünür. Hekimler menüden "
          "**PIN ile giriş** yapıp değerlendirme, ABCDE ve kesin tanı girebilir.",
          "Every analysis gets a **case number**. The **PDF report** has a QR code. When the same lesion is photographed "
          "again, **\"Start a follow-up of this lesion\"** links the cases and shows the change in a chart. Physicians can "
          "**log in with a PIN** to add a review, ABCDE and the final diagnosis."),
"tour4_t": ("4 · Model ve geri bildirim", "4 · Model and feedback"),
"tour4": ("**Model** sayfası başarıyı, sürüm geçmişini ve ayrıntılı analizi (kalibrasyon, eşikler, ten rengine göre eşitlik) gösterir. "
          "**Cilt sağlığı** sayfasında kendi kendine muayene ve fotoğraf çekme ipuçları var. Sistemi denedikten sonra "
          "**Yardım → Kullanılabilirlik anketi**'ni (2 dakika) doldurursanız projeye çok yardımcı olursunuz.",
          "The **Model** page shows performance, version history and detailed analysis (calibration, thresholds, fairness "
          "by skin tone). **Skin health** has self-examination and photo tips. After trying the system, please fill in "
          "**Help → Usability survey** (2 minutes); it helps the project a lot."),
"tour_off": ("Bu turu bir daha gösterme", "Don't show this tour again"),
"tour_skip": ("Atla", "Skip"),
"tour_next": ("İleri", "Next"),
"tour_done": ("Başla", "Start"),
"tour_again": ("Hızlı turu tekrar göster", "Show the quick tour again"),
"abcde_label": ("ABCDE bulguları (işaretleyin)", "ABCDE findings (select)"),
"abcde_help": ("Melanom için klasik klinik kontrol listesi. Modele girmez; kayıt ve rapor içindir. Melanom dışındaki türlerde "
               "(ör. bazal hücreli karsinom) sınırlı değer taşır. Hiçbiri yoksa boş bırakın.",
               "Classic clinical checklist for melanoma. Not used by the model; for the record and report. Of limited value "
               "for other lesion types (e.g. basal cell carcinoma). Leave empty if none apply."),
"abcde_short": ("ABCDE", "ABCDE"),
"abcde_A": ("A · Asimetri", "A · Asymmetry"),
"abcde_B": ("B · Düzensiz sınır", "B · Irregular border"),
"abcde_C": ("C · Renk çeşitliliği", "C · Colour variation"),
"abcde_D": ("D · Çap > 6 mm", "D · Diameter > 6 mm"),
"abcde_E": ("E · Değişim", "E · Evolving"),
"abcde_none": ("0/5 · bulgu yok", "0/5 · none"),
"abcde_conflict": ("<b>Hekim bulguları ile model uyuşmuyor:</b> ABCDE kontrol listesi ile modelin değerlendirmesi farklı yönde. "
                   "Klinik karar hekimindir; gerekirse dermatoloğa danışın.",
                   "<b>Physician findings and the model disagree:</b> the ABCDE checklist and the model's assessment point in "
                   "different directions. The clinical decision is the physician's; consult a dermatologist if needed."),
"mod_pdf": ("Görüntü uyumu", "Image agreement"),
"mod_pdf_v": ("İki görüntü farklı sonuçlara işaret ediyor; temkinli yorumlayın", "The two images disagree; interpret with caution"),
"mod_title": ("Her görüntü ne diyor?", "What does each image say?"),
"mod_top": ("en olası: {c} {p}", "most likely: {c} {p}"),
"mod_help": ("Dermoskopik ve klinik modellerin ayrı ayrı çıktısı. Sonuç bu ikisinin eşit ağırlıklı ortalamasıdır "
             "(ardından kalibre edilir); buradaki değerler kalibrasyon öncesidir.",
             "Separate outputs of the dermoscopic and clinical models. The result is their equal-weight average "
             "(then calibrated); values here are before calibration."),
"mod_disagree": ("<b>İki görüntü farklı sonuçlara işaret ediyor.</b> Görüntülerden biri lezyonu iyi göstermiyor olabilir "
                 "(bulanık, uzak, farklı lezyon) ya da lezyon gerçekten zor. Görüntüleri kontrol edin; sonucu temkinli yorumlayın.",
                 "<b>The two images point to different results.</b> One image may not show the lesion well (blurred, far away, "
                 "a different lesion) or the lesion is genuinely difficult. Check the images and interpret with caution."),
"ev_sus": ("Kullanılabilirlik anketi dolduruldu (SUS {s})", "Usability survey submitted (SUS {s})"),
"btn_rotate": ("Döndür", "Rotate"),
"rotate_help": ("Görüntüyü saat yönünde 90° döndürür (yan çekilmiş fotoğraflar için).",
                "Rotates the image 90° clockwise (for sideways photos)."),
"prev_label": ("Önceki vaka no (aynı lezyonun kontrolü, isteğe bağlı)", "Previous case no. (follow-up of the same lesion, optional)"),
"prev_help": ("Bu lezyon daha önce analiz edildiyse o vakanın numarasını yazın. Vakalar birbirine bağlanır ve lezyonun "
              "zaman içindeki değişimi grafikte görünür.",
              "If this lesion was analysed before, enter that case number. The cases are linked and the lesion's change over "
              "time is shown in a chart."),
"prev_found": ("✓ Bağlanacak vaka: {d} · en olası: {top}", "✓ Will be linked to: {d} · most likely: {top}"),
"prev_missing": ("{c} bulunamadı. Numarayı kontrol edin ya da alanı boş bırakın.",
                 "{c} was not found. Check the number or leave the field empty."),
"m_prev": ("geçerli önceki vaka no", "a valid previous case no."),
"prev_info": ("Bu vaka, aynı lezyonun önceki vakasına bağlı: {c}", "This case is linked to the same lesion's previous case: {c}"),
"ev_linked": ("Önceki vakaya bağlandı: {c}", "Linked to previous case: {c}"),
"btn_followup": ("Bu lezyonun kontrolünü başlat", "Start a follow-up of this lesion"),
"btn_followup_help": ("Yeni analiz açılır ve bu vaka \"önceki vaka\" olarak bağlanır.",
                      "Opens a new analysis linked to this case as the previous case."),
"lt_title": ("Lezyon zaman çizelgesi", "Lesion timeline"),
"lt_mal": ("Kötü huylu toplam", "Malignant total"),
"lt_risk": ("Kötü huylu + kansere dönüşebilir", "Malignant + premalignant"),
"lt_date": ("Tarih", "Date"),
"lt_value": ("Olasılık", "Probability"),
"lt_up": ("arttı", "increased"),
"lt_down": ("azaldı", "decreased"),
"lt_flat": ("belirgin değişmedi", "did not change much"),
"lt_note": ("{n} analiz ({a} – {b}). Kötü huylu toplam olasılık {trend} ({d} puan). Halkalı nokta açık olan vaka. "
            "Değişim modelin çıktısıdır; lezyondaki gerçek değişimi hekim değerlendirmelidir.",
            "{n} analyses ({a} – {b}). Malignant total {trend} ({d} pts). The ringed point is the open case. "
            "The change is the model's output; a physician must judge the real change in the lesion."),
"zip_btn": ("Toplu PDF (ZIP)", "Bulk PDF (ZIP)"),
"zip_help": ("Seçilen vakaların PDF raporlarını ve özet tabloyu tek dosyada indirir.",
             "Downloads the PDF reports of the selected cases and a summary table in one file."),
"zip_pick": ("Vakalar", "Cases"),
"zip_ph": ("Vaka seçin", "Choose cases"),
"zip_note": ("Liste, filtredeki ilk 20 vakayla gelir. Veritabanından gelen vakalarda görüntü olmadığı için raporda görüntü bulunmaz.",
             "The list starts with the first 20 filtered cases. Cases loaded from the database have no images in the report."),
"zip_make": ("ZIP hazırla ({n} vaka)", "Prepare ZIP ({n} cases)"),
"zip_wait": ("Raporlar hazırlanıyor…", "Preparing reports…"),
"zip_dl": ("ZIP'i indir ({n} rapor)", "Download ZIP ({n} reports)"),
"ev_zip": ("Toplu PDF hazırlandı ({n} vaka)", "Bulk PDF prepared ({n} cases)"),
"ev_reminder": ("Kontrol hatırlatıcısı indirildi", "Check-up reminder downloaded"),
"remind_btn": ("Kontrol hatırlatıcısı", "Check-up reminder"),
"remind_help": ("Telefonunuzun takvimine eklenir: o gün aynı ışıkta yeni fotoğraf çekip “Önceki vaka no” ile karşılaştırın. Doktor randevusunun yerine geçmez.",
                "Adds an event to your phone calendar: on that day take a new photo in the same light and compare it using “Previous case no”. It does not replace a doctor's appointment."),
"remind_when": ("Ne zaman?", "When?"),
"remind_m": ("{m} ay sonra", "In {m} months"),
"remind_dl": ("Takvime ekle (.ics)", "Add to calendar (.ics)"),
"details_more": ("Ayrıntılar (olası tanılar, ısı haritası, tüm sonuçlar)", "Details (possible diagnoses, heat map, all results)"),
"dark_toggle": ("Koyu tema", "Dark theme"),
"dark_help": ("Tercih bu tarayıcıda saklanır. Değiştirince sayfa yenilenir: kaydedilmemiş vakalar ve hekim girişi silinir.",
              "Saved in this browser. Switching reloads the page: unsaved cases and the physician login are cleared."),
"eval_sub": ("Doğrulama kümesine (1.048 lezyon) göre başarı, eşik seçimi, eşitlik ve iş yükü",
             "Performance, thresholds, fairness and workload on the validation set (1,048 lesions)"),
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
    "cam_heat_derm": ("Isı haritası (dermoskopik)", "Heat map (dermoscopic)"),
    "cam_heat_clin": ("Isı haritası (klinik)", "Heat map (clinical)"),
    "cam_note": (
        "Renkli alanlar modelin bu karar için en çok dikkat ettiği yerlerdir; beyaz çizgi en yoğun bölgeyi çevreler. Bu harita "
        "lezyonun sınırını veya tanı kanıtını göstermez; yalnızca modelin kararını anlamaya yardımcı olur (Grad-CAM yöntemi).",
        "Coloured areas are where the model focused most for this decision; the white line surrounds the strongest region. The "
        "map does not show the lesion border or diagnostic evidence; it only helps to understand the model's decision (Grad-CAM).",
    ),
    "cam_for": ("Harita neyi açıklıyor: {c}", "What the map explains: {c}"),
    "cam_t_fast": ("öncelikli sevki tetikleyen türler (melanom, SCC, diğer kötü huylu)",
                   "the types that triggered priority referral (melanoma, SCC, other malignant)"),
    "cam_t_mel": ("sevki tetikleyen melanom olasılığı", "the melanoma probability that triggered the referral"),
    "cam_t_risk": ("sevki tetikleyen kötü huylu ve kansere dönüşebilir türler",
                   "the malignant and precancerous types that triggered the referral"),
    "cam_t_top": ("en olası tür ({c})", "the most likely type ({c})"),
    "cam_low": ("düşük", "low"), "cam_high": ("yüksek ilgi", "high attention"),
    "cam_peak": ("en yoğun bölge", "strongest region"),

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
    "pdf_footer": ("Eğitim verisi: MILK10k ve ISIC 2019 (CC BY-NC 4.0), PAD-UFES-20 (CC BY 4.0). Bu uygulama bir tıbbi cihaz değildir.",
                   "Training data: MILK10k and ISIC 2019 (CC BY-NC 4.0), PAD-UFES-20 (CC BY 4.0). This application is not a medical device."),
    "pdf_footer_1": ("Bahçeşehir Üniversitesi Endüstri Mühendisliği bitirme projesi prototipi · Resmî tıbbi belge değildir",
                     "Bahçeşehir University Industrial Engineering capstone prototype · Not an official medical document"),
    "pdf_org": ("Endüstri Mühendisliği Bölümü · Bitirme Projesi", "Department of Industrial Engineering · Capstone Project"),
    "pdf_proto": ("Klinik karar destek prototipi · tanı koymaz", "Clinical decision-support prototype · not a diagnosis"),
    "pdf_f_case": ("Vaka no", "Case no"),
    "pdf_f_date": ("Tarih", "Date"),
    "pdf_f_model": ("Model", "Model"),
    "pdf_f_version": ("Sürüm", "Version"),
}


S.update({
    "up_hint": ("JPG veya PNG · telefonda kamera ya da galeri açılır", "JPG or PNG · on a phone this opens the camera or gallery"),
    "up_btn": ("Fotoğraf seç", "Choose photo"),
    "consent_how": ("<b>Nasıl kullanılır?</b> 1) Lekenin fotoğrafını yükleyin · 2) Yaş, cinsiyet gibi birkaç bilgiyi seçin · "
                    "3) Sonucu ve öneriyi okuyun. Ayrıntılı tanıtım: Yardım → Kılavuz → Hızlı tur.",
                    "<b>How to use:</b> 1) Upload a photo of the spot · 2) Pick a few details such as age and sex · "
                    "3) Read the result and advice. Full walkthrough: Help → Guide → Quick tour."),
    "role_q": ("Siteyi kim kullanıyor? Seçiminize göre doğru adımlar açılır; sonradan değiştirebilirsiniz.",
               "Who is using the site? The right steps open for your choice; you can change it later."),
    "role_patient": ("Kendim / yakınım için bakıyorum", "Checking for myself or a relative"),
    "role_patient_d": ("Yalnızca telefonla çekilmiş bir fotoğraf yeterli. Sonuç sade dille gösterilir.",
                       "A phone photo is all you need. The result is shown in plain language."),
    "role_patient_btn": ("Telefon fotoğrafıyla başla", "Start with a phone photo"),
    "role_doctor": ("Hekim / sağlık çalışanıyım", "I am a physician / health worker"),
    "role_doctor_d": ("Klinik yakın çekim + dermoskopik görüntü (dermatoskop gerekir). En güvenilir sonuç bu moddadır.",
                      "Clinical close-up + dermoscopic image (needs a dermatoscope). This mode gives the most reliable result."),
    "role_doctor_btn": ("İki görüntüyle başla", "Start with two images"),
    "show_doc": ("Doktoruma göster", "Show my doctor"),
    "mel_pdf": ("Melanom güvenlik kuralı", "Melanoma safety rule"),
    "mel_pdf_v": ("Uygulandı (melanom olasılığı {p})", "Applied (melanoma probability {p})"),
    "kind_q": ("Neye bakmak istiyorsunuz?", "What would you like to check?"),
    "kind_spot": ("Bir ben ya da tek bir leke", "A mole or a single spot"),
    "kind_rash": ("Yaygın kızarıklık, döküntü, kaşıntılı ya da pullanan alanlar", "A widespread rash, itchy or scaly patches"),
    "kind_unsure": ("Emin değilim", "Not sure"),
    "kind_rash_warn": ("<b>Bu araç bu tür sorunlar için uygun değil.</b> Benler ve tek tek lekeler için tasarlandı; egzama, sedef, "
                       "mantar, alerji ya da yayılan döküntüleri tanımak için eğitilmedi ve bu durumlarda yanıltıcı (gereksiz yere "
                       "korkutucu ya da yanlış rahatlatıcı) sonuç verebilir. Bir doktora ya da eczacıya danışmanız daha doğru olur.",
                       "<b>This tool is not suitable for this kind of problem.</b> It was designed for moles and single spots; it "
                       "was not trained to recognise eczema, psoriasis, fungal infections, allergies or spreading rashes, and can "
                       "give misleading (needlessly alarming or falsely reassuring) results. Please ask a doctor or pharmacist."),
    "kind_rash_ok": ("Anladım, yine de içindeki tek bir lekeyi analiz etmek istiyorum", "I understand; I still want to analyse a single spot in it"),
    "kind_block": ("Devam etmek için yukarıdaki uyarıyı onaylayın.", "Confirm the warning above to continue."),
    "kind_rash_result": ("Bu fotoğrafın yaygın bir döküntü ya da kızarıklıktan olduğunu belirttiniz. Araç bu durumlar için "
                         "eğitilmediğinden aşağıdaki sonuç <b>güvenilir değildir</b>; bir doktora danışın.",
                         "You said this photo shows a widespread rash or redness. The tool was not trained for this, so the "
                         "result below is <b>not reliable</b>; please ask a doctor."),
    "ev_scope_override": ("Kapsam dışı uyarısı (döküntü) onaylanarak devam edildi", "Continued after the out-of-scope (rash) warning"),
    "details_pm": ("Ayrıntılı sonuçlar (yüzdeler, benzeyen türler)", "Detailed results (percentages, similar types)"),
    "dc_title": ("Cilt lekesi ön değerlendirme özeti · tanı değildir", "Skin spot pre-assessment summary · not a diagnosis"),
    "dc_model": ("Model sonucu (en olası 3 tür)", "Model output (top 3 types)"),
    "dc_symptoms": ("Hastanın beyanı", "Patient-reported"),
    "dc_note": ("Bu özet bir bitirme projesi prototipinin çıktısıdır; tıbbi cihaz değildir ve tanı koymaz. Karar muayene eden hekimindir.",
                "This summary is the output of a capstone prototype; it is not a medical device and does not diagnose. The examining physician decides."),
    "dc_shot": ("İpucu: bu ekranın görüntüsünü alıp doktorunuza gösterebilir ya da PDF raporu indirebilirsiniz.",
                "Tip: take a screenshot of this screen to show your doctor, or download the PDF report."),
    "toast_pdf": ("PDF raporu indirildi", "PDF report downloaded"),
    "toast_reminder": ("Takvim dosyası indirildi; açınca telefon takvimine eklenir", "Calendar file downloaded; open it to add to your phone calendar"),
    "toast_fb": ("Hekim değerlendirmesi kaydedildi", "Physician review saved"),
    "toast_status": ("Vaka durumu güncellendi", "Case status updated"),
    "toast_closed": ("Vaka kapatıldı", "Case closed"),
    "toast_reopened": ("Vaka yeniden açıldı", "Case reopened"),
    "toast_saved_db": ("(veritabanına kaydedildi)", "(saved to the database)"),
    "role_note": ("Hiçbir seçenek tanı koymaz; sonuçlar hekim değerlendirmesinin yerine geçmez.",
                  "Neither option gives a diagnosis; results do not replace a physician's assessment."),
})

# Hasta modunda kullanılan sade metinler (aynı anahtar; set_pm(True) iken bunlar gösterilir)
PM = {
    "new_sub": ("Üç adım: lekenin fotoğrafını yükleyin, birkaç bilgi seçin, sonucu okuyun.",
                "Three steps: upload a photo of the spot, pick a few details, read the result."),
    "s1_title": ("Fotoğraf", "Photo"),
    "s2_title": ("Bilgiler", "Details"),
    "clin_title": ("Lekenin fotoğrafı", "Photo of the spot"),
    "clin_text": ("Lekeyi telefonla, 10–15 cm uzaktan, gün ışığında ve flaşsız çekin. Leke ortada ve net olsun; çevresindeki "
                  "deri de görünsün.",
                  "Photograph the spot with your phone from 10–15 cm, in daylight, without flash. Keep the spot centred and "
                  "sharp, with some surrounding skin visible."),
    "s2_images": ("Yüklenen fotoğraf", "Uploaded photo"),
    "pdf_images": ("Fotoğraf", "Photo"),
    "case_no": ("Kayıt no", "Record no"),
    "case_hint": ("Bu numarayı not edin; doktorunuza gösterebilir ya da <b>Vaka takibi</b> sayfasından tekrar açabilirsiniz.",
                  "Note this number; you can show it to your doctor or reopen it on the <b>Case tracking</b> page."),
    "pm_note": ("Sonuç yalnızca telefon fotoğrafından hesaplandı. Doktorların kullandığı büyüteçli görüntüyle yapılan analize göre "
                "daha az güvenilirdir.",
                "The result was computed from the phone photo only. It is less reliable than an analysis with the magnified "
                "image doctors use."),
    "top3": ("En çok benzediği 3 tür", "The 3 types it looks most like"),
    "r_urgent": ("Fotoğraftaki leke, kötü huylu (kanser) türlere belirgin şekilde benziyor (toplam benzerlik {p}). "
                 "Lekeyi kısa süre içinde bir dermatoloğa gösterin.",
                 "The spot clearly resembles malignant (cancer) types (total {p}). Please show it to a dermatologist soon."),
    "r_urgent_type": ("Fotoğraftaki leke, melanom gibi hızlı ilerleyebilen türlere benziyor (toplam benzerlik {p}). "
                      "Lekeyi 1–2 hafta içinde bir dermatoloğa gösterin.",
                      "The spot resembles faster-growing types such as melanoma (total {p}). Please show it to a "
                      "dermatologist within 1–2 weeks."),
    "r_refer": ("Fotoğraftaki leke, kanser ya da kansere dönüşebilen türlere bir ölçüde benziyor (toplam benzerlik {p}). "
                "Lekeyi 4 hafta içinde bir dermatoloğa gösterin.",
                "The spot somewhat resembles cancer or pre-cancer types (total {p}). Please show it to a dermatologist within 4 weeks."),
    "r_unsure": ("Model belirgin bir kanser bulgusu görmüyor ama sonuçtan emin değil. Lekeyi bir doktora gösterin.",
                 "The model sees no clear sign of cancer but is not sure. Please show the spot to a doctor."),
    "r_benign": ("Model fotoğrafta belirgin bir kanser bulgusu görmüyor. Yine de leke büyürse, rengi ya da şekli değişirse, "
                 "kanar, kaşınır ya da yara açarsa bir doktora gidin.",
                 "The model sees no clear sign of cancer in the photo. Still, see a doctor if the spot grows, changes colour "
                 "or shape, bleeds, itches or becomes a sore."),
    "unsure_box": ("<b>Model bu fotoğrafta emin değil.</b> En yakın sonuçlar birbirine çok yakın. Sıralamaya güvenmeyin; "
                   "fotoğrafı daha net çekmeyi deneyin ve bir doktora danışın.",
                   "<b>The model is not sure about this photo.</b> The closest results are very close to each other. Do not "
                   "rely on the order; try a sharper photo and ask a doctor."),
    "btn_followup": ("Bu lekenin kontrolünü başlat", "Start a follow-up of this spot"),
}

S.update({
    "mode_line_pm": ("<b>Telefon fotoğrafıyla bakılıyor.</b> Lekenin tek bir fotoğrafı yeterli.",
                     "<b>Checking with a phone photo.</b> One photo of the spot is enough."),
    "mode_line_dr": ("<b>İki görüntüyle bakılıyor (hekim modu).</b> Klinik yakın çekim ve dermoskopik görüntü gerekir.",
                     "<b>Checking with two images (physician mode).</b> A clinical close-up and a dermoscopic image are needed."),
    "mode_switch_pm": ("Telefon fotoğrafına geç", "Switch to phone photo"),
    "mode_switch_dr": ("İki görüntüye geç", "Switch to two images"),
    "how_1_t": ("Fotoğraf", "Photo"),
    "how_1": ("Lekenin net bir fotoğrafını yükleyin; site netlik ve ışığı kontrol eder.",
              "Upload a sharp photo of the spot; the site checks sharpness and lighting."),
    "how_2_t": ("Birkaç bilgi", "A few details"),
    "how_2": ("Yaş aralığı, cinsiyet, ten rengi ve lekenin yerini seçin.", "Pick age range, sex, skin tone and where the spot is."),
    "how_3_t": ("Sonuç ve öneri", "Result and advice"),
    "how_3": ("Lekenin doktora gösterilmesi gerekip gerekmediğini ve nedenini okuyun.",
              "Read whether the spot should be shown to a doctor, and why."),
    "role_privacy": ("Fotoğraflar kaydedilmez; ad, kimlik ya da iletişim bilgisi istenmez. İzin verirseniz yalnızca anonim sonuçlar "
                     "araştırma amacıyla saklanır.",
                     "Photos are not stored; no name, ID or contact details are asked for. If you allow it, only anonymous results are "
                     "kept for research."),
    "ru_benign": ("İyi huylu görünüm", "Benign appearance"),
    "ru_refer": ("Değerlendirme önerilir", "Review recommended"),
    "ru_urgent": ("Öncelikli değerlendirme", "Priority review"),
    "ru_pm_benign": ("Rutin takip", "Routine follow-up"),
    "ru_pm_refer": ("Doktora gösterin", "Show a doctor"),
    "ru_pm_urgent": ("Kısa sürede doktora", "See a doctor soon"),
    "ru_aria": ("Yönlendirme: {z}", "Referral: {z}"),
    "ru_note": ("Cetvel, bu fotoğraf için önerilen yönlendirmeyi gösterir; tanı değildir.",
                "The scale shows the recommended referral for this photo; it is not a diagnosis."),
    "case_info": ("Vaka bilgileri", "Case details"),
    "model_used": ("Kullanılan model: {m}", "Model used: {m}"),
    "btn_back_short": ("Vaka listesi", "Case list"),
    "more_actions": ("Diğer işlemler", "More actions"),
    "chip_real": ("Gerçek model · v3", "Real model · v3"),
})
