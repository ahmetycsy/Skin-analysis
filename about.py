"""'Model hakkında' (model kartı) ve 'Kullanım kılavuzu' sekmeleri, Türkçe ve İngilizce.

Buradaki sayılar gerçektir: veri dağılımı MILK10k eğitim dosyalarından, model sonuçları Kaggle notebook'unun
çıktısından alınmıştır. Yeni model sonuçları gelince yalnızca MODEL_STATUS listesini güncelleyin.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import i18n
from predictor import CLASSES, GROUPS

APP_VERSION = "0.9"

# MILK10k eğitim kümesi (5.240 lezyon) — MILK10k_Training_GroundTruth.csv ve _Metadata.csv'den sayıldı
CLASS_COUNTS = {"AKIEC": 303, "BCC": 2522, "BEN_OTH": 44, "BKL": 544, "DF": 52, "INF": 50,
                "MAL_OTH": 9, "MEL": 450, "NV": 746, "SCCKA": 473, "VASC": 47}
SKIN_TONE_COUNTS = {0: 6, 1: 105, 2: 506, 3: 3174, 4: 1066, 5: 383}
N_LESIONS = 5240
AGE_RANGE = (5, 85)           # veri setindeki en küçük ve en büyük yaş değeri
LOW_DATA_SKIN_TONES = {0, 1}  # eğitim verisi çok az olan ten rengi grupları

TEAM = "Ahmet Talat Yücesoy · Tuğalp Ata · Bedirhan Sevencan"
ADVISOR = "Zühal Özcan Yavuz"

# Model durumları. status: "done" | "wip" | "planned". Yeni sonuç gelince burayı güncelleyin.
# Sonuçlar Tuğalp'in 5 katlı çapraz doğrulamasından (5.240 lezyonun her biri bir kez kontrol setinde; ortalama ± sapma).
# Kaynak: tugalp dalındaki cv_table.md / ahmet_claude_ozet.md. Kullanılan ağırlıklar kat 0 modelleridir.
MODEL_STATUS = [
    {"name": ("ISIC MILK10k Benchmark — bağımsız gizli test (479 lezyon)", "ISIC MILK10k Benchmark — independent hidden test (479 lesions)"),
     "arch": ("Kullanılan sistem (hekim modu)", "Deployed system (physician mode)"), "status": "external", "f1": 0.485,
     "note": ("ISIC tarafından puanlandı · cevap anahtarı bizde değil", "Scored by ISIC · labels not available to us")},
    {"name": ("Kullanılan sistem — hekim modu (4 model)", "Deployed system — physician mode (4 models)"),
     "arch": ("B3 + V2-S, derm + klinik ortalaması", "B3 + V2-S, mean of derm + clinical"), "status": "used", "f1": 0.542,
     "note": ("±0,024 · doğruluk %77,4 · kalibre", "±0.024 · accuracy 77.4% · calibrated")},
    {"name": ("Yalnızca klinik fotoğraf (hasta modu)", "Clinical photo only (patient mode)"),
     "arch": ("B3 + V2-S, klinik", "B3 + V2-S, clinical"), "status": "done", "f1": 0.463,
     "note": ("±0,025 · doğruluk %69,9 · arayüzde henüz yok", "±0.025 · accuracy 69.9% · not in the interface yet")},
    {"name": ("Dermoskopik model", "Dermoscopic model"), "arch": "EfficientNetV2-S", "status": "done", "f1": 0.506,
     "note": ("±0,035", "±0.035")},
    {"name": ("Dermoskopik model", "Dermoscopic model"), "arch": "EfficientNet-B3", "status": "done", "f1": 0.484,
     "note": ("±0,017", "±0.017")},
    {"name": ("Klinik model", "Clinical model"), "arch": "EfficientNet-B3", "status": "done", "f1": 0.445,
     "note": ("±0,020", "±0.020")},
    {"name": ("Klinik model", "Clinical model"), "arch": "EfficientNetV2-S", "status": "done", "f1": 0.436,
     "note": ("±0,023", "±0.023")},
    {"name": ("Görüntü + hasta bilgisi füzyonu (alternatif)", "Image + patient-info fusion (alternative)"),
     "arch": ("Füzyon ağı", "Fusion network"), "status": "done", "f1": 0.497,
     "note": ("yalnızca kat 0 · aynı sette hekim modu 0,547", "fold 0 only · physician mode 0.547 on the same set")},
]
STATUS_LABEL = {"external": ("Bağımsız test", "Independent test"), "used": ("Kullanılıyor", "In use"), "done": ("Tamamlandı", "Done"), "wip": ("Devam ediyor", "In progress"), "planned": ("Planlandı", "Planned")}


def _L(pair):
    if isinstance(pair, str):
        return pair
    return pair[0] if i18n.get_lang() == "tr" else pair[1]


def _num(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if i18n.get_lang() == "tr" else s


def _int(x: int) -> str:
    s = f"{x:,}"
    return s.replace(",", ".") if i18n.get_lang() == "tr" else s


def _h(text: str):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)


TXT = {
    "lead": ("Bu sayfa sistemin ne için tasarlandığını, hangi veriyle geliştirildiğini, ne kadar iyi çalıştığını ve "
             "sınırlarını özetler (model kartı).",
             "This page summarises what the system is for, what data it was built with, how well it performs and where "
             "its limits are (model card)."),
    "k_lesions": ("lezyon (eğitim verisi)", "lesions (training data)"),
    "k_images": ("görüntü (her lezyona 2)", "images (2 per lesion)"),
    "k_classes": ("lezyon türü", "lesion types"),
    "k_best": ("kullanılan sistem (Macro F1, hedef ≥ 0,45)", "deployed system (Macro F1, target ≥ 0.45)"),
    "h_purpose": ("Amaç ve kullanım", "Purpose and intended use"),
    "purpose": (
        "- **Amaç:** Hekime ikinci bir görüş sunmak; şüpheli lezyonların önceliklendirilmesine yardımcı olmak.\n"
        "- **Hedef kullanıcı:** Hekimler ve sağlık çalışanları (eğitim ve araştırma ortamında).\n"
        "- **Kullanılmamalı:** Tek başına tanı koymak, tedavi kararı vermek veya hekime başvurmayı geciktirmek için.",
        "- **Purpose:** to offer a second opinion to physicians and help prioritise suspicious lesions.\n"
        "- **Intended users:** physicians and health workers (in education and research settings).\n"
        "- **Not for:** making a diagnosis on its own, treatment decisions, or delaying a visit to a physician.",
    ),
    "h_how": ("Sistem nasıl çalışır", "How the system works"),
    "how": (
        "1. **Dermoskopik görüntü** iki ayrı görüntü modeline (EfficientNet-B3 ve EfficientNetV2-S) girer; **klinik fotoğraf** da kendi iki modeline.\n"
        "2. Her görüntü türünün modelleri kendi içinde, sonra dermoskopik ve klinik sonuç eşit ağırlıkla ortalanır (11 tür için tek sonuç).\n"
        "3. Olasılıklar **kalibre edilir**: model “%70” dediğinde gerçekten yaklaşık %70 isabet eder.\n"
        "4. Sonuç kötü huylu / kansere dönüşebilir / iyi huylu gruplarına toplanır ve yönlendirme önerisine çevrilir.\n\n"
        "Yaş, cinsiyet, ten rengi ve bölge modele **girmez**: eklendiğinde başarı artmadı. Bu bilgiler rapor, vaka kaydı ve az veri uyarıları için kullanılır.",
        "1. The **dermoscopic image** goes into two image models (EfficientNet-B3 and EfficientNetV2-S); the **clinical photo** into its own two.\n"
        "2. Models of each image type are averaged, then the dermoscopic and clinical results are averaged with equal weight (one result for 11 types).\n"
        "3. Probabilities are **calibrated**: when the model says “70%”, it is right about 70% of the time.\n"
        "4. The result is summed into malignant / precancerous / benign groups and turned into a referral suggestion.\n\n"
        "Age, sex, skin tone and body site do **not** enter the model: adding them did not improve accuracy. They are used for the report, the case record and low-data warnings.",
    ),
    "h_status": ("Geliştirme durumu", "Development status"),
    "c_comp": ("Bileşen", "Component"), "c_arch": ("Mimari", "Architecture"), "c_status": ("Durum", "Status"),
    "c_note": ("Not", "Note"),
    "f1_note": (
        "Macro F1: 11 türün her biri için ayrı hesaplanan başarının ortalaması; nadir türleri de yaygın türler kadar önemser "
        "(1 en iyi, 0 en kötü). Değerler 5 katlı çapraz doğrulamadan (her lezyon bir kez test edildi) ortalama ± sapmadır. "
        "Füzyon modeli yalnızca tek bir bölünmede ölçüldü. Sonuçlar ~0,01–0,02 iyimser olabilir (en iyi epoch kontrol setine bakılarak seçildi).",
        "Macro F1: the average of the score computed separately for each of the 11 types; weighs rare types as much as common "
        "ones (1 best, 0 worst). Values are mean ± SD from 5-fold cross-validation (each lesion tested once). The fusion model "
        "was measured on a single split only. Results may be ~0.01–0.02 optimistic (best epoch chosen on the validation set).",
    ),
    "h_data": ("Eğitim verisi: MILK10k", "Training data: MILK10k"),
    "c_type": ("Tür", "Type"), "c_med": ("Tıbbi adı", "Medical name"), "c_group": ("Grup", "Group"),
    "c_n": ("Lezyon", "Lesions"), "c_share": ("Pay", "Share"),
    "imbalance": (
        "Veri seti çok **dengesizdir**: lezyonların yarısı yavaş büyüyen deri kanseridir (BCC), bazı türlerden ise yalnızca "
        "birkaç düzine örnek vardır. Bu nedenle nadir türlerde model daha az güvenilirdir.",
        "The dataset is highly **imbalanced**: half of the lesions are slow-growing skin cancer (BCC), while some types "
        "have only a few dozen examples. The model is therefore less reliable for rare types.",
    ),
    "c_tone": ("Ten rengi", "Skin tone"),
    "tone_note": (
        "Ten rengi dağılımı (0 en koyu, 5 en açık). Koyu tenlerde (0–1) çok az örnek vardır; bu gruplarda arayüz ayrıca "
        "uyarı gösterir.",
        "Skin-tone distribution (0 darkest, 5 lightest). There are very few examples for dark skin (0–1); the interface "
        "shows an extra warning for these groups.",
    ),
    "h_limits": ("Bilinen sınırlılıklar", "Known limitations"),
    "limits": (
        "- **Veri dengesizliği:** Nadir türlerde (ör. diğer kötü huylu oluşumlar, 9 lezyon) başarı düşüktür.\n"
        "- **Ten rengi temsili:** Eğitim verisinin büyük kısmı orta ve açık tenlidir; koyu tenlerde performans ayrıca ölçülmelidir.\n"
        "- **Bölünme:** Veri setinde hasta kimliği bulunmadığı için ayırma lezyon bazında yapılmıştır; aynı hastanın farklı "
        "lezyonları hem eğitim hem doğrulama kümesine düşmüş olabilir.\n"
        "- **Tek veri seti:** Sonuçlar yalnızca MILK10k üzerinde ölçülmüştür; farklı kamera, dermatoskop veya hasta "
        "gruplarında performans farklı olabilir.\n"
        "- **Yaş aralığı:** Veri setinde yaş 5–85 arasındadır; bu aralığın dışındaki hastalarda sonuç daha belirsizdir.\n"
        "- **Çekim koşulları:** MILK10k klinik fotoğrafları klinikte çekilmiştir; telefonla çekilen fotoğraflarda başarı daha düşük olabilir.\n"
        "- **En nadir tür:** “Diğer kötü huylu oluşumlar” (9 lezyon) hiçbir modelde öğrenilemedi.",
        "- **Class imbalance:** performance is low for rare types (e.g. other malignant growths, 9 lesions).\n"
        "- **Skin-tone representation:** most training data is medium or light skin; performance on dark skin must be "
        "measured separately.\n"
        "- **Split:** the dataset has no patient ID, so the split was made per lesion; different lesions of the same "
        "patient may appear in both training and validation sets.\n"
        "- **Single dataset:** results were measured only on MILK10k; performance may differ with other cameras, "
        "dermatoscopes or patient groups.\n"
        "- **Age range:** ages in the dataset range from 5 to 85; results are less certain outside this range.\n"
        "- **Capture conditions:** MILK10k clinical photos were taken in clinics; accuracy may be lower for phone photos.\n"
        "- **Rarest type:** “other malignant growths” (9 lesions) could not be learned by any model.",
    ),
    "h_ethics": ("Etik, gizlilik ve güvenlik", "Ethics, privacy and safety"),
    "ethics": (
        "- Sistem **tanı koymaz**; her sonuç hekim değerlendirmesiyle birlikte yorumlanmalıdır.\n"
        "- Yüklenen görüntüler yalnızca analiz sırasında bellekte işlenir; **sunucuya kaydedilmez**. İzin verilen vakalarda yalnızca anonim sonuçlar (yaş aralığı, cinsiyet, ten rengi, bölge, olasılıklar, hekim değerlendirmesi) veritabanında saklanır. Ad, kimlik numarası "
        "gibi kişisel bilgi istenmez.\n"
        "- Model emin olmadığında bunu açıkça belirtir ve hekime yönlendirir.\n"
        "- Bu uygulama bir tıbbi cihaz değildir ve klinik kullanım için onaylanmamıştır.",
        "- The system **does not make a diagnosis**; every result must be interpreted together with a physician's assessment.\n"
        "- Uploaded images are processed in memory only during analysis and are **not stored on the server**. For cases with consent, only anonymous results (age range, sex, skin tone, body site, probabilities, physician assessment) are kept in the database. No personal "
        "information such as name or ID number is requested.\n"
        "- When the model is not confident, it says so clearly and refers to a physician.\n"
        "- This application is not a medical device and is not approved for clinical use.",
    ),
    "h_project": ("Proje", "Project"),
    "project": (
        "Bahçeşehir Üniversitesi, Endüstri Mühendisliği Bitirme Projesi (Capstone I).  \n**Ekip:** {team}  \n"
        "**Danışman:** Dr. Öğr. Üyesi {adv}  \n**Veri seti:** MILK10k, ISIC Archive, CC-BY-NC 4.0 lisansı ile kullanılmıştır.",
        "Bahçeşehir University, Industrial Engineering Capstone Project (Capstone I).  \n**Team:** {team}  \n"
        "**Advisor:** Asst. Prof. Dr. {adv}  \n**Dataset:** MILK10k, ISIC Archive, used under the CC-BY-NC 4.0 licence.",
    ),
    # kılavuz
    "g_lead": ("Sistemi ilk kez kullanıyorsanız bu sayfayı bir kez okumanız yeterlidir.",
               "If you are using the system for the first time, reading this page once is enough."),
    "g1_t": ("Görüntüleri yükleyin", "Upload the images"),
    "g1": ("Aynı lezyonun bir klinik fotoğrafını ve bir dermoskopik görüntüsünü yükleyin. Gerekirse “Kırp” ile yalnızca "
           "lezyonun çevresini seçin. Sistem netliği, ışığı ve boyutu otomatik kontrol eder.",
           "Upload a clinical photo and a dermoscopic image of the same lesion. If needed, use “Crop” to keep only the "
           "area around the lesion. The system checks sharpness, lighting and size automatically."),
    "g2_t": ("Hasta bilgilerini seçin", "Enter patient information"),
    "g2": ("Yaş aralığı, cinsiyet, ten rengi ve lezyonun bulunduğu bölge. Dördü de seçilmeden analiz başlamaz.",
           "Age range, sex, skin tone and body site. Analysis does not start until all four are selected."),
    "g3_t": ("Sonucu yorumlayın", "Read the result"),
    "g3": ("Önce üstteki yönlendirme kutusuna bakın. Ardından grup çubuğu ve en olası üç tür gelir. Raporu PDF olarak "
           "indirebilirsiniz.",
           "Look at the referral box at the top first. The group bar and the three most likely types follow. You can "
           "download the report as a PDF."),
    "h_colors": ("Sonuç ekranındaki renkler", "Colours on the result screen"),
    "col_red": ("<b>Kırmızı – Öncelikli değerlendirme:</b> kötü huylu türlerin toplam olasılığı yüksek. Kısa sürede "
                "dermatoloğa yönlendirme önerilir.",
                "<b>Red – Priority review:</b> the combined probability of malignant types is high. Referral to a "
                "dermatologist soon is recommended."),
    "col_orange": ("<b>Turuncu – Değerlendirme önerilir:</b> kansere dönüşebilen veya kötü huylu türlere benzerlik var "
                   "ya da model kararsız.",
                   "<b>Orange – Review recommended:</b> resemblance to precancerous or malignant types, or the model is "
                   "uncertain."),
    "col_green": ("<b>Yeşil – İyi huylu görünüm:</b> belirgin kötü huylu özellik yok. Lezyon değişirse yine hekime "
                  "başvurulmalıdır.",
                  "<b>Green – Benign appearance:</b> no clear malignant features. See a physician if the lesion changes."),
    "h_faq": ("Sık sorulan sorular", "Frequently asked questions"),
}

FAQ = [
    (("Yüzdeler ne anlama geliyor?", "What do the percentages mean?"),
     ("Modelin her tür için hesapladığı olasılıktır; 11 türün toplamı %100'dür. Kesinlik değil, benzerlik ölçüsüdür. "
      "Örneğin %60, “kesin bu” değil, “en çok buna benziyor” demektir.",
      "They are the probabilities the model estimates for each type; the 11 add up to 100%. They measure similarity, "
      "not certainty. For example, 60% means “most similar to this”, not “definitely this”.")),
    (("“Model kararsız” uyarısı neden çıkıyor?", "Why does the “model uncertain” warning appear?"),
     ("En yüksek olasılık düşükse veya ilk iki sonuç birbirine çok yakınsa model tek bir türü ayırt edemiyor demektir. "
      "Bu durumda sıralamaya güvenilmemeli ve hekim değerlendirmesine öncelik verilmelidir.",
      "If the top probability is low or the top two results are very close, the model cannot single out one type. "
      "The ranking should then not be relied upon and a physician's assessment should come first.")),
    (("Neden iki ayrı görüntü gerekiyor?", "Why are two images needed?"),
     ("Klinik fotoğraf lezyonun genel görünüşünü, dermoskopik görüntü ise yüzeyin altındaki desenleri gösterir. "
      "Hekimler de ikisini birlikte değerlendirir; model de aynı şekilde çalışır.",
      "The clinical photo shows the overall appearance; the dermoscopic image shows patterns below the surface. "
      "Physicians assess both together, and the model works the same way.")),
    (("Kırpma ne zaman kullanılmalı?", "When should I crop?"),
     ("Lezyon görüntünün küçük bir kısmını kaplıyorsa veya kenarlarda cetvel, yazı, kıyafet gibi ilgisiz şeyler varsa. "
      "Lezyonun tamamı ve çevresinde biraz sağlam deri kutunun içinde kalmalıdır.",
      "When the lesion covers only a small part of the image, or when there are unrelated things such as rulers, text "
      "or clothing at the edges. Keep the whole lesion and a little healthy skin around it inside the box.")),
    (("Görüntülerim saklanıyor mu?", "Are my images stored?"),
     ("Hayır. Görüntüler yalnızca analiz sırasında bellekte işlenir ve sayfa kapatıldığında silinir. Analizden önce "
      "kayıt izni verirseniz yalnızca anonim sonuçlar (yaş aralığı, cinsiyet, ten rengi, bölge, olasılıklar, hekim "
      "değerlendirmesi) araştırma amacıyla saklanır; görüntü ve kişisel bilgi asla saklanmaz.",
      "No. Images are processed in memory only during analysis and are discarded when the page is closed. If you give "
      "storage consent before the analysis, only anonymous results (age range, sex, skin tone, body site, probabilities, "
      "physician assessment) are kept for research; images and personal details are never stored.")),
]


def render_about():
    st.markdown(f'<p class="lead">{_L(TXT["lead"])}</p>', unsafe_allow_html=True)
    best = next((m["f1"] for m in MODEL_STATUS if m["status"] == "used"), None)
    kpis = [(_int(N_LESIONS), _L(TXT["k_lesions"])), (_int(N_LESIONS * 2), _L(TXT["k_images"])),
            ("11", _L(TXT["k_classes"])), (_num(best) if best else "–", _L(TXT["k_best"]))]
    for col, (val, lab) in zip(st.columns(4), kpis):
        col.markdown(f'<div class="kpi"><div class="kpi-v">{val}</div><div class="kpi-l">{lab}</div></div>', unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        _h(_L(TXT["h_purpose"]))
        st.markdown(_L(TXT["purpose"]))
        _h(_L(TXT["h_how"]))
        st.markdown(_L(TXT["how"]))
    with b:
        _h(_L(TXT["h_status"]))
        rows = [{
            _L(TXT["c_comp"]): _L(m["name"]), _L(TXT["c_arch"]): _L(m["arch"]),
            _L(TXT["c_status"]): _L(STATUS_LABEL[m["status"]]),
            "Macro F1": _num(m["f1"]) if m["f1"] is not None else "–", _L(TXT["c_note"]): _L(m["note"]),
        } for m in MODEL_STATUS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(_L(TXT["f1_note"]))

    _h(_L(("Sistem mimarisi", "System architecture")))
    from architecture import render_architecture
    render_architecture()

    _h(_L(TXT["h_data"]))
    d1, d2 = st.columns([1.3, 1], gap="large")
    with d1:
        df = pd.DataFrame({
            _L(TXT["c_type"]): [i18n.plain(c) for c in CLASSES],
            _L(TXT["c_med"]): [f"{i18n.cname(c)} ({c})" for c in CLASSES],
            _L(TXT["c_group"]): [i18n.group(GROUPS[c]) for c in CLASSES],
            _L(TXT["c_n"]): [CLASS_COUNTS[c] for c in CLASSES],
        }).sort_values(_L(TXT["c_n"]), ascending=False)
        df[_L(TXT["c_share"])] = (df[_L(TXT["c_n"])] / N_LESIONS).map(i18n.pct)
        st.dataframe(df, hide_index=True, width="stretch")
    with d2:
        st.markdown(_L(TXT["imbalance"]))
        tone = pd.DataFrame({_L(TXT["c_tone"]): [str(k) for k in SKIN_TONE_COUNTS],
                             _L(TXT["c_n"]): list(SKIN_TONE_COUNTS.values())})
        st.bar_chart(tone, x=_L(TXT["c_tone"]), y=_L(TXT["c_n"]), color="#0F766E", height=190)
        st.caption(_L(TXT["tone_note"]))

    _h(_L(TXT["h_limits"]))
    st.markdown(_L(TXT["limits"]))
    _h(_L(TXT["h_ethics"]))
    st.markdown(_L(TXT["ethics"]))
    _h(_L(TXT["h_project"]))
    st.markdown(_L(TXT["project"]).format(team=TEAM, adv=ADVISOR))


def render_guide():
    st.markdown(f'<p class="lead">{_L(TXT["g_lead"])}</p>', unsafe_allow_html=True)
    for col, no in zip(st.columns(3, gap="medium"), ("1", "2", "3")):
        col.markdown(f'<div class="guide"><span class="step-no">{no}</span><div class="card-title">{_L(TXT["g" + no + "_t"])}</div>'
                     f'<p class="card-text">{_L(TXT["g" + no])}</p></div>', unsafe_allow_html=True)

    _h(_L(TXT["h_colors"]))
    st.markdown(
        '<div class="legend-big">'
        f'<div><span class="dot" style="background:#A63D57"></span>{_L(TXT["col_red"])}</div>'
        f'<div><span class="dot" style="background:#B0701A"></span>{_L(TXT["col_orange"])}</div>'
        f'<div><span class="dot" style="background:#2F7F78"></span>{_L(TXT["col_green"])}</div></div>',
        unsafe_allow_html=True,
    )
    _h(_L(TXT["h_faq"]))
    for q, ans in FAQ:
        with st.expander(_L(q)):
            st.markdown(_L(ans))
