"""'Model hakkında' ve 'Kullanım kılavuzu' sekmeleri.

Buradaki sayılar gerçektir: veri dağılımı MILK10k eğitim dosyalarından, model sonuçları Kaggle notebook'unun
çıktısından alınmıştır. Yeni model sonuçları gelince yalnızca MODEL_STATUS listesini güncelleyin.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from predictor import CLASS_NAMES, CLASSES, GROUPS, PLAIN_NAMES

APP_VERSION = "0.4"

# MILK10k eğitim kümesi (5.240 lezyon) — MILK10k_Training_GroundTruth.csv ve _Metadata.csv'den sayıldı
CLASS_COUNTS = {"AKIEC": 303, "BCC": 2522, "BEN_OTH": 44, "BKL": 544, "DF": 52, "INF": 50,
                "MAL_OTH": 9, "MEL": 450, "NV": 746, "SCCKA": 473, "VASC": 47}
SKIN_TONE_COUNTS = {0: 6, 1: 105, 2: 506, 3: 3174, 4: 1066, 5: 383}
N_LESIONS = 5240
AGE_RANGE = (5, 85)          # veri setindeki en küçük ve en büyük yaş değeri
LOW_DATA_SKIN_TONES = {0, 1}  # eğitim verisi çok az olan ten rengi grupları

# Model durumları. "durum": "Tamamlandı" | "Devam ediyor" | "Planlandı"
MODEL_STATUS = [
    {"Bileşen": "Dermoskopik görüntü modeli", "Mimari": "EfficientNet-B3", "Durum": "Tamamlandı",
     "Macro F1": "0,477", "Not": "12 epoch, 1.048 lezyonluk doğrulama kümesi"},
    {"Bileşen": "Klinik görüntü modeli", "Mimari": "EfficientNet-B3", "Durum": "Devam ediyor",
     "Macro F1": "–", "Not": ""},
    {"Bileşen": "Hasta bilgisi (metadata) modeli", "Mimari": "Lojistik regresyon", "Durum": "Devam ediyor",
     "Macro F1": "–", "Not": ""},
    {"Bileşen": "Birleşik model (füzyon)", "Mimari": "Olasılık birleştirme", "Durum": "Planlandı",
     "Macro F1": "–", "Not": "Hedef ≥ 0,45"},
]

TEAM = "Ahmet Talat Yücesoy · Tuğalp Ata · Bedirhan Sevencan"
ADVISOR = "Dr. Öğr. Üyesi Zühal Özcan Yavuz"


def _h(text: str):
    st.markdown(f'<div class="sec-label">{text}</div>', unsafe_allow_html=True)


def render_about():
    st.markdown(
        '<p class="lead">Bu sayfa sistemin ne için tasarlandığını, hangi veriyle geliştirildiğini, '
        "ne kadar iyi çalıştığını ve sınırlarını özetler (model kartı).</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lab in [
        (c1, "5.240", "lezyon (eğitim verisi)"),
        (c2, "10.480", "görüntü (her lezyona 2)"),
        (c3, "11", "lezyon türü"),
        (c4, "0,477", "en iyi ara sonuç (Macro F1)"),
    ]:
        col.markdown(f'<div class="kpi"><div class="kpi-v">{val}</div><div class="kpi-l">{lab}</div></div>',
                     unsafe_allow_html=True)

    a, b = st.columns(2, gap="large")
    with a:
        _h("Amaç ve kullanım")
        st.markdown(
            "- **Amaç:** Hekime ikinci bir görüş sunmak; şüpheli lezyonların önceliklendirilmesine yardımcı olmak.\n"
            "- **Hedef kullanıcı:** Hekimler ve sağlık çalışanları (eğitim ve araştırma ortamında).\n"
            "- **Kullanılmamalı:** Tek başına tanı koymak, tedavi kararı vermek veya hekime başvurmayı geciktirmek için."
        )
        _h("Sistem nasıl çalışır")
        st.markdown(
            "1. **Dermoskopik görüntü** ve **klinik fotoğraf** ayrı ayrı birer görüntü modeline girer.\n"
            "2. **Yaş, cinsiyet, ten rengi ve bölge** ayrı bir modele girer.\n"
            "3. Üç modelin olasılıkları birleştirilir ve 11 tür için tek bir sonuç üretilir.\n"
            "4. Sonuç; kötü huylu / kansere dönüşebilir / iyi huylu gruplarına toplanır ve yönlendirme önerisine çevrilir."
        )
    with b:
        _h("Geliştirme durumu")
        st.dataframe(pd.DataFrame(MODEL_STATUS), hide_index=True, width="stretch")
        st.caption(
            "Macro F1: 11 türün her biri için ayrı ayrı hesaplanan başarının ortalaması. "
            "Nadir türleri de yaygın türler kadar önemser; 1 en iyi, 0 en kötüdür. "
            "Sonuçlar eğitimde kullanılmayan %20'lik doğrulama kümesinden ölçülmüştür."
        )

    _h("Eğitim verisi: MILK10k")
    d1, d2 = st.columns([1.3, 1], gap="large")
    with d1:
        df = pd.DataFrame({
            "Tür": [PLAIN_NAMES[c] for c in CLASSES],
            "Tıbbi adı": [f"{CLASS_NAMES[c]} ({c})" for c in CLASSES],
            "Grup": [GROUPS[c] for c in CLASSES],
            "Lezyon": [CLASS_COUNTS[c] for c in CLASSES],
        }).sort_values("Lezyon", ascending=False)
        df["Pay"] = (df["Lezyon"] / N_LESIONS).map(lambda p: f"%{p * 100:.1f}")
        st.dataframe(df, hide_index=True, width="stretch")
    with d2:
        st.markdown(
            "Veri seti çok **dengesizdir**: lezyonların yarısı yavaş büyüyen deri kanseridir (BCC), "
            "bazı türlerden ise yalnızca birkaç düzine örnek vardır. Bu nedenle nadir türlerde model daha az güvenilirdir."
        )
        tone = pd.DataFrame({"Ten rengi": [f"{k}" for k in SKIN_TONE_COUNTS], "Lezyon": list(SKIN_TONE_COUNTS.values())})
        st.bar_chart(tone, x="Ten rengi", y="Lezyon", color="#0F766E", height=190)
        st.caption(
            "Ten rengi dağılımı (0 en koyu, 5 en açık). Koyu tenlerde (0–1) çok az örnek vardır; "
            "bu gruplarda arayüz ayrıca uyarı gösterir."
        )

    _h("Bilinen sınırlılıklar")
    st.markdown(
        "- **Veri dengesizliği:** Nadir türlerde (ör. diğer kötü huylu oluşumlar, 9 lezyon) başarı düşüktür.\n"
        "- **Ten rengi temsili:** Eğitim verisinin büyük kısmı orta ve açık tenlidir; koyu tenlerde performans ayrıca ölçülmelidir.\n"
        "- **Bölünme:** Veri setinde hasta kimliği bulunmadığı için ayırma lezyon bazında yapılmıştır; "
        "aynı hastanın farklı lezyonları hem eğitim hem doğrulama kümesine düşmüş olabilir.\n"
        "- **Tek veri seti:** Sonuçlar yalnızca MILK10k üzerinde ölçülmüştür; farklı kamera, "
        "dermatoskop veya hasta gruplarında performans farklı olabilir.\n"
        "- **Yaş aralığı:** Veri setinde yaş 5–85 arasındadır; bu aralığın dışındaki hastalarda sonuç daha belirsizdir."
    )

    _h("Etik, gizlilik ve güvenlik")
    st.markdown(
        "- Sistem **tanı koymaz**; her sonuç hekim değerlendirmesiyle birlikte yorumlanmalıdır.\n"
        "- Yüklenen görüntüler yalnızca analiz sırasında bellekte işlenir; **sunucuya kaydedilmez**. "
        "Ad, kimlik numarası gibi kişisel bilgi istenmez.\n"
        "- Model emin olmadığında bunu açıkça belirtir ve hekime yönlendirir.\n"
        "- Bu uygulama bir tıbbi cihaz değildir ve klinik kullanım için onaylanmamıştır."
    )

    _h("Proje")
    st.markdown(
        f"Bahçeşehir Üniversitesi, Endüstri Mühendisliği Bitirme Projesi (Capstone I).  \n"
        f"**Ekip:** {TEAM}  \n**Danışman:** {ADVISOR}  \n"
        "**Veri seti:** MILK10k, ISIC Archive, CC-BY-NC 4.0 lisansı ile kullanılmıştır."
    )


def render_guide():
    st.markdown(
        '<p class="lead">Sistemi ilk kez kullanıyorsanız bu sayfayı bir kez okumanız yeterlidir.</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(3, gap="medium")
    steps = [
        ("1", "Görüntüleri yükleyin", "Aynı lezyonun bir klinik fotoğrafını ve bir dermoskopik görüntüsünü yükleyin. "
         "Sistem görüntülerin netliğini, ışığını ve boyutunu otomatik kontrol eder."),
        ("2", "Hasta bilgilerini seçin", "Yaş aralığı, cinsiyet, ten rengi ve lezyonun bulunduğu bölge. "
         "Dördü de seçilmeden analiz başlamaz."),
        ("3", "Sonucu yorumlayın", "Önce üstteki yönlendirme kutusuna bakın. Ardından grup çubuğu ve en olası üç tür gelir. "
         "Özeti indirip yazdırabilirsiniz."),
    ]
    for col, (no, t, d) in zip(cols, steps):
        col.markdown(f'<div class="guide"><span class="step-no">{no}</span><div class="card-title">{t}</div>'
                     f'<p class="card-text">{d}</p></div>', unsafe_allow_html=True)

    _h("Sonuç ekranındaki renkler")
    st.markdown(
        '<div class="legend-big">'
        '<div><span class="dot" style="background:#B4234A"></span><b>Kırmızı – Öncelikli değerlendirme:</b> '
        "kötü huylu türlerin toplam olasılığı yüksek. Kısa sürede dermatoloğa yönlendirme önerilir.</div>"
        '<div><span class="dot" style="background:#C27A12"></span><b>Turuncu – Değerlendirme önerilir:</b> '
        "kansere dönüşebilen veya kötü huylu türlere benzerlik var ya da model kararsız.</div>"
        '<div><span class="dot" style="background:#2A8C84"></span><b>Yeşil – İyi huylu görünüm:</b> '
        "belirgin kötü huylu özellik yok. Lezyon değişirse yine hekime başvurulmalıdır.</div></div>",
        unsafe_allow_html=True,
    )

    _h("Sık sorulan sorular")
    faq = [
        ("Yüzdeler ne anlama geliyor?",
         "Modelin her tür için hesapladığı olasılıktır; 11 türün toplamı %100'dür. Kesinlik değil, benzerlik ölçüsüdür. "
         "Örneğin %60, 'kesin bu' değil, 'en çok buna benziyor' demektir."),
        ("'Model kararsız' uyarısı neden çıkıyor?",
         "En yüksek olasılık düşükse veya ilk iki sonuç birbirine çok yakınsa model tek bir türü ayırt edemiyor demektir. "
         "Bu durumda sıralamaya güvenilmemeli ve hekim değerlendirmesine öncelik verilmelidir."),
        ("Neden iki ayrı görüntü gerekiyor?",
         "Klinik fotoğraf lezyonun genel görünüşünü, dermoskopik görüntü ise yüzeyin altındaki desenleri gösterir. "
         "Hekimler de ikisini birlikte değerlendirir; model de aynı şekilde çalışır."),
        ("Görüntülerim saklanıyor mu?",
         "Hayır. Görüntüler yalnızca analiz sırasında bellekte işlenir ve sayfa kapatıldığında silinir. "
         "Oturum geçmişi de yalnızca sayfa açık kaldığı sürece tutulur; kimseyle paylaşılmaz."),
        ("Telefonda çalışır mı?",
         "Evet. Sayfa telefon ekranına uyum sağlar; görüntüler telefonun galerisinden yüklenebilir."),
    ]
    for q, ans in faq:
        with st.expander(q):
            st.markdown(ans)
