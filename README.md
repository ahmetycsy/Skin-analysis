---
title: Skin Lesion Decision Support
emoji: 🩺
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: cc-by-nc-4.0
short_description: Multimodal skin lesion decision-support prototype (MILK10k)
---

# Cilt Lezyonu Karar Destek Prototipi (Streamlit arayüzü)

Bu klasör Ahmet'in paketidir. Şu an SAHTE modelle çalışır; gerçek model gelince `predictor.py` içindeki
`RealPredictor` doldurulur, `app.py` değişmez.

## 1. Bilgisayarında çalıştır
1. Python 3.10+ kurulu olsun (python.org). Adım adım anlatım için "Arayüz Kurulum Rehberi" dokümanına bak.
2. Terminal'i aç, klasöre geç ve çalıştır (Mac):
   ```
   cd "~/Desktop/vlaude code"
   pip3 install -r requirements.txt
   python3 -m streamlit run app.py
   ```
   (Windows'ta `python` ve `pip` yazılır.)
3. Tarayıcıda açılan sayfada iki herhangi bir resim yükle, "Analiz et"e bas. (Sahte model olduğu uyarısı görünmeli.)

## 2. İnternette yayınla (ücretsiz)
1. Kod GitHub'da **public** bir repoda durur (GitHub Desktop ile gönderilir). Veri ve model dosyaları
   `.gitignore` sayesinde GitHub'a gitmez (MILK10k lisansı CC-BY-NC).
2. share.streamlit.io'ya GitHub hesabınla gir → "Create app" → repoyu seç, Branch `main`, Main file `app.py`, **Advanced settings → Python 3.12** → Deploy.
   (Python 3.14 seçilirse altair kütüphanesi hata verir.)
3. Birkaç dakika sonra herkese açık bir link çıkar. Bu linki rapora ve sunuma koy.
4. Kodda değişiklik olunca GitHub Desktop'ta "Commit" + "Push" yeterli; site kendini günceller.

### Hekim PIN'i
Hekim modu için Streamlit Cloud'da **Manage app → Settings → Secrets** bölümüne şunu ekleyin (kendi PIN'inizi yazın):
```
DOCTOR_PIN = "1234"
```
Tanımlanmazsa koddaki demo PIN kullanılır; kod herkese açık olduğu için yayında mutlaka değiştirin.

## 3. Eğitim notebook'u
`kaggle/milk10k_train.ipynb` Kaggle'da çalışır (`MODE = derm | clin | meta`). Kaynağı `kaggle/milk10k_train.py`.

## 4. Gerçek modeli bağlama (Bedirhan modeli teslim edince)
1. Model dosyalarını `models/` klasörüne koy.
2. `predictor.py` içindeki `RealPredictor` sınıfını modelin biçimine göre doldur (yapay zekaya bu dosyayı ve model dosyasını ver).
3. `models/` içine boş bir `READY` dosyası ekle; uygulama otomatik gerçek modele geçer.

Model sözleşmesi: girdi = klinik görsel + dermoskopik görsel + (yaş, cinsiyet, ten rengi 0–5, bölge);
çıktı = `CLASSES` sırasında 11 olasılık (toplamı 1).

## Notlar
- Streamlit Community Cloud'un belleği sınırlıdır (yaklaşık 1 GB). Model dosyası büyükse Hugging Face Spaces'e geç.
- GitHub 100 MB'tan büyük dosyayı kabul etmez; büyük model dosyasını Hugging Face'e yükle.
