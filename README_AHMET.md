# Gerçek modeli arayüze bağlama (Tuğalp'ten)

`app.py`'ye dokunmak gerekmiyor. Sadece üç dosya değişiyor, model dosyaları `models/` klasörüne giriyor.

## 1. Dosyaları yerleştir
| Dosya | Nereye | Ne değişti |
|---|---|---|
| `predictor.py` | repo kökü (eskisinin yerine) | Sadece `RealPredictor` sınıfı dolduruldu; geri kalan her şey aynı |
| `gradcam.py` | repo kökü (eskisinin yerine) | `preprocess()` artık `mean` ve `std` parametresi alıyor (varsayılan B3, yani eski davranış aynı) |
| `models/model_config.json` | `models/` | Model listesi, görüntü boyutu, normalizasyon, kalibrasyon, conformal eşiği, Grad-CAM modeli |
| 4 adet `.pt` dosyası | `models/` | B3 ve V2, dermoskopik ve klinik |
| boş `READY` dosyası | `models/` | Uygulama gerçek modele geçer |

`requirements_ek.txt` içindeki satırları `requirements.txt`'nin sonuna ekle.

## 2. Nasıl çalışıyor
- `predict(klinik, dermoskopik, meta)` → 11 kalibre olasılık (sözleşme aynı).
- **Hekim modu:** iki görüntü → 4 model. **Hasta modu:** `dermoskopik=None` → yalnızca 2 klinik model.
  Şu an arayüz iki görüntüyü zorunlu tuttuğu için hep hekim modu çalışır. Hasta modu için
  yükleme adımında dermoskopiyi isteğe bağlı yapman ve `None` göndermen yeterli
  (vaka kaydı/PDF'te dermoskopik küçük resmin boş olabilmesi gerekir).
- `explain()` → Grad-CAM. Hekim modunda **dermoskopik B3**, hasta modunda **klinik B3** modelinden;
  senin `blocks[-2]` katman seçimin aynen kullanılıyor. Dönen sınıf kodu, haritanın çizildiği sınıf.
- İsteğe bağlı: `predictor.prediction_set(probs, dermoscopic_given)` → %90 güvenli olası tanılar listesi
  (ayırıcı tanı / güven seviyesi göstermek istersen).

## 3. Bilmen gerekenler
- **Eşikler:** `app.py`'deki geçici eşikler (%30 / %40) şimdilik kalabilir. 5 katlı CV bitince
  kalibrasyonlu sonuçlarla birlikte önerilen değerleri göndereceğim.
- **Bellek:** 4 model yüklü iken ~660 MB; telefon boyutunda (12 MP) fotoğrafla tahmin + Grad-CAM'de ~1 GB'a çıkıyor.
  Streamlit Cloud'un ~1 GB sınırı için çok sıkı. **Hugging Face Spaces** (README'de zaten yazıyor) daha güvenli.
- **Model dosyaları GitHub'a gitmiyor** (`.gitignore` doğru). Hugging Face'te bir model reposu açıp 4 `.pt` dosyasını
  oraya yükle, Secrets'a `MODEL_REPO = "kullanici/repo-adi"` ekle; dosyalar ilk açılışta otomatik iner.
  Modeller CC-BY-NC lisanslı veriyle eğitildi: model kartına "yalnızca ticari olmayan araştırma" notu ekle.
- **Ön işleme eğitimle birebir aynı** (fark 0, test edildi): ortadan kare kırp → 1,15× küçült → merkez kırp → normalize.
  Kullanıcı kırpması lezyonu çok sıkı keserse model eğitimde görmediği bir çerçeve görür; gerçek görüntülerle bir dene.
- **Bölge adları** (`head/neck` vs eğitimdeki `head_neck_face`) sadece Bedirhan'ın meta modelini etkiler; bu modeller metadata kullanmıyor.

## 4. Doğrulama sonuçları (1.048 lezyonluk kontrol seti, geçici)
| Mod | Macro F1 | Doğruluk | %90 conformal küme |
|---|---|---|---|
| Hekim (4 model) | 0,547 | %76,9 | ortalama 1,6 tanı |
| Hasta (2 klinik model) | 0,486 | %70,4 | ortalama 2,3 tanı |
