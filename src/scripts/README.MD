# SCRIPTS

- [Crop Video From CVAT 1.1 Annotations](#crop-video-from-cvat-11-annotations)
- [Track Tabanlı Kişi Videolarını Çıkarma](#track-tabanlı-kişi-videolarını-çıkarma-extract_track_persons_videospy)

## Gereksinimler

- Python 3.10 veya daha yeni bir sürüm
- OpenCV (`opencv-python` paketi); betik `cv2` modülünü kullanır

Gerekli paketleri kurmak için örnek:

```bash
pip install -r requirements.txt  # Eğer gereksinim dosyanız varsa
pip install opencv-python        # Yoksa temel bağımlılık
```

# Crop Video From CVAT 1.1 Annotations

Bu depo, CVAT 1.1 formatındaki anotasyon dosyalarından her iz (track) için kırpılmış video parçaları üretmek amacıyla kullanılan `src/scripts/crop_video_from_cvat11.py` betiğini içerir. Aşağıda betiğin nasıl kullanılacağı, aldığı argümanların açıklamaları ve uçtan uca çalışma örnekleri yer almaktadır.

## Tek Video / XML Çalıştırma Örnekleri

### Basit kullanım

```bash
python src/scripts/crop_video_from_cvat11.py \
  --input-video data/video.mp4 \
  --annotations data/annotations.xml \
  --output-dir crop_videos
```

Bu komut:
- `data/video.mp4` dosyasını açar,
- `data/annotations.xml` içindeki her track için kırpılmış mp4 videolar üretir,
- çıktıları `crop_videos/` dizinine yazar.

### ZIP içindeki anotasyonla çalışma

```bash
python src/scripts/crop_video_from_cvat11.py \
  --input-video data/video.mp4 \
  --annotations data/task_annotations.zip \
  --output-dir crop_videos_zip
```

ZIP dosyasında `annotations.xml` bulunmalıdır. Betik ZIP’i açıp XML’i otomatik olarak okur.

### Etiket filtresi ile çalışma

```bash
python src/scripts/crop_video_from_cvat11.py \
  --input-video data/video.mp4 \
  --annotations data/annotations.xml \
  --labels "Birakilan Nesne" "Alinan Nesne" \
  --output-dir crop_videos_filtered
```

Sadece belirtilen etiketlere sahip track’ler işlenir; diğerleri yok sayılır.

### Başlangıç ve bitişe ekstra kare ekleme

```bash
python src/scripts/crop_video_from_cvat11.py \
  --input-video data/video.mp4 \
  --annotations data/annotations.xml \
  --start-frame-offset 15 \
  --end-frame-offset 10 \
  --output-dir crop_videos_offset
```

- Her track için kırpılmış video, anotasyonların başladığı kareden 15 kare önceye (0’dan küçükse 0’a kadar) ve bittiği kareden 10 kare sonrasına kadar uzatılır.

## CSV (Toplu) Çalıştırma Örneği

CSV dosyası `video_yolu,annotasyon_zip_yolu,split` formatında satırlar içermelidir. Örnek satır:

```
/veri/video1.mp4,/veri/video1_annotations.zip,train
/veri/video2.mp4,/veri/video2_annotations.zip,test
```

### Toplu işleme komutu

```bash
python src/scripts/crop_video_from_cvat11.py \
  --csv data/dataset.csv \
  --output-dir runs \
  --start-frame-offset 5 \
  --end-frame-offset 5
```

Bu komut:
- CSV dosyasındaki her satırı işler,
- çıktıları `runs/<split>/Abandon/` alt dizinlerine yazar,
- `--start-frame-offset` ve `--end-frame-offset` değerlerini her track için uygular.

## Tüm Argümanlar ve Açıklamaları

| Argüman | Kısa | Açıklama |
|---------|------|----------|
| `--csv PATH` | – | CSV dosyası üzerinden toplu işleme modunu etkinleştirir. Sağlanırsa tekil video argümanları yerine CSV satırları kullanılır. |
| `--input-video PATH` | `-i PATH` | Tekil modda işlenecek video dosyası. CSV modu kullanmıyorsanız zorunludur. |
| `--annotations PATH` | `-a PATH` | CVAT 1.1 XML veya ZIP dosyası. Tekil modda zorunlu. ZIP dosyasının içinde `annotations.xml` bulunmalıdır. |
| `--output-dir PATH` | `-o PATH` | Kırpılmış videoların yazılacağı dizin. Varsayılan `tracks/`. CSV modunda `split/Abandon/` alt dizinleri oluşturulur. |
| `--codec CODE` | – | Çıkış videolarının kullanacağı FourCC kodu. Varsayılan `mp4v`. |
| `--fallback-fps FLOAT` | – | Girdi videosunun FPS’i okunamazsa kullanılacak yedek FPS değeri. Varsayılan `25.0`. |
| `--start-frame-offset INT` | – | Her track için, anotasyonların başlayışından önce eklenecek ekstra kare sayısı. Negatif start kareleri 0’a kırpılır. Varsayılan `0`. |
| `--end-frame-offset INT` | – | Her track için, anotasyonların bitişinden sonra eklenecek ekstra kare sayısı. Varsayılan `0`. |
| `--labels LABEL [LABEL ...]` | – | Yalnızca belirtilen etiketlere sahip track’leri işler. Sağlanmazsa tüm etiketler dahil edilir. |
| `--verbose` | – | Daha ayrıntılı (DEBUG seviyesinde) log çıktısı üretir. |
| `-h`, `--help` | – | Yardım mesajını gösterir ve çıkar. |

## Komut Çalıştırma İpuçları

- Betik log çıktılarıyla işlem durumunu bildirir; `--verbose` ile ayrıntı seviyesini yükseltebilirsiniz.
- Çıktı dosya isimleri, video adı ve track kimliğine göre otomatik üretilir (`<video>_track_<id>_<label>.mp4`).
- CSV modunda bireysel satırlar hata verirse betik diğer satırları işlemeye devam eder ve her hata için log yazar.
- `start_frame_offset` veya `end_frame_offset` değerleri, track’in alanını genişletse bile kırpılmış videoların boyutu anotasyonlardaki kutulara göre hesaplanmaya devam eder.

## Yardım Mesajını Görüntüleme

Tüm argümanları ve varsayılan değerleri hızlıca görmek için:

```bash
python src/scripts/crop_video_from_cvat11.py -h
```

# Track Tabanlı Kişi Videolarını Çıkarma (`extract_track_persons_videos.py`)

`src/scripts/extract_track_persons_videos.py`, tek bir videodaki kişileri YOLOv11 ile takip eder ve her kişi için ayrı kırpılmış videolar üretir. Betik, belleği korumak için videoyu iki kez tarar: ilk turda sadece bounding box metadata’sını toplar, ikinci turda ise videoları üretir.

> Gereksinimler:
> - `ultralytics` paketi (`pip install ultralytics`)
> - `yolo11m.pt` ağırlık dosyası proje kökünde bulunmalı (repo ile birlikte gelir).

### Temel kullanım

```bash
python src/scripts/extract_track_persons_videos.py \
  data/input_video.mp4 \
  --output-dir outputs/person_tracks
```

Bu komut:
- Videodaki her kişiyi takip eder,
- Her kişi için `outputs/person_tracks/takip_<track_id>_kisi.mp4` dosyaları oluşturur,
- Çalışma sırasında, kaç saniyelik bölümü işlemek istediğinizi sorar (boş bırakılırsa tüm video işlenir).

### Gelişmiş örnek

```bash
python src/scripts/extract_track_persons_videos.py \
  data/input_video.mp4 \
  --output-dir outputs/person_tracks \
  --conf 0.55 \
  --min-frame-count 48 \
  --bbox-padding 40
```

- `--conf` ile daha yüksek güven eşiği ayarlanır.
- `--min-frame-count` değerini artırarak kısa süre görünen kişilerin kayıt edilmesini engelleyebilirsiniz.
- `--bbox-padding` ile kişilerin etrafına ekstra çerçeve boşluğu bırakabilirsiniz.

### Argümanlar

| Argüman | Açıklama |
|---------|----------|
| `video-path` (pozisyonel) | İşlenecek video dosyasının yolu. Betiği çağırırken ilk argüman olarak verilir. |
| `--output-dir PATH` | Kırpılmış videoların kaydedileceği klasör. Varsayılan: `kirpilmis_videolar_mem_efficient`. |
| `--conf FLOAT` | YOLO tespit güven eşiği (0.0 – 1.0). Varsayılan: `0.4`. |
| `--min-frame-count INT` | Bir kişinin videoya kaydedilmesi için gereken minimum kare sayısı. Varsayılan: `32`. |
| `--bbox-padding INT` | Bounding box etrafına eklenecek piksel cinsinden boşluk. Varsayılan: `20`. |

### Çalıştırma İpuçları

- İlk taramada bulunan track sayısı ve kare bilgileri konsola yazdırılır; minimum kare şartına uymayan track’ler otomatik olarak elenir.
- Betik interaktif olarak kaç saniyelik bölümün işleneceğini sorar. Boş bırakılırsa tüm video işlenir.
- Her track için çıktı videoları sabit boyutta tutulur; küçük bölgeler etrafına siyah padding eklenir.
- Uzun videolarda performans için GPU hızlandırması önerilir (YOLOv11, CUDA varsa otomatik kullanır).
