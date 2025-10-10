VJEPA2 Finetuning

- [Model Eğitimi (src/train.py)](#model-eğitimi-srctrainpy)
- [Model Çıkarımı (src/inference.py)](#model-çıkarımı-srcinferencepy)

## Gereksinimler

- Python 3.10 veya daha yeni bir sürüm
- OpenCV (`opencv-python`) ve PyTorch ekosistemi bileşenleri (`torch`, `torchvision`, `torchcodec`, `transformers`)

Gerekli paketleri kurmak için örnek:

```bash
pip install -r requirements.txt  # Eğer gereksinim dosyanız varsa
pip install opencv-python torch torchvision torchcodec transformers
```

## Model Eğitimi (`src/train.py`)

`src/train.py`, VJEPA2 modelini kendi veri kümeniz üzerinde ince ayar (fine-tuning) yapmak için kullanılır. Varsayılan olarak veri kümesinin `data/UCF101_subset/` altında `train/`, `val/`, `test/` yapısında bulunması beklenir.

### Temel komut

```bash
python src/train.py
```

Varsayılan konfigürasyonla eğitimi başlatır. Çıktılar `runs/vjepa2_finetune/` altında TensorBoard logları ve model kayıtları olarak saklanır.

### Epoch ve batch size ayarlama

```bash
python src/train.py --epochs 10 --batch-size 8
```

Bu komut toplam 10 epoch boyunca, batch boyutu 8 olacak şekilde eğitim yapar.

### train.py Argümanları

| Argüman | Açıklama |
|---------|----------|
| `--epochs INT` | Toplam epoch sayısını belirler. Varsayılan: `1`. |
| `--batch-size INT` | DataLoader batch boyutunu belirler. Varsayılan: `4`. |

> Eğitim betiğinin geri kalanı hiperparametreleri `TrainingConfig` sınıfında yönetir (örn. öğrenme oranı, gradyan biriktirme adımları, erken durdurma, vb.). İhtiyaç halinde kod üzerinde düzenleme yapabilirsiniz.

## Model Çıkarımı (`src/inference.py`)

`src/inference.py`, eğitilmiş modelinizi kullanarak videolarda sınıflandırma yapmak ve sonuçları videonun üzerine yazmak için kullanılır.

### Temel kullanım

```bash
python src/inference.py \
  --checkpoint runs/vjepa2_finetune/finetuned_model_best.pt \
  --video data/sample.mp4 \
  --enable-video-writer
```

Bu komut:
- `finetuned_model_best.pt` kontrol noktasını yükler,
- `data/sample.mp4` üzerinde inference yapar,
- tahmin edilen etiketi videonun üzerine yazarak kaydeder (`sample_annotated.mp4`).

### Çıkış dosyası adını belirlemek

```bash
python src/inference.py \
  --checkpoint runs/vjepa2_finetune/finetuned_model_best.pt \
  --video data/sample.mp4 \
  --output outputs/sample_result.mp4 \
  --enable-video-writer
```

`--output` parametresi yalnızca `--enable-video-writer` ile birlikte kullanılabilir; aksi takdirde hata verir.

### Etiket haritası kullanma

```bash
python src/inference.py \
  --checkpoint runs/vjepa2_finetune/finetuned_model_best.pt \
  --video data/sample.mp4 \
  --label-map runs/vjepa2_finetune/label2id.json \
  --enable-video-writer
```

Etiketleri insan okunabilir isimlere dönüştürmek için JSON formatındaki bir etiket haritası kullanabilirsiniz.

### Gerçek zamanlı önizleme

```bash
python src/inference.py \
  --checkpoint runs/vjepa2_finetune/finetuned_model_best.pt \
  --video data/sample.mp4 \
  --enable-video-writer \
  --show
```

Inference sırasında işlenen video bir pencere üzerinde gösterilir. Pencereyi kapatmak için `q` veya `Esc` tuşlarına basabilirsiniz.

### inference.py Argümanları

| Argüman | Açıklama |
|---------|----------|
| `--checkpoint PATH` | Eğitilmiş model kontrol noktası (zorunlu). |
| `--video PATH` | İnferans yapılacak video dosyası (zorunlu). |
| `--output PATH` | Annotasyonlu videonun kaydedileceği yol. `--enable-video-writer` ile birlikte kullanılmalıdır. Belirtilmezse `<video_stem>_annotated.mp4`. |
| `--enable-video-writer` | Annotasyonlu video çıktısını diske yazar. |
| `--label-map PATH` | Etiket dönüşümü için opsiyonel JSON haritası. Yoksa modelin kendi etiketleri kullanılır. |
| `--device DEVICE` | Kullanılacak PyTorch cihazı (`cuda`, `cpu`, `mps` vb.). Varsayılan: GPU varsa `cuda`, yoksa `cpu`. |
| `--frames-stride INT` | Klibi oluştururken kareler arası atlanacak adım sayısı. Varsayılan: `1`. |
| `--clip-length INT` | Her inferans klibinin kare sayısı. Modelde tanımlı değere varsayılan olarak düşer. |
| `--disable-amp` | CUDA ortamında otomatik karışık hassasiyeti (AMP) kapatır. |
| `--top-k INT` | Log’lanacak üst tahmin sayısı. Varsayılan: `5`. |
| `--show` | İnferans bittikten sonra çıktı videosunu görüntülemeye çalışır. En iyi deneyim için `--enable-video-writer` ile kullanın. |
