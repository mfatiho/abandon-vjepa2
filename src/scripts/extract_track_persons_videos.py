import cv2
from ultralytics import YOLO
import os
import argparse
from collections import defaultdict


def expand_box_with_padding(box, padding, frame_width, frame_height):
    if padding <= 0:
        return box
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(frame_width, x2 + padding),
        min(frame_height, y2 + padding),
    ]


def track_and_crop_memory_efficient(video_path, output_dir, confidence_threshold, min_frame_count, bbox_padding):
    """
    Bir videodaki kişileri takip eder, her birini kırpar ve ayrı videolar olarak kaydeder.
    Bu son derece bellek verimli versiyon, iki aşamalı bir işlem kullanarak RAM kullanımını minimize eder.

    Args:
        video_path (str): İşlenecek video dosyasının yolu.
        output_dir (str): Kırpılmış videoların kaydedileceği klasör.
        confidence_threshold (float): Nesne tespiti için minimum güven eşiği.
        min_frame_count (int): Videonun kaydedilebilmesi için gereken minimum kare sayısı.
        bbox_padding (int): BBox etrafına piksel cinsinden eklenecek boşluk.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("YOLOv11 modeli yükleniyor...")
    model = YOLO('yolo11m.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Hata: '{video_path}' video dosyası açılamadı.")
        return
    fps_value = cap.get(cv2.CAP_PROP_FPS)
    fps_value = float(fps_value) if fps_value and fps_value > 0 else 30.0
    total_frames_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames_prop if total_frames_prop > 0 else None
    if total_frames is not None:
        duration_seconds = total_frames / fps_value
        print(
            f"Video bilgisi: {total_frames} kare, yaklaşık {duration_seconds:.2f} saniye "
            f"(FPS={fps_value:.2f})."
        )
    else:
        print(f"Video FPS değeri {fps_value:.2f}. Toplam kare sayısı tespit edilemedi.")

    frame_limit = None
    while True:
        user_seconds = input(
            "İşlenecek süreyi saniye cinsinden girin (boş bırakıp Enter'a basarsanız tüm video işlenir): "
        ).strip()
        if user_seconds == "":
            frame_limit = None
            break
        try:
            seconds_to_process = float(user_seconds)
        except ValueError:
            print("Lütfen geçerli bir sayı girin.")
            continue
        if seconds_to_process <= 0:
            print("Süre pozitif olmalıdır.")
            continue
        frames_requested = int(seconds_to_process * fps_value)
        if frames_requested <= 0:
            frames_requested = 1
        frame_limit = (
            min(frames_requested, total_frames) if total_frames is not None else frames_requested
        )
        break

    if frame_limit is None:
        print("Tüm video işlenecek.")
    else:
        approx_seconds = frame_limit / fps_value
        print(f"{frame_limit} kare (~{approx_seconds:.2f} saniye) işlenecek.")

    # --- 1. AŞAMA: METADATA TOPLAMA ---
    print("1. Aşama: Metadata toplanıyor (video ilk kez taranıyor)...")
    
    # {track_id: {'boxes': {frame_num: [x1, y1, x2, y2]}, 'max_dims': [w, h]}}
    tracks_metadata = defaultdict(lambda: {'boxes': {}, 'max_dims': [0, 0]})
    
    frame_count = 0
    while True:
        if frame_limit is not None and frame_count >= frame_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[0], conf=confidence_threshold, verbose=False)

        frame_height, frame_width = frame.shape[:2]

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for track_id, box in zip(track_ids, boxes):
                x1, y1, x2, y2 = map(int, box)
                padded_box = expand_box_with_padding((x1, y1, x2, y2), bbox_padding, frame_width, frame_height)
                px1, py1, px2, py2 = map(int, padded_box)

                if px1 < px2 and py1 < py2:
                    w, h = px2 - px1, py2 - py1
                    # Metadata'yı sakla (karenin kendisini değil)
                    tracks_metadata[track_id]['boxes'][frame_count] = [px1, py1, px2, py2]
                    
                    # Maksimum boyutları güncelle
                    if w > tracks_metadata[track_id]['max_dims'][0]:
                        tracks_metadata[track_id]['max_dims'][0] = w
                    if h > tracks_metadata[track_id]['max_dims'][1]:
                        tracks_metadata[track_id]['max_dims'][1] = h
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  {frame_count} kare tarandı...")

    print("Metadata toplama tamamlandı. Toplamda {} benzersiz kişi bulundu.".format(len(tracks_metadata)))
    cap.release()

    eligible_track_ids = [
        track_id for track_id, data in tracks_metadata.items()
        if len(data['boxes']) >= min_frame_count
    ]
    skipped_tracks = len(tracks_metadata) - len(eligible_track_ids)
    if skipped_tracks > 0:
        print(f"{skipped_tracks} kişi minimum kare şartını karşılamadığı için atlandı.")
    if not eligible_track_ids:
        print("Minimum kare şartını karşılayan kişi bulunamadı. İşlem sonlandırılıyor.")
        return

    # --- 2. AŞAMA: VİDEOLARI YAZMA ---
    print("\n2. Aşama: Videolar oluşturuluyor (video ikinci kez taranıyor)...")

    # Her kişi için VideoWriter nesnelerini oluştur
    video_writers = {}
    fps = fps_value
    
    for track_id in eligible_track_ids:
        data = tracks_metadata[track_id]
        max_w, max_h = data['max_dims']
        if max_w == 0 or max_h == 0: continue # Hiç kutu bulunamadıysa atla
        
        output_path = os.path.join(output_dir, f'takip_{track_id}_kisi.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writers[track_id] = cv2.VideoWriter(output_path, fourcc, fps, (max_w, max_h))

    # Videoyu yeniden başlat
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        if frame_limit is not None and frame_count >= frame_limit:
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        # Bu karede görünen kişileri işle
        for track_id, writer in video_writers.items():
            if frame_count in tracks_metadata[track_id]['boxes']:
                box = tracks_metadata[track_id]['boxes'][frame_count]
                x1, y1, x2, y2 = box
                
                # Anlık olarak kırp
                cropped_frame = frame[y1:y2, x1:x2]
                
                # Hedef boyuta getir (padding ile)
                max_w, max_h = tracks_metadata[track_id]['max_dims']
                h, w, _ = cropped_frame.shape
                
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            
                padded_frame = cv2.copyMakeBorder(cropped_frame, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                
                # Doğrudan diske yaz
                writer.write(padded_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  {frame_count} kare yazıldı...")

    # Tüm dosyaları kapat
    cap.release()
    for writer in video_writers.values():
        writer.release()
        
    print(f"\nİşlem tamam! Videolar '{output_dir}' klasörüne başarıyla kaydedildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 ile bir videodaki kişileri takip eden ve her birini ayrı videolara kırpan, bellek verimli bir script.")
    parser.add_argument("video_path", type=str, help="İşlenecek video dosyasının yolu.")
    parser.add_argument("--output-dir", type=str, default="kirpilmis_videolar_mem_efficient", help="Kırpılmış videoların kaydedileceği klasör.")
    parser.add_argument("--conf", type=float, default=0.4, help="Tespit için minimum güven eşiği (0.0 ile 1.0 arası).")
    parser.add_argument("--min-frame-count", type=int, default=32, help="Kişi videolarının kaydedilmesi için gereken minimum kare sayısı.")
    parser.add_argument("--bbox-padding", type=int, default=20, help="Kişi videoları kaydedilirken bounding box etrafına eklenecek piksel cinsinden boşluk.")
    
    args = parser.parse_args()
    if args.min_frame_count < 1:
        parser.error("--min-frame-count parametresi pozitif bir sayı olmalıdır.")
    if args.bbox_padding < 0:
        parser.error("--bbox-padding parametresi negatif olamaz.")

    track_and_crop_memory_efficient(
        args.video_path,
        args.output_dir,
        args.conf,
        args.min_frame_count,
        args.bbox_padding,
    )
