import argparse
import pathlib
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoModelForVideoClassification, AutoVideoProcessor


HF_REPO = "facebook/vjepa2-vitg-fpc64-384-ssv2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_NAME = "V-JEPA2 Top-1 Prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V-JEPA2 on full videos and overlay the running top-1 label."
    )
    parser.add_argument("--video", type=pathlib.Path, help="Path to a single video file to process.")
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        help="Root directory of an 'abandon' style dataset containing video files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Optional directory where annotated videos will be written.",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        help="Override model frames_per_clip (default: value from the checkpoint).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a window that streams annotated frames (press 'q' to stop early).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output so only warnings/errors are printed.",
    )
    args = parser.parse_args()

    if not args.video and not args.dataset:
        parser.error("Please provide either --video or --dataset.")
    return args


def discover_videos(video: Optional[pathlib.Path], dataset: Optional[pathlib.Path]) -> List[pathlib.Path]:
    if video is not None:
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")
        if not video.is_file():
            raise ValueError(f"--video must point to a file, received: {video}")
        return [video]

    assert dataset is not None  # guarded by parse_args
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset}")
    if not dataset.is_dir():
        raise ValueError(f"--dataset must point to a directory, received: {dataset}")

    exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [
        path
        for path in sorted(dataset.rglob("*"))
        if path.is_file() and path.suffix.lower() in exts
    ]
    if not videos:
        raise FileNotFoundError(f"No video files with extensions {sorted(exts)} found under {dataset}")
    return videos


def ensure_relative(path: pathlib.Path, root: Optional[pathlib.Path]) -> Optional[pathlib.Path]:
    if root is None:
        return None
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def build_output_path(
    video_path: pathlib.Path,
    dataset_root: Optional[pathlib.Path],
    output_dir: Optional[pathlib.Path],
) -> Optional[pathlib.Path]:
    if output_dir is None:
        return None

    rel = ensure_relative(video_path, dataset_root)
    if rel is None:
        rel = pathlib.Path(video_path.stem + ".mp4")
    else:
        rel = rel.with_suffix(".mp4")

    output_path = output_dir / rel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def to_model_batch(frames_rgb: List[np.ndarray], frames_per_clip: int) -> np.ndarray:
    padded = frames_rgb
    if len(frames_rgb) < frames_per_clip and frames_rgb:
        padded = frames_rgb + [frames_rgb[-1]] * (frames_per_clip - len(frames_rgb))
    array = np.stack(padded, axis=0)  # (T, H, W, C)
    return array.transpose(0, 3, 1, 2)  # (T, C, H, W)


def predict_clip(
    frames_rgb: List[np.ndarray],
    *,
    frames_per_clip: int,
    processor: AutoVideoProcessor,
    model: AutoModelForVideoClassification,
) -> Tuple[str, float]:
    video_batch = to_model_batch(frames_rgb, frames_per_clip)
    inputs = processor(video_batch, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top_prob, top_idx = torch.max(probs, dim=-1)
    label = model.config.id2label[top_idx.item()]
    return label, float(top_prob.item())


def annotate_and_emit(
    frames_bgr: Iterable[np.ndarray],
    *,
    label: str,
    probability: float,
    writer: Optional[cv2.VideoWriter],
    show: bool,
) -> bool:
    text = f"{label} ({probability * 100:.1f}%)"
    for frame in frames_bgr:
        cv2.putText(
            frame,
            text,
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 220, 20),
            2,
            cv2.LINE_AA,
        )
        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True
    return False


def process_video(
    video_path: pathlib.Path,
    *,
    model: AutoModelForVideoClassification,
    processor: AutoVideoProcessor,
    frames_per_clip: int,
    show: bool,
    output_path: Optional[pathlib.Path],
    quiet: bool,
) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 25.0

    ret, frame = capture.read()
    if not ret:
        capture.release()
        raise RuntimeError(f"No frames read from {video_path}")

    height, width = frame.shape[:2]
    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to create writer for {output_path}")

    if show:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frames_rgb: List[np.ndarray] = []
    frames_bgr: List[np.ndarray] = []
    total_frames = 0
    processed_clips = 0

    def flush_clip() -> bool:
        nonlocal frames_rgb, frames_bgr, processed_clips
        if not frames_rgb:
            return False
        label, prob = predict_clip(
            frames_rgb,
            frames_per_clip=frames_per_clip,
            processor=processor,
            model=model,
        )
        processed_clips += 1
        stop_requested = annotate_and_emit(
            frames_bgr,
            label=label,
            probability=prob,
            writer=writer,
            show=show,
        )
        frames_rgb = []
        frames_bgr = []
        return stop_requested

    frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frames_bgr.append(frame)
    total_frames += 1

    stop = False
    while True:
        ret, frame = capture.read()
        if not ret:
            stop = flush_clip()
            break
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames_bgr.append(frame)
        total_frames += 1
        if len(frames_rgb) == frames_per_clip:
            stop = flush_clip()
            if stop:
                break

    capture.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyWindow(WINDOW_NAME)

    if not quiet:
        print(
            f"{video_path.name}: processed {total_frames} frames as {processed_clips} clip(s) "
            f"with frames_per_clip={frames_per_clip}"
        )

    return stop


def main() -> None:
    args = parse_args()
    model = AutoModelForVideoClassification.from_pretrained(HF_REPO).to(DEVICE)
    processor = AutoVideoProcessor.from_pretrained(HF_REPO)

    frames_per_clip = args.frames_per_clip or model.config.frames_per_clip
    videos = discover_videos(args.video, args.dataset)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for video_path in videos:
        output_path = build_output_path(video_path, args.dataset, args.output_dir)
        if not args.quiet:
            destination = f" -> {output_path}" if output_path else ""
            print(f"Processing {video_path}{destination}")
        stop_requested = process_video(
            video_path,
            model=model,
            processor=processor,
            frames_per_clip=frames_per_clip,
            show=args.show,
            output_path=output_path,
            quiet=args.quiet,
        )
        if stop_requested:
            break


if __name__ == "__main__":
    main()
