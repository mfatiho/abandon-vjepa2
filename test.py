import argparse
import pathlib
from collections import deque
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

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


def predict_clip(
    frames_rgb: Sequence[np.ndarray],
    *,
    frames_per_clip: int,
    processor: AutoVideoProcessor,
    model: AutoModelForVideoClassification,
) -> Tuple[str, float]:
    clip = list(frames_rgb)
    if clip and len(clip) < frames_per_clip:
        clip = clip + [clip[-1]] * (frames_per_clip - len(clip))
    inputs = processor(videos=[clip[:frames_per_clip]], return_tensors="pt")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    amp_enabled = model.device.type == "cuda"
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
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

    writer = None
    window_open = False

    clip_buffer: Deque[np.ndarray] = deque()
    total_frames = 0
    processed_clips = 0

    def ensure_writer(width: int, height: int) -> None:
        nonlocal writer
        if writer is None and output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to create writer for {output_path}")

    def ensure_window() -> None:
        nonlocal window_open
        if show and not window_open:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            window_open = True

    def prepare_clip_rgb(frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    def emit_clip(frames: Sequence[np.ndarray], label: str, prob: float) -> bool:
        return annotate_and_emit(
            frames,
            label=label,
            probability=prob,
            writer=writer,
            show=window_open,
        )

    def flush_clip() -> bool:
        nonlocal processed_clips
        if not clip_buffer:
            return False
        frames = list(clip_buffer)
        clip_rgb = prepare_clip_rgb(frames)
        label, prob = predict_clip(
            clip_rgb,
            frames_per_clip=frames_per_clip,
            processor=processor,
            model=model,
        )
        processed_clips += 1
        stop_requested = emit_clip(frames, label, prob)
        clip_buffer.clear()
        return stop_requested

    stop = False
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                stop = flush_clip()
                break

            total_frames += 1

            ensure_writer(frame.shape[1], frame.shape[0])
            ensure_window()

            clip_buffer.append(frame)
            if len(clip_buffer) == frames_per_clip:
                stop = flush_clip()
                if stop:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if window_open:
            cv2.destroyWindow(WINDOW_NAME)
            window_open = False

    if total_frames == 0:
        raise RuntimeError(f"No frames read from {video_path}")

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
