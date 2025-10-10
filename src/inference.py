from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time
from collections import deque
from typing import List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForVideoClassification, AutoVideoProcessor


LOGGER = logging.getLogger(__name__)

HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEFAULT_OUTPUT_SUFFIX = "_annotated.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a finetuned VJEPA2 model and annotate the video."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=pathlib.Path,
        help="Path to the finetuned model checkpoint produced by train.py.",
    )
    parser.add_argument(
        "--video",
        required=True,
        type=pathlib.Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help=(
            "Where to write the annotated video. "
            "Defaults to <video_stem>_annotated.mp4 alongside the input video."
        ),
    )
    parser.add_argument(
        "--enable-video-writer",
        action="store_true",
        help="Write the annotated video to disk.",
    )
    parser.add_argument(
        "--label-map",
        type=pathlib.Path,
        help=(
            "Optional JSON file describing the label mapping. "
            "Should map class ids (ints) to names (strings) or vice versa."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device to use (default: cuda if available, otherwise cpu).",
    )
    parser.add_argument(
        "--frames-stride",
        type=int,
        default=1,
        help="Number of frames to skip between sampled frames while forming each clip.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        help=(
            "How many frames each inference clip should contain. "
            "Defaults to the model's configured frames_per_clip."
        ),
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision even if CUDA is available.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many predictions to log to stdout (default: 5).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video after writing it to disk.",
    )
    return parser.parse_args()


def resolve_output_path(video_path: pathlib.Path, output_path: Optional[pathlib.Path]) -> pathlib.Path:
    if output_path is not None:
        return output_path
    return video_path.with_name(video_path.stem + DEFAULT_OUTPUT_SUFFIX)


def discover_label_map_path(checkpoint_path: pathlib.Path) -> Optional[pathlib.Path]:
    candidates = [
        checkpoint_path.with_suffix(".labels.json"),
        checkpoint_path.with_suffix(".label_map.json"),
        checkpoint_path.parent / "label2id.json",
        checkpoint_path.parent / "id2label.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_label_map(
    model: AutoModelForVideoClassification,
    *,
    checkpoint_path: pathlib.Path,
    provided_path: Optional[pathlib.Path],
) -> Mapping[int, str]:
    if provided_path and not provided_path.exists():
        raise FileNotFoundError(f"Label map file {provided_path} does not exist.")
    label_map_path = provided_path or discover_label_map_path(checkpoint_path)
    if isinstance(label_map_path, pathlib.Path):
        return parse_label_map_json(label_map_path)

    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, Mapping) and id2label:
        return {int(idx): str(name) for idx, name in id2label.items()}

    num_labels = model.config.num_labels
    LOGGER.warning(
        "Falling back to numeric label names because no label map was provided or discovered."
    )
    return {idx: f"class_{idx}" for idx in range(num_labels)}


def parse_label_map_json(path: pathlib.Path) -> Mapping[int, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, Mapping):
        # Accept either id->label or label->id.
        if all(isinstance(key, str) and key.isdigit() for key in payload.keys()):
            return {int(k): str(v) for k, v in payload.items()}
        if all(isinstance(value, int) for value in payload.values()):
            inverse = {int(v): str(k) for k, v in payload.items()}
            LOGGER.info("Loaded label map from %s (inverted label->id mapping).", path)
            return inverse
        raise ValueError(f"Unsupported label map structure in {path}")

    if isinstance(payload, Sequence):
        return {idx: str(label) for idx, label in enumerate(payload)}

    raise ValueError(f"Unsupported label map format in {path}")


def compute_clip_span(frames_per_clip: int, stride: int) -> int:
    frames_per_clip = max(1, frames_per_clip)
    stride = max(1, stride)
    return (frames_per_clip - 1) * stride + 1


def prepare_clip_from_buffer(
    buffer: Sequence[np.ndarray],
    *,
    frames_per_clip: int,
    stride: int,
) -> List[np.ndarray]:
    expected = compute_clip_span(frames_per_clip, stride)
    if len(buffer) < expected:
        raise ValueError("Insufficient frames available to build a clip.")

    frames = list(buffer)
    indices = range(len(frames) - expected, len(frames), max(1, stride))
    clip_rgb = [cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB) for idx in indices]
    if len(clip_rgb) < frames_per_clip:
        raise RuntimeError(
            "Internal error while constructing the clip: "
            f"expected at least {frames_per_clip} frames, got {len(clip_rgb)}."
        )
    return clip_rgb[:frames_per_clip]


def annotate_frame(frame: np.ndarray, text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    padding = 10
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    origin = (padding, padding + text_height)
    background_tl = (origin[0] - padding // 2, origin[1] - text_height - padding // 2)
    background_br = (origin[0] + text_width + padding // 2, origin[1] + padding // 2)

    cv2.rectangle(frame, background_tl, background_br, color=(0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        origin,
        font,
        font_scale,
        color=(255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def overlay_fps(frame: np.ndarray, fps_value: float) -> None:
    """Render running FPS estimate in the bottom-left corner of the frame."""
    if fps_value <= 0:
        return
    text = f"FPS: {fps_value:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    padding = 10
    height, width = frame.shape[:2]
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    origin = (padding, height - padding)
    background_tl = (origin[0] - padding // 2, origin[1] - text_height - padding // 2)
    background_br = (origin[0] + text_width + padding // 2, origin[1] + padding // 2)
    cv2.rectangle(frame, background_tl, background_br, color=(0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        text,
        origin,
        font,
        font_scale,
        color=(255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def prepare_clip_tensor(
    processor: AutoVideoProcessor,
    sampled_frames: Sequence[np.ndarray],
    *,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    inputs = processor(
        videos=[sampled_frames],
        return_tensors="pt",
    )
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def try_show_video(window_title: str, video_path: pathlib.Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        LOGGER.warning("Unable to open %s for display.", video_path)
        return

    LOGGER.info("Press 'q' or ESC to close the preview window.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    except cv2.error as exc:
        LOGGER.warning("Display not available: %s", exc)
    finally:
        capture.release()
        cv2.destroyWindow(window_title)


def load_model(
    checkpoint_path: pathlib.Path,
    *,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[AutoModelForVideoClassification, AutoVideoProcessor]:
    LOGGER.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    saved_config = None
    if isinstance(checkpoint_payload, Mapping) and "state_dict" in checkpoint_payload:
        state_dict = checkpoint_payload["state_dict"]
        saved_config = checkpoint_payload.get("config")
    else:
        state_dict = checkpoint_payload

    num_labels = infer_num_labels(state_dict)

    model_kwargs = {"ignore_mismatched_sizes": True}
    if saved_config:
        base_config = AutoConfig.from_pretrained(HF_REPO)
        base_config.update(saved_config)
        model_kwargs["config"] = base_config
        LOGGER.info(
            "Loaded config from checkpoint (num_labels=%s).",
            getattr(base_config, "num_labels", "unknown"),
        )
    elif num_labels:
        LOGGER.info("Detected %d labels from checkpoint", num_labels)
        model_kwargs["num_labels"] = num_labels

    model = AutoModelForVideoClassification.from_pretrained(HF_REPO, **model_kwargs)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    processor = AutoVideoProcessor.from_pretrained(HF_REPO)

    if saved_config:
        id2label = getattr(model.config, "id2label", None)
        if isinstance(id2label, Mapping):
            remapped = {}
            for key, value in id2label.items():
                try:
                    remapped[int(key)] = str(value)
                except (TypeError, ValueError):
                    remapped[str(key)] = str(value)
            model.config.id2label = remapped
        label2id = getattr(model.config, "label2id", None)
        if isinstance(label2id, Mapping):
            remapped = {}
            for key, value in label2id.items():
                try:
                    remapped[str(key)] = int(value)
                except (TypeError, ValueError):
                    remapped[str(key)] = value
            model.config.label2id = remapped

    LOGGER.info(
        "Model ready (num_labels=%d, frames_per_clip=%d, amp=%s)",
        model.config.num_labels,
        getattr(model.config, "frames_per_clip", -1),
        amp_enabled,
    )
    return model, processor


def infer_num_labels(state_dict: Mapping[str, torch.Tensor]) -> Optional[int]:
    candidate_keys = [
        "classifier.weight",
        "classifier.bias",
        "score.weight",
        "score.bias",
        "head.weight",
        "head.bias",
    ]
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if tensor is not None and tensor.ndim > 0:
            return tensor.shape[0]
    return None


def postprocess_predictions(
    logits: torch.Tensor,
    *,
    id_to_label: Mapping[int, str],
    top_k: int,
) -> List[Tuple[str, float]]:
    probabilities = torch.softmax(logits, dim=-1)
    top_k = min(top_k, probabilities.shape[-1])
    values, indices = torch.topk(probabilities, k=top_k, dim=-1)
    results: List[Tuple[str, float]] = []
    for score, idx in zip(values[0].tolist(), indices[0].tolist()):
        label = id_to_label.get(idx, f"class_{idx}")
        results.append((label, float(score)))
    return results


def run_inference(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.disable_amp

    model, processor = load_model(args.checkpoint, device=device, amp_enabled=amp_enabled)
    id_to_label = load_label_map(
        model,
        checkpoint_path=args.checkpoint,
        provided_path=args.label_map,
    )

    config_clip_length = getattr(model.config, "frames_per_clip", None)
    if isinstance(config_clip_length, int) and config_clip_length > 0:
        default_clip_length = config_clip_length
    else:
        default_clip_length = 32

    if args.clip_length is not None:
        if args.clip_length <= 0:
            raise ValueError("--clip-length must be a positive integer.")
        frames_per_clip = args.clip_length
    else:
        frames_per_clip = default_clip_length

    stride = max(1, args.frames_stride)
    clip_span = compute_clip_span(frames_per_clip, stride)
    LOGGER.info(
        "Streaming inference with clip_length=%d, stride=%d (span=%d frames).",
        frames_per_clip,
        stride,
        clip_span,
    )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video at {args.video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 30.0

    if args.output and not args.enable_video_writer:
        raise ValueError("--output requires --enable-video-writer.")

    output_path: Optional[pathlib.Path] = None
    if args.enable_video_writer:
        output_path = resolve_output_path(args.video, args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frame_buffer: deque[np.ndarray] = deque(maxlen=clip_span)
    overlay_text = "Collecting frames..."
    last_top_label: Optional[str] = None
    performed_inference = False
    fps_samples: deque[float] = deque(maxlen=60)
    fps_sum = 0.0
    last_frame_time: Optional[float] = None
    min_fps_samples = 10

    window_title = "VJEPA2 Prediction"
    show_enabled = bool(args.show)
    show_window_open = False
    postprocess_preview = False
    primary_score = 0.0
    try:
        if show_enabled:
            try:
                cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                show_window_open = True
            except cv2.error as exc:
                LOGGER.warning("Unable to open display window: %s", exc)
                show_enabled = False
                postprocess_preview = True

        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            now = time.perf_counter()
            if last_frame_time is not None:
                delta = now - last_frame_time
                if delta > 0:
                    sample = 1.0 / delta
                    sample = min(sample, 240.0)
                    if len(fps_samples) == fps_samples.maxlen:
                        fps_sum -= fps_samples[0]
                    fps_samples.append(sample)
                    fps_sum += sample
            last_frame_time = now

            frame_buffer.append(frame_bgr)

            if args.enable_video_writer and writer is None:
                height, width = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {output_path}")

            if len(frame_buffer) == clip_span:
                clip_rgb = prepare_clip_from_buffer(
                    frame_buffer,
                    frames_per_clip=frames_per_clip,
                    stride=stride,
                )
                inputs = prepare_clip_tensor(processor, clip_rgb, device=device)

                with torch.inference_mode():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        outputs = model(**inputs)
                logits = outputs.logits
                top_predictions = postprocess_predictions(
                    logits,
                    id_to_label=id_to_label,
                    top_k=args.top_k,
                )

                performed_inference = True
                primary_label, primary_score = top_predictions[0]
                overlay_text = f"{primary_label} ({primary_score * 100:.1f}%)"

                if primary_label != last_top_label:
                    LOGGER.info("Top prediction: %s (%.2f%%)", primary_label, primary_score * 100)
                    for rank, (label, score) in enumerate(top_predictions, start=1):
                        LOGGER.info("  #%d %s — %.2f%%", rank, label, score * 100)
                    last_top_label = primary_label
            if primary_score > 0.7:
                annotate_frame(frame_bgr, overlay_text)
            enough_samples = len(fps_samples) >= min_fps_samples
            avg_fps = (fps_sum / len(fps_samples)) if enough_samples else 0.0
            if args.show and avg_fps > 0:
                overlay_fps(frame_bgr, avg_fps)
            if show_enabled and show_window_open:
                try:
                    cv2.imshow(window_title, frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    LOGGER.warning("Display error: %s", exc)
                    show_enabled = False
                    if show_window_open:
                        cv2.destroyWindow(window_title)
                        show_window_open = False
                    postprocess_preview = True
                else:
                    if key in (27, ord("q")):
                        LOGGER.info("Display window closed by user input.")
                        cv2.destroyWindow(window_title)
                        show_window_open = False
                        show_enabled = False
                        postprocess_preview = False

            if writer is not None:
                writer.write(frame_bgr)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show_window_open:
            cv2.destroyWindow(window_title)

    if args.enable_video_writer and writer is None:
        raise RuntimeError(f"Video {args.video} does not contain any frames.")

    if args.enable_video_writer:
        LOGGER.info("Annotated video written to %s", output_path)
    else:
        LOGGER.info("Video writer disabled; annotated frames were not saved to disk.")
    if not performed_inference:
        LOGGER.warning(
            "The video never reached %d frames; no model inference was performed.",
            clip_span,
        )

    if args.show and postprocess_preview:
        if args.enable_video_writer and output_path is not None:
            try_show_video("VJEPA2 Prediction", output_path)
        else:
            LOGGER.info("Cannot preview output video because the writer is disabled.")


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
