#!/usr/bin/env python3
"""Generate per-track cropped videos from a CVAT 1.1 annotation file."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, IO, Iterable, List, Tuple, Union

import cv2

@dataclass
class CropBox:
    """Integer crop rectangle in pixel coordinates (inclusive-exclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


@dataclass
class TrackData:
    """Stores metadata extracted from the CVAT track."""

    track_id: int
    label: str
    frame_boxes: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    min_xtl: float = field(default_factory=lambda: float("inf"))
    min_ytl: float = field(default_factory=lambda: float("inf"))
    max_xbr: float = field(default_factory=lambda: float("-inf"))
    max_ybr: float = field(default_factory=lambda: float("-inf"))
    crop_box: CropBox | None = None
    frames_written: int = 0
    output_path: Path | None = None
    adjusted_start_frame: int | None = None
    adjusted_end_frame: int | None = None

    def update_bounds(self, xtl: float, ytl: float, xbr: float, ybr: float) -> None:
        self.min_xtl = min(self.min_xtl, xtl)
        self.min_ytl = min(self.min_ytl, ytl)
        self.max_xbr = max(self.max_xbr, xbr)
        self.max_ybr = max(self.max_ybr, ybr)

    def finalize_crop(self, frame_width: int, frame_height: int) -> None:
        if not self.frame_boxes:
            return
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive when finalizing crop boxes.")

        x1 = max(0, min(frame_width - 1, math.floor(self.min_xtl)))
        y1 = max(0, min(frame_height - 1, math.floor(self.min_ytl)))
        x2 = min(frame_width, max(x1 + 1, math.ceil(self.max_xbr)))
        y2 = min(frame_height, max(y1 + 1, math.ceil(self.max_ybr)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Computed invalid crop for track {self.track_id}: "
                f"({x1}, {y1}, {x2}, {y2})."
            )

        self.crop_box = CropBox(x1, y1, x2, y2)

    @property
    def start_frame(self) -> int:
        if self.adjusted_start_frame is not None:
            return self.adjusted_start_frame
        return min(self.frame_boxes)

    @property
    def end_frame(self) -> int:
        if self.adjusted_end_frame is not None:
            return self.adjusted_end_frame
        return max(self.frame_boxes)


def slugify_text(text: str, default: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug or default


def build_output_filename(video_prefix: str, track: TrackData) -> str:
    label_slug = slugify_text(track.label, "track")
    return f"{video_prefix}_track_{track.track_id:03d}_{label_slug}.mp4"


def parse_cvat_annotations(
    source: Union[Path, str, IO[str], IO[bytes]]
) -> Tuple[Dict[int, TrackData], Dict[int, List[int]]]:
    """Parse the CVAT 1.1 XML file into per-track metadata."""
    if isinstance(source, (str, Path)):
        tree = ET.parse(str(source))
    else:
        tree = ET.parse(source)
    root = tree.getroot()

    tracks: Dict[int, TrackData] = {}
    frames_to_tracks: Dict[int, List[int]] = defaultdict(list)

    for track_el in root.findall("track"):
        try:
            track_id = int(track_el.attrib["id"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Encountered track with invalid id attribute: {track_el.attrib}") from exc
        label = track_el.attrib.get("label", f"track_{track_id}")
        track = TrackData(track_id=track_id, label=label)

        for box_el in track_el.findall("box"):
            if box_el.attrib.get("outside", "0") == "1":
                continue
            try:
                frame_index = int(box_el.attrib["frame"])
                xtl = float(box_el.attrib["xtl"])
                ytl = float(box_el.attrib["ytl"])
                xbr = float(box_el.attrib["xbr"])
                ybr = float(box_el.attrib["ybr"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Invalid box attributes in track {track_id}: {box_el.attrib}"
                ) from exc

            if xbr <= xtl or ybr <= ytl:
                logging.warning(
                    "Skipping degenerate box at frame %d for track %d (coords: %s).",
                    frame_index,
                    track_id,
                    box_el.attrib,
                )
                continue

            track.frame_boxes[frame_index] = (xtl, ytl, xbr, ybr)
            track.update_bounds(xtl, ytl, xbr, ybr)
            frames_to_tracks[frame_index].append(track_id)

        if track.frame_boxes:
            tracks[track_id] = track
        else:
            logging.info("Ignoring track %d (%s) with no valid boxes.", track_id, label)

    return tracks, frames_to_tracks


def apply_frame_offsets(
    tracks: Dict[int, TrackData],
    frames_to_tracks: Dict[int, List[int]],
    start_offset: int,
    end_offset: int,
) -> Dict[int, List[int]]:
    """Extend track coverage by adding frames before and after annotated range."""
    if start_offset == 0 and end_offset == 0:
        for track in tracks.values():
            track.adjusted_start_frame = None
            track.adjusted_end_frame = None
        return frames_to_tracks

    extended: Dict[int, List[int]] = defaultdict(list)
    for frame, ids in frames_to_tracks.items():
        extended[frame].extend(ids)

    for track in tracks.values():
        if not track.frame_boxes:
            track.adjusted_start_frame = None
            track.adjusted_end_frame = None
            continue

        original_start = min(track.frame_boxes)
        original_end = max(track.frame_boxes)

        raw_start = original_start - start_offset
        effective_start = max(0, raw_start)
        extended_end = original_end + end_offset

        track.adjusted_start_frame = effective_start if start_offset > 0 else None
        track.adjusted_end_frame = extended_end if end_offset > 0 else None

        if effective_start < original_start:
            for frame in range(effective_start, original_start):
                extended[frame].append(track.track_id)

        if extended_end > original_end:
            for frame in range(original_end + 1, extended_end + 1):
                extended[frame].append(track.track_id)

    deduped: Dict[int, List[int]] = {}
    for frame in sorted(extended):
        ids = extended[frame]
        seen: set[int] = set()
        ordered: List[int] = []
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                ordered.append(tid)
        deduped[frame] = ordered

    return deduped


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_video_capture(path: Path) -> Tuple[cv2.VideoCapture, int, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file at {path}.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_width <= 0 or frame_height <= 0:
        # Fallback: read a frame to inspect dimensions.
        ret, sample = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Failed to read frame for determining video dimensions.")
        frame_height, frame_width = sample.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return cap, frame_width, frame_height, fps


def initialize_writers(
    tracks: Dict[int, TrackData],
    output_dir: Path,
    fourcc: int,
    fps: float,
    video_prefix: str,
) -> Dict[int, cv2.VideoWriter]:
    writers: Dict[int, cv2.VideoWriter] = {}
    for track in tracks.values():
        if track.crop_box is None:
            raise ValueError(f"Track {track.track_id} has no crop box defined.")
        output_name = build_output_filename(video_prefix, track)
        output_path = output_dir / output_name
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (track.crop_box.width, track.crop_box.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize writer for {output_path}.")
        track.output_path = output_path
        writers[track.track_id] = writer
        logging.info(
            "Prepared writer for track %d (%s) -> %s, size=%dx%d",
            track.track_id,
            track.label,
            output_path,
            track.crop_box.width,
            track.crop_box.height,
        )
    return writers


def process_video(
    cap: cv2.VideoCapture,
    tracks: Dict[int, TrackData],
    frames_to_tracks: Dict[int, List[int]],
    writers: Dict[int, cv2.VideoWriter],
    fallback_fps: float,
) -> None:
    if not frames_to_tracks:
        logging.warning("No frames to process; frames_to_tracks is empty.")
        return

    max_needed_frame = max(frames_to_tracks)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logging.warning(
            "Falling back to %.2f FPS because input FPS could not be determined.",
            fallback_fps,
        )
        fps = fallback_fps

    frame_index = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Video stream ended at frame %d.", frame_index)
            break

        frame_index += 1
        if frame_index > max_needed_frame:
            logging.info("Reached all required frames (%d). Stopping.", max_needed_frame)
            break

        track_ids = frames_to_tracks.get(frame_index)
        if not track_ids:
            continue

        for track_id in track_ids:
            track = tracks.get(track_id)
            if track is None or track.crop_box is None:
                continue

            crop = frame[
                track.crop_box.y1 : track.crop_box.y2,
                track.crop_box.x1 : track.crop_box.x2,
            ]
            if crop.size == 0:
                logging.warning(
                    "Skipping frame %d for track %d due to empty crop after clipping.",
                    frame_index,
                    track_id,
                )
                continue

            writer = writers.get(track_id)
            if writer is None:
                logging.error("Missing writer for track %d; skipping frame %d.", track_id, frame_index)
                continue

            if crop.shape[1] != track.crop_box.width or crop.shape[0] != track.crop_box.height:
                logging.debug(
                    "Resizing crop for track %d at frame %d from %sx%s to %sx%s.",
                    track_id,
                    frame_index,
                    crop.shape[1],
                    crop.shape[0],
                    track.crop_box.width,
                    track.crop_box.height,
                )
                crop = cv2.resize(
                    crop,
                    (track.crop_box.width, track.crop_box.height),
                    interpolation=cv2.INTER_LINEAR,
                )

            writer.write(crop)
            track.frames_written += 1


def crop_video_with_annotations(
    video_path: Path,
    annotation_source: Union[Path, str, IO[str], IO[bytes]],
    output_dir: Path,
    fourcc: int,
    fallback_fps: float,
    start_frame_offset: int,
    end_frame_offset: int,
    label_filter: set[str] | None = None,
) -> Dict[int, TrackData]:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    tracks, frames_to_tracks = parse_cvat_annotations(annotation_source)
    if not tracks:
        logging.warning("No valid tracks found in annotations for %s.", video_path)
        return {}

    if label_filter:
        tracks = {tid: track for tid, track in tracks.items() if track.label in label_filter}
        if not tracks:
            logging.warning(
                "No tracks with labels %s found in annotations for %s.",
                ", ".join(sorted(label_filter)),
                video_path,
            )
            return {}
        frames_to_tracks = {
            frame: [tid for tid in ids if tid in tracks]
            for frame, ids in frames_to_tracks.items()
        }
        frames_to_tracks = {frame: ids for frame, ids in frames_to_tracks.items() if ids}
        if not frames_to_tracks:
            logging.warning(
                "Label filter %s removed all frames for %s.",
                ", ".join(sorted(label_filter)),
                video_path,
            )
            return {}

    frames_to_tracks = apply_frame_offsets(tracks, frames_to_tracks, start_frame_offset, end_frame_offset)
    if not tracks:
        logging.warning(
            "No tracks remain for %s after applying frame offsets (start=%d, end=%d).",
            video_path,
            start_frame_offset,
            end_frame_offset,
        )
        return {}
    if not frames_to_tracks:
        logging.warning(
            "No frames remain for %s after applying frame offsets (start=%d, end=%d).",
            video_path,
            start_frame_offset,
            end_frame_offset,
        )
        return {}

    logging.info("Preparing video capture for %s.", video_path)
    cap, frame_w, frame_h, fps = prepare_video_capture(video_path)
    if not fps or fps <= 0:
        logging.warning(
            "Input video FPS unavailable for %s; falling back to %.2f if needed.",
            video_path,
            fallback_fps,
        )

    for track in tracks.values():
        track.finalize_crop(frame_w, frame_h)
        logging.debug(
            "Track %d (%s): frames %d-%d, crop x=[%d,%d) y=[%d,%d).",
            track.track_id,
            track.label,
            track.start_frame,
            track.end_frame,
            track.crop_box.x1,
            track.crop_box.x2,
            track.crop_box.y1,
            track.crop_box.y2,
        )

    ensure_output_directory(output_dir)
    video_prefix = slugify_text(video_path.stem, "video")
    fps_to_use = fps if fps and fps > 0 else fallback_fps
    writers = initialize_writers(tracks, output_dir, fourcc, fps_to_use, video_prefix)

    try:
        process_video(cap, tracks, frames_to_tracks, writers, fallback_fps)
    finally:
        cap.release()
        close_writers(writers.values())

    return tracks


def close_writers(writers: Iterable[cv2.VideoWriter]) -> None:
    for writer in writers:
        writer.release()


def non_negative_int(value: str) -> int:
    """argparse helper that ensures provided integer values are non-negative."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got {value!r}.") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Value must be >= 0, got {parsed}.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-track cropped videos from CVAT 1.1 annotations. "
            "Process a single video/XML pair or bulk-process entries defined in a CSV file."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "Path to a CSV file containing rows of 'video_path,annotations_zip_path,split'. "
            "When provided, the script processes each entry and writes outputs under "
            "split/Abandon directories."
        ),
    )
    parser.add_argument(
        "--input-video",
        "-i",
        type=Path,
        help="Path to a single source video (e.g. test/eyp.mp4).",
    )
    parser.add_argument(
        "--annotations",
        "-a",
        type=Path,
        help="Path to a CVAT 1.1 annotations XML file (e.g. test/annotations.xml).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("tracks"),
        help="Directory where the cropped track videos will be written.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec identifier for the output video (default: mp4v).",
    )
    parser.add_argument(
        "--fallback-fps",
        type=float,
        default=25.0,
        help="FPS to use if the input video FPS cannot be read (default: 25).",
    )
    parser.add_argument(
        "--start-frame-offset",
        type=non_negative_int,
        default=0,
        help="Number of extra frames to include before the first annotated frame of each track.",
    )
    parser.add_argument(
        "--end-frame-offset",
        type=non_negative_int,
        default=0,
        help="Number of extra frames to include after the last annotated frame of each track.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help=(
            "Only export tracks whose label matches one of the provided names (case-sensitive). "
            "Provide multiple labels separated by spaces, e.g. --labels \"Birakilan Nesne\" \"Alinan Nesne\"."
        ),
    )
    return parser.parse_args()


def process_dataset(
    csv_path: Path,
    output_root: Path,
    fourcc: int,
    fallback_fps: float,
    start_frame_offset: int,
    end_frame_offset: int,
    label_filter: set[str] | None,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    processed_rows = 0
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row_idx, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 3:
                logging.warning("Row %d skipped: expected 3 columns, got %d.", row_idx, len(row))
                continue

            video_path = Path(row[0].strip())
            annotations_path = Path(row[1].strip())
            split_raw = row[2].strip()
            if not split_raw:
                logging.warning("Row %d skipped: split value is empty.", row_idx)
                continue

            split = split_raw.lower()
            if split not in {"train", "test", "val"}:
                logging.warning(
                    "Row %d skipped: split '%s' is not one of train/test/val.",
                    row_idx,
                    split_raw,
                )
                continue

            split_dir = output_root / split / "Abandon"
            ensure_output_directory(split_dir)

            if not video_path.exists():
                logging.error("Row %d skipped: video %s not found.", row_idx, video_path)
                continue
            if not annotations_path.exists():
                logging.error("Row %d skipped: annotations %s not found.", row_idx, annotations_path)
                continue

            if label_filter:
                logging.info(
                    "Row %d: processing %s with annotations %s (split=%s, labels=%s).",
                    row_idx,
                    video_path,
                    annotations_path,
                    split,
                    ", ".join(sorted(label_filter)),
                )
            else:
                logging.info(
                    "Row %d: processing %s with annotations %s (split=%s).",
                    row_idx,
                    video_path,
                    annotations_path,
                    split,
                )

            try:
                if annotations_path.suffix.lower() == ".zip":
                    with zipfile.ZipFile(annotations_path) as archive:
                        try:
                            with archive.open("annotations.xml") as xml_file:
                                tracks = crop_video_with_annotations(
                                    video_path,
                                    xml_file,
                                    split_dir,
                                    fourcc,
                                    fallback_fps,
                                    start_frame_offset,
                                    end_frame_offset,
                                    label_filter,
                                )
                        except KeyError:
                            logging.error(
                                "Row %d skipped: annotations.xml not found inside %s.",
                                row_idx,
                                annotations_path,
                            )
                            continue
                else:
                    tracks = crop_video_with_annotations(
                        video_path,
                        annotations_path,
                        split_dir,
                        fourcc,
                        fallback_fps,
                        start_frame_offset,
                        end_frame_offset,
                        label_filter,
                    )
            except zipfile.BadZipFile as exc:
                logging.error("Row %d skipped: invalid zip file %s (%s).", row_idx, annotations_path, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive logging
                logging.exception(
                    "Row %d failed while processing %s: %s",
                    row_idx,
                    video_path,
                    exc,
                )
                continue

            if not tracks:
                logging.warning("Row %d: no tracks produced for %s.", row_idx, video_path)
                continue

            processed_rows += 1
            for track in tracks.values():
                logging.info(
                    "Row %d track %d (%s): wrote %d frames to %s.",
                    row_idx,
                    track.track_id,
                    track.label,
                    track.frames_written,
                    track.output_path,
                )

    if processed_rows == 0:
        logging.warning("No dataset rows were processed successfully from %s.", csv_path)
    else:
        logging.info("Finished batch processing: %d rows handled successfully.", processed_rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    label_filter = set(args.labels) if args.labels else None
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    if args.csv:
        process_dataset(
            args.csv,
            args.output_dir,
            fourcc,
            args.fallback_fps,
            args.start_frame_offset,
            args.end_frame_offset,
            label_filter,
        )
        return

    if not args.input_video or not args.annotations:
        raise SystemExit(
            "Provide either --csv for batch mode or both --input-video and --annotations for single mode."
        )

    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")

    if args.annotations.suffix.lower() == ".zip":
        with zipfile.ZipFile(args.annotations) as archive:
            try:
                with archive.open("annotations.xml") as xml_file:
                    tracks = crop_video_with_annotations(
                        args.input_video,
                        xml_file,
                        args.output_dir,
                        fourcc,
                        args.fallback_fps,
                        args.start_frame_offset,
                        args.end_frame_offset,
                        label_filter,
                    )
            except KeyError as exc:
                raise FileNotFoundError("annotations.xml not found inside the provided zip file.") from exc
    else:
        tracks = crop_video_with_annotations(
            args.input_video,
            args.annotations,
            args.output_dir,
            fourcc,
            args.fallback_fps,
            args.start_frame_offset,
            args.end_frame_offset,
            label_filter,
        )

    for track in tracks.values():
        logging.info(
            "Track %d (%s): wrote %d frames to %s.",
            track.track_id,
            track.label,
            track.frames_written,
            track.output_path,
        )


if __name__ == "__main__":
    main()
