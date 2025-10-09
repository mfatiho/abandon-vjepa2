from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, replace
import argparse
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_random_indices
from torchvision.transforms import v2
from transformers import AutoModelForVideoClassification, AutoVideoProcessor


LOGGER = logging.getLogger(__name__)

if torch.cuda.is_available():  # enable cuDNN autotuner for faster kernels on GPUs
    torch.backends.cudnn.benchmark = True

DATASET_ROOT = pathlib.Path("data/UCF101_subset")
HF_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
RUN_DIR = pathlib.Path("runs/vjepa2_finetune")


@dataclass
class TrainingConfig:
    """Container for the knobs we tweak during finetuning."""

    learning_rate: float = 1e-5
    batch_size: int = 1
    num_workers: int = 8
    epochs: int = 5
    accumulation_steps: int = 4
    frames_stride: int = 3
    run_dir: pathlib.Path = RUN_DIR
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = True
    zero_grad_set_to_none: bool = True
    lr_decay_factor: float = 0.15
    lr_patience: int = 7
    max_lr_reductions: int = 3
    min_learning_rate: float = 1e-7
    early_stopping_patience: int = 11


class VideoClassificationDataset(Dataset):
    """Wraps a list of video paths and exposes them as (decoder, label_id) pairs."""

    def __init__(self, video_paths: Sequence[pathlib.Path], label_to_id: Mapping[str, int]):
        self._video_paths = sorted(video_paths)
        self._label_to_id = label_to_id

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._video_paths)

    def __getitem__(self, idx: int) -> Tuple[VideoDecoder, int]:
        video_path = self._video_paths[idx]
        label_name = video_path.parent.name
        decoder = VideoDecoder(str(video_path))
        return decoder, self._label_to_id[label_name]


def split_dataset_paths(root: pathlib.Path) -> Dict[str, List[pathlib.Path]]:
    """Return video paths grouped by split (train/val/test)."""

    splits = {"train": [], "val": [], "test": []}
    for path in root.glob("**/*.avi"):
        try:
            split_name, label_name, _ = path.relative_to(root).parts[:3]
        except ValueError as err:  # pragma: no cover - defensive
            raise ValueError(f"Unexpected dataset layout for {path}") from err

        if split_name not in splits:
            raise ValueError(f"Unknown split '{split_name}' derived from {path}")

        splits[split_name].append(path)

    total = sum(len(items) for items in splits.values())
    LOGGER.info(
        "Discovered %d videos (%d train / %d val / %d test)",
        total,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return splits


def build_label_maps(paths: Iterable[pathlib.Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Compute label <-> id mapping once so it can be reused everywhere."""

    labels = sorted({path.parent.name for path in paths})
    if not labels:
        raise ValueError("No class labels found in dataset")

    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def build_collate_fn(
    *,
    frames_per_clip: int,
    transforms: v2.Compose,
    frames_stride: int,
) -> Callable[[Sequence[Tuple[VideoDecoder, int]]], Tuple[torch.Tensor, torch.Tensor]]:
    """Generate a collate function that samples clips and applies transforms."""

    def collate(samples: Sequence[Tuple[VideoDecoder, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        clips, labels = [], []
        for decoder, label in samples:
            clip = clips_at_random_indices(
                decoder,
                num_clips=1,
                num_frames_per_clip=frames_per_clip,
                num_indices_between_frames=frames_stride,
            ).data
            clips.append(clip)
            labels.append(label)

        videos = torch.cat(clips, dim=0)
        videos = transforms(videos)
        return videos, torch.tensor(labels)

    return collate


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    collate_fn,
) -> DataLoader:
    """Create a DataLoader with the configuration we use in training/eval."""

    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )


def freeze_backbone(model: AutoModelForVideoClassification) -> None:
    """Freeze the heavy backbone so that only the classifier head trains."""

    if hasattr(model, "vjepa2"):
        for param in model.vjepa2.parameters():
            param.requires_grad = False


def evaluate(
    loader: DataLoader,
    model: AutoModelForVideoClassification,
    processor: AutoVideoProcessor,
    device: torch.device,
    *,
    amp_enabled: bool,
) -> float:
    """Compute accuracy over a dataloader."""

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for videos, labels in loader:
            inputs = processor(list(videos), return_tensors="pt").to(device)
            targets = labels.to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(**inputs).logits
            predictions = logits.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    return correct / total if total else 0.0


def train_one_epoch(
    *,
    loader: DataLoader,
    model: AutoModelForVideoClassification,
    processor: AutoVideoProcessor,
    optimizer: torch.optim.Optimizer,
    accumulation_steps: int,
    epoch_index: int,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    zero_grad_set_to_none: bool,
    amp_enabled: bool,
) -> None:
    """Run a single training epoch with gradient accumulation."""

    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

    steps_ran = 0
    for step, (videos, labels) in enumerate(loader, start=1):
        steps_ran = step
        inputs = processor(list(videos), return_tensors="pt").to(device)
        targets = labels.to(device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(**inputs, labels=targets)
        loss = outputs.loss
        if loss.ndim > 0:
            loss = loss.mean()
        loss_to_backprop = loss / accumulation_steps
        if scaler is not None:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()
        running_loss += loss.item()

        if step % accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=zero_grad_set_to_none)
            LOGGER.info(
                "Epoch %d | Step %d | Accumulated loss %.4f",
                epoch_index,
                step,
                running_loss / accumulation_steps,
            )
            running_loss = 0.0

    # Flush remaining gradients if the last batch did not align with accumulation steps.
    if steps_ran and steps_ran % accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=zero_grad_set_to_none)


def maybe_run_demo(
    *,
    model: AutoModelForVideoClassification,
    processor: AutoVideoProcessor,
    id_to_label: Mapping[int, str],
    device: torch.device,
    amp_enabled: bool,
) -> None:
    """Optional convenience demo using a single remote video."""

    video_url = "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/IMG_3830.mp4"
    try:
        decoder = VideoDecoder(video_url)
    except Exception as exc:  # pragma: no cover - best effort helper
        LOGGER.warning("Skipping demo video: %s", exc)
        return

    frame_indices = np.arange(0, 32)
    video_frames = decoder.get_frames_at(indices=frame_indices).data
    inputs = processor(video_frames, return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(**inputs).logits

    predicted_class_idx = logits.argmax(dim=-1).item()
    LOGGER.info("Demo prediction: %s", id_to_label[predicted_class_idx])


def run_training(config: TrainingConfig) -> None:
    """Main entry point for setting up data, model, and training loop."""

    splits = split_dataset_paths(DATASET_ROOT)
    label_to_id, id_to_label = build_label_maps(
        path for split_paths in splits.values() for path in split_paths
    )

    datasets = {
        split: VideoClassificationDataset(paths, label_to_id)
        for split, paths in splits.items()
    }

    num_labels = len(label_to_id)
    model = AutoModelForVideoClassification.from_pretrained(
        HF_REPO,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(config.device)
    model_config = model.config

    processor = AutoVideoProcessor.from_pretrained(HF_REPO)
    model_config.id2label = {idx: label for idx, label in id_to_label.items()}
    model_config.label2id = {label: idx for idx, label in id_to_label.items()}

    freeze_backbone(model)
    if torch.cuda.device_count() > 1 and config.device.type == "cuda":
        LOGGER.info(
            "Wrapping model with DataParallel across %d GPUs", torch.cuda.device_count()
        )
        model = torch.nn.DataParallel(model)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)
    amp_enabled = config.use_amp and config.device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                (processor.crop_size["height"], processor.crop_size["width"])
            ),
            v2.RandomHorizontalFlip(),
        ]
    )
    eval_transforms = v2.Compose(
        [v2.CenterCrop((processor.crop_size["height"], processor.crop_size["width"]))]
    )

    train_loader = build_dataloader(
        datasets["train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        collate_fn=build_collate_fn(
            frames_per_clip=model_config.frames_per_clip,
            transforms=train_transforms,
            frames_stride=config.frames_stride,
        ),
    )
    val_loader = build_dataloader(
        datasets["val"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=build_collate_fn(
            frames_per_clip=model_config.frames_per_clip,
            transforms=eval_transforms,
            frames_stride=config.frames_stride,
        ),
    )
    test_loader = build_dataloader(
        datasets["test"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        collate_fn=build_collate_fn(
            frames_per_clip=model_config.frames_per_clip,
            transforms=eval_transforms,
            frames_stride=config.frames_stride,
        ),
    )

    best_val_acc = 0.0
    epochs_without_improvement = 0
    plateau_epochs = 0
    lr_reductions = 0

    epochs_ran = 0
    best_checkpoint_epoch: Optional[int] = None

    config.run_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = config.run_dir / "finetuned_model_best.pt"

    with SummaryWriter(config.run_dir) as writer:
        for epoch in range(1, config.epochs + 1):
            train_one_epoch(
                loader=train_loader,
                model=model,
                processor=processor,
                optimizer=optimizer,
                accumulation_steps=config.accumulation_steps,
                epoch_index=epoch,
                device=config.device,
                scaler=scaler,
                zero_grad_set_to_none=config.zero_grad_set_to_none,
                amp_enabled=amp_enabled,
            )
            val_acc = evaluate(
                val_loader,
                model,
                processor,
                config.device,
                amp_enabled=amp_enabled,
            )
            LOGGER.info("Epoch %d | Validation accuracy %.4f", epoch, val_acc)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            epochs_ran = epoch

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                plateau_epochs = 0
                model_to_snapshot = (
                    model.module if isinstance(model, torch.nn.DataParallel) else model
                )
                state_dict_cpu = {
                    key: value.detach().cpu()
                    for key, value in model_to_snapshot.state_dict().items()
                }
                snapshot_payload = {
                    "state_dict": state_dict_cpu,
                    "config": model_to_snapshot.config.to_dict(),
                }
                torch.save(snapshot_payload, best_checkpoint_path)
                best_checkpoint_epoch = epoch
                del state_dict_cpu, snapshot_payload
                LOGGER.info(
                    "New best validation accuracy %.4f at epoch %d; saved checkpoint to %s.",
                    best_val_acc,
                    epoch,
                    best_checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
                plateau_epochs += 1
                LOGGER.info(
                    "Validation accuracy did not improve for %d epoch(s)",
                    epochs_without_improvement,
                )

                reduced_lr_this_round = False
                if plateau_epochs >= config.lr_patience:
                    current_lr = optimizer.param_groups[0]["lr"]
                    can_reduce_lr = (
                        lr_reductions < config.max_lr_reductions
                        and config.lr_decay_factor < 1.0
                        and current_lr > config.min_learning_rate
                    )
                    if can_reduce_lr:
                        new_lr_target = current_lr * config.lr_decay_factor
                        new_lr = max(new_lr_target, config.min_learning_rate)
                        if new_lr < current_lr:
                            for group in optimizer.param_groups:
                                group["lr"] = new_lr
                            lr_reductions += 1
                            plateau_epochs = 0
                            epochs_without_improvement = 0
                            reduced_lr_this_round = True
                            LOGGER.info(
                                "Reducing learning rate from %e to %e (%d/%d reduction)",
                                current_lr,
                                new_lr,
                                lr_reductions,
                                config.max_lr_reductions,
                            )
                        else:
                            can_reduce_lr = False
                            LOGGER.info(
                                "Learning rate already at minimum threshold (%e); cannot reduce further.",
                                current_lr,
                            )
                    else:
                        reasons = []
                        if lr_reductions >= config.max_lr_reductions:
                            reasons.append("maximum reductions reached")
                        if config.lr_decay_factor >= 1.0:
                            reasons.append("decay factor >= 1.0")
                        if current_lr <= config.min_learning_rate:
                            reasons.append("current lr at minimum")
                        LOGGER.info(
                            "Skipping learning rate reduction (%s).",
                            ", ".join(reasons) if reasons else "constraints not met",
                        )
                    if not reduced_lr_this_round:
                        plateau_epochs = 0

                if epochs_without_improvement >= config.early_stopping_patience:
                    LOGGER.info(
                        "Early stopping triggered after %d non-improving epoch(s); current lr %e",
                        epochs_without_improvement,
                        optimizer.param_groups[0]["lr"],
                    )
                    break

        test_acc = None
        if best_checkpoint_epoch is not None:
            LOGGER.info(
                "Loading best model from epoch %d for final evaluation (%s)",
                best_checkpoint_epoch,
                best_checkpoint_path,
            )
            checkpoint_payload = torch.load(best_checkpoint_path, map_location=config.device)
            if isinstance(checkpoint_payload, Mapping) and "state_dict" in checkpoint_payload:
                best_state = checkpoint_payload["state_dict"]
            else:
                best_state = checkpoint_payload
            model_to_eval = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_eval.load_state_dict(best_state, strict=False)
            test_acc = evaluate(
                test_loader,
                model,
                processor,
                config.device,
                amp_enabled=amp_enabled,
            )
            LOGGER.info("Final test accuracy (best model) %.4f", test_acc)
            writer.add_scalar("test/accuracy", test_acc, max(epochs_ran, 1))
        else:
            LOGGER.info("Best model snapshot not available; skipping final test evaluation.")

    maybe_run_demo(
        model=model,
        processor=processor,
        id_to_label=id_to_label,
        device=config.device,
        amp_enabled=amp_enabled,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune VJEPA2 on a local dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs to run")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for DataLoader instances",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - script entry point
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = TrainingConfig()
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.batch_size is not None:
        config = replace(config, batch_size=args.batch_size)

    LOGGER.info("Using device: %s", config.device)
    run_training(config)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
