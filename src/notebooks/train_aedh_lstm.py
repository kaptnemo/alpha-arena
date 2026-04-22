from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from alpha_arena.models.aedh_lstm import (
    AEDH_LSTMConfig,
    AttentionEnhancedDualHeadLSTM,
)
from alpha_arena.train.dataset.loader import (
    GroupedByDateBatchSampler,
    SequenceDataset,
    collate_fn,
)
from alpha_arena.train.trainer import train_model_ddp
from alpha_arena.utils import configure_logging, get_logger


DEFAULT_DATASET_NAME = "csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5"
DEFAULT_CHECKPOINT_DIR = "checkpoints/notebooks/aedh_lstm"


def build_loader_kwargs(num_workers: int, pin_memory: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def create_notebook_dataloaders(
    train_dataset: SequenceDataset,
    valid_dataset: SequenceDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> dict[str, DataLoader]:
    loader_kwargs = build_loader_kwargs(num_workers=num_workers, pin_memory=pin_memory)

    train_grouped_loader = DataLoader(
        train_dataset,
        batch_sampler=GroupedByDateBatchSampler(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        ),
        **loader_kwargs,
    )

    train_random_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return {
        "train_grouped_loader": train_grouped_loader,
        "train_random_loader": train_random_loader,
        "valid_loader": valid_loader,
    }


def infer_model_config(dataset: SequenceDataset) -> AEDH_LSTMConfig:
    sample = dataset[0]
    return AEDH_LSTMConfig(
        input_dim=int(sample["x_seq"].shape[-1]),
        hidden_dim=256,
        num_layers=2,
        attn_dim=64,
        head_hidden_dim=64,
        dropout=0.1,
        use_last_state=True,
        cs_feature_dim=int(sample["x_cs"].shape[-1]),
        cs_feature_mask=bool(sample["x_cs"].shape[-1]),
    )


def history_to_frame(history: list[dict[str, Any]]) -> pd.DataFrame:
    if not history:
        raise ValueError("Training history is empty, cannot build loss curve.")

    history_frame = pd.DataFrame(history).sort_values("epoch").reset_index(drop=True)
    history_frame["epoch"] = history_frame["epoch"] + 1
    return history_frame


def plot_loss_curves(
    history: list[dict[str, Any]],
    output_path: Path | None = None,
    show: bool = True,
) -> pd.DataFrame:
    history_frame = history_to_frame(history)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history_frame["epoch"], history_frame["train/loss"], label="train/loss", linewidth=2)
    ax.plot(history_frame["epoch"], history_frame["valid/loss"], label="valid/loss", linewidth=2)
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return history_frame


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Notebook-friendly AEDH-LSTM training script.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--mid-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument("--no-show", action="store_true", help="Save loss curve without calling plt.show().")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    configure_logging(log_dir=PROJECT_ROOT / "logs", file_name="train_notebook.log")
    logger = get_logger(__name__)

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    plot_path = Path(args.plot_path) if args.plot_path else checkpoint_dir / "loss_curve.png"
    if not plot_path.is_absolute():
        plot_path = PROJECT_ROOT / plot_path

    logger.info("loading_dataset", dataset_name=args.dataset_name)
    train_dataset = SequenceDataset(dataset_name=args.dataset_name, split_name="train")
    valid_dataset = SequenceDataset(dataset_name=args.dataset_name, split_name="evaluate")

    model_config = infer_model_config(train_dataset)
    logger.info(
        "dataset_loaded",
        train_size=len(train_dataset),
        valid_size=len(valid_dataset),
        input_dim=model_config.input_dim,
        cs_feature_dim=model_config.cs_feature_dim,
        device=str(device),
    )

    dataloaders = create_notebook_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = AttentionEnhancedDualHeadLSTM(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    logger.info(
        "start_training",
        checkpoint_dir=str(checkpoint_dir),
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
    )
    result = train_model_ddp(
        model=model,
        optimizer=optimizer,
        train_random_loader=dataloaders["train_random_loader"],
        train_grouped_loader=dataloaders["train_grouped_loader"],
        valid_loader=dataloaders["valid_loader"],
        device=device,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        mid_epochs=args.mid_epochs,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm if args.max_grad_norm > 0 else None,
        checkpoint_dir=checkpoint_dir,
        use_ddp=False,
        verbose=True,
    )

    history = result["history"] or []
    history_frame = plot_loss_curves(
        history=history,
        output_path=plot_path,
        show=not args.no_show,
    )

    logger.info(
        "training_completed",
        best_epoch=result["best_epoch"],
        best_metric=result["best_metric"],
        history_path=str(checkpoint_dir / "history.json"),
        plot_path=str(plot_path),
    )
    print(history_frame[["epoch", "stage", "train/loss", "valid/loss", "lr"]].tail().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
