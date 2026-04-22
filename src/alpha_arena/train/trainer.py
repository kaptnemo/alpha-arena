from __future__ import annotations

import os
import time
import copy
import json
from pathlib import Path
from typing import Any
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

from dataclasses import asdict
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from alpha_arena.models.aedh_lstm import AttentionEnhancedDualHeadLSTM, AEDH_LSTMConfig

    
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def ddp_broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_dist_available_and_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def ddp_reduce_op(value: float, device: torch.device, op: dist.ReduceOp = dist.ReduceOp.SUM) -> float:
    if not is_dist_available_and_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=op)
    return float(t.item())


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def broadcast_model_parameters(model: nn.Module, src: int = 0) -> None:
    if not is_dist_available_and_initialized():
        return

    raw_model = unwrap_model(model)
    state_dict = raw_model.state_dict()
    for v in state_dict.values():
        if torch.is_tensor(v):
            dist.broadcast(v.data, src=src)


# =========================
# io / checkpoint utils
# =========================
def save_checkpoint(
    save_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    best_metric: float,
    history: list[dict[str, Any]],
    model_config: dict[str, Any],
    # feature_config: dict[str, Any],
    extra_state: dict[str, Any] | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),

        "model_config": model_config,
        # "feature_config": feature_config,

        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,

        "best_metric": best_metric,
        "history": history,
    }
    if extra_state is not None:
        ckpt["extra_state"] = extra_state

    torch.save(ckpt, save_path)


def save_model(
    save_path: str | Path,
    model: nn.Module,
    model_config: dict[str, Any],
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": unwrap_model(model).state_dict(),
        "model_config": model_config,
    }

    torch.save(ckpt, save_path)


def load_model(path):
    import torch

    ckpt = torch.load(path, map_location="cpu")

    model_config = ckpt["model_config"]
    #feature_config = ckpt["feature_config"]
    config = AEDH_LSTMConfig(**model_config)
    model = AttentionEnhancedDualHeadLSTM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, model_config  


def load_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=map_location)

    unwrap_model(model).load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return ckpt


# =========================
# misc utils
# =========================
def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)

    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
        return

    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
        return


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def detach_loss_dict(loss_dict: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu().item())
        elif isinstance(v, (float, int)):
            out[k] = float(v)
    return out


def compute_cross_sectional_metrics(
    pred_return,
    target_return,
    label_date,
    top_frac: float = 0.2,
    min_group_size: int = 20,
) -> dict[str, float]:
    df = pd.DataFrame({
        "pred": np.asarray(pred_return),
        "target": np.asarray(target_return),
        "label_date": np.asarray(label_date),
    })

    rank_ics = []
    ics = []
    spreads = []

    for _, g in df.groupby("label_date"):
        if len(g) < min_group_size:
            continue

        pred = g["pred"].to_numpy()
        target = g["target"].to_numpy()

        if np.std(pred) < 1e-12 or np.std(target) < 1e-12:
            continue

        rank_ic = spearmanr(pred, target).correlation
        ic = pearsonr(pred, target)[0]

        if np.isfinite(rank_ic):
            rank_ics.append(rank_ic)
        if np.isfinite(ic):
            ics.append(ic)

        g_sorted = g.sort_values("pred")
        k = max(int(len(g_sorted) * top_frac), 1)

        bottom_ret = g_sorted.head(k)["target"].mean()
        top_ret = g_sorted.tail(k)["target"].mean()
        spreads.append(top_ret - bottom_ret)

    rank_ics = np.asarray(rank_ics, dtype=np.float64)
    ics = np.asarray(ics, dtype=np.float64)
    spreads = np.asarray(spreads, dtype=np.float64)

    if len(rank_ics) == 0:
        return {
            "rank_ic_mean": float("nan"),
            "rank_ic_std": float("nan"),
            "rank_ic_ir": float("nan"),
            "rank_ic_pos_ratio": float("nan"),
            "ic_mean": float("nan"),
            "top_bottom_spread": float("nan"),
        }

    return {
        "rank_ic_mean": float(np.mean(rank_ics)),
        "rank_ic_std": float(np.std(rank_ics)),
        "rank_ic_ir": float(np.mean(rank_ics) / (np.std(rank_ics) + 1e-8)),
        "rank_ic_pos_ratio": float(np.mean(rank_ics > 0)),
        "ic_mean": float(np.mean(ics)) if len(ics) > 0 else float("nan"),
        "top_bottom_spread": float(np.mean(spreads)) if len(spreads) > 0 else float("nan"),
    }

mean_keys = {
    "loss",
    "base_loss",
    "nll_loss",
    "mse_loss",
    "pred_var_mean",
    "pred_std_mean",
    "pred_return_mean",
    "pred_return_std",
}

batch_mean_keys = {
    "rank_loss",
}

min_keys = {
    "pred_var_min",
}

max_keys = {
    "pred_var_max",
}

# =========================
# one epoch: AMP + DDP
# =========================
def run_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_type: str,
    alpha_rank: float,
    alpha_mse: float,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    """
    optimizer is None -> eval
    optimizer is not None -> train

    AMP:
      - train: autocast + GradScaler
      - eval : autocast only
    DDP:
      - 所有 scalar metric 按 sum(loss * batch_size) / sum(batch_size) 做全局聚合
    """
    is_train = optimizer is not None
    model.train(is_train)

    local_sum: dict[str, float] = {}
    local_batch_sum = {}
    local_min = {}
    local_max = {}
    local_num_samples = 0
    local_num_batches = 0

    all_pred_return = []
    all_target_return = []
    all_label_date = []

    grad_context = torch.enable_grad() if is_train else torch.no_grad()

    # 只让 rank0 显示 tqdm
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    if is_main:
        pbar = tqdm(loader, desc="train" if is_train else "valid", leave=False)
    else:
        pbar = loader

    with grad_context:
        for step, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            autocast_enabled = use_amp and (device.type == "cuda")
            with torch.amp.autocast(
                device_type=device.type,
                enabled=autocast_enabled,
                dtype=amp_dtype if device.type == "cuda" else None,
            ):
                # torch.cuda.synchronize(device)
                # t_opt0 = time.time()
                outputs = model(
                    x_seq=batch["x_seq"],
                    x_cs=batch["x_cs"],
                    x_cs_mask=batch["x_cs_mask"],
                )
                if not is_train:
                    all_pred_return.append(outputs["pred_return"].detach().float().cpu())
                    all_target_return.append(batch["y_return"].detach().float().cpu())
                    all_label_date.extend(batch["label_date"])
                # torch.cuda.synchronize(device)
                # t_opt1 = time.time()

                loss_dict = unwrap_model(model).compute_loss(
                    outputs=outputs,
                    target_return=batch["y_return"],
                    loss_type=loss_type,
                    alpha_rank=alpha_rank,
                    alpha_mse=alpha_mse,
                )
                loss = loss_dict["loss"]
                # torch.cuda.synchronize(device)
                # t_opt2 = time.time()
            if is_train:
                if scaler is not None and autocast_enabled:

                    scaler.scale(loss).backward()
                    # torch.cuda.synchronize(device)
                    # t_opt3 = time.time()
                    if max_grad_norm is not None and max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # torch.cuda.synchronize(device)
                    # t_opt4 = time.time()
                    scaler.step(optimizer)
                    # torch.cuda.synchronize(device)
                    # t_opt5 = time.time()
                    scaler.update()
                    # torch.cuda.synchronize(device)
                    # t_opt6 = time.time()
                else:
                    loss.backward()
                    if max_grad_norm is not None and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()

            if is_main:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    # "load": f"{(t1 - t0)*1000:.1f}ms",
                    # "step": f"{(t6 - t1)*1000:.1f}ms",
                })
                pbar.update(1)


            batch_size = int(batch["x_seq"].shape[0])
            scalar_losses = detach_loss_dict(loss_dict)

            for k, v in scalar_losses.items():
                if k in mean_keys:
                    local_sum[k] = local_sum.get(k, 0.0) + v * batch_size
                elif k in batch_mean_keys:
                    local_batch_sum[k] = local_batch_sum.get(k, 0.0) + v
                elif k in min_keys:
                    local_min[k] = min(local_min.get(k, float("inf")), v)
                elif k in max_keys:
                    local_max[k] = max(local_max.get(k, float("-inf")), v)

            local_num_samples += batch_size
            local_num_batches += 1

    metrics = {}

    global_num_samples = ddp_reduce_op(float(local_num_samples), device=device, op=dist.ReduceOp.SUM)
    global_num_batches = ddp_reduce_op(float(local_num_batches), device=device, op=dist.ReduceOp.SUM)

    for k, v in local_sum.items():
        global_sum = ddp_reduce_op(float(v), device=device, op=dist.ReduceOp.SUM)
        metrics[k] = global_sum / max(global_num_samples, 1.0)

    for k, v in local_batch_sum.items():
        global_sum = ddp_reduce_op(float(v), device=device, op=dist.ReduceOp.SUM)
        metrics[k] = global_sum / max(global_num_batches, 1.0)

    for k, v in local_min.items():
        global_min = ddp_reduce_op(float(v), device=device, op=dist.ReduceOp.MIN)
        metrics[k] = global_min
    
    for k, v in local_max.items():
        global_max = ddp_reduce_op(float(v), device=device, op=dist.ReduceOp.MAX)
        metrics[k] = global_max

    if not is_train and all_pred_return:
        pred_return_np = torch.cat(all_pred_return).numpy()
        target_return_np = torch.cat(all_target_return).numpy()

        cs_metrics = compute_cross_sectional_metrics(
            pred_return=pred_return_np,
            target_return=target_return_np,
            label_date=all_label_date,
            top_frac=0.2,
            min_group_size=20,
        )

        metrics.update(cs_metrics)

    return metrics


def set_optimizer_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr

# =========================
# main trainer: AMP + DDP
# =========================
def train_model_ddp(
    *,
    train_task_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_random_loader: DataLoader,
    train_grouped_loader: DataLoader,
    valid_loader: DataLoader,
    device: str | torch.device,
    num_epochs: int,
    warmup_epochs: int = 0,
    mid_epochs: int = 0,
    loss_type: str = "gaussian_nll",
    warmup_alpha_rank: float = 0.0,
    mid_alpha_rank: float = 0.05,
    main_alpha_rank: float = 0.1,
    warmup_alpha_mse: float = 100.0,
    mid_alpha_mse: float = 50.0,
    main_alpha_mse: float = 10.0,
    warmup_lr: float = 1e-3,
    mid_lr: float = 3e-4,
    main_lr: float = 1e-4,
    scheduler: Any | None = None,
    scheduler_step_on: str = "epoch",   # "epoch" | "valid_metric"
    monitor: str = "valid/loss",
    monitor_mode: str = "min",          # "min" | "max"
    patience: int = 10,
    min_delta: float = 0.0,
    max_grad_norm: float | None = None,
    checkpoint_dir: str | Path = "./checkpoints",
    save_best: bool = True,
    save_last: bool = True,
    resume_from: str | Path | None = None,
    use_ddp: bool = True,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    verbose: bool = True,
) -> dict[str, Any]:
    
    model_config = asdict(model.config)
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir) / train_task_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    if use_ddp and is_dist_available_and_initialized() and not isinstance(model, DDP):
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,  # 推荐，能节省显存
            static_graph=True,  # 推荐，能提升性能（前提是模型前向图是静态的）
            bucket_cap_mb=50,  # 根据模型大小调整，16MB 通常是个不错的起点
        )

    if monitor_mode not in {"min", "max"}:
        raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode}")
    if scheduler_step_on not in {"epoch", "valid_metric"}:
        raise ValueError(f"scheduler_step_on must be 'epoch' or 'valid_metric', got {scheduler_step_on}")

    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)

    start_epoch = 0
    history: list[dict[str, Any]] = []
    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_without_improve = 0

    # ---------------- resume ----------------
    if resume_from is not None:
        ckpt = load_checkpoint(
            ckpt_path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        history = ckpt.get("history", [])
        monitor = ckpt.get("extra_state", {}).get("monitor", monitor)
        monitor_mode = ckpt.get("extra_state", {}).get("monitor_mode", monitor_mode)

        if history:
            best_record = None
            for rec in history:
                if monitor not in rec:
                    continue
                if best_record is None:
                    best_record = rec
                else:
                    if monitor_mode == "min" and rec[monitor] < best_record[monitor]:
                        best_record = rec
                    elif monitor_mode == "max" and rec[monitor] > best_record[monitor]:
                        best_record = rec
            if best_record is not None:
                best_epoch = int(best_record["epoch"])

        if is_main_process() and verbose:
            print(f"[Resume] from={resume_from} start_epoch={start_epoch} best_metric={best_metric:.6f}")

    sync_resume_state = {
        "start_epoch": int(start_epoch),
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "epochs_without_improve": int(epochs_without_improve),
    }
    sync_resume_state = ddp_broadcast_object(sync_resume_state, src=0)

    start_epoch = sync_resume_state["start_epoch"]
    best_metric = sync_resume_state["best_metric"]
    best_epoch = sync_resume_state["best_epoch"]
    epochs_without_improve = sync_resume_state["epochs_without_improve"]


    barrier()

    # ---------------- train loop ----------------
    for epoch in range(start_epoch, num_epochs):
        if epoch < warmup_epochs:
            train_loader = train_random_loader
            alpha_rank = warmup_alpha_rank
            alpha_mse = warmup_alpha_mse
            lr = warmup_lr
            monitor = "valid/loss"
            monitor_mode = "min"
            stage = "warmup"
        elif epoch < warmup_epochs + mid_epochs:
            train_loader = train_grouped_loader
            alpha_rank = mid_alpha_rank
            alpha_mse = mid_alpha_mse
            lr = mid_lr
            monitor = "valid/rank_ic_mean"
            monitor_mode = "max"
            stage = "main_mid"
        else:
            train_loader = train_grouped_loader
            alpha_rank = main_alpha_rank
            alpha_mse = main_alpha_mse
            lr = main_lr
            monitor = "valid/rank_ic_mean"
            monitor_mode = "max"
            stage = "main"

        is_better = (
            (lambda current, best: current < best - min_delta)
            if monitor_mode == "min"
            else (lambda current, best: current > best + min_delta)
        )

        set_optimizer_lr(optimizer, lr)
        set_loader_epoch(train_loader, epoch)
        set_loader_epoch(valid_loader, epoch)

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_type=loss_type,
            alpha_rank=alpha_rank,
            alpha_mse=alpha_mse,
            scaler=scaler,
            use_amp=amp_enabled,
            amp_dtype=amp_dtype,
            max_grad_norm=max_grad_norm,
        )

        valid_metrics = run_one_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            optimizer=None,
            loss_type=loss_type,
            alpha_rank=alpha_rank,
            alpha_mse=main_alpha_mse,
            scaler=None,
            use_amp=amp_enabled,
            amp_dtype=amp_dtype,
            max_grad_norm=None,
        )

        epoch_record = {
            "epoch": epoch,
            "stage": stage,
            "alpha_rank": alpha_rank,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        epoch_record.update({f"train/{k}": v for k, v in train_metrics.items()})
        epoch_record.update({f"valid/{k}": v for k, v in valid_metrics.items()})

        current_metric = epoch_record.get(monitor)
        if current_metric is None:
            raise KeyError(f"monitor='{monitor}' not found in epoch_record: {list(epoch_record.keys())}")

        sync_state = None

        if is_main_process():
            history.append(epoch_record)

            improved = is_better(current_metric, best_metric)
            should_stop = False

            if improved:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improve = 0
                best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())

                if save_best:
                    save_checkpoint(
                        save_path=checkpoint_dir / "best.pt",
                        model=model,
                        model_config=model_config,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        best_metric=best_metric,
                        history=history,
                        extra_state={
                            "best_epoch": best_epoch,
                            "monitor": monitor,
                            "monitor_mode": monitor_mode,
                            "world_size": get_world_size(),
                            "amp_enabled": amp_enabled,
                        },
                    )
                    save_model(
                        save_path=checkpoint_dir / "best_model.pt",
                        model=model,
                        model_config=model_config,
                    )
            else:
                epochs_without_improve += 1

            if save_last:
                save_checkpoint(
                    save_path=checkpoint_dir / "last.pt",
                    model=model,
                    model_config=model_config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    history=history,
                    extra_state={
                        "best_epoch": best_epoch,
                        "monitor": monitor,
                        "monitor_mode": monitor_mode,
                        "world_size": get_world_size(),
                        "amp_enabled": amp_enabled,
                    },
                )

            if patience > 0 and epochs_without_improve >= patience:
                should_stop = True

            if verbose:
                def g(k):
                    return epoch_record.get(k, float("nan"))

                print(
                    f"[Epoch {epoch+1:03d}/{num_epochs:03d}] "
                    f"stage={stage} lr={epoch_record['lr']:.6e} "
                    f"{monitor}={current_metric:.6f} best={best_metric:.6f} "
                    f"patience={epochs_without_improve}/{patience}"
                )

                print(
                    f"  loss: "
                    f"train={g('train/loss'):.4f} "
                    f"valid={g('valid/loss'):.4f} "
                    f"train_nll={g('train/nll_loss'):.4f} "
                    f"valid_nll={g('valid/nll_loss'):.4f}"
                )

                print(
                    f"  mse/rank: "
                    f"train_mse={g('train/mse_loss'):.4e} "
                    f"valid_mse={g('valid/mse_loss'):.4e} "
                    f"train_rank={g('train/rank_loss'):.4f} "
                    f"valid_rank={g('valid/rank_loss'):.4f}"
                )

                print(
                    f"  pred_var: "
                    f"mean={g('train/pred_var_mean'):.4e} "
                    f"std={g('train/pred_std_mean'):.4f} "
                    f"min={g('train/pred_var_min'):.2e} "
                    f"max={g('train/pred_var_max'):.2e}"
                )

            if should_stop and verbose:
                print(
                    f"[EarlyStopping] stop at epoch={epoch}, "
                    f"best_epoch={best_epoch}, best_metric={best_metric:.6f}"
                )

            sync_state = {
                "current_metric": float(current_metric),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "epochs_without_improve": int(epochs_without_improve),
                "should_stop": bool(should_stop),
            }

        sync_state = ddp_broadcast_object(sync_state, src=0)
        current_metric = sync_state["current_metric"]
        best_metric = sync_state["best_metric"]
        best_epoch = sync_state["best_epoch"]
        epochs_without_improve = sync_state["epochs_without_improve"]
        should_stop = sync_state["should_stop"]

        # 所有 rank 同步 step scheduler
        if scheduler is not None:
            if scheduler_step_on == "epoch":
                scheduler.step()
            elif scheduler_step_on == "valid_metric":
                scheduler.step(current_metric)

        barrier()

        if should_stop:
            break

    # ---------------- load best at end ----------------
    best_ckpt_path = checkpoint_dir / "best.pt"

    if is_main_process():
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device)
            unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        elif best_state_dict is not None:
            unwrap_model(model).load_state_dict(best_state_dict)

    broadcast_model_parameters(model, src=0)
    barrier()

    if is_main_process():
        history_json_path = checkpoint_dir / "history.json"
        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history if is_main_process() else None,
    }