可以。下面我直接给你一个 **支持 DDP 的完整训练版本**，特点是：

* 支持 **warmup 阶段 random loader**，之后切 **grouped-by-date loader**
* 支持 **DDP**
* 支持 **train / valid**
* 支持 **best / last checkpoint**
* 支持 **early stopping**
* 支持 **只在 rank0 保存和打印**
* 支持 **所有 rank 同步停止**
* 单卡 / 非 DDP 也能跑

---

# 一版可直接用的实现

```python
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# =========================
# distributed utils
# =========================
def is_dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_available_and_initialized():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def ddp_reduce_scalar(value: float, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    """
    Reduce a scalar across all processes and return the reduced value.
    """
    if not is_dist_available_and_initialized():
        return value

    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def ddp_broadcast_object(obj: Any, src: int = 0) -> Any:
    if not is_dist_available_and_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# =========================
# batch / meter utils
# =========================
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


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """
    For DistributedSampler, call set_epoch(epoch) every epoch.
    """
    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)

    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
        return

    inner_sampler = getattr(batch_sampler, "sampler", None)
    if isinstance(inner_sampler, DistributedSampler):
        inner_sampler.set_epoch(epoch)
        return

    # 某些自定义 sampler 也可能自己实现 set_epoch
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
        return

    if hasattr(batch_sampler, "set_epoch"):
        batch_sampler.set_epoch(epoch)
        return


# =========================
# checkpoint utils
# =========================
def save_checkpoint(
    save_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    history: list[dict[str, Any]],
    extra_state: dict[str, Any] | None = None,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    raw_model = unwrap_model(model)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_metric": best_metric,
        "history": history,
    }
    if extra_state is not None:
        ckpt["extra_state"] = extra_state

    torch.save(ckpt, save_path)


def load_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=map_location)

    raw_model = unwrap_model(model)
    raw_model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt


# =========================
# one epoch
# =========================
def run_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    loss_type: str,
    alpha_rank: float,
    max_grad_norm: float | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    local_sum: dict[str, float] = {}
    local_num_samples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)

            outputs = model(
                x_seq=batch["x_seq"],
                x_cs=batch["x_cs"],
                x_cs_mask=batch["x_cs_mask"],
            )

            loss_dict = unwrap_model(model).compute_loss(
                outputs=outputs,
                target_return=batch["y_return"],
                loss_type=loss_type,
                alpha_rank=alpha_rank,
            )

            loss = loss_dict["loss"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()

            batch_size = int(batch["x_seq"].shape[0])
            scalar_losses = detach_loss_dict(loss_dict)

            for k, v in scalar_losses.items():
                local_sum[k] = local_sum.get(k, 0.0) + v * batch_size
            local_num_samples += batch_size

    # ---- DDP reduction ----
    # 所有 rank 汇总 sum 和 num_samples，再求全局平均
    global_num_samples = ddp_reduce_scalar(float(local_num_samples), device=device, op=dist.ReduceOp.SUM)
    result = {}

    for k, v in local_sum.items():
        global_sum = ddp_reduce_scalar(float(v), device=device, op=dist.ReduceOp.SUM)
        result[k] = global_sum / max(global_num_samples, 1.0)

    return result


# =========================
# main train
# =========================
def train_model_ddp(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_random_loader: DataLoader,
    train_grouped_loader: DataLoader,
    valid_loader: DataLoader,
    device: str | torch.device,
    num_epochs: int,
    warmup_epochs: int = 0,
    loss_type: str = "gaussian_nll",
    warmup_alpha_rank: float = 0.0,
    main_alpha_rank: float = 0.05,
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
    verbose: bool = True,
) -> dict[str, Any]:
    """
    支持单卡和 DDP。
    要求外部已完成：
      - init_process_group(...)
      - 设置 LOCAL_RANK / device
      - DataLoader 使用 DistributedSampler（若是普通 batch sampler 需自行保证 DDP 切分）
    """
    device = torch.device(device)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)

    if use_ddp and is_dist_available_and_initialized():
        if not isinstance(model, DDP):
            model = DDP(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                output_device=device.index if device.type == "cuda" else None,
                broadcast_buffers=broadcast_buffers,
                find_unused_parameters=find_unused_parameters,
            )

    start_epoch = 0
    history: list[dict[str, Any]] = []

    if monitor_mode not in {"min", "max"}:
        raise ValueError(f"monitor_mode must be 'min' or 'max', got {monitor_mode}")
    if scheduler_step_on not in {"epoch", "valid_metric"}:
        raise ValueError(f"scheduler_step_on must be 'epoch' or 'valid_metric', got {scheduler_step_on}")

    is_better = (
        (lambda current, best: current < best - min_delta)
        if monitor_mode == "min"
        else (lambda current, best: current > best + min_delta)
    )

    best_metric = float("inf") if monitor_mode == "min" else -float("inf")
    best_epoch = -1
    best_state_dict = None
    epochs_without_improve = 0

    # ---- resume ----
    if resume_from is not None:
        ckpt = load_checkpoint(
            ckpt_path=resume_from,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(ckpt["epoch"]) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))
        history = ckpt.get("history", [])

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

    # resume 后同步一次，避免不同 rank 状态不一致
    if is_dist_available_and_initialized():
        barrier()

    for epoch in range(start_epoch, num_epochs):
        if epoch < warmup_epochs:
            train_loader = train_random_loader
            alpha_rank = warmup_alpha_rank
            stage = "warmup"
        else:
            train_loader = train_grouped_loader
            alpha_rank = main_alpha_rank
            stage = "main"

        # DDP sampler 需要每个 epoch 设置
        set_loader_epoch(train_loader, epoch)
        set_loader_epoch(valid_loader, epoch)

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_type=loss_type,
            alpha_rank=alpha_rank,
            max_grad_norm=max_grad_norm,
        )

        valid_metrics = run_one_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            optimizer=None,
            loss_type=loss_type,
            alpha_rank=0.0,   # 更推荐验证时固定 0，避免 rank loss 干扰 early stopping
            max_grad_norm=None,
        )

        epoch_record = {
            "epoch": epoch,
            "stage": stage,
            "alpha_rank": alpha_rank,
        }
        epoch_record.update({f"train/{k}": v for k, v in train_metrics.items()})
        epoch_record.update({f"valid/{k}": v for k, v in valid_metrics.items()})

        current_metric = epoch_record.get(monitor)
        if current_metric is None:
            raise KeyError(f"monitor='{monitor}' not found in epoch_record: {list(epoch_record.keys())}")

        improved = False
        should_stop = False

        # 只在 rank0 决策保存 / early stop
        if is_main_process():
            history.append(epoch_record)

            improved = is_better(current_metric, best_metric)
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improve = 0
                best_state_dict = copy.deepcopy(unwrap_model(model).state_dict())

                if save_best:
                    save_checkpoint(
                        save_path=checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        best_metric=best_metric,
                        history=history,
                        extra_state={
                            "best_epoch": best_epoch,
                            "monitor": monitor,
                            "monitor_mode": monitor_mode,
                            "world_size": get_world_size(),
                        },
                    )
            else:
                epochs_without_improve += 1

            if save_last:
                save_checkpoint(
                    save_path=checkpoint_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    history=history,
                    extra_state={
                        "best_epoch": best_epoch,
                        "monitor": monitor,
                        "monitor_mode": monitor_mode,
                        "world_size": get_world_size(),
                    },
                )

            if scheduler is not None:
                if scheduler_step_on == "epoch":
                    scheduler.step()
                elif scheduler_step_on == "valid_metric":
                    scheduler.step(current_metric)

            if verbose:
                print(
                    f"[Epoch {epoch+1:03d}/{num_epochs:03d}] "
                    f"stage={stage} "
                    f"train_loss={epoch_record.get('train/loss', float('nan')):.6f} "
                    f"valid_loss={epoch_record.get('valid/loss', float('nan')):.6f} "
                    f"{monitor}={current_metric:.6f} "
                    f"best={best_metric:.6f} "
                    f"patience={epochs_without_improve}/{patience}"
                )

            if patience > 0 and epochs_without_improve >= patience:
                should_stop = True
                if verbose:
                    print(
                        f"[EarlyStopping] stop at epoch={epoch}, "
                        f"best_epoch={best_epoch}, best_metric={best_metric:.6f}"
                    )

        # ---- broadcast from rank0 to all ranks ----
        sync_state = {
            "should_stop": should_stop,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        sync_state = ddp_broadcast_object(sync_state, src=0)

        should_stop = sync_state["should_stop"]
        best_metric = sync_state["best_metric"]
        best_epoch = sync_state["best_epoch"]

        if is_dist_available_and_initialized():
            barrier()

        if should_stop:
            break

    # ---- load best weights on all ranks ----
    best_ckpt_path = checkpoint_dir / "best.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
    elif best_state_dict is not None:
        unwrap_model(model).load_state_dict(best_state_dict)

    if is_dist_available_and_initialized():
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
```

---

# 外部怎么初始化 DDP

你外面一般这样写：

```python
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return local_rank, device


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
```

---

# DataLoader 也要改

## 1）普通 random loader

这个最标准，直接用 `DistributedSampler`：

```python
train_random_sampler = DistributedSampler(
    train_dataset,
    shuffle=True,
    drop_last=False,
)

train_random_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_random_sampler,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
)
```

---

## 2）valid loader

验证集一般也建议用 `DistributedSampler`，但 `shuffle=False`：

```python
valid_sampler = DistributedSampler(
    valid_dataset,
    shuffle=False,
    drop_last=False,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
)
```

---

## 3）grouped_by_date_loader 是重点

你这个最关键。

如果你当前 `GroupedByDateBatchSampler` 是自己按日期吐出 batch index，那么 **原版不能直接无脑用于 DDP**，因为：

* 每个 rank 都会遍历同样的 batch
* 导致样本重复训练
* 等于没有做分布式切分

所以你需要把它改成 **DDP-aware 的 batch sampler**。

---

# DDP 版 GroupedByDateBatchSampler

思路：

* 先按 date 分组，得到 `date -> indices`
* 每个 epoch 打乱日期顺序
* 每个日期内部切成多个 batch
* 然后把所有 batch 列表按 `rank/world_size` 均匀切分
* 每个 rank 只拿属于自己的 batch

下面是一个能直接用的版本。

```python
import math
import numpy as np
from torch.utils.data import Sampler


class DistributedGroupedByDateBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: int | None = None,
        rank: int | None = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        if world_size is None:
            world_size = dist.get_world_size() if is_dist_available_and_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if is_dist_available_and_initialized() else 0

        self.world_size = world_size
        self.rank = rank

        self.date_to_indices = self._group_indices_by_date()

    def _group_indices_by_date(self):
        date_to_indices = {}
        for idx in range(len(self.dataset)):
            # 你原来的 metadata 取法
            date = self.dataset[idx]["metadata"]["label_date"]
            if date not in date_to_indices:
                date_to_indices[date] = []
            date_to_indices[date].append(idx)
        return date_to_indices

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _build_all_batches(self) -> list[list[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)

        dates = list(self.date_to_indices.keys())
        if self.shuffle:
            rng.shuffle(dates)

        all_batches = []

        for date in dates:
            indices = list(self.date_to_indices[date])

            if self.shuffle:
                rng.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        return all_batches

    def __iter__(self):
        all_batches = self._build_all_batches()

        # 按 batch 维度切分给各 rank
        if self.drop_last:
            num_batches_per_rank = len(all_batches) // self.world_size
            total_size = num_batches_per_rank * self.world_size
            all_batches = all_batches[:total_size]
        else:
            num_batches_per_rank = math.ceil(len(all_batches) / self.world_size)
            total_size = num_batches_per_rank * self.world_size

            if len(all_batches) < total_size:
                padding = total_size - len(all_batches)
                all_batches.extend(all_batches[:padding])

        rank_batches = all_batches[self.rank:total_size:self.world_size]
        return iter(rank_batches)

    def __len__(self):
        all_batches = self._build_all_batches()
        if self.drop_last:
            return len(all_batches) // self.world_size
        return math.ceil(len(all_batches) / self.world_size)
```

然后这么用：

```python
train_grouped_batch_sampler = DistributedGroupedByDateBatchSampler(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
)

train_grouped_loader = DataLoader(
    train_dataset,
    batch_sampler=train_grouped_batch_sampler,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=collate_fn,
)
```

---

# 训练入口示例

```python
def main():
    local_rank, device = setup_ddp()

    model = MyModel(...)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    train_random_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=False,
    )
    train_random_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_random_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    train_grouped_batch_sampler = DistributedGroupedByDateBatchSampler(
        train_dataset,
        batch_size=64,
        shuffle=True,
        drop_last=False,
    )
    train_grouped_loader = DataLoader(
        train_dataset,
        batch_sampler=train_grouped_batch_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    valid_sampler = DistributedSampler(
        valid_dataset,
        shuffle=False,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=64,
        sampler=valid_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    result = train_model_ddp(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_random_loader=train_random_loader,
        train_grouped_loader=train_grouped_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=100,
        warmup_epochs=5,
        loss_type="gaussian_nll",
        warmup_alpha_rank=0.0,
        main_alpha_rank=0.05,
        scheduler_step_on="valid_metric",
        monitor="valid/loss",
        monitor_mode="min",
        patience=10,
        min_delta=1e-4,
        max_grad_norm=1.0,
        checkpoint_dir="./checkpoints/exp_ddp",
        use_ddp=True,
        broadcast_buffers=False,
        find_unused_parameters=False,
        verbose=True,
    )

    cleanup_ddp()
```

启动：

```bash
torchrun --nproc_per_node=4 train.py
```

---

# 你这个场景下，DDP 最容易踩的坑

## 1）`GroupedByDateBatchSampler` 不能直接复用单卡版

这是你这里最重要的点。

你原来那种 sampler 如果不做 rank 切分，DDP 下每张卡都会训练到同一批数据，白跑。

所以必须像上面这样做 **DistributedGroupedByDateBatchSampler**。

---

## 2）checkpoint 只能 rank0 保存

否则你会得到：

* 多张卡同时写文件
* checkpoint 冲突
* history.json 被反复覆盖

我上面的实现已经处理了：只有 `rank0` 保存。

---

## 3）early stopping 只能 rank0 决策，再广播

否则不同 rank 会在不同时间 break，直接卡死。

我上面的实现里：

* rank0 决定 `should_stop`
* `broadcast_object_list` 广播给所有 rank
* 所有 rank 同步退出

这个是必须的。

---

## 4）验证指标必须全局聚合

DDP 下每个 rank 只看到自己的那部分验证集。

如果你不 all-reduce：

* `valid/loss` 只是本卡 loss
* early stopping 不可靠

我上面的 `run_one_epoch` 已经做了：

* 按 `sum(loss * batch_size)` 聚合
* 按 `sum(num_samples)` 聚合
* 最后算全局平均

这个是对的。

---

## 5）`model.compute_loss(...)` 要通过 `unwrap_model(model)` 调

因为 DDP 包装后，自定义方法通常在 `model.module` 上。

所以我写成：

```python
unwrap_model(model).compute_loss(...)
```

---

# 我建议你再加的两个增强

## A. AMP 混合精度

DDP 下很常见，尤其你是时序模型，显存通常是瓶颈。

可以后面改成：

* `torch.cuda.amp.autocast()`
* `GradScaler`

## B. 监控 `valid/ic` 而不是只看 `valid/loss`

如果你最终目标是排序选股，很多时候：

* `loss` 更低
* 不代表 `IC` 更高

所以更推荐后续把验证阶段补成：

* `valid/loss`
* `valid/ic`
* `valid/rank_ic`

然后 early stopping 监控 `valid/ic`。

---

# 一个很关键的工程建议

你现在有两种 train loader：

* `random_loader`
* `grouped_by_date_loader`

建议你把它们的构造统一成一个工厂函数，比如：

```python
def build_train_loaders_ddp(...):
    ...
    return train_random_loader, train_grouped_loader
```

否则后面很容易出现：

* random_loader 正确做了 `DistributedSampler`
* grouped_loader 忘了做 distributed batch split

这种 bug 很隐蔽。

---

# 最后给你一个结论

你的训练框架如果要支持 DDP，核心不是 `train()` 里加 `DDP(model)` 就完了，真正关键的是这 4 件事：

1. **sampler 必须分布式切分**
2. **指标必须 all-reduce**
3. **checkpoint 只能 rank0 写**
4. **early stopping 必须全局同步**

你这里最特殊的点，就是 **按日期分组的 batch sampler 必须重写成 distributed 版本**。这个比 train loop 本身更重要。

如果你要，我下一条可以直接继续给你一版 **AMP + DDP + gradient accumulation + CSV logger + resume 完整工业 Trainer**。
