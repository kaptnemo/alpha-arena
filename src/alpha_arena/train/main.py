import torch
from torch.utils.data import DataLoader, DistributedSampler

from alpha_arena.train.dataset.loader import (
    SequenceDataset,
    DistributedGroupedByDateBatchSampler,
    collate_fn,
)

from alpha_arena.train.trainer import (
    train_model_ddp,
    setup_ddp,
    cleanup_ddp,
)

from alpha_arena.models.aedh_lstm import (
    AttentionEnhancedDualHeadLSTM,
    AEDH_LSTMConfig,
)


def create_dataloader(
    train_dataset: SequenceDataset,
    valid_dataset: SequenceDataset,
    batch_size: int
) -> dict[str, DataLoader]:
    # 创建分布式采样器
    distributed_sampler = DistributedSampler(train_dataset)
    
    # 创建分布式按日期分组的批次采样器
    grouped_batch_sampler = DistributedGroupedByDateBatchSampler(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 是否在每个 epoch 重新打乱日期顺序
        drop_last=False,  # 是否丢弃最后一个不足 batch_size 的批次
    )
    
    # 创建 DataLoader，使用分布式按日期分组的批次采样器
    train_grouped_loader = DataLoader(
        train_dataset,
        batch_sampler=grouped_batch_sampler,
        num_workers=4,  # 根据需要调整
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,    # 很推荐
        prefetch_factor=2,          # 默认通常够，必要时可试 4
    )

    train_random_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=distributed_sampler,  # 使用分布式采样器进行随机采样
        num_workers=4,  # 根据需要调整
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,    # 很推荐
        prefetch_factor=2,          # 默认通常够，必要时可试 4
    )

    valid_distributed_sampler = DistributedSampler(valid_dataset, shuffle=False)  # 验证集使用分布式采样器但不打乱
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_distributed_sampler,  # 验证集使用分布式采样器但不打乱
        shuffle=False,  # 验证集通常不需要打乱
        num_workers=4,  # 根据需要调整
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,    # 很推荐
        prefetch_factor=2,          # 默认通常够，必要时可试 4
    )


    return {
        "train_grouped_loader": train_grouped_loader,
        "train_random_loader": train_random_loader,
        "valid_loader": valid_loader,
    }


def debug_prediction_batch(batch, output, target_key="y_return"):
    import torch

    pred = output["pred_return"] if isinstance(output, dict) else output
    pred = pred.detach().float().cpu().view(-1)

    if isinstance(batch, dict):
        x = batch["x_seq"].detach().float().cpu()
        y = batch[target_key].detach().float().cpu().view(-1)
    else:
        x, y = batch
        x = x.detach().float().cpu()
        y = y.detach().float().cpu().view(-1)

    print("=== X ===")
    print("shape:", tuple(x.shape))
    print("mean:", x.mean().item())
    print("std :", x.std().item())
    print("min :", x.min().item())
    print("max :", x.max().item())
    print("nan :", torch.isnan(x).any().item())
    print("inf :", torch.isinf(x).any().item())

    print("\n=== Y TRUE ===")
    print("mean:", y.mean().item())
    print("std :", y.std().item())
    print("min :", y.min().item())
    print("max :", y.max().item())
    print("pos ratio:", (y > 0).float().mean().item())

    print("\n=== PRED ===")
    print("mean:", pred.mean().item())
    print("std :", pred.std().item())
    print("min :", pred.min().item())
    print("max :", pred.max().item())
    print("pos ratio:", (pred > 0).float().mean().item())

    print("\n=== COMPARE ===")
    print("pred first 20:", pred[:20])
    print("y first 20   :", y[:20])


def load_model_and_predict(path):
    import torch

    ckpt = torch.load(path, map_location="cpu")

    model_config = ckpt["model_config"]
    # feature_config = ckpt["feature_config"]

    config = AEDH_LSTMConfig(**model_config)
    model = AttentionEnhancedDualHeadLSTM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to("cuda:1")  # 根据需要调整设备
    dataset_name = "csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5"
    dataset_test = SequenceDataset(dataset_name=dataset_name, split_name="test")

    data_loader = DataLoader(
        dataset_test,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )


    # === 推理 ===
    with torch.no_grad():
        for batch in data_loader:
            x_seq = batch["x_seq"].to("cuda:1")  # 根据需要调整设备
            x_cs = batch["x_cs"].to("cuda:1")  # 根据需要调整设备
            x_cs_mask = batch["x_cs_mask"].to("cuda:1")  # 根据需要调整设备
            out = model(x_seq, x_cs, x_cs_mask)
            debug_prediction_batch(batch, out)
            break  # 这里只看一个批次



def main():
    import datetime
    # 先设置好分布式环境
    setup_ddp()

    train_task_name = f'train_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # 创建模型
    config = AEDH_LSTMConfig(
        input_dim=464,  # 根据实际特征维度调整
        hidden_dim=128,
        num_layers=2,
        attn_dim=64,
        head_hidden_dim=64,
        dropout=0.1,
        use_last_state = True,
        cs_feature_dim = 36,
        cs_feature_mask = True,
    )
    model = AttentionEnhancedDualHeadLSTM(config)

    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 加载数据集
    dataset_name = "csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5"
    train_dataset = SequenceDataset(dataset_name=dataset_name, split_name="train")
    valid_dataset = SequenceDataset(dataset_name=dataset_name, split_name="evaluate")

    dataloaders = create_dataloader(train_dataset, valid_dataset, batch_size=512)
    train_grouped_loader = dataloaders["train_grouped_loader"]
    train_random_loader = dataloaders["train_random_loader"]
    valid_loader = dataloaders["valid_loader"]

    num_epochs = 50
    mid_epochs = num_epochs // 2
    warmup_epochs = 10

    # 训练模型
    result = train_model_ddp(
        train_task_name=train_task_name,
        model=model,
        optimizer=optimizer,
        train_random_loader=train_random_loader,
        train_grouped_loader=train_grouped_loader,
        valid_loader=valid_loader,
        device=torch.device("cuda"),
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        mid_epochs=mid_epochs,
        warmup_alpha_rank=0.0,
        mid_alpha_rank=0.02,
        main_alpha_rank=0.05,
        warmup_alpha_mse=50.0,
        mid_alpha_mse=20.0,
        main_alpha_mse=10.0,
        warmup_lr=1e-3,
        mid_lr=3e-4,
        main_lr=1e-4,
        patience=20,
    )

    print("Training completed. Final results:", result)


if __name__ == "__main__":
    main()
    # load_model_and_predict("/data/study/alpha-arena/checkpoints/best_model.pt")
