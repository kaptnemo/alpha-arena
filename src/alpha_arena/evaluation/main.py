from alpha_arena.train.dataset.loader import (
    SequenceDataset,
    collate_fn,
    GroupedByDateBatchSampler,
)
from torch.utils.data import DataLoader
from alpha_arena.models.aedh_lstm import AttentionEnhancedDualHeadLSTM
from alpha_arena.train.trainer import load_model_from_pretrained
from alpha_arena.evaluation.metrics_main import daily_ic_rankic, summarize_ic
from alpha_arena.evaluation.grouping import add_daily_grouping_by_prediction
import pandas as pd


def test_dataloader(dataset_name: str, batch_size: int) -> DataLoader:

    dataset = SequenceDataset(
        dataset_name=dataset_name,
        split_name="test",
    )

    # group_by_date_sampler = GroupedByDateBatchSampler(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=False,  # 测试集通常不需要打乱
    #     drop_last=False,
    # )

    # test_loader = DataLoader(
    #     dataset,
    #     batch_sampler=group_by_date_sampler,
    #     num_workers=4,
    #     pin_memory=True,
    #     collate_fn=collate_fn,
    #     persistent_workers=True,
    #     prefetch_factor=2,
    # )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 测试集通常不需要打乱
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return test_loader


def predict_with_model(model: AttentionEnhancedDualHeadLSTM, test_loader: DataLoader):
    model = model.to("cuda:1")
    model.eval()
    predictions = []

    for batch in test_loader:
        x_seq = batch["x_seq"].to("cuda:1")
        x_cs = batch["x_cs"].to("cuda:1")
        x_cs_mask = batch["x_cs_mask"].to("cuda:1")
        label_date = batch["label_date"]
        ts_code = batch["ts_code"]
        outputs = model.predict(
            x_seq=x_seq,
            x_cs=x_cs,
            x_cs_mask=x_cs_mask,
        )

        df = pd.DataFrame({
            "ts_code": ts_code,
            "label_date": label_date,
            "pred_return": outputs["pred_return"].cpu().numpy(),
            "pred_var": outputs["pred_var"].cpu().numpy(),
            "y_return": batch["y_return"].cpu().numpy(),
            "y_risk": batch["y_risk"].cpu().numpy(),
        })
        predictions.append(df)
        # break  # 先测试一个批次，确认流程正确后再去掉这个 break 来跑完整个测试集
    try:
        predictions_df = pd.concat(predictions, ignore_index=True)
    except Exception as e:
        print("Error concatenating predictions:", e)
        for i, df in enumerate(predictions):
            print(f"Batch {i} shape: {df.shape}")
        raise e
    return predictions_df


def compute_metrics(predictions_df: pd.DataFrame):
    ic_rankic_df = daily_ic_rankic(
        predictions_df,
        date_col="label_date",
        pred_col="pred_return",
        target_col="y_return",
    )
    print(ic_rankic_df.head())
    summary = summarize_ic(ic_rankic_df)


    predictions_df = add_daily_grouping_by_prediction(
        predictions_df,
        date_col="label_date",
        pred_col="pred_return",
        group_col="pred_group",
        n_groups=5,
    )

    return ic_rankic_df, summary, predictions_df



def main():
    dataset_name = "csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5"
    batch_size = 1024

    test_loader = test_dataloader(dataset_name, batch_size)

    model_path = "checkpoints/train_20260424_192645/best_model.pt"

    model, model_config = load_model_from_pretrained(model_path)

    predictions_df = predict_with_model(model, test_loader)

    ic_rankic_df, summary, predictions_df = compute_metrics(predictions_df)

    ic_rankic_df.to_csv("ic_rankic_by_date.csv", index=False)
    predictions_df.to_csv("predictions_with_groups.csv", index=False)

    print(predictions_df.head())
    print(ic_rankic_df.head())
    print(summary)

if __name__ == "__main__":
    main()