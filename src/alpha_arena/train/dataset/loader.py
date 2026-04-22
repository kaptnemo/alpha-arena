import json

import torch
from torch.utils.data import (
    Dataset,
    DataLoader,
    Sampler,
)
import numpy as np
import pandas as pd

from alpha_arena.data import (
    DATASET_DATA_DIR,
)
from alpha_arena.utils import get_logger
from alpha_arena.data.loader import load_from_parquet

logger = get_logger(__name__)


class SequenceDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 split_name: str,
                 y_return_col: str = "y_ret_5",
                 y_risk_col: str = "y_risk_vol_5",
                 **kwargs):
        self.y_return_col = y_return_col
        self.y_risk_col = y_risk_col
        self.metadata_path = DATASET_DATA_DIR / f"{dataset_name}_{split_name}_metadata.parquet"
        self.features_path = DATASET_DATA_DIR / f"{dataset_name}_features.parquet"
        self.config_path = DATASET_DATA_DIR / f"{dataset_name}_build_result.json"
        self.features = load_from_parquet(self.features_path, **kwargs)
        self.data = load_from_parquet(self.metadata_path, **kwargs)
        self.config = self.load_config()

        # 在 __post_init__ 中根据 self.data 计算得到
        self.label_dates = None
        # 在 __post_init__ 中根据 self.config 和 self.features.columns 计算得到
        self.feature_columns = None
        self.feature_mask_columns = None
        self.seq_columns = None
        self.cross_sectional_columns = None
        self.__post_init__()

    def __post_init__(self):
        required_cols = {"start_idx", "end_idx", self.y_return_col, self.y_risk_col}
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns in metadata: {sorted(missing)}")

        feature_columns = self.config["feature_columns"]
        missing_feature_cols = set(feature_columns) - set(self.features.columns)
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns in features: {sorted(missing_feature_cols)}")

        feature_mask_columns = [f"{col}_mask" for col in feature_columns]
        missing_mask_cols = set(feature_mask_columns) - set(self.features.columns)
        if missing_mask_cols:
            raise ValueError(f"Missing feature mask columns in features: {sorted(missing_mask_cols)}")

        self.data = (
            self.data[self.data[self.y_return_col].notna() & self.data[self.y_risk_col].notna()]
            .reset_index(drop=True)
        )
        if self.data.empty:
            raise ValueError("No valid samples after filtering out rows with NaN targets.")

        self.label_dates = self.data["label_date"].to_numpy()
        self.feature_columns = self.config["feature_columns"]
        self.feature_mask_columns = [f"{col}_mask" for col in self.feature_columns]
        self.seq_columns = self.feature_columns + self.feature_mask_columns
        self.cross_sectional_columns = self.config.get("cross_sectional_columns", [])

        logger.info(f"Loaded dataset with {len(self.data)} samples after filtering.")

    def load_config(self):

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as fh:
            config = json.load(fh)

        if "feature_columns" not in config:
            raise ValueError("Config missing 'feature_columns'")
        if not isinstance(config["feature_columns"], list) or not config["feature_columns"]:
            raise ValueError("'feature_columns' must be a non-empty list")

        return config

    def __len__(self):
        return len(self.data)

    def check_sample_validity(self, sample):
        if np.isnan(sample["x_seq"]).any():
            logger.warning("NaN values found in x_seq")
            return False
        if np.isnan(sample["x_cs"]).any():
            logger.warning("NaN values found in x_cs")
            return False
        if np.isnan(sample["x_cs_mask"]).any():
            logger.warning("NaN values found in x_cs_mask")
            return False
        if np.isnan(sample["y_return"]):
            logger.warning("NaN value found in y_return")
            return False
        if np.isnan(sample["y_risk"]):
            logger.warning("NaN value found in y_risk")
            return False
        return True

    def __getitem__(self, idx):
        metadata = self.data.iloc[idx]
        start_idx = int(metadata["start_idx"])
        end_idx = int(metadata["end_idx"])
        
        x_seq = self.features.iloc[start_idx:end_idx + 1][self.seq_columns].to_numpy()

        y_return = metadata[self.y_return_col]
        y_risk = metadata[self.y_risk_col]

        cross_sectional_columns = [col for col in self.cross_sectional_columns if col in metadata.index]
        cross_sectional_mask_columns = [f"{col}_mask" for col in cross_sectional_columns]

        if cross_sectional_columns:
            x_cs_series = metadata[cross_sectional_columns]
            x_cs = x_cs_series.to_numpy(dtype=np.float32)

            if all(col in metadata.index for col in cross_sectional_mask_columns):
                x_cs_mask = metadata[cross_sectional_mask_columns].to_numpy(dtype=np.float32)
            else:
                x_cs_mask = (~x_cs_series.isna().to_numpy()).astype(np.float32)
        else:
            x_cs = np.empty(0, dtype=np.float32)
            x_cs_mask = np.empty(0, dtype=np.float32)

        sample = {
            "x_seq": x_seq.astype(np.float32),
            "x_cs": x_cs.astype(np.float32),
            "x_cs_mask": x_cs_mask.astype(np.float32),
            "y_return": np.float32(y_return),
            "y_risk": np.float32(y_risk),
            "metadata": {
                "label_date": metadata["label_date"],
            }
        }

        if not self.check_sample_validity(sample):
            raise ValueError(f"Invalid sample at index {idx}: contains NaN values in features or targets.")

        return sample


def collate_fn(batch):
    return {
        "x_seq": torch.as_tensor(np.stack([item["x_seq"] for item in batch]), dtype=torch.float32),
        "x_cs": torch.as_tensor(np.stack([item["x_cs"] for item in batch]), dtype=torch.float32),
        "x_cs_mask": torch.as_tensor(np.stack([item["x_cs_mask"] for item in batch]), dtype=torch.float32),
        "y_return": torch.as_tensor([item["y_return"] for item in batch], dtype=torch.float32),
        "y_risk": torch.as_tensor([item["y_risk"] for item in batch], dtype=torch.float32),
    }


class GroupedByDateBatchSampler(Sampler):
    def __init__(self,
                 dataset: SequenceDataset,
                 batch_size: int,
                 shuffle: bool = True,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.date_to_indices = self._group_indices_by_date()

    def _group_indices_by_date(self):
        date_to_indices = {}
        for idx in range(len(self.dataset)):
            date = self.dataset.label_dates[idx]
            if date not in date_to_indices:
                date_to_indices[date] = []
            date_to_indices[date].append(idx)
        return date_to_indices

    def __iter__(self):
        dates = list(self.date_to_indices.keys())
        if self.shuffle:
            np.random.shuffle(dates)

        for date in dates:
            indices = self.date_to_indices[date].copy()
            if self.shuffle:
                np.random.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        total = 0
        for indices in self.date_to_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


if __name__ == "__main__":
    dataset = SequenceDataset(
        dataset_name="csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5",
        split_name="test",
    )

    for i in range(20):
        assert dataset[i]['metadata']['label_date'] == dataset.label_dates[i], f"Metadata label_date {dataset[i]['metadata']['label_date']} does not match label_dates[{i}] = {dataset.label_dates[i]}"

    random_dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn)
    
    grouped_by_date_sampler = GroupedByDateBatchSampler(dataset, batch_size=64, shuffle=True)
    grouped_dataloader = DataLoader(
        dataset,
        batch_sampler=grouped_by_date_sampler,
        num_workers=4,
        collate_fn=collate_fn)

    for batch in random_dataloader:
        print(batch)
        break

    for batch in grouped_dataloader:
        print(batch)
        break
