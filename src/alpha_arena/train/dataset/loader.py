import json

from torch.utils.data import Dataset, DataLoader
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
        self.data = self.data[self.data[self.y_return_col].notna() & self.data[self.y_risk_col].notna()].reset_index(drop=True)
        if self.data.empty:
            raise ValueError("No valid samples after filtering out rows with NaN targets.")
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
        feature_columns = self.config["feature_columns"]
        x_seq = self.features.iloc[start_idx:end_idx + 1][feature_columns].to_numpy()

        y_return = metadata[self.y_return_col]
        y_risk = metadata[self.y_risk_col]

        config_cross_sectional_columns = self.config.get("cross_sectional_columns", [])
        cross_sectional_columns = [col for col in config_cross_sectional_columns if col in metadata]
        cross_sectional_mask_columns = [f'{col}_mask' for col in cross_sectional_columns]
        if cross_sectional_columns:
            x_cs = metadata[cross_sectional_columns].to_numpy()
            if cross_sectional_mask_columns and all(col in metadata for col in cross_sectional_mask_columns):
                x_cs_mask = metadata[cross_sectional_mask_columns].to_numpy(dtype=np.float32)
            else:
                x_cs_mask = (~np.isnan(x_cs)).astype(np.float32)
        else:
            x_cs = np.empty(0, dtype=np.float32)
            x_cs_mask = np.empty(0, dtype=np.float32)

        sample = {
            "x_seq": x_seq.astype(np.float32),
            "x_cs": x_cs.astype(np.float32),
            "x_cs_mask": x_cs_mask.astype(np.float32),
            "y_return": np.float32(y_return),
            "y_risk": np.float32(y_risk),
        }

        if not self.check_sample_validity(sample):
            logger.error(f"Invalid sample at index {idx}: {sample}")
            import pdb; pdb.set_trace()

        return sample


if __name__ == "__main__":
    dataset = SequenceDataset(
        dataset_name="csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5",
        split_name="test",
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in dataloader:
        print(batch)