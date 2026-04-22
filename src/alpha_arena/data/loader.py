import pandas as pd
from pathlib import Path
# from torch.utils.data import Dataset, DataLoader
from alpha_arena.data import (
    RAW_DATA_DIR,
)

from alpha_arena.utils import get_logger

logger = get_logger(__name__)


def is_full_path(path_str: str) -> bool:
    """如果包含目录信息（绝对或相对路径）返回 True，纯文件名返回 False"""
    p = Path(path_str)
    # 方法1：检查是否包含父目录（即 name 不等于原始字符串）
    if p.name != path_str:
        return True
    # 方法2：检查是否为绝对路径（以 / 或盘符开头）
    if p.is_absolute():
        return True
    return False


def load_from_parquet(file_path: str | Path, **kwargs):
    """Load data from a parquet file and return a DataFrame."""
    file_path = Path(file_path)
    if file_path.suffix != '.parquet':
        raise ValueError("Unsupported file format. Only parquet files are supported.")
    if is_full_path(str(file_path)):
        logger.info("Loading data from specified file path", file_path=str(file_path))
    else:
        file_path = RAW_DATA_DIR / file_path
        logger.info("Loading data from RAW_DATA_DIR", file_path=file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_parquet(file_path, **kwargs)


if __name__ == "__main__":
    from alpha_arena.data import DATASET_DATA_DIR
    # Example usage
    new_file_path = DATASET_DATA_DIR / "csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5_train_metadata.parquet"
    df = load_from_parquet(new_file_path)
    print(df.head())
    print(df.tail())
