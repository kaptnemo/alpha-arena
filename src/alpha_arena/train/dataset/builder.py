from __future__ import annotations

from collections.abc import Iterator
import json
import re
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
import pyarrow as pa
import pyarrow.parquet as pq

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from alpha_arena.data import (
    DATASET_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from alpha_arena.data.helpers.tushare_helper import TuShareHelper
from alpha_arena.data.loader import load_from_parquet
from alpha_arena.features.builder import (
    build_panel_features,
    build_panel_features_multiprocess,
)
from alpha_arena.features.config import FeatureConfig, FeatureSpec
from alpha_arena.features.selector import select_lstm_feature_columns
from alpha_arena.features.targets import add_targets
from alpha_arena.utils import get_logger

logger = get_logger(__name__)

_RAW_COLS = frozenset({"ts_code", "date", "open", "high", "low", "close", "volume"})

# Tracks which steps have already been logged in this process.
# In multiprocessing, each worker gets its own copy, so each process logs only once.
_logged_steps: set[str] = set()


def _log_columns_once(
    step: str,
    before: list[str] | None,
    after: list[str],
) -> None:
    """Log column lists before/after a pipeline step, at most once per process."""
    if step in _logged_steps:
        return
    _logged_steps.add(step)
    pid = os.getpid()
    if before is not None:
        logger.info(
            f"[pid={pid}] [step={step}] BEFORE columns ({len(before)}): {before}"
        )
    logger.info(
        f"[pid={pid}] [step={step}] AFTER  columns ({len(after)}): {after}"
    )
_PROCESSED_CONFIG_SUFFIX = ".config.json"

TradeCalendarProvider = Callable[
    [pd.Timestamp, pd.Timestamp, str],
    pd.DataFrame | pd.DatetimeIndex | Sequence[pd.Timestamp | str],
]


@dataclass(frozen=True)
class DatasetYearSplitConfig:
    train_years: Sequence[int]
    evaluate_years: Sequence[int]
    test_years: Sequence[int]

    def __post_init__(self) -> None:
        split_map = self.as_dict()
        overlap_pairs = []
        split_names = tuple(split_map)
        for idx, left_name in enumerate(split_names):
            left_years = set(split_map[left_name])
            for right_name in split_names[idx + 1 :]:
                overlap = left_years.intersection(split_map[right_name])
                if overlap:
                    overlap_pairs.append(
                        f"{left_name}/{right_name} overlap on years {sorted(overlap)}"
                    )
        if overlap_pairs:
            raise ValueError("; ".join(overlap_pairs))

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        return {
            "train": tuple(sorted(set(self.train_years))),
            "evaluate": tuple(sorted(set(self.evaluate_years))),
            "test": tuple(sorted(set(self.test_years))),
        }


@dataclass(frozen=True)
class SequenceSliceConfig:
    sequence_length: int
    start_interval: int = 1
    target_horizons: Sequence[int] = (1, 5, 10)
    max_missing_trade_days_per_gap: int = 0
    max_missing_gaps: int = 0

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if self.start_interval <= 0:
            raise ValueError("start_interval must be positive.")
        if any(h <= 0 for h in self.target_horizons):
            raise ValueError("All target_horizons must be positive.")
        if self.max_missing_trade_days_per_gap < 0:
            raise ValueError("max_missing_trade_days_per_gap must be >= 0.")
        if self.max_missing_gaps < 0:
            raise ValueError("max_missing_gaps must be >= 0.")


@dataclass
class ProcessedPanelConfig:
    raw_file_path: str
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    target_horizons: Sequence[int] = (1, 5, 10)
    processed_file_name: str = "panel_features.parquet"
    processed_dir: Path = field(default_factory=lambda: PROCESSED_DATA_DIR)
    use_multiprocess_features: bool = False
    num_workers: int | None = None
    # 以下字段不接受外部传入，而是由 build_processed_panel 内部根据计算结果填充，供后续 dataset 构建使用
    target_columns: list[str] = field(default_factory=list)
    feature_specs: list[FeatureSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.processed_dir = Path(self.processed_dir)
        self.target_horizons = tuple(
            sorted({int(horizon) for horizon in self.target_horizons})
        )
        if self.target_columns:
            raise ValueError("target_columns should not be set in ProcessedPanelConfig; it is determined by target_horizons.")
        if self.feature_specs:
            raise ValueError("feature_specs should not be set in ProcessedPanelConfig; it is determined by the features built in build_processed_panel.")
        if not self.target_horizons:
            raise ValueError("target_horizons must not be empty.")
        if any(horizon <= 0 for horizon in self.target_horizons):
            raise ValueError("All target_horizons must be positive.")
        if not self.processed_file_name.endswith(".parquet"):
            raise ValueError("processed_file_name must end with '.parquet'.")
        self.raw_file_path = str(Path(self.raw_file_path))

    @property
    def processed_path(self) -> Path:
        return self.processed_dir / self.processed_file_name

    @property
    def config_path(self) -> Path:
        return self.processed_dir / (
            f"{Path(self.processed_file_name).stem}{_PROCESSED_CONFIG_SUFFIX}"
        )

    def artifact_signature(self) -> dict[str, Any]:
        return {
            "raw_file_path": self.raw_file_path,
            "feature_config": _to_json_compatible(asdict(self.feature_config)),
            "target_horizons": list(self.target_horizons),
            "target_columns": self.target_columns,
            "feature_specs": [asdict(spec) for spec in self.feature_specs],
        }

    def build_options(self) -> dict[str, Any]:
        return {
            "use_multiprocess_features": self.use_multiprocess_features,
            "num_workers": self.num_workers,
        }


@dataclass
class DatasetBuilderConfig:
    processed: ProcessedPanelConfig
    splits: DatasetYearSplitConfig
    sequence: SequenceSliceConfig
    label_column: str | None = None
    exchange: str = "SSE"
    dataset_name: str = "sequence_dataset"
    dataset_dir: Path = field(default_factory=lambda: DATASET_DATA_DIR)

    def __post_init__(self) -> None:
        self.dataset_dir = Path(self.dataset_dir)
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty.")
        sequence_horizons = tuple(
            sorted({int(horizon) for horizon in self.sequence.target_horizons})
        )
        if sequence_horizons != tuple(self.processed.target_horizons):
            raise ValueError(
                "sequence.target_horizons must exactly match processed.target_horizons."
            )
        object.__setattr__(self.sequence, "target_horizons", self.processed.target_horizons)
        if self.label_column is None:
            self.label_column = f"y_ret_{min(self.processed.target_horizons)}"
        label_horizon = _target_column_horizon(self.label_column)
        if label_horizon is None:
            raise ValueError("label_column must be y_ret_{h} or y_ret_5_ra.")
        if label_horizon not in self.processed.target_horizons:
            raise ValueError(
                "label_column horizon must be present in processed.target_horizons."
            )


@dataclass(frozen=True)
class DatasetSplitArgs:
    symbol_df: pd.DataFrame
    sequence_config: SequenceSliceConfig
    split_config: DatasetYearSplitConfig
    ts_code: str
    calendar_positions: dict[pd.Timestamp, int]
    feature_specs: Sequence[FeatureSpec | dict]
    trading_calendar: pd.DatetimeIndex
    target_columns: Sequence[str]
    label_column: str


@dataclass(frozen=True)
class DatasetStreamingWriteArgs:
    dataset_dir: Path
    dataset_name: str
    scaler: StandardScaler
    dataset_split_args: DatasetSplitArgs


@dataclass(frozen=True)
class DatasetMetadataArgs:
    symbol_tasks: tuple[DatasetSplitArgs, ...]


@dataclass(frozen=True)
class ProcessedPanelBuildResult:
    processed_path: Path
    config_path: Path


@dataclass(frozen=True)
class DatasetSplitFrames:
    features: pd.DataFrame
    metadata: pd.DataFrame


@dataclass(frozen=True)
class DatasetSplitPaths:
    features_path: Path   # directory containing parquet parts
    metadata_path: Path   # directory containing parquet parts


@dataclass(frozen=True)
class DatasetBuildResult:
    processed_path: Path
    processed_config_path: Path
    dataset_paths: dict[str, Path]
    feature_columns: list[str]
    target_columns: list[str]
    cross_sectional_columns: list[str]
    sample_counts: dict[str, int]


@dataclass
class FeatureStats:
    count: int
    sum_: np.ndarray
    sumsq_: np.ndarray


def update_feature_stats(stats, X):
    if stats is None:
        return FeatureStats(
            count=X.shape[0],
            sum_=X.sum(axis=0),
            sumsq_=(X ** 2).sum(axis=0),
        )
    stats.count += X.shape[0]
    stats.sum_ += X.sum(axis=0)
    stats.sumsq_ += (X ** 2).sum(axis=0)
    return stats


def merge_feature_stats(stats_list: list[FeatureStats], n_features: int) -> FeatureStats:
    total_count = 0
    total_sum = np.zeros(n_features, dtype=np.float64)
    total_sumsq = np.zeros(n_features, dtype=np.float64)

    for stat in stats_list:
        total_count += stat.count
        total_sum += stat.sum_
        total_sumsq += stat.sumsq_

    return FeatureStats(
        count=total_count,
        sum_=total_sum,
        sumsq_=total_sumsq,
    )


def resolve_split_by_end_date(
    end_date: pd.Timestamp,
    split_config: dict[str, tuple[int, ...]],
) -> str | None:
    year = pd.Timestamp(end_date).year
    for split_name, years in split_config.items():
        if year in years:
            return split_name
    return None


def iter_symbol_samples(args: DatasetSplitArgs) -> Iterator[tuple[str, dict]]:
    symbol_df = args.symbol_df
    sequence_config = args.sequence_config
    ts_code = args.ts_code
    calendar_positions = args.calendar_positions
    trading_calendar = args.trading_calendar
    target_columns = args.target_columns
    label_column = args.label_column
    split_config = args.split_config.as_dict()
    if len(symbol_df) < sequence_config.sequence_length:
        # Not enough data to form a single sequence; skip this symbol.
        return
    date_to_row_index = {
        pd.Timestamp(date).normalize(): idx
        for idx, date in enumerate(symbol_df["date"])
    }
    cross_sectional_columns = select_cross_sectional_feature_columns(args.feature_specs)
    max_start = len(symbol_df) - sequence_config.sequence_length + 1
    for start_idx in range(0, max_start, sequence_config.start_interval):
        end_idx = start_idx + sequence_config.sequence_length - 1
        window = symbol_df.iloc[start_idx : end_idx + 1].copy()

        if not _is_window_trade_continuous(
            window["date"],
            calendar_positions,
            sequence_config,
        ):
            continue

        target_info = _compute_targets_for_window(
            symbol_df=symbol_df,
            end_idx=end_idx,
            date_to_row_index=date_to_row_index,
            trading_calendar=trading_calendar,
            calendar_positions=calendar_positions,
            target_columns=target_columns,
            cross_sectional_columns=cross_sectional_columns,
        )
        if target_info is None:
            continue

        label_date = target_info["label_dates"].get(label_column, pd.NaT)
        target_value = target_info["targets"].get(label_column, np.nan)
        target_mask = target_info["target_mask"].get(label_column, 0.0)
        if pd.isna(label_date) or pd.isna(target_value) or target_mask <= 0.0:
            continue

        start_date = window.iloc[0]["date"]
        end_date = window.iloc[-1]["date"]
        sample_id = (
            f"{ts_code}:{start_date.strftime('%Y%m%d')}:"
            f"{end_date.strftime('%Y%m%d')}"
        )

        metadata_row: dict[str, object] = {
            "sample_id": sample_id,
            "ts_code": ts_code,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "sequence_start_date": start_date,
            "sequence_end_date": end_date,
            "label_date": label_date,
        }
        for target_col in target_columns:
            metadata_row[target_col] = target_info["targets"].get(target_col, np.nan)
            metadata_row[_target_mask_column(target_col)] = target_info[
                "target_mask"
            ].get(target_col, 0.0)

        for cross_col in cross_sectional_columns:
            metadata_row[cross_col] = target_info["cross_sectional"].get(cross_col, np.nan)
            metadata_row[f"{cross_col}_mask"] = target_info["cross_sectional_mask"].get(cross_col, 0.0)

        split_name = resolve_split_by_end_date(
            end_date,
            split_config)
        yield split_name, metadata_row


def select_cross_sectional_feature_columns(feature_specs: Sequence[FeatureSpec | dict]) -> list[str]:
    """选择需要参与横截面特征计算的数值特征列。"""
    cross_sectional_columns = []
    for spec in feature_specs:
        kind = spec['kind'] if isinstance(spec, dict) else spec.kind
        name = spec['name'] if isinstance(spec, dict) else spec.name
        if kind == "cross_sectional":
            cross_sectional_columns.append(name)
    return cross_sectional_columns


def select_scaler_feature_columns(feature_specs: Sequence[FeatureSpec | dict]) -> list[str]:
    """只选择需要参与 StandardScaler 的连续数值特征。"""
    scaler_feature_columns = []
    for spec in feature_specs:
        kind = spec['kind'] if isinstance(spec, dict) else spec.kind
        name = spec['name'] if isinstance(spec, dict) else spec.name
        if kind == "numeric" and name not in _RAW_COLS:
            scaler_feature_columns.append(name)
    return scaler_feature_columns


def _subset_by_years(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    years = tuple(sorted(set(int(year) for year in years)))
    if not years:
        return df.iloc[0:0].copy()
    mask = pd.to_datetime(df["date"]).dt.year.isin(years)
    return df.loc[mask].copy()


def fit_train_scaler(
    processed_df: pd.DataFrame,
    scaler_feature_columns: Sequence[str],
    split_config: DatasetYearSplitConfig,
) -> StandardScaler:
    train_df = _subset_by_years(processed_df, split_config.train_years)
    scaler_feature_columns = list(scaler_feature_columns)

    if train_df.empty:
        raise ValueError("No training samples found in the processed DataFrame for the specified train_years.")
    
    missing_cols = [c for c in scaler_feature_columns if c not in train_df.columns]
    if missing_cols:
        raise ValueError(
            f"Scaler columns missing in processed_df: {missing_cols}"
        )

    X_train = train_df.loc[:, scaler_feature_columns].to_numpy(dtype=np.float64, copy=False)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def _worker_get_dataset_metadata_task(
    args: DatasetMetadataArgs,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    symbol_tasks = args.symbol_tasks

    if not symbol_tasks:
        return {}, {}

    split_config = symbol_tasks[0].split_config.as_dict()

    sample_counts = {split_name: 0 for split_name in split_config}

    metadata_rows: dict[str, list[dict]] = {split_name: [] for split_name in split_config}
    for dataset_split_args in symbol_tasks:
        for split_name, metadata_row in iter_symbol_samples(dataset_split_args):
            if split_name is None or metadata_row is None:
                continue
            metadata_rows[split_name].append(metadata_row)
            sample_counts[split_name] += 1

    split_dfs = {}
    for split_name in split_config:
        split_dfs[split_name] = pd.DataFrame(metadata_rows[split_name])

    return split_dfs, sample_counts


def build_and_save_dataset(
    source_df: pd.DataFrame,
    feature_specs: Sequence[FeatureSpec | dict],
    sequence_config: SequenceSliceConfig,
    trading_calendar: pd.DatetimeIndex,
    target_columns: Sequence[str],
    label_column: str,
    split_config: DatasetYearSplitConfig,
    dataset_dir: Path,
    dataset_name: str,
    multiprocess: bool = True,
    num_workers: int | None = None,
) -> tuple[dict[str, Path], dict[str, int]]:
    split_names = tuple(split_config.as_dict().keys())
    dataset_paths: dict[str, Path] = {
        split_name: dataset_dir / f"{dataset_name}_{split_name}_metadata.parquet"
        for split_name in split_names
    }
    sample_counts: dict[str, int] = {split_name: 0 for split_name in split_names}

    calendar_positions = {
        pd.Timestamp(date).normalize(): idx for idx, date in enumerate(trading_calendar)
    }

    groups = [(ts_code, group) for ts_code, group in source_df.groupby("ts_code", sort=False)]
    max_workers = num_workers or max(1, (os.cpu_count() or 1) - 1) if multiprocess else 1
    worker_count = max(1, min(max_workers, len(groups)))

    split_arg_list = [
        DatasetSplitArgs(
            symbol_df=group,
            sequence_config=sequence_config,
            split_config=split_config,
            ts_code=ts_code,
            calendar_positions=calendar_positions,
            feature_specs=feature_specs,
            trading_calendar=trading_calendar,
            target_columns=target_columns,
            label_column=label_column,
        )
        for ts_code, group in groups
    ]

    batch_size = max(1, (len(split_arg_list) + worker_count - 1) // worker_count)
    batch_tasks = [
        DatasetMetadataArgs(
            symbol_tasks=tuple(split_arg_list[i:i + batch_size]),
        )
        for _, i in enumerate(range(0, len(split_arg_list), batch_size))
    ]

    if worker_count == 1:
        results = [_worker_get_dataset_metadata_task(task) for task in batch_tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(_worker_get_dataset_metadata_task, batch_tasks))

    split_metadata_dfs: dict[str, list[pd.DataFrame]] = {split_name: [] for split_name in split_names}
    for metadata_dfs, counts in results:
        for split_name, metadata_df in metadata_dfs.items():
            if not metadata_df.empty:
                split_metadata_dfs[split_name].append(metadata_df)
        for split_name, count in counts.items():
            sample_counts[split_name] += count

    for split_name in split_names:
        if split_metadata_dfs[split_name]:
            full_metadata_df = pd.concat(split_metadata_dfs[split_name], ignore_index=True)
        else:
            full_metadata_df = _empty_metadata_frame(target_columns)

        metadata_path = dataset_paths[split_name]
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        full_metadata_df.to_parquet(metadata_path, index=False, engine="pyarrow")
        logger.info(
            f"Saved {split_name} metadata with {len(full_metadata_df)} samples to {metadata_path}"
        )

    return dataset_paths, sample_counts


def _normalize_calendar_dates(
    calendar: pd.DataFrame | pd.DatetimeIndex | Sequence[pd.Timestamp | str],
) -> pd.DatetimeIndex:
    if isinstance(calendar, pd.DataFrame):
        if "cal_date" not in calendar.columns:
            raise ValueError("Trade calendar DataFrame must contain 'cal_date'.")
        open_mask = calendar["is_open"].eq(1) if "is_open" in calendar.columns else True
        dates = pd.to_datetime(calendar.loc[open_mask, "cal_date"])
    else:
        dates = pd.to_datetime(pd.Index(calendar))

    normalized = pd.DatetimeIndex(pd.Series(dates).dropna().drop_duplicates().sort_values())
    if normalized.empty:
        raise ValueError("Trade calendar is empty.")
    return normalized.normalize()


def fetch_trade_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    exchange: str = "SSE",
    provider: TradeCalendarProvider | None = None,
) -> pd.DatetimeIndex:
    if provider is not None:
        return _normalize_calendar_dates(provider(start_date, end_date, exchange))

    with TuShareHelper() as tushare:
        calendar_result = tushare.trade_cal(
            exchange=exchange,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
        calendar_df = calendar_result.data
    return _normalize_calendar_dates(calendar_df)


def optimize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # float64 → float32
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    # int downcast
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # string
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype("string")

    return df


def build_processed_panel(config: ProcessedPanelConfig) -> ProcessedPanelBuildResult:
    raw_df = load_from_parquet(config.raw_file_path)
    raw_df = raw_df.sort_values(["ts_code", "date"], ignore_index=True)
    _log_columns_once("load_raw", None, list(raw_df.columns))
    if config.use_multiprocess_features:
        panel_df, feature_specs = build_panel_features_multiprocess(
            raw_df,
            cfg=config.feature_config,
            num_workers=config.num_workers,
        )
    else:
        panel_df, feature_specs = build_panel_features(raw_df, cfg=config.feature_config)
    _log_columns_once("build_panel_features", list(raw_df.columns), list(panel_df.columns))
    processed_df, target_columns = add_targets(panel_df, horizons=config.target_horizons)
    _log_columns_once("add_targets", list(panel_df.columns), list(processed_df.columns))
    logger.info(f"raw_df shape: {raw_df.shape}, panel_df shape: {panel_df.shape}, processed_df shape: {processed_df.shape}")
    processed_path = config.processed_path
    config_path = config.config_path
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df = optimize_df(processed_df)
    processed_df = processed_df.sort_values(["ts_code", "date"], ignore_index=True)
    processed_df.to_parquet(processed_path, index=False, engine="pyarrow")
    config.target_columns = target_columns
    config.feature_specs = feature_specs
    _save_processed_config(config)
    logger.info(
        "Processed panel saved",
        file_path=str(processed_path),
        config_path=str(config_path),
        rows=len(processed_df),
        columns=len(processed_df.columns),
    )
    return ProcessedPanelBuildResult(
        processed_path=processed_path,
        config_path=config_path,
    )


def load_processed_panel(
    config: ProcessedPanelConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed_path = config.processed_path
    config_path = config.config_path
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed panel not found: {processed_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Processed config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        persisted_config = json.load(fh)

    # _assert_processed_config_consistent(config, persisted_config)
    processed_df = load_from_parquet(str(processed_path))
    return processed_df, persisted_config


def _save_processed_config(config: ProcessedPanelConfig) -> None:
    payload = {
        "version": 1,
        "artifact_signature": config.artifact_signature(),
        "build_options": config.build_options(),
    }
    config.config_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _assert_processed_config_consistent(
    expected_config: ProcessedPanelConfig,
    persisted_config: dict[str, Any],
) -> None:
    persisted_signature = persisted_config.get("artifact_signature")
    expected_signature = expected_config.artifact_signature()
    if not isinstance(persisted_signature, dict):
        raise ValueError("Processed config is missing 'artifact_signature'.")
    if persisted_signature != expected_signature:
        raise ValueError(
            "Processed artifact config mismatch: "
            f"expected {json.dumps(expected_signature, sort_keys=True)} "
            f"but found {json.dumps(persisted_signature, sort_keys=True)}."
        )


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_json_compatible(item) for item in value]
    return value


def _empty_feature_frame(
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "sequence_position",
            "ts_code",
            "date",
            *feature_columns,
            *(f"{col}_mask" for col in feature_columns),
        ]
    )


def _empty_metadata_frame(target_columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "sample_id",
            "ts_code",
            "start_idx",
            "end_idx",
            "sequence_start_date",
            "sequence_end_date",
            "label_date",
            *target_columns,
            *(_target_mask_column(target_column) for target_column in target_columns),
        ]
    )


def _is_window_trade_continuous(
    dates: pd.Series,
    calendar_positions: dict[pd.Timestamp, int],
    sequence_config: SequenceSliceConfig,
) -> bool:
    gap_count = 0
    normalized_dates = [pd.Timestamp(date).normalize() for date in dates]

    for prev_date, curr_date in zip(normalized_dates[:-1], normalized_dates[1:]):
        prev_pos = calendar_positions.get(prev_date)
        curr_pos = calendar_positions.get(curr_date)
        if prev_pos is None or curr_pos is None or curr_pos <= prev_pos:
            return False
        missing_trade_days = curr_pos - prev_pos - 1
        if missing_trade_days > sequence_config.max_missing_trade_days_per_gap:
            return False
        if missing_trade_days > 0:
            gap_count += 1
            if gap_count > sequence_config.max_missing_gaps:
                return False

    return True


def _compute_targets_for_window(
    symbol_df: pd.DataFrame,
    end_idx: int,
    date_to_row_index: dict[pd.Timestamp, int],
    trading_calendar: pd.DatetimeIndex,
    calendar_positions: dict[pd.Timestamp, int],
    target_columns: Sequence[str],
    cross_sectional_columns: Sequence[str],
) -> dict | None:
    """
    基于已 add_targets 的 symbol_df，在窗口末尾 end_idx 处读取多目标标签，
    以及 anchor 时刻的 cross-sectional sample-level 特征。

    返回：
    {
        "anchor_date": ...,
        "label_dates": {...},
        "targets": {...},
        "target_mask": {...},
        "cross_sectional": {...},
        "cross_sectional_mask": {...},
    }

    若所有 target 和 cross-sectional 特征都无效，则返回 None。
    """
    anchor_row = symbol_df.iloc[end_idx]
    anchor_date = pd.Timestamp(anchor_row["date"]).normalize()
    anchor_position = calendar_positions.get(anchor_date)
    if anchor_position is None:
        return None

    label_dates: dict[str, pd.Timestamp] = {}
    targets: dict[str, float] = {}
    target_mask: dict[str, float] = {}

    cross_sectional: dict[str, float] = {}
    cross_sectional_mask: dict[str, float] = {}

    valid_any = False

    # 1. label / target
    for col in target_columns:
        h = _target_column_horizon(col)
        if h is None:
            label_dates[col] = pd.NaT
            targets[col] = np.nan
            target_mask[col] = 0.0
            continue

        future_position = anchor_position + h
        if future_position >= len(trading_calendar):
            label_dates[col] = pd.NaT
            targets[col] = np.nan
            target_mask[col] = 0.0
            continue

        label_date = pd.Timestamp(trading_calendar[future_position]).normalize()
        label_dates[col] = label_date

        future_row_index = date_to_row_index.get(label_date)
        if future_row_index is None or future_row_index <= end_idx:
            targets[col] = np.nan
            target_mask[col] = 0.0
            continue

        value = anchor_row.get(col, np.nan)
        if pd.isna(value):
            targets[col] = np.nan
            target_mask[col] = 0.0
            continue

        targets[col] = float(value)
        target_mask[col] = 1.0
        valid_any = True

    # 2. sample-level cross-sectional features at anchor time
    for col in cross_sectional_columns:
        if col in target_columns:
            continue

        value = anchor_row.get(col, np.nan)
        if pd.isna(value):
            cross_sectional[col] = 0.0
            cross_sectional_mask[col] = 0.0
            continue
        
        cross_sectional[col] = float(value)
        cross_sectional_mask[col] = 1.0

    if not valid_any:
        return None

    return {
        "anchor_date": anchor_date,
        "label_dates": label_dates,
        "targets": targets,
        "target_mask": target_mask,
        "cross_sectional": cross_sectional,
        "cross_sectional_mask": cross_sectional_mask,
    }

_RETURN_LABEL_PATTERN = re.compile(r"^y_ret_(\d+)$")
_RETURN_RA_LABEL_PATTERN = re.compile(r"^y_ret_(\d+)_ra$")
_RISK_VOL_LABEL_PATTERN = re.compile(r"^y_risk_vol_(\d+)$")
_RISK_LABEL_PATTERN = re.compile(r"^y_risk_(\d+)$")


def _target_column_horizon(target_column: str) -> int | None:
    if (match := _RETURN_LABEL_PATTERN.fullmatch(target_column)) is not None:
        return int(match.group(1))
    if (match := _RETURN_RA_LABEL_PATTERN.fullmatch(target_column)) is not None:
        return int(match.group(1))
    if (match := _RISK_VOL_LABEL_PATTERN.fullmatch(target_column)) is not None:
        return int(match.group(1))
    if (match := _RISK_LABEL_PATTERN.fullmatch(target_column)) is not None:
        return int(match.group(1))
    return None


def _target_mask_column(target_column: str) -> str:
    if target_column == "y_ret_5_ra":
        return "ret_5_ra_mask"
    match = _RETURN_LABEL_PATTERN.fullmatch(target_column)
    if match is not None:
        return f"ret_{match.group(1)}_mask"
    return f"{target_column}_mask"


# def _resolve_target_columns(config: DatasetBuilderConfig) -> list[str]:
#     target_columns = [f"y_ret_{horizon}" for horizon in config.processed.target_horizons]
#     if config.label_column not in target_columns:
#         target_columns.append(config.label_column)
#     return target_columns


def _apply_feature_scaler(
    dataset_df: pd.DataFrame,
    scaler_feature_columns: Sequence[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    if dataset_df.empty:
        return dataset_df

    scaled_df = dataset_df.copy()
    scaler_feature_columns = list(scaler_feature_columns)
    scaled_df[scaler_feature_columns] = scaled_df[scaler_feature_columns].astype(np.float64)
    scaled_df.loc[:, scaler_feature_columns] = scaler.transform(
        scaled_df.loc[:, scaler_feature_columns].to_numpy()
    )
    scaled_df = optimize_df(scaled_df)
    return scaled_df


def build_and_save_features(
    processed_df: pd.DataFrame,
    feature_columns: Sequence[str],
    scaler_feature_columns: Sequence[str],
    scaler: StandardScaler,
    features_dir: Path,
    dataset_name: str,
) -> Path:
    features_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = list(feature_columns)
    scaler_feature_columns = list(scaler_feature_columns)

    missing_feature_cols = [c for c in feature_columns if c not in processed_df.columns]
    if missing_feature_cols:
        raise ValueError(
            f"Feature columns missing in processed_df: {missing_feature_cols}"
        )

    missing_scaler_cols = [c for c in scaler_feature_columns if c not in processed_df.columns]
    if missing_scaler_cols:
        raise ValueError(
            f"Scaler columns missing in processed_df: {missing_scaler_cols}"
        )

    # scaler 列必须是 feature 列子集
    extra_scaler_cols = [c for c in scaler_feature_columns if c not in feature_columns]
    if extra_scaler_cols:
        raise ValueError(
            f"Scaler columns are not a subset of feature_columns: {extra_scaler_cols}"
        )

    # 底表必须包含所有模型输入列
    features_df = processed_df.loc[:, ["ts_code", "date", *feature_columns]].copy()
    features_df["date"] = pd.to_datetime(features_df["date"]).dt.normalize()
    features_df = features_df.sort_values(["ts_code", "date"], ignore_index=True)

    # 1) 强校验：所有 feature_columns 必须是数值型
    non_numeric_cols = (
        features_df[feature_columns]
        .select_dtypes(exclude=[np.number])
        .columns
        .tolist()
    )
    if non_numeric_cols:
        raise ValueError(f"Non-numeric columns in feature_columns: {non_numeric_cols}")

    # 2) 记录填补前缺失情况
    raw_mask = ~features_df[feature_columns].isna()

    # 3) 对所有 feature_columns 做组内前向填充 + 余下补0
    #    这样才能保证 x_seq 真正无 NaN
    features_df[feature_columns] = (
        features_df.groupby("ts_code", sort=False)[feature_columns]
        .ffill()
        .fillna(0.0)
    )

    # 4) 只对 scaler_feature_columns 做标准化
    if scaler_feature_columns:
        features_df = _apply_feature_scaler(
            features_df,
            scaler_feature_columns=scaler_feature_columns,
            scaler=scaler,
        )

    # 5) 生成 mask 列
    mask_df = raw_mask.astype("int8").add_suffix("_mask")
    features_df = pd.concat([features_df, mask_df], axis=1)

    # 6) 最终硬校验：不允许 feature_columns 里残留 NaN
    remaining_nan_cols = features_df[feature_columns].columns[
        features_df[feature_columns].isna().any()
    ].tolist()
    if remaining_nan_cols:
        raise ValueError(f"NaN still exists in feature_columns after fill: {remaining_nan_cols}")

    features_df = optimize_df(features_df)

    features_path = features_dir / f"{dataset_name}_features.parquet"
    features_df.to_parquet(features_path, index=False, engine="pyarrow", compression="snappy")
    return features_path


def build_datasets(
    config: DatasetBuilderConfig,
    trade_calendar_provider: TradeCalendarProvider | None = None,
    multiprocess: bool = True,
    num_workers: int | None = None,
) -> DatasetBuildResult:
    logger.info(
        "Starting dataset build",
        dataset_name=config.dataset_name,
        splits={k: len(v) for k, v in config.splits.as_dict().items()},
        sequence_length=config.sequence.sequence_length,
        target_horizons=config.sequence.target_horizons,
        label_column=config.label_column,
    )
    processed_df, persisted_config = load_processed_panel(config.processed)
    _log_columns_once("load_processed_panel", None, list(processed_df.columns))
    target_columns = persisted_config["artifact_signature"]["target_columns"]
    feature_columns = select_lstm_feature_columns(processed_df, target_columns=target_columns)
    _log_columns_once("select_lstm_feature_columns", list(processed_df.columns), list(feature_columns))
    if not feature_columns:
        raise ValueError("No LSTM feature columns were selected from processed_df.")

    processed_df["date"] = pd.to_datetime(processed_df["date"]).dt.normalize()
    trading_calendar = fetch_trade_calendar(
        start_date=processed_df["date"].min(),
        end_date=processed_df["date"].max(),
        exchange=config.exchange,
        provider=trade_calendar_provider,
    )

    logger.info(
        "Fetched trading calendar",
        start_date=trading_calendar.min(),
        end_date=trading_calendar.max(),
        total_days=len(trading_calendar),
    )
    feature_specs = persisted_config["artifact_signature"]["feature_specs"]

    # 先确定最终真正进入模型、且需要 scaler 的列
    scaler_feature_columns = [
        c for c in select_scaler_feature_columns(feature_specs)
        if c in feature_columns
    ]

    logger.info(
        "Resolved feature groups",
        feature_columns_count=len(feature_columns),
        scaler_feature_columns_count=len(scaler_feature_columns),
    )

    scaler = fit_train_scaler(
        processed_df=processed_df,
        scaler_feature_columns=scaler_feature_columns,
        split_config=config.splits,
    )
    logger.info("Save dataset with feature_base + metadata")

    config.dataset_dir.mkdir(parents=True, exist_ok=True)

    # 先构建并保存全局 feature base
    features_path = build_and_save_features(
        processed_df=processed_df,
        feature_columns=feature_columns,
        scaler_feature_columns=scaler_feature_columns,
        scaler=scaler,
        features_dir=config.dataset_dir,
        dataset_name=config.dataset_name,
    )

    # metadata 生成仍然必须基于保留 target 列的 source_df，而不是 features_df
    cross_sectional_columns = select_cross_sectional_feature_columns(feature_specs)
    metadata_columns = ["ts_code", "date", *target_columns, *cross_sectional_columns]
    metadata_columns = list(dict.fromkeys(metadata_columns))

    metadata_source_df = processed_df.loc[:, metadata_columns].copy()
    metadata_source_df["date"] = pd.to_datetime(metadata_source_df["date"]).dt.normalize()
    metadata_source_df = metadata_source_df.sort_values(["ts_code", "date"], ignore_index=True)

    dataset_paths, sample_counts = build_and_save_dataset(
        source_df=metadata_source_df,
        feature_specs=feature_specs,
        sequence_config=config.sequence,
        trading_calendar=trading_calendar,
        target_columns=target_columns,
        label_column=config.label_column,
        split_config=config.splits,
        dataset_dir=config.dataset_dir,
        dataset_name=config.dataset_name,
        multiprocess=multiprocess,
        num_workers=num_workers,
    )

    dataset_paths["features"] = features_path

    result = DatasetBuildResult(
        processed_path=config.processed.processed_path,
        processed_config_path=config.processed.config_path,
        dataset_paths=dataset_paths,
        feature_columns=feature_columns,
        target_columns=target_columns,
        cross_sectional_columns=select_cross_sectional_feature_columns(feature_specs),
        sample_counts=sample_counts,
    )

    # save result to a json file for easy loading in training script
    result_path = config.dataset_dir / f"{config.dataset_name}_build_result.json"
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "processed_path": str(result.processed_path),
            "processed_config_path": str(result.processed_config_path),
            "dataset_paths": {k: str(v) for k, v in result.dataset_paths.items()},
            "feature_columns": result.feature_columns,
            "target_columns": result.target_columns,
            "cross_sectional_columns": result.cross_sectional_columns,
            "sample_counts": result.sample_counts,
        }, fh, ensure_ascii=True, indent=2, sort_keys=True)
    logger.info(
        "Dataset build completed",
        dataset_name=config.dataset_name,
        result_path=str(result_path),
    )
    return result


__all__ = [
    "DatasetBuildResult",
    "DatasetBuilderConfig",
    "DatasetSplitArgs",
    "DatasetSplitFrames",
    "DatasetSplitPaths",
    "ProcessedPanelBuildResult",
    "ProcessedPanelConfig",
    "DatasetYearSplitConfig",
    "SequenceSliceConfig",
    "build_datasets",
    "build_processed_panel",
    "fetch_trade_calendar",
    "load_processed_panel",
]


if __name__ == "__main__":
    multiprocessing.freeze_support()

    processed_config = ProcessedPanelConfig(
        raw_file_path=str(RAW_DATA_DIR / "csi300_stocks_2017_2025.parquet"),
        feature_config=FeatureConfig(),
        target_horizons=(5, 10, 20),
        processed_file_name="panel_features.parquet",
        use_multiprocess_features=True,
        num_workers=None,
    )

    dataset_config = DatasetBuilderConfig(
        processed=processed_config,
        splits=DatasetYearSplitConfig(
            train_years=range(2017, 2023),
            evaluate_years=range(2023, 2024),
            test_years=range(2024, 2026),
        ),
        sequence=SequenceSliceConfig(
            sequence_length=60,
            start_interval=5,
            target_horizons=(5, 10, 20),
            max_missing_trade_days_per_gap=2,
            max_missing_gaps=1,
        ),
        label_column="y_ret_5",
        exchange="SSE",
        dataset_name="csi300_2017_2025_seq60_step5_targets_5_10_20_label_y_ret_5",
    )

    rebuild_processed_panel = False

    if rebuild_processed_panel:
        build_processed_panel(processed_config)

    result = build_datasets(
        dataset_config,
        trade_calendar_provider=None,
        multiprocess=True,
        num_workers=20,
    )

    print("dataset build done")
    print(result)

