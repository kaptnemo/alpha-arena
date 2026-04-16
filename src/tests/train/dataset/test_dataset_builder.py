from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_arena.features.config import FeatureConfig
from alpha_arena.train.dataset.builder import (
    DatasetBuilderConfig,
    ProcessedPanelConfig,
    DatasetYearSplitConfig,
    SequenceSliceConfig,
    build_dataset_split,
    build_datasets,
    build_processed_panel,
    load_processed_panel,
)


def _business_day_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    exchange: str,
) -> pd.DatetimeIndex:
    del exchange
    return pd.date_range(start_date, end_date, freq="B")


def _make_raw_panel() -> pd.DataFrame:
    dates = (
        list(pd.date_range("2022-01-03", periods=12, freq="B"))
        + list(pd.date_range("2023-01-02", periods=12, freq="B"))
        + list(pd.date_range("2024-01-02", periods=12, freq="B"))
    )
    symbols = ["000001.SZ", "000002.SZ"]

    rows: list[dict[str, object]] = []
    for symbol_idx, ts_code in enumerate(symbols):
        base_price = 10.0 + symbol_idx * 2.0
        prev_close = base_price
        for date_idx, date in enumerate(dates):
            close = base_price + 0.15 * date_idx + symbol_idx * 0.05
            open_ = prev_close + 0.01 * ((date_idx % 3) - 1)
            high = max(open_, close) + 0.1
            low = min(open_, close) - 0.1
            volume = 1_000_000 + symbol_idx * 10_000 + date_idx * 1_000

            rows.append(
                {
                    "ts_code": ts_code,
                    "date": date,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "pre_close": prev_close,
                    "volume": volume,
                    "in_csi300": True,
                }
            )
            prev_close = close

    return pd.DataFrame(rows)


def _make_feature_config(*, cross_sectional_rank: bool = False) -> FeatureConfig:
    return FeatureConfig(
        price_windows=(2,),
        vol_windows=(2,),
        zscore_windows=(2,),
        cross_sectional_rank=cross_sectional_rank,
        add_time_features=False,
        add_risk_adjusted_features=True,
        add_ta_features=False,
        add_pandas_ta_features=False,
        fill_method="ffill",
        clip_return=0.2,
    )


def _make_processed_config(
    *,
    raw_path: Path,
    tmp_path: Path,
    feature_config: FeatureConfig | None = None,
) -> ProcessedPanelConfig:
    return ProcessedPanelConfig(
        raw_file_path=str(raw_path),
        feature_config=feature_config or _make_feature_config(),
        target_horizons=(1,),
        processed_file_name="processed.parquet",
        processed_dir=tmp_path / "processed",
    )


def test_build_processed_panel_saves_processed_parquet_and_config(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_panel.parquet"
    _make_raw_panel().to_parquet(raw_path, index=False)

    processed_config = _make_processed_config(raw_path=raw_path, tmp_path=tmp_path)

    result = build_processed_panel(processed_config)

    assert result.processed_path.exists()
    assert result.config_path.exists()

    processed_df, persisted_config = load_processed_panel(processed_config)
    assert "y_ret_1" in processed_df.columns
    assert persisted_config["artifact_signature"] == processed_config.artifact_signature()
    assert json.loads(result.config_path.read_text(encoding="utf-8"))["build_options"] == (
        processed_config.build_options()
    )


def test_build_datasets_reads_processed_artifact_without_cross_split_leakage(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_panel.parquet"
    _make_raw_panel().to_parquet(raw_path, index=False)

    processed_config = _make_processed_config(raw_path=raw_path, tmp_path=tmp_path)
    build_processed_panel(processed_config)

    config = DatasetBuilderConfig(
        processed=processed_config,
        splits=DatasetYearSplitConfig(
            train_years=(2022,),
            evaluate_years=(2023,),
            test_years=(2024,),
        ),
        sequence=SequenceSliceConfig(
            sequence_length=3,
            start_interval=2,
            target_horizons=(1,),
        ),
        label_column="y_ret_1",
        dataset_name="unit",
        dataset_dir=tmp_path / "dataset",
    )

    result = build_datasets(config, trade_calendar_provider=_business_day_calendar)

    assert result.processed_path.exists()
    assert result.processed_config_path.exists()
    processed_df = pd.read_parquet(result.processed_path)
    assert "y_ret_1" in processed_df.columns
    assert result.feature_columns

    for split_name, split_year in {"train": 2022, "evaluate": 2023, "test": 2024}.items():
        dataset_paths = result.dataset_paths[split_name]
        assert dataset_paths.features_path.exists()
        assert dataset_paths.metadata_path.exists()
        features_df = pd.read_parquet(dataset_paths.features_path)
        metadata_df = pd.read_parquet(dataset_paths.metadata_path)
        assert not features_df.empty
        assert not metadata_df.empty
        assert len(metadata_df) == result.sample_counts[split_name]
        assert metadata_df["split"].eq(split_name).all()
        assert metadata_df["sequence_start_date"].dt.year.eq(split_year).all()
        assert metadata_df["sequence_end_date"].dt.year.eq(split_year).all()
        assert metadata_df["label_date"].dt.year.eq(split_year).all()
        assert metadata_df["mask_ret_1"].eq(1.0).all()
        assert {"sample_id", "ts_code", "start_idx", "end_idx", "y_ret_1", "mask_ret_1"} <= set(
            metadata_df.columns
        )
        assert features_df["sample_id"].isin(metadata_df["sample_id"]).all()
        assert features_df["date"].dt.year.eq(split_year).all()

        sample_lengths = features_df.groupby("sample_id")["sequence_position"].agg(["count", "max"])
        assert sample_lengths["count"].eq(config.sequence.sequence_length).all()
        assert sample_lengths["max"].eq(config.sequence.sequence_length - 1).all()

    train_dataset_df = pd.read_parquet(result.dataset_paths["train"].features_path)
    assert np.allclose(
        train_dataset_df[result.feature_columns].to_numpy().mean(axis=0),
        0.0,
        atol=1e-6,
    )
    evaluate_dataset_df = pd.read_parquet(result.dataset_paths["evaluate"].features_path)
    test_dataset_df = pd.read_parquet(result.dataset_paths["test"].features_path)
    assert not np.allclose(
        evaluate_dataset_df[result.feature_columns].to_numpy().mean(axis=0),
        0.0,
        atol=1e-3,
    )
    assert not np.allclose(
        test_dataset_df[result.feature_columns].to_numpy().mean(axis=0),
        0.0,
        atol=1e-3,
    )


def test_build_datasets_rejects_processed_config_mismatch(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_panel.parquet"
    _make_raw_panel().to_parquet(raw_path, index=False)

    saved_processed_config = _make_processed_config(raw_path=raw_path, tmp_path=tmp_path)
    build_processed_panel(saved_processed_config)
    mismatched_processed_config = _make_processed_config(
        raw_path=raw_path,
        tmp_path=tmp_path,
        feature_config=_make_feature_config(cross_sectional_rank=True),
    )

    config = DatasetBuilderConfig(
        processed=mismatched_processed_config,
        splits=DatasetYearSplitConfig(
            train_years=(2022,),
            evaluate_years=(2023,),
            test_years=(2024,),
        ),
        sequence=SequenceSliceConfig(
            sequence_length=3,
            start_interval=2,
            target_horizons=(1,),
        ),
        label_column="y_ret_1",
        dataset_name="unit",
        dataset_dir=tmp_path / "dataset",
    )

    with pytest.raises(ValueError, match="Processed artifact config mismatch"):
        build_datasets(config, trade_calendar_provider=_business_day_calendar)


def test_build_dataset_split_respects_gap_tolerance() -> None:
    split_df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 4,
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"]
            ),
            "close": [10.0, 10.2, 10.4, 10.6],
            "ret_1": [0.01, 0.02, 0.03, 0.04],
            "y_ret_1": [0.02, 10.4 / 10.2 - 1.0, 10.6 / 10.4 - 1.0, np.nan],
        }
    )
    trading_calendar = pd.date_range("2024-01-02", "2024-01-08", freq="B")

    strict_df = build_dataset_split(
        split_name="train",
        split_df=split_df,
        feature_columns=["ret_1"],
        sequence_config=SequenceSliceConfig(
            sequence_length=3,
            start_interval=1,
            target_horizons=(1,),
            max_missing_trade_days_per_gap=0,
            max_missing_gaps=0,
        ),
        trading_calendar=trading_calendar,
        label_column="y_ret_1",
    )
    tolerant_df = build_dataset_split(
        split_name="train",
        split_df=split_df,
        feature_columns=["ret_1"],
        sequence_config=SequenceSliceConfig(
            sequence_length=3,
            start_interval=1,
            target_horizons=(1,),
            max_missing_trade_days_per_gap=1,
            max_missing_gaps=1,
        ),
        trading_calendar=trading_calendar,
        label_column="y_ret_1",
    )

    assert strict_df.features.empty
    assert strict_df.metadata.empty
    assert tolerant_df.metadata["sample_id"].nunique() == 1
    assert tolerant_df.features["sequence_position"].tolist() == [0, 1, 2]
    assert tolerant_df.metadata["label_date"].dt.strftime("%Y-%m-%d").unique().tolist() == [
        "2024-01-08"
    ]
    assert tolerant_df.metadata["mask_ret_1"].tolist() == [1.0]
