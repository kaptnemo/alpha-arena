from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_arena.features.config import FeatureConfig
from alpha_arena.train.dataset.builder import (
    DatasetBuilderConfig,
    DatasetYearSplitConfig,
    ProcessedPanelConfig,
    SequenceSliceConfig,
    apply_grouped_rolling_normalization,
    build_datasets,
    build_processed_panel,
    classify_preprocess_feature_columns,
    filter_derived_zscore_feature_columns,
    generate_split_anchor_calendars,
)


def _trade_calendar(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    exchange: str,
) -> pd.DatetimeIndex:
    del start_date, end_date, exchange
    return pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2023-12-26",
                "2023-12-27",
                "2023-12-28",
                "2023-12-29",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
            ]
        )
    )


def _make_feature_config() -> FeatureConfig:
    return FeatureConfig(
        price_windows=(2,),
        vol_windows=(2,),
        zscore_windows=(2,),
        cross_sectional_rank=False,
        add_time_features=False,
        add_risk_adjusted_features=True,
        add_ta_features=False,
        add_pandas_ta_features=False,
        fill_method="ffill",
        clip_return=0.2,
    )


def _make_processed_config(raw_path: Path, tmp_path: Path) -> ProcessedPanelConfig:
    return ProcessedPanelConfig(
        raw_file_path=str(raw_path),
        feature_config=_make_feature_config(),
        target_horizons=(1,),
        processed_file_name="processed.parquet",
        processed_dir=tmp_path / "processed",
    )


def _make_cross_year_raw_panel() -> pd.DataFrame:
    dates = _trade_calendar(pd.Timestamp("2023-12-26"), pd.Timestamp("2024-01-09"), "SSE")
    rows: list[dict[str, object]] = []
    for symbol_idx, ts_code in enumerate(["000001.SZ", "000002.SZ"]):
        base_price = 10.0 + symbol_idx
        prev_close = base_price
        for date_idx, date in enumerate(dates):
            close = base_price + 0.1 * date_idx
            open_ = prev_close
            rows.append(
                {
                    "ts_code": ts_code,
                    "date": date,
                    "open": open_,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "pre_close": prev_close,
                    "volume": 1_000_000 + date_idx * 1_000,
                    "in_csi300": True,
                }
            )
            prev_close = close
    return pd.DataFrame(rows)


def _make_universe_filter_panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=70, freq="B")
    rows: list[dict[str, object]] = []
    in_csi300 = [True] * len(dates)
    in_csi300[60] = False
    for symbol_idx, ts_code in enumerate(["000001.SZ", "000002.SZ"]):
        prev_close = 10.0 + symbol_idx
        for date_idx, (date, in_universe) in enumerate(zip(dates, in_csi300, strict=True)):
            close = 10.0 + symbol_idx + 0.1 * date_idx
            rows.append(
                {
                    "ts_code": ts_code,
                    "date": date,
                    "open": prev_close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "pre_close": prev_close,
                    "volume": 1_000_000 + symbol_idx * 10_000 + date_idx * 1_000,
                    "in_csi300": in_universe,
                }
            )
            prev_close = close
    return pd.DataFrame(rows)


def test_generate_split_anchor_calendars_uses_label_date_and_split_specific_intervals(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_panel.parquet"
    _make_cross_year_raw_panel().to_parquet(raw_path, index=False)

    processed_config = _make_processed_config(raw_path=raw_path, tmp_path=tmp_path)
    config = DatasetBuilderConfig(
        processed=processed_config,
        splits=DatasetYearSplitConfig(
            train_years=(2023,),
            evaluate_years=(),
            test_years=(2024,),
        ),
        sequence=SequenceSliceConfig(
            sequence_length=3,
            start_interval=5,
            target_horizons=(1,),
        ),
        label_column="y_ret_1",
        dataset_name="unit",
        dataset_dir=tmp_path / "dataset",
        split_on="label_date",
        train_start_interval=2,
        test_start_interval=1,
    )

    split_anchor_calendars, anchor_split_map = generate_split_anchor_calendars(
        trading_calendar=_trade_calendar(pd.Timestamp("2023-12-26"), pd.Timestamp("2024-01-09"), "SSE"),
        sequence_config=config.sequence,
        split_config=config.splits,
        label_column=config.label_column,
        split_on=config.split_on,
        start_interval_by_split=config.start_interval_by_split(),
    )

    assert split_anchor_calendars["train"].strftime("%Y-%m-%d").tolist() == ["2023-12-28"]
    assert split_anchor_calendars["test"].strftime("%Y-%m-%d").tolist() == [
        "2023-12-29",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
    ]
    assert anchor_split_map[pd.Timestamp("2023-12-29")] == "test"


def test_build_datasets_supports_anchor_date_universe_filter_and_anchor_metadata(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_panel.parquet"
    _make_universe_filter_panel().to_parquet(raw_path, index=False)

    processed_config = _make_processed_config(raw_path=raw_path, tmp_path=tmp_path)
    build_processed_panel(processed_config)

    base_kwargs = dict(
        processed=processed_config,
        splits=DatasetYearSplitConfig(
            train_years=(2024,),
            evaluate_years=(),
            test_years=(),
        ),
        sequence=SequenceSliceConfig(
            sequence_length=3,
            start_interval=1,
            target_horizons=(1,),
        ),
        label_column="y_ret_1",
        dataset_name="unit",
        dataset_dir=tmp_path / "dataset",
        split_on="label_date",
    )

    anchor_mode_result = build_datasets(
        DatasetBuilderConfig(
            **base_kwargs,
            universe_filter_mode="anchor_date",
        ),
        trade_calendar_provider=lambda *_args: pd.DatetimeIndex(
            pd.date_range("2024-01-02", periods=70, freq="B")
        ),
        multiprocess=False,
    )
    full_window_result = build_datasets(
        DatasetBuilderConfig(
            dataset_name="unit_full_window",
            universe_filter_mode="full_window",
            **{k: v for k, v in base_kwargs.items() if k != "dataset_name"},
        ),
        trade_calendar_provider=lambda *_args: pd.DatetimeIndex(
            pd.date_range("2024-01-02", periods=70, freq="B")
        ),
        multiprocess=False,
    )

    assert anchor_mode_result.sample_counts["train"] == 132
    assert full_window_result.sample_counts["train"] == 128
    assert all(not column.endswith("_z2") for column in anchor_mode_result.feature_columns)

    anchor_metadata = pd.read_parquet(anchor_mode_result.dataset_paths["train"])
    assert "anchor_date" in anchor_metadata.columns
    anchor_dates = sorted(anchor_metadata["anchor_date"].dt.strftime("%Y-%m-%d").unique().tolist())
    assert anchor_dates[0] == "2024-01-04"
    assert anchor_dates[-1] == "2024-04-05"
    assert "2024-03-26" not in anchor_dates


def test_apply_grouped_rolling_normalization_uses_per_symbol_history() -> None:
    dates = pd.date_range("2024-01-02", periods=70, freq="B")
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(dates) + ["000002.SZ"] * len(dates),
            "date": list(dates) * 2,
            "feature_a": list(range(70)) + list(range(100, 170)),
        }
    )

    result = apply_grouped_rolling_normalization(
        df,
        feature_columns=["feature_a"],
        window=252,
        min_periods=60,
    )

    first_symbol = result[result["ts_code"] == "000001.SZ"].reset_index(drop=True)
    assert not first_symbol["feature_a"].isna().any()
    assert first_symbol["date"].iloc[0] == dates[60]

    window_values = pd.Series(range(60), dtype="float64")
    expected_day_60 = (60.0 - window_values.mean()) / window_values.std(ddof=0)
    assert abs(first_symbol.loc[0, "feature_a"] - expected_day_60) < 1e-6

    second_symbol = result[result["ts_code"] == "000002.SZ"].reset_index(drop=True)
    assert len(first_symbol) == len(dates) - 60
    assert first_symbol.loc[0, "feature_a"] == second_symbol.loc[0, "feature_a"]


def test_feature_preprocess_groups_drop_derived_zscores() -> None:
    filtered = filter_derived_zscore_feature_columns(
        [
            "ret_1",
            "ret_1_z10",
            "ma_ratio_5",
            "rsi_14",
            "bb_pos",
            "obv",
            "macd_hist_z20",
        ]
    )
    assert filtered == ["ret_1", "ma_ratio_5", "rsi_14", "bb_pos", "obv"]

    bounded_features, ratio_features, normal_features = classify_preprocess_feature_columns(filtered)
    assert bounded_features == ["rsi_14", "bb_pos"]
    assert ratio_features == ["ret_1", "ma_ratio_5"]
    assert normal_features == ["obv"]
