from __future__ import annotations

import pandas as pd
from pandas.testing import assert_frame_equal

from alpha_arena.features.builder import (
    build_panel_features,
    build_panel_features_multiprocess,
)
from alpha_arena.features.config import FeatureConfig


def _make_panel_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=15, freq="B")
    ts_codes = ["000001.SZ", "000002.SZ", "000004.SZ"]

    rows: list[dict[str, object]] = []
    for ts_code_idx, ts_code in enumerate(ts_codes):
        base_price = 10.0 + ts_code_idx * 3.0
        prev_close = base_price

        for date_idx, date in enumerate(dates):
            drift = 0.12 * date_idx + 0.03 * ts_code_idx
            close = base_price + drift + (date_idx % 3) * 0.05
            open_ = prev_close + 0.02 * ((date_idx % 2) - 0.5)
            high = max(open_, close) + 0.08 + 0.01 * ts_code_idx
            low = min(open_, close) - 0.07 - 0.01 * ts_code_idx
            volume = 1_000_000 + ts_code_idx * 10_000 + date_idx * 2_500

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


def test_build_panel_features_multiprocess_matches_single_process() -> None:
    test_df = _make_panel_df()
    cfg = FeatureConfig(
        price_windows=(3, 5),
        vol_windows=(3, 5),
        zscore_windows=(5,),
        cross_sectional_rank=True,
        add_time_features=True,
        add_risk_adjusted_features=True,
        add_ta_features=False,
        add_pandas_ta_features=False,
        fill_method="ffill",
        clip_return=0.2,
    )

    panel_feat_single = build_panel_features(test_df, cfg)
    panel_feat_mp = build_panel_features_multiprocess(test_df, cfg, num_workers=2)

    assert_frame_equal(
        panel_feat_single,
        panel_feat_mp,
        check_dtype=True,
        check_like=False,
        atol=1e-6,
        rtol=1e-7,
    )
