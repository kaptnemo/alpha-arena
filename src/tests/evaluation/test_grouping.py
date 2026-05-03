from __future__ import annotations

import pandas as pd

from alpha_arena.evaluation.grouping import add_daily_grouping_by_prediction


def test_add_daily_grouping_by_prediction_splits_each_day_into_five_groups() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 10
            + [pd.Timestamp("2024-01-03")] * 5,
            "ts_code": [f"{i:06d}.SZ" for i in range(15)],
            "y_pred": [0.1, 0.4, 0.2, 0.8, 0.7, 0.5, 0.6, 0.9, 0.3, 0.0, 5, 4, 3, 2, 1],
        }
    )

    result = add_daily_grouping_by_prediction(df)

    first_day = result[result["date"] == pd.Timestamp("2024-01-02")].sort_values(
        "y_pred"
    )
    assert first_day["pred_group"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    second_day = result[result["date"] == pd.Timestamp("2024-01-03")].sort_values(
        "y_pred"
    )
    assert second_day["pred_group"].tolist() == [1, 2, 3, 4, 5]


def test_add_daily_grouping_by_prediction_preserves_missing_predictions() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 4,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "y_pred": [0.3, None, 0.1, 0.2],
        }
    )

    result = add_daily_grouping_by_prediction(df)

    assert result["pred_group"].dtype == "Int64"
    assert result.loc[result["ts_code"] == "000002.SZ", "pred_group"].isna().all()
    valid_groups = (
        result.loc[result["pred_group"].notna()]
        .sort_values("y_pred")["pred_group"]
        .tolist()
    )
    assert valid_groups == [1, 2, 3]
