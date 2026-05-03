from __future__ import annotations

import pandas as pd
import pytest

from alpha_arena.evaluation.report import (
    analyze_ic_rankic_by_date,
    analyze_predictions_with_groups,
    render_ic_rankic_report_text,
    render_predictions_with_groups_report_text,
)


def test_analyze_ic_rankic_by_date_builds_expected_sections() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-02-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "n": [300, 299, 298, 297, 296],
            "ic": [0.10, -0.05, 0.02, -0.01, 0.03],
            "rank_ic": [0.20, -0.10, 0.05, 0.04, 0.01],
        }
    )

    report = analyze_ic_rankic_by_date(df, rolling_window=2, top_n=2)

    assert report.overview["rows"] == 5
    assert report.overview["rank_ic_summary"]["count"] == 5
    assert report.overview["ic_pos_rank_ic_neg_days"] == 0
    assert report.overview["ic_neg_rank_ic_pos_days"] == 1
    assert report.yearly["year"].tolist() == [2024, 2025]
    assert report.monthly["month"].tolist() == ["2024-01", "2024-02", "2025-01"]
    assert report.weekday["weekday"].tolist() == [
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]
    assert "ic_2d" in report.rolling.columns
    assert len(report.best_rank_ic_days) == 2
    assert len(report.worst_rank_ic_days) == 2
    assert report.correlations.shape == (3, 3)


def test_render_ic_rankic_report_text_contains_core_summary() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "n": [300, 300],
            "ic": [0.1, -0.1],
            "rank_ic": [0.2, 0.0],
        }
    )

    report = analyze_ic_rankic_by_date(df, rolling_window=2, top_n=1)
    text = render_ic_rankic_report_text(report)

    assert "rows=2" in text
    assert "rank_ic_mean=" in text
    assert "sign_mismatch_ratio=" in text


def test_analyze_predictions_with_groups_builds_group_sections() -> None:
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "label_date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "pred_return": [0.1, 0.2, 0.3, 0.4],
            "y_return": [0.01, 0.03, -0.02, 0.05],
            "pred_group": [1, 2, 1, 2],
        }
    )

    report = analyze_predictions_with_groups(df)

    assert report.overview["rows"] == 4
    assert report.overview["group_count"] == 2
    assert report.overview["complete_group_days"] == 2
    assert report.by_group["pred_group"].tolist() == [1, 2]
    assert report.daily_group_returns.columns.tolist() == [
        "label_date",
        "group_1",
        "group_2",
    ]
    assert report.daily_group_counts.columns.tolist() == [
        "label_date",
        "group_1",
        "group_2",
    ]
    assert "long_short_return" in report.daily_spread.columns
    assert report.daily_spread["long_short_return"].tolist() == pytest.approx(
        [0.02, 0.07]
    )


def test_render_predictions_with_groups_report_text_contains_core_summary() -> None:
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "label_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "pred_return": [0.1, 0.2],
            "y_return": [0.01, 0.03],
            "pred_group": [1, 2],
        }
    )

    report = analyze_predictions_with_groups(df)
    text = render_predictions_with_groups_report_text(report)

    assert "rows=2" in text
    assert "long_short_mean=" in text
    assert "monotonic_ratio=" in text
