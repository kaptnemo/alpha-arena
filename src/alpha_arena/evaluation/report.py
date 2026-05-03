from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ICAnalysisReport:
    overview: dict[str, Any]
    yearly: pd.DataFrame
    monthly: pd.DataFrame
    weekday: pd.DataFrame
    rolling: pd.DataFrame
    best_rank_ic_days: pd.DataFrame
    worst_rank_ic_days: pd.DataFrame
    correlations: pd.DataFrame


@dataclass(frozen=True)
class PredictionGroupAnalysisReport:
    overview: dict[str, Any]
    by_group: pd.DataFrame
    daily_group_returns: pd.DataFrame
    daily_group_counts: pd.DataFrame
    daily_spread: pd.DataFrame


def load_ic_rankic_by_date(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return prepare_ic_rankic_frame(df)


def load_predictions_with_groups(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return prepare_predictions_with_groups_frame(df)


def prepare_ic_rankic_frame(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    required_columns = {date_col, "n", "ic", "rank_ic"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    prepared = df.copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col])
    prepared = prepared.sort_values(date_col).reset_index(drop=True)
    return prepared


def prepare_predictions_with_groups_frame(
    df: pd.DataFrame,
    date_col: str = "label_date",
    pred_col: str = "pred_return",
    target_col: str = "y_return",
    group_col: str = "pred_group",
) -> pd.DataFrame:
    required_columns = {date_col, pred_col, target_col, group_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    prepared = df.copy()
    prepared[date_col] = pd.to_datetime(prepared[date_col])
    prepared[group_col] = pd.to_numeric(prepared[group_col], errors="coerce").astype(
        "Int64"
    )
    prepared = prepared.sort_values([date_col, group_col]).reset_index(drop=True)
    return prepared


def _summarize_metric(series: pd.Series) -> dict[str, float | int]:
    s = series.dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "ir": float("nan"),
            "positive_ratio": float("nan"),
            "t_stat": float("nan"),
            "q10": float("nan"),
            "median": float("nan"),
            "q90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    mean = float(s.mean())
    std = float(s.std(ddof=1))
    has_dispersion = bool(np.isfinite(std) and std > 0)
    return {
        "count": int(s.count()),
        "mean": mean,
        "std": std,
        "ir": float(mean / std) if has_dispersion else float("nan"),
        "positive_ratio": float((s > 0).mean()),
        "t_stat": float(mean / (std / np.sqrt(len(s))))
        if has_dispersion and len(s) > 1
        else float("nan"),
        "q10": float(s.quantile(0.1)),
        "median": float(s.median()),
        "q90": float(s.quantile(0.9)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def _compute_sign_streaks(series: pd.Series) -> dict[str, int]:
    s = series.dropna()
    if s.empty:
        return {"max_positive_streak": 0, "max_negative_streak": 0}

    signs = np.sign(s)
    change_groups = (signs != signs.shift()).cumsum()
    streaks = s.groupby(change_groups).agg(["size", "first"])
    positive = streaks[streaks["first"] > 0]
    negative = streaks[streaks["first"] < 0]

    return {
        "max_positive_streak": int(positive["size"].max()) if not positive.empty else 0,
        "max_negative_streak": int(negative["size"].max()) if not negative.empty else 0,
    }


def _build_overview(df: pd.DataFrame, date_col: str) -> dict[str, Any]:
    n_summary = df["n"].describe()
    ic_summary = _summarize_metric(df["ic"])
    rank_ic_summary = _summarize_metric(df["rank_ic"])

    sign_mismatch_mask = np.sign(df["ic"]) * np.sign(df["rank_ic"]) < 0
    overview = {
        "rows": int(len(df)),
        "date_start": df[date_col].min(),
        "date_end": df[date_col].max(),
        "n_summary": {key: float(value) for key, value in n_summary.to_dict().items()},
        "ic_summary": ic_summary,
        "rank_ic_summary": rank_ic_summary,
        "ic_streaks": _compute_sign_streaks(df["ic"]),
        "rank_ic_streaks": _compute_sign_streaks(df["rank_ic"]),
        "sign_mismatch_ratio": float(sign_mismatch_mask.mean()),
        "ic_pos_rank_ic_neg_days": int(((df["ic"] > 0) & (df["rank_ic"] < 0)).sum()),
        "ic_neg_rank_ic_pos_days": int(((df["ic"] < 0) & (df["rank_ic"] > 0)).sum()),
    }
    return overview


def summarize_by_year(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    yearly = (
        df.assign(year=df[date_col].dt.year)
        .groupby("year", as_index=False)
        .agg(
            days=(date_col, "size"),
            n_mean=("n", "mean"),
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            rank_ic_mean=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            ic_pos_ratio=("ic", lambda s: (s > 0).mean()),
            rank_ic_pos_ratio=("rank_ic", lambda s: (s > 0).mean()),
        )
    )
    yearly["ic_ir"] = yearly["ic_mean"] / yearly["ic_std"]
    yearly["rank_ic_ir"] = yearly["rank_ic_mean"] / yearly["rank_ic_std"]
    return yearly


def summarize_by_month(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    return (
        df.assign(month=df[date_col].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            days=(date_col, "size"),
            n_mean=("n", "mean"),
            ic_mean=("ic", "mean"),
            rank_ic_mean=("rank_ic", "mean"),
            ic_pos_ratio=("ic", lambda s: (s > 0).mean()),
            rank_ic_pos_ratio=("rank_ic", lambda s: (s > 0).mean()),
        )
    )


def summarize_by_weekday(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekday = (
        df.assign(weekday=df[date_col].dt.day_name())
        .groupby("weekday", as_index=False)
        .agg(
            days=(date_col, "size"),
            n_mean=("n", "mean"),
            ic_mean=("ic", "mean"),
            rank_ic_mean=("rank_ic", "mean"),
        )
    )
    weekday["weekday"] = pd.Categorical(
        weekday["weekday"], categories=weekday_order, ordered=True
    )
    return weekday.sort_values("weekday").reset_index(drop=True)


def add_rolling_means(
    df: pd.DataFrame,
    date_col: str = "date",
    window: int = 20,
) -> pd.DataFrame:
    rolling = df[[date_col, "ic", "rank_ic"]].copy()
    rolling[f"ic_{window}d"] = rolling["ic"].rolling(window).mean()
    rolling[f"rank_ic_{window}d"] = rolling["rank_ic"].rolling(window).mean()
    return rolling


def select_extreme_rank_ic_days(
    df: pd.DataFrame,
    date_col: str = "date",
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [date_col, "n", "ic", "rank_ic"]
    worst = df.nsmallest(top_n, "rank_ic")[columns].reset_index(drop=True)
    best = df.nlargest(top_n, "rank_ic")[columns].reset_index(drop=True)
    return best, worst


def summarize_prediction_groups(
    df: pd.DataFrame,
    *,
    date_col: str = "label_date",
    pred_col: str = "pred_return",
    target_col: str = "y_return",
    group_col: str = "pred_group",
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            observations=(target_col, "size"),
            dates=(date_col, "nunique"),
            pred_mean=(pred_col, "mean"),
            pred_std=(pred_col, "std"),
            return_mean=(target_col, "mean"),
            return_std=(target_col, "std"),
            positive_ratio=(target_col, lambda s: (s > 0).mean()),
        )
        .reset_index()
        .sort_values(group_col)
        .reset_index(drop=True)
    )
    return grouped


def compute_daily_group_returns(
    df: pd.DataFrame,
    *,
    date_col: str = "label_date",
    target_col: str = "y_return",
    group_col: str = "pred_group",
) -> pd.DataFrame:
    grouped = (
        df.groupby([date_col, group_col], dropna=False)[target_col]
        .mean()
        .unstack(group_col)
        .sort_index()
    )
    grouped.columns = [f"group_{int(col)}" for col in grouped.columns]
    return grouped.reset_index()


def compute_daily_group_counts(
    df: pd.DataFrame,
    *,
    date_col: str = "label_date",
    group_col: str = "pred_group",
) -> pd.DataFrame:
    grouped = (
        df.groupby([date_col, group_col], dropna=False)
        .size()
        .unstack(group_col)
        .sort_index()
        .fillna(0)
        .astype(int)
    )
    grouped.columns = [f"group_{int(col)}" for col in grouped.columns]
    return grouped.reset_index()


def compute_daily_group_spread(
    df: pd.DataFrame,
    *,
    date_col: str = "label_date",
    target_col: str = "y_return",
    group_col: str = "pred_group",
) -> pd.DataFrame:
    daily_group_returns = compute_daily_group_returns(
        df,
        date_col=date_col,
        target_col=target_col,
        group_col=group_col,
    )
    group_columns = [c for c in daily_group_returns.columns if c != date_col]
    if not group_columns:
        return pd.DataFrame(
            columns=[
                date_col,
                "top_group",
                "bottom_group",
                "long_short_return",
                "monotonic",
            ]
        )

    bottom_group = group_columns[0]
    top_group = group_columns[-1]
    spread = daily_group_returns[[date_col, bottom_group, top_group]].copy()
    spread["top_group"] = spread[top_group]
    spread["bottom_group"] = spread[bottom_group]
    spread["long_short_return"] = spread["top_group"] - spread["bottom_group"]
    monotonic_pairs = [
        daily_group_returns[group_columns[idx]]
        <= daily_group_returns[group_columns[idx + 1]]
        for idx in range(len(group_columns) - 1)
    ]
    spread["monotonic"] = (
        pd.concat(monotonic_pairs, axis=1).all(axis=1) if monotonic_pairs else True
    )
    return spread[
        [date_col, "top_group", "bottom_group", "long_short_return", "monotonic"]
    ]


def _build_prediction_group_overview(
    df: pd.DataFrame,
    *,
    date_col: str,
    pred_col: str,
    target_col: str,
    group_col: str,
) -> dict[str, Any]:
    daily_group_returns = compute_daily_group_returns(
        df,
        date_col=date_col,
        target_col=target_col,
        group_col=group_col,
    )
    daily_spread = compute_daily_group_spread(
        df,
        date_col=date_col,
        target_col=target_col,
        group_col=group_col,
    )
    group_values = df[group_col].dropna().astype(int)
    return {
        "rows": int(len(df)),
        "date_start": df[date_col].min(),
        "date_end": df[date_col].max(),
        "date_count": int(df[date_col].nunique()),
        "ts_code_count": int(df["ts_code"].nunique())
        if "ts_code" in df.columns
        else None,
        "group_count": int(group_values.nunique()),
        "groups": sorted(group_values.unique().tolist()),
        "pred_mean": float(df[pred_col].mean()),
        "target_mean": float(df[target_col].mean()),
        "daily_mean_n": float(df.groupby(date_col).size().mean()),
        "long_short_mean": float(daily_spread["long_short_return"].mean()),
        "long_short_std": float(daily_spread["long_short_return"].std(ddof=1)),
        "long_short_positive_ratio": float(
            (daily_spread["long_short_return"] > 0).mean()
        ),
        "monotonic_ratio": float(daily_spread["monotonic"].mean()),
        "complete_group_days": int(
            daily_group_returns.drop(columns=[date_col]).notna().all(axis=1).sum()
        ),
    }


def analyze_predictions_with_groups(
    df: pd.DataFrame,
    *,
    date_col: str = "label_date",
    pred_col: str = "pred_return",
    target_col: str = "y_return",
    group_col: str = "pred_group",
) -> PredictionGroupAnalysisReport:
    prepared = prepare_predictions_with_groups_frame(
        df,
        date_col=date_col,
        pred_col=pred_col,
        target_col=target_col,
        group_col=group_col,
    )
    return PredictionGroupAnalysisReport(
        overview=_build_prediction_group_overview(
            prepared,
            date_col=date_col,
            pred_col=pred_col,
            target_col=target_col,
            group_col=group_col,
        ),
        by_group=summarize_prediction_groups(
            prepared,
            date_col=date_col,
            pred_col=pred_col,
            target_col=target_col,
            group_col=group_col,
        ),
        daily_group_returns=compute_daily_group_returns(
            prepared,
            date_col=date_col,
            target_col=target_col,
            group_col=group_col,
        ),
        daily_group_counts=compute_daily_group_counts(
            prepared,
            date_col=date_col,
            group_col=group_col,
        ),
        daily_spread=compute_daily_group_spread(
            prepared,
            date_col=date_col,
            target_col=target_col,
            group_col=group_col,
        ),
    )


def analyze_predictions_with_groups_file(
    csv_path: str | Path,
    *,
    date_col: str = "label_date",
    pred_col: str = "pred_return",
    target_col: str = "y_return",
    group_col: str = "pred_group",
) -> PredictionGroupAnalysisReport:
    df = load_predictions_with_groups(csv_path)
    return analyze_predictions_with_groups(
        df,
        date_col=date_col,
        pred_col=pred_col,
        target_col=target_col,
        group_col=group_col,
    )


def analyze_ic_rankic_by_date(
    df: pd.DataFrame,
    date_col: str = "date",
    rolling_window: int = 20,
    top_n: int = 10,
) -> ICAnalysisReport:
    prepared = prepare_ic_rankic_frame(df, date_col=date_col)
    best_rank_ic_days, worst_rank_ic_days = select_extreme_rank_ic_days(
        prepared,
        date_col=date_col,
        top_n=top_n,
    )

    return ICAnalysisReport(
        overview=_build_overview(prepared, date_col=date_col),
        yearly=summarize_by_year(prepared, date_col=date_col),
        monthly=summarize_by_month(prepared, date_col=date_col),
        weekday=summarize_by_weekday(prepared, date_col=date_col),
        rolling=add_rolling_means(prepared, date_col=date_col, window=rolling_window),
        best_rank_ic_days=best_rank_ic_days,
        worst_rank_ic_days=worst_rank_ic_days,
        correlations=prepared[["n", "ic", "rank_ic"]].corr(),
    )


def analyze_ic_rankic_file(
    csv_path: str | Path,
    date_col: str = "date",
    rolling_window: int = 20,
    top_n: int = 10,
) -> ICAnalysisReport:
    df = load_ic_rankic_by_date(csv_path)
    return analyze_ic_rankic_by_date(
        df,
        date_col=date_col,
        rolling_window=rolling_window,
        top_n=top_n,
    )


def render_ic_rankic_report_text(report: ICAnalysisReport) -> str:
    ic_summary = report.overview["ic_summary"]
    rank_ic_summary = report.overview["rank_ic_summary"]
    return "\n".join(
        [
            f"rows={report.overview['rows']}",
            f"date_range={report.overview['date_start'].date()}..{report.overview['date_end'].date()}",
            (
                f"ic_mean={ic_summary['mean']:.6f}, ic_ir={ic_summary['ir']:.6f}, "
                f"ic_positive_ratio={ic_summary['positive_ratio']:.4f}"
            ),
            (
                f"rank_ic_mean={rank_ic_summary['mean']:.6f}, "
                f"rank_ic_ir={rank_ic_summary['ir']:.6f}, "
                f"rank_ic_positive_ratio={rank_ic_summary['positive_ratio']:.4f}"
            ),
            f"sign_mismatch_ratio={report.overview['sign_mismatch_ratio']:.4f}",
        ]
    )


def render_predictions_with_groups_report_text(
    report: PredictionGroupAnalysisReport,
) -> str:
    overview = report.overview
    by_group = report.by_group
    top_group = by_group.iloc[-1] if not by_group.empty else None
    bottom_group = by_group.iloc[0] if not by_group.empty else None
    lines = [
        f"rows={overview['rows']}",
        f"date_range={overview['date_start'].date()}..{overview['date_end'].date()}",
        f"date_count={overview['date_count']}, group_count={overview['group_count']}",
        f"long_short_mean={overview['long_short_mean']:.6f}, long_short_positive_ratio={overview['long_short_positive_ratio']:.4f}",
        f"monotonic_ratio={overview['monotonic_ratio']:.4f}, complete_group_days={overview['complete_group_days']}",
    ]
    if top_group is not None and bottom_group is not None:
        lines.append(
            "top_minus_bottom_group_mean="
            f"{top_group['return_mean'] - bottom_group['return_mean']:.6f}"
        )
    return "\n".join(lines)


__all__ = [
    "ICAnalysisReport",
    "PredictionGroupAnalysisReport",
    "add_rolling_means",
    "analyze_ic_rankic_by_date",
    "analyze_ic_rankic_file",
    "analyze_predictions_with_groups",
    "analyze_predictions_with_groups_file",
    "compute_daily_group_counts",
    "compute_daily_group_returns",
    "compute_daily_group_spread",
    "load_ic_rankic_by_date",
    "load_predictions_with_groups",
    "prepare_ic_rankic_frame",
    "prepare_predictions_with_groups_frame",
    "render_ic_rankic_report_text",
    "render_predictions_with_groups_report_text",
    "select_extreme_rank_ic_days",
    "summarize_prediction_groups",
    "summarize_by_month",
    "summarize_by_weekday",
    "summarize_by_year",
]


if __name__ == "__main__":
    # path = "/data/study/alpha-arena/evaluations/train_20260424_192645_ic_rankic_by_date.csv"
    path = "/data/study/alpha-arena/evaluations/train_20260503_125825_predictions_with_groups.csv"
    report = analyze_predictions_with_groups_file(path)
    print(render_predictions_with_groups_report_text(report))
