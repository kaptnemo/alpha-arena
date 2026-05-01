from __future__ import annotations

import numpy as np
import pandas as pd


def add_daily_grouping_by_prediction(
    df: pd.DataFrame,
    date_col: str = "date",
    pred_col: str = "y_pred",
    n_groups: int = 5,
    group_col: str = "pred_group",
) -> pd.DataFrame:
    """
    Cross-sectionally sort stocks by daily prediction and assign quantile groups.

    Group ids start at 1. Higher predicted values are assigned to higher groups.
    When the number of valid predictions in a day is smaller than ``n_groups``,
    the function falls back to the maximum feasible number of non-empty groups.
    """
    required_columns = {date_col, pred_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if n_groups <= 0:
        raise ValueError(f"n_groups must be positive, got {n_groups}")

    grouped_df = df.copy()
    grouped_df[group_col] = pd.Series(pd.NA, index=grouped_df.index, dtype="Int64")

    for _, index in grouped_df.groupby(date_col, sort=False).groups.items():
        day_index = pd.Index(index)
        day_slice = grouped_df.loc[day_index, pred_col]
        valid_mask = day_slice.notna()
        if not valid_mask.any():
            continue

        valid_values = day_slice[valid_mask]
        effective_groups = min(n_groups, len(valid_values))
        ranks = valid_values.rank(method="first", ascending=True)
        group_ids = np.ceil(ranks * effective_groups / len(valid_values)).astype(np.int64)
        grouped_df.loc[valid_values.index, group_col] = pd.array(group_ids, dtype="Int64")

    return grouped_df


def add_daily_grouping_by_prediction(
    df: pd.DataFrame,
    date_col: str = "date",
    pred_col: str = "pred_return",
    target_col: str | None = None,
    ts_code_col: str | None = "ts_code",
    n_groups: int = 5,
    group_col: str = "pred_group",
) -> pd.DataFrame:
    required_columns = {date_col, pred_col}
    if target_col is not None:
        required_columns.add(target_col)
    if ts_code_col is not None:
        required_columns.add(ts_code_col)

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if n_groups <= 0:
        raise ValueError(f"n_groups must be positive, got {n_groups}")

    out = df.copy()
    out[group_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    valid_base = out[pred_col].notna()
    if target_col is not None:
        valid_base &= out[target_col].notna()

    for _, day_idx in out.groupby(date_col, sort=False).groups.items():
        day_idx = pd.Index(day_idx)
        valid_idx = day_idx[valid_base.loc[day_idx]]

        if len(valid_idx) == 0:
            continue

        if ts_code_col is not None:
            valid_idx = (
                out.loc[valid_idx, [pred_col, ts_code_col]]
                .sort_values([pred_col, ts_code_col], ascending=[True, True])
                .index
            )
        else:
            valid_idx = out.loc[valid_idx, pred_col].sort_values().index

        effective_groups = min(n_groups, len(valid_idx))
        ranks = np.arange(1, len(valid_idx) + 1)
        group_ids = np.ceil(ranks * effective_groups / len(valid_idx)).astype("int64")

        out.loc[valid_idx, group_col] = pd.array(group_ids, dtype="Int64")

    return out