from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def build_features(
    df: pd.DataFrame,
    *,
    price_col: str = "close",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    vol_col: str = "vol",
    amount_col: str = "amount",
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    in_universe_col: Optional[str] = "in_csi300",
    lookback_zscore: int = 60,
    future_return_horizon: int = 5,
    clip_value: float = 5.0,
    min_history: int = 80,
    do_ts_zscore: bool = True,
    do_cs_zscore: bool = True,
    keep_raw_cols: bool = True,
) -> pd.DataFrame:
    """
    Build panel features for stock prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel data. Expected columns include:
        [trade_date, ts_code, open, high, low, close, vol, amount]
    price_col, open_col, high_col, low_col, vol_col, amount_col : str
        Column names.
    date_col, code_col : str
        Date and stock code column names.
    in_universe_col : Optional[str]
        Universe mask column, e.g. "in_csi300".
        If provided and exists, cross-sectional normalization will only use rows where mask == 1.
    lookback_zscore : int
        Rolling window for time-series z-score.
    future_return_horizon : int
        Future return horizon used to build label.
    clip_value : float
        Clip z-scored features into [-clip_value, clip_value].
    min_history : int
        Minimum history per stock before sample is considered usable.
    do_ts_zscore : bool
        Whether to apply time-series rolling z-score.
    do_cs_zscore : bool
        Whether to apply cross-sectional z-score by date.
    keep_raw_cols : bool
        Whether to keep original OHLCV columns in output.

    Returns
    -------
    pd.DataFrame
        Feature dataframe with engineered features and labels.
    """
    required_cols = {date_col, code_col, open_col, high_col, low_col, price_col, vol_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()

    # -------------------------
    # Basic cleanup
    # -------------------------
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values([code_col, date_col]).reset_index(drop=True)

    numeric_cols = [open_col, high_col, low_col, price_col, vol_col]
    if amount_col in data.columns:
        numeric_cols.append(amount_col)

    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # Drop obviously invalid rows
    data = data.dropna(subset=[date_col, code_col, price_col, open_col, high_col, low_col, vol_col]).copy()
    data = data[(data[price_col] > 0) & (data[open_col] > 0) & (data[high_col] > 0) & (data[low_col] > 0)].copy()

    # -------------------------
    # Helpers
    # -------------------------
    def groupby_code() -> pd.core.groupby.generic.DataFrameGroupBy:
        return data.groupby(code_col, group_keys=False, sort=False)

    def pct_change_by_code(col: str, periods: int = 1) -> pd.Series:
        return groupby_code()[col].pct_change(periods)

    def shift_by_code(col: str, periods: int = 1) -> pd.Series:
        return groupby_code()[col].shift(periods)

    def rolling_mean_by_code(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        if min_periods is None:
            min_periods = max(3, window // 3)
        return (
            series.groupby(data[code_col], group_keys=False)
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )

    def rolling_std_by_code(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        if min_periods is None:
            min_periods = max(3, window // 3)
        return (
            series.groupby(data[code_col], group_keys=False)
            .rolling(window=window, min_periods=min_periods)
            .std()
            .reset_index(level=0, drop=True)
        )

    def rolling_sum_by_code(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        if min_periods is None:
            min_periods = max(3, window // 3)
        return (
            series.groupby(data[code_col], group_keys=False)
            .rolling(window=window, min_periods=min_periods)
            .sum()
            .reset_index(level=0, drop=True)
        )

    def ewm_by_code(series: pd.Series, span: int) -> pd.Series:
        return (
            series.groupby(data[code_col], group_keys=False)
            .apply(lambda x: x.ewm(span=span, adjust=False, min_periods=span).mean())
            .reset_index(level=0, drop=True)
        )

    def rolling_min_by_code(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        if min_periods is None:
            min_periods = max(3, window // 3)
        return (
            series.groupby(data[code_col], group_keys=False)
            .rolling(window=window, min_periods=min_periods)
            .min()
            .reset_index(level=0, drop=True)
        )

    def rolling_max_by_code(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
        if min_periods is None:
            min_periods = max(3, window // 3)
        return (
            series.groupby(data[code_col], group_keys=False)
            .rolling(window=window, min_periods=min_periods)
            .max()
            .reset_index(level=0, drop=True)
        )

    def ts_zscore(series: pd.Series, window: int) -> pd.Series:
        mean_ = rolling_mean_by_code(series, window)
        std_ = rolling_std_by_code(series, window)
        z = (series - mean_) / (std_ + 1e-8)
        return z

    def cs_zscore(series: pd.Series, mask: Optional[pd.Series] = None) -> pd.Series:
        out = pd.Series(index=series.index, dtype="float64")

        if mask is None:
            grp = data.groupby(date_col)[series.name]
            return data.groupby(date_col)[series.name].transform(
                lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
            )

        # only use masked universe rows to compute cs stats
        for dt, idx in data.groupby(date_col).groups.items():
            idx = list(idx)
            x = series.loc[idx]
            m = mask.loc[idx].fillna(0).astype(bool)

            if m.sum() <= 1:
                out.loc[idx] = np.nan
                continue

            mu = x[m].mean()
            sigma = x[m].std(ddof=0) + 1e-8
            out.loc[idx] = (x - mu) / sigma

        return out

    # -------------------------
    # Basic price/return features
    # -------------------------
    data["ret_1"] = pct_change_by_code(price_col, 1)
    data["ret_2"] = pct_change_by_code(price_col, 2)
    data["ret_3"] = pct_change_by_code(price_col, 3)
    data["ret_5"] = pct_change_by_code(price_col, 5)
    data["ret_10"] = pct_change_by_code(price_col, 10)
    data["ret_20"] = pct_change_by_code(price_col, 20)

    data["log_ret_1"] = np.log(data[price_col] / shift_by_code(price_col, 1))
    data["log_ret_5"] = np.log(data[price_col] / shift_by_code(price_col, 5))
    data["log_ret_20"] = np.log(data[price_col] / shift_by_code(price_col, 20))

    data["intraday_ret"] = data[price_col] / data[open_col] - 1.0
    data["hl_spread"] = data[high_col] / data[low_col] - 1.0
    data["co_gap"] = data[open_col] / shift_by_code(price_col, 1) - 1.0

    # -------------------------
    # Trend / momentum
    # -------------------------
    ma_5 = rolling_mean_by_code(data[price_col], 5)
    ma_10 = rolling_mean_by_code(data[price_col], 10)
    ma_20 = rolling_mean_by_code(data[price_col], 20)
    ma_60 = rolling_mean_by_code(data[price_col], 60)

    data["ma_gap_5"] = data[price_col] / ma_5 - 1.0
    data["ma_gap_10"] = data[price_col] / ma_10 - 1.0
    data["ma_gap_20"] = data[price_col] / ma_20 - 1.0
    data["ma_gap_60"] = data[price_col] / ma_60 - 1.0

    data["ma_cross_5_20"] = ma_5 / ma_20 - 1.0
    data["ma_cross_10_60"] = ma_10 / ma_60 - 1.0

    data["mom_20"] = data[price_col] / shift_by_code(price_col, 20) - 1.0
    data["mom_60"] = data[price_col] / shift_by_code(price_col, 60) - 1.0

    # -------------------------
    # Volatility features
    # -------------------------
    data["realized_vol_5"] = rolling_std_by_code(data["ret_1"], 5)
    data["realized_vol_10"] = rolling_std_by_code(data["ret_1"], 10)
    data["realized_vol_20"] = rolling_std_by_code(data["ret_1"], 20)
    data["realized_vol_60"] = rolling_std_by_code(data["ret_1"], 60)

    prev_close = shift_by_code(price_col, 1)
    tr1 = data[high_col] - data[low_col]
    tr2 = (data[high_col] - prev_close).abs()
    tr3 = (data[low_col] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    data["atr_14"] = rolling_mean_by_code(true_range, 14) / (data[price_col] + 1e-8)
    data["atr_20"] = rolling_mean_by_code(true_range, 20) / (data[price_col] + 1e-8)

    # -------------------------
    # Volume / liquidity
    # -------------------------
    data["log_vol"] = np.log1p(data[vol_col])
    data["vol_chg_1"] = pct_change_by_code(vol_col, 1)

    vol_ma_5 = rolling_mean_by_code(data[vol_col], 5)
    vol_ma_20 = rolling_mean_by_code(data[vol_col], 20)
    data["vol_ratio_5"] = data[vol_col] / (vol_ma_5 + 1e-8)
    data["vol_ratio_20"] = data[vol_col] / (vol_ma_20 + 1e-8)

    if amount_col in data.columns:
        data["log_amount"] = np.log1p(data[amount_col])
        amt_ma_20 = rolling_mean_by_code(data[amount_col], 20)
        data["amount_ratio_20"] = data[amount_col] / (amt_ma_20 + 1e-8)

    # -------------------------
    # RSI
    # -------------------------
    delta = groupby_code()[price_col].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain_14 = rolling_mean_by_code(gain, 14)
    avg_loss_14 = rolling_mean_by_code(loss, 14)
    rs_14 = avg_gain_14 / (avg_loss_14 + 1e-8)
    data["rsi_14"] = 100 - (100 / (1 + rs_14))

    # -------------------------
    # MACD
    # -------------------------
    ema_12 = ewm_by_code(data[price_col], 12)
    ema_26 = ewm_by_code(data[price_col], 26)
    macd_line = ema_12 - ema_26
    signal_line = (
        macd_line.groupby(data[code_col], group_keys=False)
        .apply(lambda x: x.ewm(span=9, adjust=False, min_periods=9).mean())
        .reset_index(level=0, drop=True)
    )
    data["macd_line"] = macd_line / (data[price_col] + 1e-8)
    data["macd_signal"] = signal_line / (data[price_col] + 1e-8)
    data["macd_hist"] = (macd_line - signal_line) / (data[price_col] + 1e-8)

    # -------------------------
    # Bollinger-style
    # -------------------------
    bb_mid_20 = ma_20
    bb_std_20 = rolling_std_by_code(data[price_col], 20)
    bb_up_20 = bb_mid_20 + 2.0 * bb_std_20
    bb_dn_20 = bb_mid_20 - 2.0 * bb_std_20

    data["bb_pos_20"] = (data[price_col] - bb_mid_20) / (2.0 * bb_std_20 + 1e-8)
    data["bb_width_20"] = (bb_up_20 - bb_dn_20) / (bb_mid_20 + 1e-8)

    # -------------------------
    # Stochastic / price position
    # -------------------------
    low_14 = rolling_min_by_code(data[low_col], 14)
    high_14 = rolling_max_by_code(data[high_col], 14)
    data["stoch_k_14"] = (data[price_col] - low_14) / (high_14 - low_14 + 1e-8)

    low_20 = rolling_min_by_code(data[low_col], 20)
    high_20 = rolling_max_by_code(data[high_col], 20)
    data["price_pos_20"] = (data[price_col] - low_20) / (high_20 - low_20 + 1e-8)

    # -------------------------
    # Mean reversion / skewness proxy
    # -------------------------
    data["ret_mean_5"] = rolling_mean_by_code(data["ret_1"], 5)
    data["ret_mean_20"] = rolling_mean_by_code(data["ret_1"], 20)
    data["ret_std_20"] = rolling_std_by_code(data["ret_1"], 20)
    data["reversal_5_1"] = data["ret_1"] - data["ret_mean_5"]

    # -------------------------
    # Label
    # -------------------------
    future_price = shift_by_code(price_col, -future_return_horizon)
    data[f"label_fwd_ret_{future_return_horizon}"] = future_price / data[price_col] - 1.0
    data[f"label_fwd_logret_{future_return_horizon}"] = np.log(future_price / data[price_col])

    # next-day label, too
    future_price_1 = shift_by_code(price_col, -1)
    data["label_fwd_ret_1"] = future_price_1 / data[price_col] - 1.0
    data["label_fwd_logret_1"] = np.log(future_price_1 / data[price_col])

    # -------------------------
    # Count history
    # -------------------------
    data["history_count"] = data.groupby(code_col).cumcount() + 1

    # -------------------------
    # Feature list
    # -------------------------
    feature_cols = [
        "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
        "log_ret_1", "log_ret_5", "log_ret_20",
        "intraday_ret", "hl_spread", "co_gap",
        "ma_gap_5", "ma_gap_10", "ma_gap_20", "ma_gap_60",
        "ma_cross_5_20", "ma_cross_10_60",
        "mom_20", "mom_60",
        "realized_vol_5", "realized_vol_10", "realized_vol_20", "realized_vol_60",
        "atr_14", "atr_20",
        "log_vol", "vol_chg_1", "vol_ratio_5", "vol_ratio_20",
        "rsi_14",
        "macd_line", "macd_signal", "macd_hist",
        "bb_pos_20", "bb_width_20",
        "stoch_k_14", "price_pos_20",
        "ret_mean_5", "ret_mean_20", "ret_std_20", "reversal_5_1",
    ]
    if "log_amount" in data.columns:
        feature_cols += ["log_amount", "amount_ratio_20"]

    # -------------------------
    # Time-series rolling z-score
    # -------------------------
    if do_ts_zscore:
        for col in feature_cols:
            z_col = f"{col}_tsz"
            data[z_col] = ts_zscore(data[col], lookback_zscore)
        feature_cols = [f"{c}_tsz" for c in feature_cols]

    # -------------------------
    # Cross-sectional z-score
    # -------------------------
    if do_cs_zscore:
        mask = None
        if in_universe_col is not None and in_universe_col in data.columns:
            mask = data[in_universe_col].fillna(0).astype(bool)

        cs_feature_cols = []
        for col in feature_cols:
            temp_name = f"{col}_csz"
            tmp = data[col].rename(col)
            data[temp_name] = cs_zscore(tmp, mask=mask)
            cs_feature_cols.append(temp_name)
        feature_cols = cs_feature_cols

    # -------------------------
    # Clip extreme values
    # -------------------------
    for col in feature_cols:
        data[col] = data[col].clip(-clip_value, clip_value)

    # -------------------------
    # Replace inf with nan
    # -------------------------
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # -------------------------
    # Valid sample mask
    # -------------------------
    data["is_feature_valid"] = True
    data.loc[data["history_count"] < min_history, "is_feature_valid"] = False

    # Require enough non-null feature ratio
    non_null_ratio = data[feature_cols].notna().mean(axis=1)
    data.loc[non_null_ratio < 0.8, "is_feature_valid"] = False

    # Require label available
    data.loc[data[f"label_fwd_ret_{future_return_horizon}"].isna(), "is_feature_valid"] = False

    # -------------------------
    # Optional: drop raw cols from final output
    # -------------------------
    base_cols = [date_col, code_col]
    if in_universe_col is not None and in_universe_col in data.columns:
        base_cols.append(in_universe_col)

    label_cols = [
        f"label_fwd_ret_{future_return_horizon}",
        f"label_fwd_logret_{future_return_horizon}",
        "label_fwd_ret_1",
        "label_fwd_logret_1",
    ]

    meta_cols = ["history_count", "is_feature_valid"]

    if keep_raw_cols:
        raw_cols = [c for c in [open_col, high_col, low_col, price_col, vol_col, amount_col] if c in data.columns]
    else:
        raw_cols = []

    final_cols = base_cols + raw_cols + feature_cols + label_cols + meta_cols
    final_cols = [c for c in final_cols if c in data.columns]

    out = data[final_cols].copy()
    out = out.sort_values([date_col, code_col]).reset_index(drop=True)

    return out