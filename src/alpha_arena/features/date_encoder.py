import numpy as np
import pandas as pd


def _add_time_features(df, date_col='date'):
    df = df.copy()
    dt = pd.to_datetime(df[date_col])

    # basic calendar
    dow = dt.dt.weekday          # 0=Mon, 4=Fri for trading days
    month = dt.dt.month
    doy = dt.dt.dayofyear
    dom = dt.dt.day

    # cyclic encodings
    df['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)

    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12.0)

    df['doy_sin'] = np.sin(2 * np.pi * doy / 365.0)
    df['doy_cos'] = np.cos(2 * np.pi * doy / 365.0)

    df['dom_sin'] = np.sin(2 * np.pi * (dom - 1) / 31.0)
    df['dom_cos'] = np.cos(2 * np.pi * (dom - 1) / 31.0)

    # boundary flags
    df['is_month_start'] = dt.dt.is_month_start.astype(int)
    df['is_month_end'] = dt.dt.is_month_end.astype(int)
    df['is_quarter_start'] = dt.dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = dt.dt.is_quarter_end.astype(int)
    df['is_year_start'] = dt.dt.is_year_start.astype(int)
    df['is_year_end'] = dt.dt.is_year_end.astype(int)

    # natural day gap between trading dates
    df['gap_days'] = dt.diff().dt.days.fillna(1).clip(lower=1)

    time_feature_cols = [
        'dow_sin', 'dow_cos',
        'month_sin', 'month_cos',
        'doy_sin', 'doy_cos',
        'dom_sin', 'dom_cos',
        'is_month_start', 'is_month_end',
        'is_quarter_start', 'is_quarter_end',
        'is_year_start', 'is_year_end',
        'gap_days',
    ]

    return df, time_feature_cols
