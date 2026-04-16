"""
utils.py
========
内部工具函数，供 pipeline 各模块共享调用。

所有函数均以下划线开头，表示模块内部使用，不对外暴露为公开 API。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_arena.features.config import REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def _check_input(df: pd.DataFrame) -> pd.DataFrame:
    """校验输入 DataFrame 并做基础规范化。

    检查必要列是否存在，将 date 列转为 datetime 类型，
    并按 (symbol, date) 升序排序，保证后续时序操作的正确性。

    Parameters
    ----------
    df:
        原始输入 DataFrame。

    Returns
    -------
    pd.DataFrame
        规范化后的副本，索引已重置。

    Raises
    ------
    ValueError
        当任意必要列缺失时抛出，并在消息中列出所有缺失列名。
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ts_code", "date"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 数值工具
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    """安全除法：分母加微小扰动，避免除零产生 inf / NaN。

    Parameters
    ----------
    a, b:
        分子与分母 Series，形状须相同。
    eps:
        分母加法平滑项，默认 1e-12。
    """
    return a / (b.abs() + eps)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """对单列 Series 计算滚动 Z-score 标准化。

    使用滚动均值和标准差将序列标准化，使每个时间步的值
    相对于过去 ``window`` 个交易日的分布保持可比性。
    前 ``window-1`` 行因样本不足返回 NaN（min_periods=window 严格执行）。

    Parameters
    ----------
    s:
        输入时间序列（单只股票的某个特征列）。
    window:
        滚动窗口长度（交易日数）。
    """
    mean_ = s.rolling(window, min_periods=window).mean()
    std_ = s.rolling(window, min_periods=window).std()
    return (s - mean_) / (std_ + 1e-12)


# ---------------------------------------------------------------------------
# 横截面工具
# ---------------------------------------------------------------------------

def _cross_sectional_zscore(
    df: pd.DataFrame,
    col: str,
    date_col: str = "date",
    universe_col: str = "in_csi300",
) -> pd.Series:
    """在每个交易日截面内对指定列做 Z-score 标准化。

    只对 in_csi300 的股票进行标准化，消除不同交易日整体市场水平差异，使因子值在截面上可横向比较。
    标准差为零时（极端情形：所有股票当天值相同）加 1e-12 保护。

    Parameters
    ----------
    df:
        panel DataFrame，包含多只股票多个交易日的数据。
    col:
        要标准化的列名。
    date_col:
        日期列名，默认 ``"date"``。
    universe_col:
        股票池列名，默认 ``"in_csi300"``，只对该股票池内的股票进行标准化。
    """
    masked = df[col].where(df[universe_col])
    grp = masked.groupby(df[date_col])
    mean_ = grp.transform("mean")
    std_ = grp.transform("std")
    return (masked - mean_) / (std_ + 1e-12)


def _cross_sectional_rank(
    df: pd.DataFrame,
    col: str,
    date_col: str = "date",
    universe_col: str = "in_csi300",
    pct: bool = True,
) -> pd.Series:
    """在每个交易日截面内对指定列计算百分比排名。

    每个交易日只对 in_csi300 的股票进行排名，返回值范围 (0, 1]，1 表示当日最高值。
    百分比排名相比绝对排名对样本量变化更鲁棒。

    Parameters
    ----------
    df:
        panel DataFrame。
    col:
        要排名的列名。
    date_col:
        日期列名，默认 ``"date"``。
    pct:
        是否返回百分比排名，默认 ``True``。
    universe_col:
        股票池列名，默认 ``"in_csi300"``，只对该股票池内的股票进行排名。
    """
    masked = df[col].where(df[universe_col])

    cs_rank = masked.groupby(df[date_col]).rank(pct=pct)

    # 防止某些日期只有1个样本 → rank=1 → 过拟合
    if pct:
        cs_rank = cs_rank.clip(0.0, 1.0)

    return cs_rank

# ---------------------------------------------------------------------------
# 时间特征
# ---------------------------------------------------------------------------

def _add_time_features(g: pd.DataFrame) -> pd.DataFrame:
    """为单只股票的 DataFrame 追加日历周期特征。

    使用正弦/余弦编码将离散日历变量（星期几、月份、年内第几天）
    映射为连续的周期信号，避免序数编码带来的"1 月与 12 月不相邻"等问题。

    生成的列：
    - ``dow_sin / dow_cos``    : 星期几周期编码（周期 = 7 天）
    - ``month_sin / month_cos``: 月份周期编码（周期 = 12 个月）
    - ``doy_sin / doy_cos``    : 年内第几天周期编码（周期 = 365.25 天）

    Parameters
    ----------
    g:
        单只股票的 DataFrame，必须包含 ``date`` 列（datetime 类型）。
    """
    g = g.copy()
    dt = g["date"]

    dow = dt.dt.dayofweek          # 0=周一, 6=周日
    month = dt.dt.month            # 1–12
    day_of_year = dt.dt.dayofyear  # 1–366

    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    g["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    g["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    g["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    g["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return g
