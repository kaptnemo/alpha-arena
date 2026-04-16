"""
targets.py
==========
目标变量（Label）构造模块。

将未来价格信息转化为监督学习的标签，并严格通过 shift(-h)
保证标签只依赖未来数据，不污染特征列。

注意：含目标变量的行在训练时须根据 horizon 去掉末尾 h 行，
防止使用 NaN 或填充值作为监督信号。
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from alpha_arena.features.utils import _safe_div


def add_targets(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10),
    price_col: str = "close",
) -> pd.DataFrame:
    """为 panel DataFrame 构造多周期未来收益率标签。

    对每个预测期 h，计算 h 日后的简单收益率作为回归目标，
    同时构造一个风险调整目标（当 ``volatility_20`` 列存在时）。

    **无数据泄漏保证**：使用 ``groupby("symbol").shift(-h)``，
    仅在同一股票内部向前移位，不会跨股票错误对齐。
    序列末尾 h 行因无未来数据而产生 NaN，训练时需截断。

    Parameters
    ----------
    df:
        已完成特征工程的 panel DataFrame，须包含
        ``symbol``、``date``、``price_col`` 三列，
        且已按 (symbol, date) 排序。
    horizons:
        预测期列表（单位：交易日）。
        默认 (1, 5, 10) 对应短期、周度、半月度预测。
    price_col:
        用于计算收益率的价格列名，默认 ``"close"``。

    Returns
    -------
    pd.DataFrame
        追加了以下目标列的副本：

        - ``y_ret_{h}``   : h 日后简单收益率 = P_{t+h} / P_t - 1
        - ``y_ret_5_ra``  : 5 日收益率除以 volatility_20（仅当该列存在时），
          风险调整收益率，可作为更稳定的训练目标
    """
    out = df.sort_values(["symbol", "date"]).copy()

    for h in horizons:
        # shift(-h)：将未来第 h 期的价格对齐到当前行
        # groupby 保证跨股票边界不连通
        future_price = out.groupby("symbol")[price_col].shift(-h)
        out[f"y_ret_{h}"] = future_price / out[price_col] - 1.0

    # 风险调整目标：5 日收益 / 近期波动率
    # 对高波动时期的大收益进行"惩罚"，使模型关注稳定性而非绝对幅度
    if "volatility_20" in out.columns and "y_ret_5" in out.columns:
        out["y_ret_5_ra"] = _safe_div(out["y_ret_5"], out["volatility_20"])

    return out
