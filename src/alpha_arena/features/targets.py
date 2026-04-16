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
import numpy as np

from alpha_arena.features.utils import _safe_div


def add_targets(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10),
    price_col: str = "close",
    add_risk_target: bool = True,
    add_risk_adjusted_return: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """为 panel DataFrame 构造未来收益与未来风险标签。

    目标定义
    --------
    对每个预测期 h：

    - y_ret_{h}:
        未来 h 日简单收益率 = P[t+h] / P[t] - 1

    - y_risk_vol_{h}:
        未来 h 日实现波动率（realized volatility），
        基于未来逐日收益 ret_1 在区间 (t, t+h] 内的标准差计算。
        这是真正可作为 risk head 的监督信号。

    - y_ret_{h}_ra:
        风险调整收益 = y_ret_{h} / volatility_20
        仅当当前行存在 volatility_20 时构造。
        这是 reward-to-risk 风格目标，不是纯 risk。

    无数据泄漏说明
    --------------
    1. 收益标签 y_ret_{h} 通过 groupby("ts_code").shift(-h) 构造，
       仅使用同一股票未来价格。
    2. 风险标签 y_risk_vol_{h} 仅使用未来窗口内的逐日收益，
       不使用未来窗口之外的信息。
    3. 若 df 中已有 ret_1 / volatility_20，它们应当由历史数据计算得到。

    Parameters
    ----------
    df:
        panel DataFrame，至少包含 ts_code, date, price_col。
        且建议已按 (ts_code, date) 排序。
    horizons:
        预测期（交易日）。
    price_col:
        价格列，默认 close。
    add_risk_target:
        是否构造未来实现波动率标签。
    add_risk_adjusted_return:
        是否构造风险调整收益标签。

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        增广后的 DataFrame 与目标列名列表。
    """
    required_cols = {"ts_code", "date", price_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.sort_values(["ts_code", "date"]).copy()
    target_columns: list[str] = []

    # 若没有 ret_1，则先按股票内部构造逐日收益
    if "ret_1" not in out.columns:
        out["ret_1"] = out.groupby("ts_code", sort=False)[price_col].pct_change()

    grouped_price = out.groupby("ts_code", sort=False)[price_col]

    for h in horizons:
        # -------- 1) future return target --------
        future_price = grouped_price.shift(-h)
        ret_col = f"y_ret_{h}"
        out[ret_col] = future_price / out[price_col] - 1.0
        target_columns.append(ret_col)

        # -------- 2) future realized volatility target --------
        # 定义：用未来区间 (t, t+h] 的逐日收益 ret_1 的标准差作为未来风险
        # 对每个股票：
        #   risk_t = std(ret_{t+1}, ..., ret_{t+h})
        if add_risk_target:
            risk_col = f"y_risk_vol_{h}"

            def _future_realized_vol(g: pd.DataFrame) -> pd.Series:
                r = g["ret_1"].to_numpy(dtype=np.float64)
                n = len(r)
                out_arr = np.full(n, np.nan, dtype=np.float64)

                for i in range(n):
                    start = i + 1
                    end = i + h + 1  # python slice end exclusive
                    if end <= n:
                        window = r[start:end]
                        if np.isfinite(window).all():
                            # ddof=0 更稳，避免小窗口不必要的 NaN
                            out_arr[i] = float(np.std(window, ddof=0))
                return pd.Series(out_arr, index=g.index)

            out[risk_col] = (
                out.groupby("ts_code", sort=False, group_keys=False)
                .apply(_future_realized_vol)
                .astype("float32")
            )
            target_columns.append(risk_col)

        # -------- 3) risk-adjusted return target --------
        if add_risk_adjusted_return and "volatility_20" in out.columns:
            ra_col = f"y_ret_{h}_ra"
            out[ra_col] = _safe_div(out[ret_col], out["volatility_20"]).astype("float32")
            target_columns.append(ra_col)

    return out, target_columns