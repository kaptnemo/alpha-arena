"""
base_features.py
================
基础价格、收益率及风险调整特征构造。

本模块仅依赖 OHLCV 原始数据，不引入任何技术指标库，
适用于快速验证和轻量级场景。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from alpha_arena.features.config import FeatureConfig, FeatureSpec
from alpha_arena.features.utils import _safe_div


def _add_base_features(g: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, List[FeatureSpec]]:
    """为单只股票构造基础价格 / 收益率 / 风险调整特征。

    所有特征均严格使用历史数据（无未来信息泄漏）：
    - 收益率基于当前及过去的收盘价计算；
    - 滚动统计量的 min_periods 由 window 本身隐式控制
      （rolling 默认 min_periods=1，但窗口未满时结果会有偏，
      可在 FeatureConfig 中扩展严格模式）。

    Parameters
    ----------
    g:
        单只股票的 DataFrame，已按日期升序排列，
        必须包含 ``open / high / low / close / volume`` 列。
    cfg:
        特征工程配置，控制窗口大小、开关等。

    Returns
    -------
    pd.DataFrame
        追加了以下特征列的副本：

        **收益率**
        - ``ret_1``       : 单日对数收益的线性近似（pct_change），可按 cfg.clip_return 截断
        - ``log_ret_1``   : 精确单日对数收益，用于需要可加性的场景
        - ``ret_{w}``     : 多周期简单收益率（w ∈ cfg.price_windows）

        **日内 / 跳空结构**
        - ``oc_ratio``    : (close - open) / open，衡量日内方向强度
        - ``hl_ratio``    : (high - low) / close，衡量日内波动幅度
        - ``co_gap``      : (open - prev_close) / prev_close，隔夜跳空幅度

        **均线相对位置**
        - ``ma_ratio_{w}``: close / rolling_mean(w) - 1，偏离均线程度

        **区间位置**
        - ``price_pos_{w}``: (close - rolling_min) / (rolling_max - rolling_min)，
          Williams %R 的对称版，反映价格在近期区间内的相对高低

        **波动率**
        - ``volatility_{w}`` : ret_1 的滚动标准差，衡量近期价格波动强度

        **成交量比率**
        - ``volume_ma_ratio_{w}`` : volume / rolling_mean(volume) - 1，
          量能放大/萎缩程度

        **风险调整指标**（仅当 cfg.add_risk_adjusted_features=True）
        - ``sharpe_like_{w}``  : 滚动均收益 / 滚动标准差，类 Sharpe 比率
        - ``sortino_like_{w}`` : 滚动均收益 / 下行标准差，类 Sortino 比率
        - ``drawdown_{w}``     : close / rolling_max - 1，近期最大回撤幅度
    """
    g = g.copy()

    feature_specs = [
        FeatureSpec(name="ret_1", kind="numeric", dtype="float32"),
        FeatureSpec(name="log_ret_1", kind="numeric", dtype="float32"),
        FeatureSpec(name="oc_ratio", kind="numeric", dtype="float32"),
        FeatureSpec(name="hl_ratio", kind="numeric", dtype="float32"),
        FeatureSpec(name="co_gap", kind="numeric", dtype="float32"),
    ]
    # ------------------------------------------------------------------
    # 单日收益率
    # ------------------------------------------------------------------
    g["ret_1"] = g["close"].pct_change()
    if cfg.clip_return is not None:
        # 截断极端值，防止单日涨跌停或复权数据异常污染后续滚动统计
        g["ret_1"] = g["ret_1"].clip(-cfg.clip_return, cfg.clip_return)

    # 对数收益率：ln(P_t / P_{t-1})，具有时间可加性，用于累计收益计算
    g["log_ret_1"] = np.log(g["close"] / g["close"].shift(1))

    # ------------------------------------------------------------------
    # 日内结构特征
    # ------------------------------------------------------------------
    # 日内涨跌幅：反映开盘后市场的定向力量
    g["oc_ratio"] = _safe_div(g["close"] - g["open"], g["open"])
    # 日内振幅：反映当日价格波动区间的相对大小
    g["hl_ratio"] = _safe_div(g["high"] - g["low"], g["close"])
    # 隔夜跳空：反映非交易时段信息的冲击强度
    g["co_gap"] = _safe_div(g["open"] - g["pre_close"], g["pre_close"])

    # ------------------------------------------------------------------
    # 多周期价格特征
    # ------------------------------------------------------------------
    for w in cfg.price_windows:
        # 多周期简单收益率
        g[f"ret_{w}"] = g["close"].pct_change(w)
        # 偏离滚动均线程度（动量 / 均值回归信号）
        g[f"ma_ratio_{w}"] = _safe_div(g["close"], g["close"].rolling(w).mean()) - 1.0
        # 价格在近期 [low, high] 区间内的相对位置，类似 Williams %R（反向）
        roll_low = g["low"].rolling(w).min()
        roll_high = g["high"].rolling(w).max()
        g[f"price_pos_{w}"] = _safe_div(g["close"] - roll_low, roll_high - roll_low)
        feature_specs.append(FeatureSpec(name=f"ret_{w}", kind="numeric", dtype="float32"))
        feature_specs.append(FeatureSpec(name=f"ma_ratio_{w}", kind="numeric", dtype="float32"))
        feature_specs.append(FeatureSpec(name=f"price_pos_{w}", kind="numeric", dtype="float32"))

    # ------------------------------------------------------------------
    # 多周期波动率 / 量能特征
    # ------------------------------------------------------------------
    for w in cfg.vol_windows:
        # 滚动收益率标准差：衡量近期价格不确定性
        g[f"volatility_{w}"] = g["ret_1"].rolling(w).std()
        # 相对于近期均量的偏离度：量能放大为正，萎缩为负
        g[f"volume_ma_ratio_{w}"] = (
            _safe_div(g["volume"], g["volume"].rolling(w).mean()) - 1.0
        )
        feature_specs.append(FeatureSpec(name=f"volatility_{w}", kind="numeric", dtype="float32"))
        feature_specs.append(FeatureSpec(name=f"volume_ma_ratio_{w}", kind="numeric", dtype="float32"))

    # ------------------------------------------------------------------
    # 风险调整指标
    # ------------------------------------------------------------------
    if cfg.add_risk_adjusted_features:
        for w in cfg.vol_windows:
            mean_ret = g["ret_1"].rolling(w).mean()
            std_ret = g["ret_1"].rolling(w).std()
            # 下行标准差：仅统计负收益的波动，Sortino 比率的分母
            downside_std = g["ret_1"].where(g["ret_1"] < 0, 0.0).rolling(w).std()

            # 类 Sharpe：均收益 / 总波动，越高表示单位风险收益越强
            g[f"sharpe_like_{w}"] = _safe_div(mean_ret, std_ret)
            # 类 Sortino：均收益 / 下行波动，对上涨波动更宽容
            g[f"sortino_like_{w}"] = _safe_div(mean_ret, downside_std)

            # 相对于近期高点的回撤幅度，反映趋势持续 / 反转的强度
            roll_max = g["close"].rolling(w).max()
            g[f"drawdown_{w}"] = _safe_div(g["close"], roll_max) - 1.0
            feature_specs.append(FeatureSpec(name=f"sharpe_like_{w}", kind="numeric", dtype="float32"))
            feature_specs.append(FeatureSpec(name=f"sortino_like_{w}", kind="numeric", dtype="float32"))
            feature_specs.append(FeatureSpec(name=f"drawdown_{w}", kind="numeric", dtype="float32"))

    return g, feature_specs
