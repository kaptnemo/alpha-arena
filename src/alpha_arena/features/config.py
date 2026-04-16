"""
config.py
=========
全局配置与常量。

FeatureConfig 通过 dataclass 集中管理特征工程的所有开关和参数，
使调用方只需传递一个对象即可控制整个 pipeline 的行为，避免散乱的关键字参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal
from dataclasses import dataclass

FeatureKind = Literal["numeric", "boolean", "cyclic", "mask", "target", "id", "cross_sectional", "other"]

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: FeatureKind
    dtype: str

# ---------------------------------------------------------------------------
# 输入列校验白名单：pipeline 入口处强制要求这些列存在
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["ts_code", "date", "open", "high", "low", "close", "volume"]


@dataclass
class FeatureConfig:
    """控制特征工程 pipeline 的全局配置。

    Attributes
    ----------
    price_windows:
        价格类滚动窗口列表（单位：交易日）。
        用于计算多周期收益率、均线比率、价格位置等。
    vol_windows:
        波动率 / 成交量类滚动窗口列表。
        用于计算滚动标准差、成交量比率、Sharpe-like 等指标。
    zscore_windows:
        时间序列 Z-score 标准化的滚动窗口列表。
        对每个数值特征分别计算滚动均值和标准差后标准化，
        使模型输入在时间维度上量纲一致。
    cross_sectional_rank:
        是否在 panel 级别追加横截面 rank 和 z-score 特征。
        开启后对常用因子列（ret_1、rsi_14 等）计算同日排名，
        有助于消除市场整体涨跌对信号的干扰。
    add_time_features:
        是否追加日历周期特征（星期几、月份、年内第几天的 sin/cos 编码）。
    add_return_features:
        保留字段，预留给未来扩展的更多收益率特征开关。
    add_risk_adjusted_features:
        是否在基础特征中计算 Sharpe-like、Sortino-like、
        最大回撤等风险调整指标。
    add_ta_features:
        是否调用 `ta` 库计算动量、趋势、波动率、成交量等技术指标。
    add_pandas_ta_features:
        是否调用 `pandas-ta` / `pandas-ta-classic` 计算 Supertrend、
        KDJ、Efficiency Ratio、归一化 ATR 等补充指标。
    fill_method:
        NaN 填充策略。
        - ``"none"`` ：不填充，让模型自行处理（推荐配合 mask/weight）。
        - ``"ffill"``：前向填充，适用于低频数据或需要连续输入的场景。
    clip_return:
        对单日收益率 ``ret_1`` 的截断阈值（绝对值）。
        ``None`` 表示不截断；设为 0.2 则将收益率限制在 [-20%, +20%]，
        防止极端行情对归一化统计量造成污染。
    """

    price_windows: Sequence[int] = (5, 10, 20)
    vol_windows: Sequence[int] = (5, 10, 20)
    zscore_windows: Sequence[int] = (10, 20,)
    cross_sectional_rank: bool = True
    add_time_features: bool = True
    add_return_features: bool = True
    add_risk_adjusted_features: bool = True
    add_ta_features: bool = True
    add_pandas_ta_features: bool = True
    fill_method: str = "none"       # "none" | "ffill"
    clip_return: Optional[float] = 0.2
