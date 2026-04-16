"""
selector.py
===========
LSTM 输入特征列筛选模块。

提供基于命名约定的白名单过滤器，从完整 feature DataFrame
中选出适合输入 LSTM 的数值特征子集，排除原始 OHLCV 价格列
（避免尺度不一致）和横截面排名列（默认排除，可按需保留）。
"""

from __future__ import annotations

from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# 特征前缀白名单
# 新增特征时，只需在此处追加对应前缀，selector 会自动纳入
# ---------------------------------------------------------------------------
_FEATURE_PREFIXES = (
    # 基础价格 / 收益率
    "ret_", "log_ret_", "oc_ratio", "hl_ratio", "co_gap",
    "ma_ratio_", "price_pos_",
    # 波动率 / 量能
    "volatility_", "volume_ma_ratio_",
    # 风险调整
    "sharpe_like_", "sortino_like_", "drawdown_",
    # ta 库指标
    "rsi_", "roc_", "willr_", "stoch_",
    "macd", "cci_", "adx_", "atr_", "bb_",
    "obv", "cmf_", "mfi_",
    # pandas-ta 指标
    "supertrend", "er_", "natr_",
    # 时间编码
    "dow_", "month_", "doy_",
)

# 即使前缀匹配，也排除含以下关键词的列（横截面特征不直接输入 LSTM）
# _EXCLUDE_KEYWORDS = ("_cs_rank", "_cs_z")
_EXCLUDE_KEYWORDS = []

# 始终排除的原始列
_RAW_COLS = frozenset({"ts_code", "date", "open", "high", "low", "close", "volume", "pre_close", "change", "pct_chg", "amount", "in_csi300"})


def select_lstm_feature_columns(
    df: pd.DataFrame,
    target_columns: tuple[str, ...],
    feature_prefixes: tuple[str, ...] | None = None,
    exclude_keywords: tuple[str, ...] | None = None,

) -> List[str]:
    """从 feature DataFrame 中筛选适合 LSTM 输入的特征列名。

    筛选规则（按优先级）：
    1. 排除 ``ts_code / date / open / high / low / close / volume``（原始价格列
       量纲差异大，直接输入会干扰 LSTM 学习，建议归一化后另行处理）。
    2. 排除包含 ``_cs_rank`` 或 ``_cs_z`` 的横截面列
       （横截面特征依赖同期其他股票，推理时难以保证实时可用；
       如需做横截面建模可注释掉此规则）。
    3. 保留前缀匹配 ``_FEATURE_PREFIXES`` 的列。

    返回列表已排序，保证多次调用结果确定性一致，
    方便与 numpy/tensor 索引对齐。

    Parameters
    ----------
    df:
        经过 ``build_panel_features`` 和/或 ``add_targets`` 处理后的 DataFrame。

    Returns
    -------
    List[str]
        排序后的特征列名列表，可直接用于 ``df[cols].values`` 构造 LSTM 输入张量。

    Examples
    --------
    >>> feat_cols = select_lstm_feature_columns(panel_df)
    >>> X = panel_df[feat_cols].values  # shape: (N, len(feat_cols))
    """
    exclude_keywords = exclude_keywords or []
    feature_prefixes = feature_prefixes or []
    cols = []
    for c in df.columns:
        if c in _RAW_COLS:
            continue
        if any(k in c for k in exclude_keywords):
            continue
        if c in target_columns:
            continue
        if (feature_prefixes and any(c.startswith(p) for p in feature_prefixes)) or not feature_prefixes:
            cols.append(c)

    return sorted(cols)
