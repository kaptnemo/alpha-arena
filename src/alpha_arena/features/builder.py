"""
builder.py
==========
特征组装模块：将各子模块的构造函数串联成完整 pipeline。

- ``build_features_for_one_symbol``              : 单只股票特征计算（含 Z-score 标准化）
- ``build_panel_features``                       : panel 级批量处理（含横截面排名/Z-score）
- ``build_panel_features_multiprocess``         : panel 级多进程特征处理，主进程聚合

这些函数均不修改原始 DataFrame，所有操作在副本上进行。
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional

import pandas as pd

from alpha_arena.features.config import FeatureConfig
from alpha_arena.features.utils import (
    _check_input,
    _rolling_zscore,
    _cross_sectional_rank,
    _cross_sectional_zscore,
    # _add_time_features,
)
from alpha_arena.features.date_encoder import _add_time_features
from alpha_arena.features.base_features import _add_base_features
from alpha_arena.features.ta_features import _add_ta_library_features, _add_pandas_ta_features


def _build_features_for_one_symbol_task(
    args: tuple[pd.DataFrame, FeatureConfig],
) -> pd.DataFrame:
    """供多进程池调用的顶层 worker，保证可 pickling。"""
    g, cfg = args
    return build_features_for_one_symbol(g, cfg)


def _add_cross_sectional_features(feat: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """在主进程内追加横截面特征，避免跨进程共享大对象。"""
    if not cfg.cross_sectional_rank:
        return feat

    # 候选列：常用因子，从实际列名中过滤（部分指标可能未开启）
    candidate_cols = [
        "ret_1", "ret_5", "ret_10", "ret_20",
        "volatility_5", "volatility_10", "volatility_20",
        "rsi_14", "macd_hist", "cci_20", "adx_14",
        "atr_14", "bb_width", "cmf_20", "mfi_14",
        "sharpe_like_20", "sortino_like_20", "drawdown_20",
    ]
    candidate_cols = [c for c in candidate_cols if c in feat.columns]

    new_columns = {}
    for c in candidate_cols:
        rank_col = f"{c}_cs_rank"
        z_col = f"{c}_cs_z"

        # 百分比排名：1 = 当日最高，0 ≈ 当日最低
        new_columns[rank_col] = _cross_sectional_rank(feat, c)
        # 截面 Z-score：消除当日市场整体水平的影响
        new_columns[z_col] = _cross_sectional_zscore(feat, c)

    return pd.concat([feat, pd.DataFrame(new_columns, index=feat.index)], axis=1)


# ---------------------------------------------------------------------------
# 单只股票
# ---------------------------------------------------------------------------

def build_features_for_one_symbol(g: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """对单只股票按顺序执行全量特征工程。

    调用顺序及必要性说明：

    1. **_add_base_features** ：所有场景必须执行，产生收益率等基础特征，
       后续 ta 指标、Z-score 均依赖其输出。

    2. **_add_ta_library_features** ：调用 `ta` 库计算动量/趋势/波动/量能指标，
       需要完整 OHLCV 数据，执行前不依赖步骤 1 的输出。

    3. **_add_pandas_ta_features** ：扩展指标，可选，失败不影响主流程。

    4. **_add_time_features** ：日历特征，与价格数据无关，顺序无约束。

    5. **Rolling Z-score** ：对当前所有数值列做时序标准化，
       必须在所有原始特征构造完成后执行，避免对 Z-score 再做 Z-score。

    6. **ffill** ：前向填充，若开启则在 Z-score 之后执行，
       确保填充的是标准化后的值而非原始值。

    Parameters
    ----------
    g:
        单只股票的 DataFrame，已按 date 排序，须包含基础 OHLCV 列。
    cfg:
        特征工程配置。

    Returns
    -------
    pd.DataFrame
        追加了全量特征列的副本，行数与输入相同。

        **列命名约定**：
        - 原始特征：无后缀（e.g., ``ret_1``, ``rsi_14``）
        - 时序 Z-score：原列名 + ``_z{window}``（e.g., ``ret_1_z20``）
    """
    g = g.sort_values("date").copy()

    # 1. 基础价格 / 收益率 / 风险调整特征
    g = _add_base_features(g, cfg)

    # 2. ta 库技术指标（动量、趋势、波动率、量能）
    if cfg.add_ta_features:
        g = _add_ta_library_features(g)

    # 3. pandas-ta 扩展指标（Supertrend、KDJ 等）
    if cfg.add_pandas_ta_features:
        g = _add_pandas_ta_features(g)

    # 4. 日历周期特征（sin/cos 编码）
    time_feature_cols = []
    if cfg.add_time_features:
        g, time_feature_cols = _add_time_features(g)

    # 5. 对所有数值特征做滚动 Z-score 标准化
    #    目的：消除不同特征量纲差异，使 LSTM 的梯度更稳定
    #    排除 symbol / date 等非数值列，以及 Z-score 本身（防止二次标准化）
    numeric_cols: List[str] = [
        c for c in g.columns
        if c not in ("symbol", "date")
        and pd.api.types.is_numeric_dtype(g[c])
    ]

    new_columns = {}
    for c in numeric_cols:
        if c in time_feature_cols:
            # 时间特征不做 Z-score，直接保留原值
            continue
        for w in cfg.zscore_windows:
            col_name = f"{c}_z{w}"
            new_columns[col_name] = _rolling_zscore(g[c], w)
    new_df = pd.DataFrame(new_columns, index=g.index)
    g = pd.concat([g, new_df], axis=1)
    # 6. 前向填充：将滚动窗口预热期产生的前几行 NaN 传播至下一个有效值
    if cfg.fill_method == "ffill":
        g[numeric_cols] = g[numeric_cols].ffill()

    return g


# ---------------------------------------------------------------------------
# Panel 级封装
# ---------------------------------------------------------------------------

def build_panel_features(
    df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """对多股票 panel DataFrame 批量执行特征工程。

    Pipeline 流程：

    1. **输入校验** ：检查必要列，统一日期类型，按 (symbol, date) 排序。
    2. **逐股票特征构造** ：调用 ``build_features_for_one_symbol``，
       各股票独立计算，保证时序隔离（无跨股票数据泄漏）。
    3. **拼接** ：按 (date, symbol) 排序，重置索引，便于后续切片。
    4. **横截面标准化（可选）** ：对指定因子列在每个交易日截面内计算
       百分比排名（``_cs_rank``）和 Z-score（``_cs_z``），
       用于构建截面因子或多任务学习的辅助目标。

    Parameters
    ----------
    df:
        原始 panel DataFrame，必须包含
        ``symbol / date / open / high / low / close / volume``。
    cfg:
        特征工程配置，默认使用 ``FeatureConfig()``（全部开启）。

    Returns
    -------
    pd.DataFrame
        包含全量特征的 panel DataFrame，以 (date, symbol) 为逻辑顺序排列。

        **横截面列**（``cfg.cross_sectional_rank=True`` 时追加）：
        - ``{col}_cs_rank`` : 当日该因子在全体股票中的百分比排名
        - ``{col}_cs_z``    : 当日该因子的截面 Z-score
    """
    cfg = cfg or FeatureConfig()
    df = _check_input(df)

    parts = []
    # sort=False：数据已按 ts_code 排序，跳过 groupby 内部重复排序
    for _, g in df.groupby("ts_code", sort=False):
        parts.append(build_features_for_one_symbol(g, cfg))

    feat = pd.concat(parts, axis=0, ignore_index=True)
    feat = feat.sort_values(["date", "ts_code"]).reset_index(drop=True)
    return _add_cross_sectional_features(feat, cfg)


def build_panel_features_multiprocess(
    df: pd.DataFrame,
    cfg: Optional[FeatureConfig] = None,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """多进程版 panel 特征工程。

    处理流程与 ``build_panel_features`` 一致，但会将每只股票的
    ``build_features_for_one_symbol`` 分发给子进程执行，最终在主进程中：

    1. 聚合各子进程结果
    2. 按 ``(date, ts_code)`` 排序
    3. 追加横截面 rank / z-score 特征

    Parameters
    ----------
    df:
        原始 panel DataFrame。
    cfg:
        特征工程配置。
    num_workers:
        进程数。默认为 ``min(os.cpu_count(), 股票数)``；若为 1，则退化为单进程实现。
    """
    cfg = cfg or FeatureConfig()
    df = _check_input(df)

    groups = [g for _, g in df.groupby("ts_code", sort=False)]
    if not groups:
        return _add_cross_sectional_features(df.iloc[0:0].copy(), cfg)

    max_workers = num_workers or max(1, (os.cpu_count() or 1) - 1)
    worker_count = max(1, min(max_workers, len(groups)))
    if worker_count == 1:
        return build_panel_features(df, cfg)

    tasks = [(g, cfg) for g in groups]
    chunksize = max(1, len(tasks) // (worker_count * 4))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        parts = list(executor.map(
            _build_features_for_one_symbol_task,
            tasks,
            chunksize=chunksize
        ))

    feat = pd.concat(parts, axis=0, ignore_index=True)
    feat = feat.sort_values(["date", "ts_code"]).reset_index(drop=True)
    return _add_cross_sectional_features(feat, cfg)


if __name__ == "__main__":
    # 简单测试：加载原始数据，构造特征，检查输出
    from alpha_arena.data.loader import load_from_parquet
    
    # 测试单只股票特征构造
    file_name = "csi300_stocks_2017_2025.parquet"
    df = load_from_parquet(file_name)

    # sample_symbol = df["ts_code"].iloc[0]
    # sample_df = df[df["ts_code"] == sample_symbol].copy()
    cfg = FeatureConfig(
        add_ta_features=True,
        add_pandas_ta_features=True,
        add_time_features=True,
        zscore_windows=[20],
        fill_method="ffill",
        cross_sectional_rank=True,
    )
    # feat = build_features_for_one_symbol(sample_df, cfg)
    # print(f"单只股票特征构造完成，样本行数: {len(feat)}, 特征列数: {len(feat.columns)}")
    # print("示例特征列:", feat.columns.tolist()[:10])

    # # 测试 panel 级特征构造
    # test_df = df[df["date"] >= "20230101"].copy()  # 取部分数据加快测试
    # panel_feat = build_panel_features(test_df, cfg)
    # import pdb; pdb.set_trace()  # --- IGNORE ---
    # print(f"Panel 级特征构造完成，样本行数: {len(panel_feat)}, 特征列数: {len(panel_feat.columns)}")
    # print("示例特征列:", panel_feat.columns.tolist()[:10])

    # 对比测试多进程 panel 级特征构造记录时间，并与单进程结果对比
    import time
    test_df = df[df["date"] >= "20230101"].copy()  # 取部分数据加快测试
    t0 = time.perf_counter()
    panel_feat_mp = build_panel_features_multiprocess(test_df, cfg)
    t1 = time.perf_counter()
    elapsed_mp = t1 - t0
    print(f"多进程 Panel 级特征构造完成，样本行数: {len(panel_feat_mp)}, 特征列数: {len(panel_feat_mp.columns)}")
    print("示例特征列:", panel_feat_mp.columns.tolist()[:10])
    print(f"多进程耗时: {elapsed_mp:.2f} 秒")

    # 单进程与多进程结果对比
    # 注意：由于涉及浮点数计算和多进程调度，结果可能存在微小差异，以下对比允许一定的数值误差。
    t0 = time.perf_counter()
    panel_feat_single = build_panel_features(test_df, cfg)
    t1 = time.perf_counter()
    elapsed_single = t1 - t0
    print(f"单进程 Panel 级特征构造完成，样本行数: {len(panel_feat_single)}, 特征列数: {len(panel_feat_single.columns)}")
    print("示例特征列:", panel_feat_single.columns.tolist()[:10])
    print(f"单进程耗时: {elapsed_single:.2f} 秒")
    # 对比数值列，允许微小误差
    numeric_cols = [c for c in panel_feat_single.columns if pd.api.types.is_numeric_dtype(panel_feat_single[c])]
    for col in numeric_cols:
        if col in panel_feat_mp.columns:
            s1 = panel_feat_single[col]
            s2 = panel_feat_mp[col]
            if not s1.equals(s2):
                diff = (s1 - s2).abs()
                max_diff = diff.max()
                if max_diff > 1e-6:
                    print(f"列 {col} 存在差异，最大绝对差异: {max_diff:.2e}")
                else:
                    print(f"列 {col} 数值基本一致，最大绝对差异: {max_diff:.2e}")
    

    