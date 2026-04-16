import pandas as pd

from stockstats import wrap
from alpha_arena.data.loader import load_from_parquet


DEFAULT_INDICATORS = ['macd', 'macdh', 'macds', 'rsi']

import pandas_ta_classic as ta  # 推荐导入别名

# 定义需要计算的指标及其参数
# pandas-ta 的指标函数名通常是小写，如 'macd', 'rsi'
INDICATOR_CONFIG = [
    {"kind": "macd", "fast": 12, "slow": 26, "signal": 9},
    {"kind": "rsi", "length": 14},
]

def calculate_indicators_pdta(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 pandas-ta 为多只股票批量计算技术指标。

    优化点：
    - ta.Strategy 在部分环境触发内部锁卡死，改为遍历 INDICATOR_CONFIG 逐一调用
    - set_index 返回新 DataFrame，无需额外 copy()
    - set_index 返回新 DataFrame，无需额外 copy()
    - groupby(sort=False) 跳过对已排序数据的重复排序

    参数:
        df: 原始DataFrame，必须包含 'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol' 列。

    返回:
        pd.DataFrame: 以 (ts_code, date) 为多级索引，包含原始列及所有新增技术指标。
    """
    required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ts_code', 'date']).reset_index(drop=True)

    results = []
    for _, group in df.groupby('ts_code', sort=False):
        # set_index 返回新 DataFrame，不影响原 df，无需 copy()
        stock_df = group.set_index('date')
        for cfg in INDICATOR_CONFIG:
            method = getattr(stock_df.ta, cfg['kind'])
            params = {k: v for k, v in cfg.items() if k != 'kind'}
            method(**params, append=True)
        results.append(stock_df.reset_index())

    return pd.concat(results, ignore_index=True).set_index(['ts_code', 'date'])


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 stockstats 为多只股票计算技术指标。

    优化点：
    - stockstats 会将 'date' 列设为 DatetimeIndex，sdf.get() 结果与 df 的整数索引不一致
      修正为 reset_index() 还原 'date' 列，再以 ['ts_code', 'date'] 做精确 merge
    - groupby(sort=False) 跳过对已排序数据的重复排序
    - 去除多余的中间 reset_index / set_index 转换

    参数
    ----------
    df : pd.DataFrame
        必须包含列：'ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol'

    返回
    -------
    pd.DataFrame
        以 (ts_code, date) 为多级索引，包含原始列及 DEFAULT_INDICATORS 中的指标列。
    """
    required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.sort_values(['ts_code', 'date']).reset_index(drop=True)

    indicator_parts = []
    for ts_code, group in df.groupby('ts_code', sort=False):
        sdf = wrap(group.copy())
        ind_df = sdf.get(DEFAULT_INDICATORS).reset_index()
        ind_df['ts_code'] = ts_code
        indicator_parts.append(ind_df)

    all_indicators = pd.concat(indicator_parts, ignore_index=True)
    result_df = df.merge(all_indicators, on=['ts_code', 'date'], how='left')
    return result_df.set_index(['ts_code', 'date'])


if __name__ == "__main__":
    import time
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    file_path = "csi300_stocks_2021_2022.parquet"
    df = load_from_parquet(file_path)

    logger.info(f"数据规模: {df.shape[0]} 行, {df['ts_code'].nunique()} 只股票")

    # --- 测试 calculate_indicators_pdta ---
    t0 = time.perf_counter()
    result_pdta = calculate_indicators_pdta(df.copy())
    t1 = time.perf_counter()
    elapsed_pdta = t1 - t0
    logger.info(f"[pandas-ta] 耗时: {elapsed_pdta:.3f}s, 输出形状: {result_pdta.shape}")
    logger.info(f"输出示例:\n{result_pdta.tail()}")

    # --- 测试 calculate_indicators ---
    t0 = time.perf_counter()
    result_v2 = calculate_indicators(df.copy())
    t1 = time.perf_counter()
    elapsed_v2 = t1 - t0
    logger.info(f"[stockstats] 耗时: {elapsed_v2:.3f}s, 输出形状: {result_v2.shape}")
    logger.info(f"输出示例:\n{result_v2.tail()}")

    # --- 汇总对比 ---
    faster = "pandas-ta" if elapsed_pdta < elapsed_v2 else "stockstats"
    ratio = max(elapsed_pdta, elapsed_v2) / min(elapsed_pdta, elapsed_v2)
    logger.info(f"结论: {faster} 更快，速度比约 {ratio:.2f}x")

    print("\n=== 效率对比汇总 ===")
    print(f"  calculate_indicators_pdta : {elapsed_pdta:.3f}s")
    print(f"  calculate_indicators   : {elapsed_v2:.3f}s")
    print(f"  更快的方法: {faster} ({ratio:.2f}x)")