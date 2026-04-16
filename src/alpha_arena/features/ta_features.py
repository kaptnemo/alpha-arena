"""
ta_features.py
==============
技术指标特征构造，分两层：

1. ``_add_ta_library_features``   — 基于 `ta` 库（稳定 API，生产推荐）
2. ``_add_pandas_ta_features``    — 基于 `pandas-ta` / `pandas-ta-classic`（扩展指标）

两个函数均接收单只股票的 DataFrame，返回追加了指标列的副本。
"""

from __future__ import annotations

import pandas as pd

from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator

from alpha_arena.features.utils import _safe_div

# 延迟导入：pandas-ta 有两个可用分支，优先尝试 classic 版本
try:
    import pandas_ta_classic as pta
except ImportError:
    try:
        import pandas_ta as pta
    except ImportError:
        pta = None


# ---------------------------------------------------------------------------
# ta 库指标
# ---------------------------------------------------------------------------

def _add_ta_library_features(g: pd.DataFrame) -> pd.DataFrame:
    """使用 `ta` 库为单只股票追加标准技术指标。

    `ta` 库对每个指标提供独立的类，接口稳定、无副作用，
    适合生产环境使用。本函数覆盖四大类指标：

    **动量（Momentum）**
    - ``rsi_14``        : RSI(14)，衡量近期涨跌幅度的超买超卖信号，范围 [0, 100]
    - ``roc_10``        : ROC(10)，10 日价格变化率，纯动量指标
    - ``willr_14``      : Williams %R(14)，范围 [-100, 0]，与 RSI 互补
    - ``stoch_k_14``    : Stochastic %K(14)，当前收盘价在近期区间的相对位置
    - ``stoch_d_14``    : Stochastic %D(14)，%K 的 3 日平滑信号线

    **MACD（趋势动量复合）**
    - ``macd``          : DIF 线（快线 EMA12 - 慢线 EMA26）
    - ``macd_signal``   : DEA 线（DIF 的 9 日 EMA），即信号线
    - ``macd_hist``     : MACD 柱（DIF - DEA），正值表示多头动能增强

    **趋势（Trend）**
    - ``cci_20``        : CCI(20)，商品通道指数，±100 为超买超卖阈值
    - ``adx_14``        : ADX(14)，趋势强度指数，仅衡量强弱不判断方向（>25 为强趋势）

    **波动率（Volatility）**
    - ``atr_14``        : ATR(14)，平均真实波动幅度，用于止损和仓位管理
    - ``bb_h / bb_l / bb_m`` : 布林带上轨、下轨、中轨（均线）
    - ``bb_width``      : 布林带宽度 = (上轨 - 下轨) / 中轨，反映波动率扩张/收窄
    - ``bb_pos``        : 收盘价在布林带内的相对位置，0 = 下轨，1 = 上轨

    **量能（Volume）**
    - ``obv``           : OBV，累计量价关系，趋势确认指标
    - ``cmf_20``        : CMF(20)，Chaikin 资金流指数，正值表示资金净流入
    - ``mfi_14``        : MFI(14)，资金流量指数，类似 RSI 但纳入成交量权重

    Parameters
    ----------
    g:
        单只股票的 DataFrame，必须包含 ``high / low / close / volume`` 列。

    Returns
    -------
    pd.DataFrame
        追加了上述指标列的副本。
    """
    g = g.copy()

    high = g["high"]
    low = g["low"]
    close = g["close"]
    volume = g["volume"]

    # --- 动量 ---
    g["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    g["roc_10"] = ROCIndicator(close=close, window=10).roc()
    g["willr_14"] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    g["stoch_k_14"] = stoch.stoch()
    g["stoch_d_14"] = stoch.stoch_signal()

    # --- MACD ---
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    g["macd"] = macd.macd()
    g["macd_signal"] = macd.macd_signal()
    g["macd_hist"] = macd.macd_diff()

    # --- 趋势 ---
    g["cci_20"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()
    g["adx_14"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # --- 波动率 ---
    g["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    g["bb_h"] = bb.bollinger_hband()
    g["bb_l"] = bb.bollinger_lband()
    g["bb_m"] = bb.bollinger_mavg()
    # 布林带宽度：值越大，近期波动越剧烈
    g["bb_width"] = _safe_div(
        bb.bollinger_hband() - bb.bollinger_lband(),
        bb.bollinger_mavg()
    )
    # 收盘价在布林带内的位置：>0.8 视为偏上轨，<0.2 视为偏下轨
    g["bb_pos"] = _safe_div(
        close - bb.bollinger_lband(),
        bb.bollinger_hband() - bb.bollinger_lband()
    )

    # --- 量能 ---
    g["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    g["cmf_20"] = ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, window=20
    ).chaikin_money_flow()
    g["mfi_14"] = MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14
    ).money_flow_index()

    return g


# ---------------------------------------------------------------------------
# pandas-ta 扩展指标
# ---------------------------------------------------------------------------

def _add_pandas_ta_features(g: pd.DataFrame) -> pd.DataFrame:
    """使用 `pandas-ta` 追加补充技术指标。

    依赖 `pandas-ta-classic` 或 `pandas-ta`，任一不可用时直接返回原 DataFrame。
    各指标计算均包裹在 try/except 中，保证单个指标失败不影响整体 pipeline。

    **Supertrend（超级趋势线）**
    - ``supertrend``    : 动态支撑/压力线，基于 ATR 计算（参数：length=10, mult=3.0）
    - ``supertrend_dir``: 趋势方向，+1 表示上升趋势，-1 表示下降趋势
      （不同版本列名差异较大，内部做了自动 rename）

    **KDJ（随机指标变体）**
    - ``k / d / j``     : KDJ 三线，J 线对超买超卖更灵敏（参数：length=9, signal=3）

    **Efficiency Ratio（效率比率）**
    - ``er_10``         : Kaufman 效率比率（length=10），范围 [0, 1]；
      接近 1 表示价格在高效单向运动，接近 0 表示震荡

    **Normalized ATR（归一化 ATR）**
    - ``natr_14``       : NATR(14) = ATR(14) / close * 100，
      对不同价格区间的股票具有可比性

    Parameters
    ----------
    g:
        单只股票的 DataFrame，必须包含 ``high / low / close`` 列。

    Returns
    -------
    pd.DataFrame
        追加了可用指标列的副本；若 pandas-ta 未安装则原样返回。
    """
    if pta is None:
        return g

    g = g.copy()

    # --- Supertrend ---
    try:
        st = pta.supertrend(
            high=g["high"], low=g["low"], close=g["close"],
            length=10, multiplier=3.0
        )
        if isinstance(st, pd.DataFrame):
            # 不同版本的列名形如：SUPERT_10_3.0 / SUPERTd_10_3.0 等
            # 统一 rename 为 supertrend / supertrend_dir
            rename_map = {}
            for c in st.columns:
                lc = c.lower()
                if "sup" in lc and "d" not in lc:
                    rename_map[c] = "supertrend"
                elif "direction" in lc or lc.endswith("_d") or "supd" in lc:
                    rename_map[c] = "supertrend_dir"
            st = st.rename(columns=rename_map)
            g = pd.concat([g, st], axis=1)
    except Exception:
        pass

    # --- KDJ ---
    try:
        kdj = pta.kdj(
            high=g["high"], low=g["low"], close=g["close"],
            length=9, signal=3
        )
        if isinstance(kdj, pd.DataFrame):
            # 统一转小写列名，避免版本差异
            kdj = kdj.rename(columns={c: c.lower() for c in kdj.columns})
            for c in kdj.columns:
                g[c] = kdj[c]
    except Exception:
        pass

    # --- Efficiency Ratio ---
    try:
        er = pta.er(g["close"], length=10)
        if isinstance(er, pd.Series):
            g["er_10"] = er
    except Exception:
        pass

    # --- Normalized ATR ---
    try:
        natr = pta.natr(high=g["high"], low=g["low"], close=g["close"], length=14)
        if isinstance(natr, pd.Series):
            g["natr_14"] = natr
    except Exception:
        pass

    return g
