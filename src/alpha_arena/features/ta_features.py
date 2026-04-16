"""
ta_features.py
==============
技术指标特征构造，分两层：

1. ``_add_ta_library_features``   — 基于 `ta` 库（稳定 API，生产推荐）
2. ``_add_pandas_ta_features``    — 基于 `pandas-ta` / `pandas-ta-classic`（扩展指标）

两个函数均接收单只股票的 DataFrame，返回追加了指标列的副本。
"""

from __future__ import annotations

import re
import warnings
from typing import Callable

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator

from alpha_arena.features.utils import _safe_div
from alpha_arena.features.config import FeatureSpec

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

def _add_ta_library_features(g: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
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
    feature_specs: list[FeatureSpec] = [
        FeatureSpec(name="rsi_14", kind="numeric", dtype="float32"),
        FeatureSpec(name="roc_10", kind="numeric", dtype="float32"),
        FeatureSpec(name="willr_14", kind="numeric", dtype="float32"),
        FeatureSpec(name="stoch_k_14", kind="numeric", dtype="float32"),
        FeatureSpec(name="stoch_d_14", kind="numeric", dtype="float32"),
        FeatureSpec(name="macd", kind="numeric", dtype="float32"),
        FeatureSpec(name="macd_signal", kind="numeric", dtype="float32"),
        FeatureSpec(name="macd_hist", kind="numeric", dtype="float32"),
        FeatureSpec(name="cci_20", kind="numeric", dtype="float32"),
        FeatureSpec(name="adx_14", kind="numeric", dtype="float32"),
        FeatureSpec(name="atr_14", kind="numeric", dtype="float32"),
        FeatureSpec(name="bb_h", kind="numeric", dtype="float32"),
        FeatureSpec(name="bb_l", kind="numeric", dtype="float32"),
        FeatureSpec(name="bb_m", kind="numeric", dtype="float32"),
        FeatureSpec(name="bb_width", kind="numeric", dtype="float32"),
        FeatureSpec(name="bb_pos", kind="numeric", dtype="float32"),
        FeatureSpec(name="obv", kind="numeric", dtype="float32"),
        FeatureSpec(name="cmf_20", kind="numeric", dtype="float32"),
        FeatureSpec(name="mfi_14", kind="numeric", dtype="float32"),
    ]

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

    return g, feature_specs


# ---------------------------------------------------------------------------
# pandas-ta 扩展指标
# ---------------------------------------------------------------------------

def _add_pandas_ta_features(g: pd.DataFrame) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """为单只股票追加 pandas-ta 技术指标。

    特性
    ----
    - 对 pandas-ta / pandas-ta-classic 做尽量兼容
    - 单指标失败不影响整体 pipeline
    - 严格避免重复列名
    - 重复运行时按“覆盖同名列”处理，而不是产生重复列
    - 输出列名稳定、可解释、适合生产环境落盘

    依赖列
    ------
    必须包含: high, low, close
    建议包含: date（若存在则会先排序）

    Returns
    -------
    pd.DataFrame
        追加技术指标后的副本
    """
    if pta is None:
        return g, []

    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(g.columns):
        return g, []

    out = g.copy()

    feature_specs: list[FeatureSpec] = []
    # 保证时间顺序稳定，避免技术指标在乱序数据上计算错误
    if "date" in out.columns:
        out = out.sort_values("date", kind="stable").reset_index(drop=True)

    # 统一转数值，避免 object dtype 干扰指标计算
    for col in ("high", "low", "close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    high = out["high"]
    low = out["low"]
    close = out["close"]

    def _safe_run(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:
            warnings.warn(f"[pandas-ta] feature '{name}' failed: {e}", RuntimeWarning)

    def _safe_assign(df: pd.DataFrame, col_name: str, values: pd.Series | np.ndarray) -> None:
        """安全写列：同名覆盖，不制造重复列。"""
        if isinstance(values, np.ndarray):
            values = pd.Series(values, index=df.index)
        else:
            values = values.reindex(df.index)

        df[col_name] = pd.to_numeric(values, errors="coerce")

    def _normalize_multiplier_str(x: float | int) -> str:
        """3.0 -> 3, 2.5 -> 2p5"""
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return str(xf).replace(".", "p")

    def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
        """最终防御：若外部已有重复列名，自动去重重命名。"""
        cols = list(df.columns)
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_cols.append(f"{c}__dup{seen[c]}")
        df.columns = new_cols
        return df

    # ------------------------------------------------------------------
    # Supertrend
    # ------------------------------------------------------------------
    def _add_supertrend() -> None:
        length = 10
        multiplier = 3.0
        mult_str = _normalize_multiplier_str(multiplier)
        prefix = f"supertrend_{length}_{mult_str}"

        st = pta.supertrend(
            high=high,
            low=low,
            close=close,
            length=length,
            multiplier=multiplier,
        )

        if st is None or not isinstance(st, pd.DataFrame) or st.empty:
            return

        # pandas-ta 常见输出示例：
        # SUPERT_10_3.0   -> line
        # SUPERTd_10_3.0  -> direction
        # SUPERTl_10_3.0  -> long band
        # SUPERTs_10_3.0  -> short band
        #
        # 兼容不同大小写 / 小变体
        assigned = set()

        for c in st.columns:
            lc = str(c).lower()

            if re.search(r"supertd|supertrend.*dir|direction", lc):
                _safe_assign(out, f"{prefix}_dir", st[c])
                assigned.add(f"{prefix}_dir")
                feature_specs.append(FeatureSpec(name=f"{prefix}_dir", kind="numeric", dtype="float32"))
            elif re.search(r"supertl|supertrend.*long", lc):
                _safe_assign(out, f"{prefix}_long", st[c])
                assigned.add(f"{prefix}_long")
                feature_specs.append(FeatureSpec(name=f"{prefix}_long", kind="numeric", dtype="float32"))
            elif re.search(r"superts|supertrend.*short", lc):
                _safe_assign(out, f"{prefix}_short", st[c])
                assigned.add(f"{prefix}_short")
                feature_specs.append(FeatureSpec(name=f"{prefix}_short", kind="numeric", dtype="float32"))
            elif re.search(r"supert(?![dls])|supertrend(?!.*(dir|long|short))", lc):
                _safe_assign(out, f"{prefix}_line", st[c])
                assigned.add(f"{prefix}_line")
                feature_specs.append(FeatureSpec(name=f"{prefix}_line", kind="numeric", dtype="float32"))

        # 万一版本返回列名不标准，按列位置兜底
        if not assigned:
            cols = list(st.columns)
            if len(cols) >= 1:
                _safe_assign(out, f"{prefix}_line", st[cols[0]])
                feature_specs.append(FeatureSpec(name=f"{prefix}_line", kind="numeric", dtype="float32"))
            if len(cols) >= 2:
                _safe_assign(out, f"{prefix}_dir", st[cols[1]])
                feature_specs.append(FeatureSpec(name=f"{prefix}_dir", kind="numeric", dtype="float32"))
            if len(cols) >= 3:
                _safe_assign(out, f"{prefix}_long", st[cols[2]])
                feature_specs.append(FeatureSpec(name=f"{prefix}_long", kind="numeric", dtype="float32"))
            if len(cols) >= 4:
                _safe_assign(out, f"{prefix}_short", st[cols[3]])
                feature_specs.append(FeatureSpec(name=f"{prefix}_short", kind="numeric", dtype="float32"))

    _safe_run("supertrend", _add_supertrend)

    # ------------------------------------------------------------------
    # KDJ
    # ------------------------------------------------------------------
    def _add_kdj() -> None:
        length = 9
        signal = 3
        prefix = f"kdj_{length}_{signal}"

        kdj = pta.kdj(
            high=high,
            low=low,
            close=close,
            length=length,
            signal=signal,
        )

        if kdj is None or not isinstance(kdj, pd.DataFrame) or kdj.empty:
            return

        # 常见列名：
        # K_9_3 / D_9_3 / J_9_3
        # 或小写变体
        assigned = set()
        for c in kdj.columns:
            lc = str(c).lower()

            if re.fullmatch(r"k(_.*)?", lc) or lc.startswith("k_"):
                _safe_assign(out, f"{prefix}_k", kdj[c])
                assigned.add(f"{prefix}_k")
                feature_specs.append(FeatureSpec(name=f"{prefix}_k", kind="numeric", dtype="float32"))
            elif re.fullmatch(r"d(_.*)?", lc) or lc.startswith("d_"):
                _safe_assign(out, f"{prefix}_d", kdj[c])
                assigned.add(f"{prefix}_d")
                feature_specs.append(FeatureSpec(name=f"{prefix}_d", kind="numeric", dtype="float32"))
            elif re.fullmatch(r"j(_.*)?", lc) or lc.startswith("j_"):
                _safe_assign(out, f"{prefix}_j", kdj[c])
                assigned.add(f"{prefix}_j")
                feature_specs.append(FeatureSpec(name=f"{prefix}_j", kind="numeric", dtype="float32"))
        # 兜底：按前三列位置映射
        if not assigned:
            cols = list(kdj.columns)
            if len(cols) >= 1:
                _safe_assign(out, f"{prefix}_k", kdj[cols[0]])
                feature_specs.append(FeatureSpec(name=f"{prefix}_k", kind="numeric", dtype="float32"))
            if len(cols) >= 2:
                _safe_assign(out, f"{prefix}_d", kdj[cols[1]])
                feature_specs.append(FeatureSpec(name=f"{prefix}_d", kind="numeric", dtype="float32"))
            if len(cols) >= 3:
                _safe_assign(out, f"{prefix}_j", kdj[cols[2]])
                feature_specs.append(FeatureSpec(name=f"{prefix}_j", kind="numeric", dtype="float32"))

    _safe_run("kdj", _add_kdj)

    # ------------------------------------------------------------------
    # Efficiency Ratio
    # ------------------------------------------------------------------
    def _add_er() -> None:
        length = 10
        er = pta.er(close, length=length)
        if er is not None and isinstance(er, pd.Series):
            _safe_assign(out, f"er_{length}", er)
            feature_specs.append(FeatureSpec(name=f"er_{length}", kind="numeric", dtype="float32"))
    _safe_run("er", _add_er)

    # ------------------------------------------------------------------
    # NATR
    # ------------------------------------------------------------------
    def _add_natr() -> None:
        length = 14
        natr = pta.natr(high=high, low=low, close=close, length=length)
        if natr is not None and isinstance(natr, pd.Series):
            _safe_assign(out, f"natr_{length}", natr)
            feature_specs.append(FeatureSpec(name=f"natr_{length}", kind="numeric", dtype="float32"))
    _safe_run("natr", _add_natr)

    # 最后一层保险，防止外部已有脏数据导致重复列
    out = _dedup_columns(out)

    return out, feature_specs