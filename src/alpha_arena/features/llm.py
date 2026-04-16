from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# ta
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator

# pandas-ta classic / pandas-ta style
try:
    import pandas_ta_classic as pta
except ImportError:
    try:
        import pandas_ta as pta
    except ImportError:
        pta = None


# =========================
# 1) 配置
# =========================

@dataclass
class FeatureConfig:
    price_windows: Sequence[int] = (5, 10, 20, 60)
    vol_windows: Sequence[int] = (5, 10, 20, 60)
    zscore_windows: Sequence[int] = (20, 60)
    cross_sectional_rank: bool = True
    add_time_features: bool = True
    add_return_features: bool = True
    add_risk_adjusted_features: bool = True
    add_ta_features: bool = True
    add_pandas_ta_features: bool = True
    fill_method: str = "none"   # "none" | "ffill"
    clip_return: Optional[float] = 0.2


# =========================
# 2) 工具函数
# =========================

REQUIRED_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def _check_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return a / (b.abs() + eps)


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mean_ = s.rolling(window, min_periods=window).mean()
    std_ = s.rolling(window, min_periods=window).std()
    return (s - mean_) / (std_ + 1e-12)


def _cross_sectional_zscore(df: pd.DataFrame, col: str, date_col: str = "date") -> pd.Series:
    grp = df.groupby(date_col)[col]
    mean_ = grp.transform("mean")
    std_ = grp.transform("std")
    return (df[col] - mean_) / (std_ + 1e-12)


def _cross_sectional_rank(df: pd.DataFrame, col: str, date_col: str = "date", pct: bool = True) -> pd.Series:
    return df.groupby(date_col)[col].rank(pct=pct)


def _add_time_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    dt = g["date"]

    dow = dt.dt.dayofweek
    month = dt.dt.month
    day_of_year = dt.dt.dayofyear

    g["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    g["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    g["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    g["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    g["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    g["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    return g


# =========================
# 3) 单股票基础价格/收益特征
# =========================

def _add_base_features(g: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = g.copy()

    # returns
    g["ret_1"] = g["close"].pct_change()
    if cfg.clip_return is not None:
        g["ret_1"] = g["ret_1"].clip(-cfg.clip_return, cfg.clip_return)

    g["log_ret_1"] = np.log(g["close"] / g["close"].shift(1))

    # intraday / range
    g["oc_ratio"] = _safe_div(g["close"] - g["open"], g["open"])
    g["hl_ratio"] = _safe_div(g["high"] - g["low"], g["close"])
    g["co_gap"] = _safe_div(g["open"] - g["close"].shift(1), g["close"].shift(1))

    for w in cfg.price_windows:
        g[f"ret_{w}"] = g["close"].pct_change(w)
        g[f"ma_ratio_{w}"] = _safe_div(g["close"], g["close"].rolling(w).mean()) - 1.0
        g[f"price_pos_{w}"] = _safe_div(
            g["close"] - g["low"].rolling(w).min(),
            g["high"].rolling(w).max() - g["low"].rolling(w).min()
        )

    for w in cfg.vol_windows:
        g[f"volatility_{w}"] = g["ret_1"].rolling(w).std()
        g[f"volume_ma_ratio_{w}"] = _safe_div(g["volume"], g["volume"].rolling(w).mean()) - 1.0

    if cfg.add_risk_adjusted_features:
        for w in cfg.vol_windows:
            mean_ret = g["ret_1"].rolling(w).mean()
            std_ret = g["ret_1"].rolling(w).std()
            downside_std = g["ret_1"].where(g["ret_1"] < 0, 0.0).rolling(w).std()

            g[f"sharpe_like_{w}"] = _safe_div(mean_ret, std_ret)
            g[f"sortino_like_{w}"] = _safe_div(mean_ret, downside_std)

            roll_max = g["close"].rolling(w).max()
            g[f"drawdown_{w}"] = _safe_div(g["close"], roll_max) - 1.0

    return g


# =========================
# 4) ta 库指标
# =========================

def _add_ta_library_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    high = g["high"]
    low = g["low"]
    close = g["close"]
    volume = g["volume"]

    # momentum
    g["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    g["roc_10"] = ROCIndicator(close=close, window=10).roc()
    g["willr_14"] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    g["stoch_k_14"] = stoch.stoch()
    g["stoch_d_14"] = stoch.stoch_signal()

    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    g["macd"] = macd.macd()
    g["macd_signal"] = macd.macd_signal()
    g["macd_hist"] = macd.macd_diff()

    # trend
    g["cci_20"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()
    g["adx_14"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()

    # volatility
    g["atr_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    g["bb_h"] = bb.bollinger_hband()
    g["bb_l"] = bb.bollinger_lband()
    g["bb_m"] = bb.bollinger_mavg()
    g["bb_width"] = _safe_div(bb.bollinger_hband() - bb.bollinger_lband(), bb.bollinger_mavg())
    g["bb_pos"] = _safe_div(close - bb.bollinger_lband(), bb.bollinger_hband() - bb.bollinger_lband())

    # volume
    g["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    g["cmf_20"] = ChaikinMoneyFlowIndicator(
        high=high, low=low, close=close, volume=volume, window=20
    ).chaikin_money_flow()
    g["mfi_14"] = MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=14
    ).money_flow_index()

    return g


# =========================
# 5) pandas-ta 指标
# =========================

def _add_pandas_ta_features(g: pd.DataFrame) -> pd.DataFrame:
    if pta is None:
        return g

    g = g.copy()

    # 注意：不同分支返回列名可能有差异，所以这里统一做 rename
    # supertrend
    try:
        st = pta.supertrend(high=g["high"], low=g["low"], close=g["close"], length=10, multiplier=3.0)
        if isinstance(st, pd.DataFrame):
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

    # kdj / stochastic variants
    try:
        kdj = pta.kdj(high=g["high"], low=g["low"], close=g["close"], length=9, signal=3)
        if isinstance(kdj, pd.DataFrame):
            cols = {c: c.lower() for c in kdj.columns}
            kdj = kdj.rename(columns=cols)
            for c in kdj.columns:
                g[c] = kdj[c]
    except Exception:
        pass

    # efficiency ratio
    try:
        er = pta.er(g["close"], length=10)
        if isinstance(er, pd.Series):
            g["er_10"] = er
    except Exception:
        pass

    # normalised ATR if available
    try:
        natr = pta.natr(high=g["high"], low=g["low"], close=g["close"], length=14)
        if isinstance(natr, pd.Series):
            g["natr_14"] = natr
    except Exception:
        pass

    return g


# =========================
# 6) 单股票总装
# =========================

def build_features_for_one_symbol(g: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = g.sort_values("date").copy()

    g = _add_base_features(g, cfg)

    if cfg.add_ta_features:
        g = _add_ta_library_features(g)

    if cfg.add_pandas_ta_features:
        g = _add_pandas_ta_features(g)

    if cfg.add_time_features:
        g = _add_time_features(g)

    # rolling zscore (per symbol)
    numeric_cols = [
        c for c in g.columns
        if c not in ["symbol", "date"]
        and pd.api.types.is_numeric_dtype(g[c])
    ]

    for c in numeric_cols:
        for w in cfg.zscore_windows:
            g[f"{c}_z{w}"] = _rolling_zscore(g[c], w)

    if cfg.fill_method == "ffill":
        g[numeric_cols] = g[numeric_cols].ffill()

    return g


# =========================
# 7) panel 级封装
# =========================

def build_panel_features(df: pd.DataFrame, cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    cfg = cfg or FeatureConfig()
    df = _check_input(df)

    parts = []
    for symbol, g in df.groupby("symbol", sort=False):
        out = build_features_for_one_symbol(g, cfg)
        parts.append(out)

    feat = pd.concat(parts, axis=0, ignore_index=True)
    feat = feat.sort_values(["date", "symbol"]).reset_index(drop=True)

    # 横截面 rank / zscore
    if cfg.cross_sectional_rank:
        candidate_cols = [
            "ret_1", "ret_5", "ret_10", "ret_20",
            "volatility_5", "volatility_10", "volatility_20",
            "rsi_14", "macd_hist", "cci_20", "adx_14",
            "atr_14", "bb_width", "cmf_20", "mfi_14",
            "sharpe_like_20", "sortino_like_20", "drawdown_20"
        ]
        candidate_cols = [c for c in candidate_cols if c in feat.columns]

        for c in candidate_cols:
            feat[f"{c}_cs_rank"] = _cross_sectional_rank(feat, c)
            feat[f"{c}_cs_z"] = _cross_sectional_zscore(feat, c)

    return feat


# =========================
# 8) 目标变量构造
# =========================

def add_targets(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 5, 10),
    price_col: str = "close"
) -> pd.DataFrame:
    out = df.sort_values(["symbol", "date"]).copy()

    for h in horizons:
        out[f"y_ret_{h}"] = out.groupby("symbol")[price_col].shift(-h) / out[price_col] - 1.0

    # 一个简单 risk-adjusted target
    if "volatility_20" in out.columns:
        out["y_ret_5_ra"] = _safe_div(out["y_ret_5"], out["volatility_20"])

    return out


# =========================
# 9) LSTM 用特征清单
# =========================

def select_lstm_feature_columns(df: pd.DataFrame) -> List[str]:
    whitelist_prefix = (
        "ret_", "log_ret_", "ma_ratio_", "price_pos_", "volatility_",
        "volume_ma_ratio_", "sharpe_like_", "sortino_like_", "drawdown_",
        "rsi_", "roc_", "willr_", "stoch_", "macd", "cci_", "adx_",
        "atr_", "bb_", "obv", "cmf_", "mfi_", "supertrend", "er_", "natr_",
        "dow_", "month_", "doy_"
    )
    exclude_keywords = ("_cs_rank", "_cs_z")  # 也可保留，看你是否做横截面建模

    cols = []
    for c in df.columns:
        if c in ("symbol", "date", "open", "high", "low", "close", "volume"):
            continue
        if any(c.startswith(p) for p in whitelist_prefix) and not any(k in c for k in exclude_keywords):
            cols.append(c)

    return sorted(cols)

if __name__ == "__main__":
    # 简单测试
    from alpha_arena.data.loader import load_from_parquet

    df = load_from_parquet("daily_data.parquet")
    # df: [symbol, date, open, high, low, close, volume]
    cfg = FeatureConfig(
        price_windows=(5, 10, 20, 60),
        vol_windows=(5, 10, 20, 60),
        zscore_windows=(20, 60),
        cross_sectional_rank=True,
        add_time_features=True,
        add_return_features=True,
        add_risk_adjusted_features=True,
        add_ta_features=True,
        add_pandas_ta_features=True,
        fill_method="none",
    )

    feat = build_panel_features(df, cfg)
    feat = add_targets(feat, horizons=(1, 5, 10))

    feature_cols = select_lstm_feature_columns(feat)

    train_df = feat.dropna(subset=feature_cols + ["y_ret_5"]).copy()
    print("num_features =", len(feature_cols))
    print(feature_cols[:30])