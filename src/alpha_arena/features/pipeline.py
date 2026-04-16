"""
pipeline.py
===========
特征工程统一入口（向后兼容保留文件）。

本文件不再包含实现代码，所有逻辑已按功能拆分到以下子模块：

+------------------+--------------------------------------------+
| 模块             | 内容                                       |
+==================+============================================+
| config.py        | FeatureConfig 配置类，REQUIRED_COLUMNS 常量 |
+------------------+--------------------------------------------+
| utils.py         | 内部工具函数（校验、safe_div、Z-score 等） |
+------------------+--------------------------------------------+
| base_features.py | 基础价格 / 收益率 / 风险调整特征           |
+------------------+--------------------------------------------+
| ta_features.py   | ta 库 + pandas-ta 技术指标特征             |
+------------------+--------------------------------------------+
| targets.py       | 目标变量（未来收益率 label）构造           |
+------------------+--------------------------------------------+
| selector.py      | LSTM 输入特征列筛选                        |
+------------------+--------------------------------------------+
| builder.py       | 单股 / panel 级特征组装 pipeline           |
+------------------+--------------------------------------------+

所有原始公开符号均在此处重新导出，外部代码无需修改 import 路径。
"""

from __future__ import annotations

# 配置与常量
from alpha_arena.features.config import FeatureConfig, REQUIRED_COLUMNS  # noqa: F401

# 工具函数（供外部调试或扩展使用）
from alpha_arena.features.utils import (  # noqa: F401
    _check_input,
    _safe_div,
    _rolling_zscore,
    _cross_sectional_zscore,
    _cross_sectional_rank,
    _add_time_features,
)

# 特征构造函数
from alpha_arena.features.base_features import _add_base_features          # noqa: F401
from alpha_arena.features.ta_features import (                             # noqa: F401
    _add_ta_library_features,
    _add_pandas_ta_features,
)

# 组装 pipeline
from alpha_arena.features.builder import (                                 # noqa: F401
    build_features_for_one_symbol,
    build_panel_features,
    build_panel_features_multiprocess,
)

# 目标变量
from alpha_arena.features.targets import add_targets                       # noqa: F401

# LSTM 特征列筛选
from alpha_arena.features.selector import select_lstm_feature_columns      # noqa: F401

__all__ = [
    # 配置
    "FeatureConfig",
    "REQUIRED_COLUMNS",
    # 工具
    "_check_input",
    "_safe_div",
    "_rolling_zscore",
    "_cross_sectional_zscore",
    "_cross_sectional_rank",
    "_add_time_features",
    # 特征
    "_add_base_features",
    "_add_ta_library_features",
    "_add_pandas_ta_features",
    # pipeline
    "build_features_for_one_symbol",
    "build_panel_features",
    "build_panel_features_multiprocess",
    # 目标
    "add_targets",
    # 筛选
    "select_lstm_feature_columns",
]
