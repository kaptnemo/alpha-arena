import numpy as np
import pandas as pd


def daily_ci(
    df: pd.DataFrame,
    group_col: str = "date",
    pred_col: str = "pred",
    target_col: str = "target",
    metric_func=lambda y_true, y_pred: (y_true * y_pred).mean(),
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
):
    """
    计算每日指标的置信区间。

    参数:
    - df: 包含日期、预测值和目标值的DataFrame。
    - group_col: 日期列的名称。
    - pred_col: 预测值列的名称。
    - target_col: 目标值列的名称。
    - metric_func: 用于计算指标的函数，默认是IC。
    - n_bootstrap: 引导法的迭代次数。
    - ci_level: 置信水平，默认为95%。

    返回:
    包含日期、指标均值、下限和上限的DataFrame。
    """

    results = []
    for date, group in df.groupby(group_col):
        y_true = group[target_col].values
        y_pred = group[pred_col].values
        metric_values = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(group), size=len(group), replace=True)
            metric_value = metric_func(y_true[indices], y_pred[indices])
            metric_values.append(metric_value)
        lower_bound = np.percentile(metric_values, (1 - ci_level) / 2 * 100)
        upper_bound = np.percentile(metric_values, (1 + ci_level) / 2 * 100)
        mean_metric = np.mean(metric_values)
        results.append((date, mean_metric, lower_bound, upper_bound))

    return pd.DataFrame(
        results, columns=[group_col, "metric_mean", "ci_lower", "ci_upper"]
    )
