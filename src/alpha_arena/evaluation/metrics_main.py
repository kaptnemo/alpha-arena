import pandas as pd


def daily_ic_rankic(
    df: pd.DataFrame,
    date_col: str = "date",
    pred_col: str = "y_pred",
    target_col: str = "y_true",
    min_count: int = 30,
) -> pd.DataFrame:
    rows = []

    for date, g in df.groupby(date_col):
        g = g[[pred_col, target_col]].dropna()

        if len(g) < min_count:
            continue

        ic = g[pred_col].corr(g[target_col], method="pearson")
        rank_ic = g[pred_col].corr(g[target_col], method="spearman")

        rows.append(
            {
                "date": date,
                "n": len(g),
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )

    return pd.DataFrame(rows)


def summarize_ic(ic_df: pd.DataFrame) -> dict:
    out = {}

    for col in ["ic", "rank_ic"]:
        s = ic_df[col].dropna()

        out[f"{col}_mean"] = s.mean()
        out[f"{col}_std"] = s.std(ddof=1)
        out[f"{col}_ir"] = (
            s.mean() / s.std(ddof=1) if s.std(ddof=1) > 0 else float("nan")
        )
        out[f"{col}_positive_ratio"] = (s > 0).mean()
        out[f"{col}_count"] = len(s)

    return out
