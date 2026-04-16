from pathlib import Path
from alpha_arena.data.helpers.baostock_helper import BaostockHelper
import typer
import pandas as pd
import time
from alpha_arena.data import (
    RAW_DATA_DIR,
)
from alpha_arena.utils import get_logger

app = typer.Typer(help="Data ingestion utilities for AMC-LSTM project.")
logger = get_logger(__name__)


@app.command()
def stock_daily(
    code: str = typer.Argument(..., help="Stock code (e.g., 'sh.000001')"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    storage_format: str = typer.Option("csv", "--storage-format", "-s", help="Storage format (csv or parquet)"),
    file_path: Path = typer.Option(None, "--file-path", "-f", help="Output file path"),
):
    """Ingest daily stock data for a given stock code and date range."""
    if storage_format not in ["csv", "parquet"]:
        raise ValueError("Unsupported storage format. Use 'csv' or 'parquet'.")
    if file_path is None:
        file_path = RAW_DATA_DIR / f"{code}_daily_{start_date}_{end_date}.{storage_format}"
    with BaostockHelper() as helper:
        daily_result = helper.daily(code, start_date, end_date)
        if storage_format == "csv":
            daily_result.save_to_csv(file_path)
        else:
            daily_result.save_to_parquet(file_path)


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first_day = pd.Timestamp(year=year, month=month, day=1)
    days_until_weekday = (weekday - first_day.weekday()) % 7
    first_target = first_day + pd.Timedelta(days=days_until_weekday)
    return first_target + pd.Timedelta(weeks=n - 1)


# 生成所有调整生效日（6月和12月的第三个周一）
# 沪深300指数调整是6月和12月的第二个周五，但是baostock的接口是每周一更新，所以我们取第三个周一作为调整生效日的近似值
def get_adjustment_dates(start_year: str, end_year: str):
    adjustment_dates = []
    for year in range(int(start_year) - 1, int(end_year) + 1):
        for month in [6, 12]:
            # 获取当月的第三个周一
            third_monday = nth_weekday_of_month(year, month, weekday=2, n=3)  # weekday=0表示周一
            adjustment_dates.append(third_monday)
    return adjustment_dates[1:]


def get_csi300_stocks(
    helper: BaostockHelper,
    date: str,

):
    """Ingest csi300 stock list for a given date."""
    csi300_stocks_result = helper.query_csi300_stocks(date=date.strftime('%Y-%m-%d'))
    return csi300_stocks_result.data['code'].tolist()


def csi300_stocks_by_date_range(
    helper: BaostockHelper,
    start_year: str = typer.Argument(..., help="Start year (YYYY) for csi300 stock list"),
    end_year: str = typer.Argument(..., help="End year (YYYY) for csi300 stock list"),
):
    """Ingest csi300 stock lists for multiple date ranges defined by adjustment dates."""
    adjust_dates = get_adjustment_dates(start_year, end_year)
    date_range_stocks = {}
    for i in range(len(adjust_dates)):
        start_date = adjust_dates[i]
        end_date = adjust_dates[i + 1] - pd.Timedelta(days=1) if i + 1 < len(adjust_dates) else pd.to_datetime(f'{end_year}-12-31')
        stocks = get_csi300_stocks(helper, start_date)
        if 'sh.600004' in stocks:
            logger.info("Sample stock is in csi300", stock_code="sh.600004", start_date=str(start_date), end_date=str(end_date))
        else:
            logger.warning("Sample stock is not in csi300", stock_code="sh.600004", start_date=str(start_date), end_date=str(end_date))
        date_range_stocks[(start_date, end_date)] = stocks
        logger.info("Fetched csi300 constituents", start_date=str(start_date), end_date=str(end_date), stock_count=len(stocks))
        time.sleep(1)  # 避免请求过快
    
    stock_date_ranges = {}
    for (start_date, end_date), stocks in date_range_stocks.items():
        for stock in stocks:
            if stock not in stock_date_ranges:
                stock_date_ranges[stock] = [(start_date, end_date)]
            else:
                stock_date_ranges[stock].append((start_date, end_date))
    return stock_date_ranges

def check_stock_in_csi300(stock_date_ranges, stock_code, date):
    """Check if a stock was in csi300 on a specific date."""
    if stock_code not in stock_date_ranges:
        return False
    for start_date, end_date in stock_date_ranges[stock_code]:
        if start_date <= pd.to_datetime(date) <= end_date:
            return True
    return False


@app.command()
def csi300_stocks(
    start_year: str = typer.Argument(..., help="Start year (YYYY) for csi300 stock list"),
    end_year: str = typer.Argument(..., help="End year (YYYY) for csi300 stock list"),
    storage_format: str = typer.Option("csv", "--storage-format", "-s", help="Storage format (csv or parquet)"),
    file_path: Path = typer.Option(None, "--file-path", "-f", help="Output file path"),
):
    if storage_format not in ["csv", "parquet"]:
        raise ValueError("Unsupported storage format. Use 'csv' or 'parquet'.")
    if file_path is None:
        if not RAW_DATA_DIR.exists():
            RAW_DATA_DIR.mkdir(parents=True)
        file_path = RAW_DATA_DIR / f"csi300_stocks_{start_year}_{end_year}.{storage_format}"

    all_fds = []
    with BaostockHelper() as helper:
        stock_date_ranges = csi300_stocks_by_date_range(helper, start_year, end_year)
        all_stocks = list(stock_date_ranges.keys())
        logger.info(
            "Collected csi300 universe",
            start_year=start_year,
            end_year=end_year,
            total_unique_stocks=len(all_stocks),
        )

        for stock in all_stocks[:2]:
            start_date = f'{start_year}-01-01'
            end_date = f'{end_year}-12-31'
            daily_result = helper.daily(stock, start_date, end_date)
            df = daily_result.data
            df['in_csi300'] = df['date'].apply(lambda x: check_stock_in_csi300(stock_date_ranges, stock, x))
            all_fds.append(df)
            logger.info("Processed stock daily data", stock_code=stock, records=len(df))
            logger.debug("Sample dataframe preview", stock_code=stock, head=df.head().to_dict("records"), tail=df.tail().to_dict("records"))
            time.sleep(0.5)  # 避免请求过快
    final_df = pd.concat(all_fds, ignore_index=True)
    if storage_format == "csv":
        final_df.to_csv(file_path, index=False)
    else:
        final_df.to_parquet(file_path, index=False)


def main():
    app()

if __name__ == "__main__":
    main()