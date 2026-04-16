import pandas as pd
import time
from pathlib import Path
from collections import OrderedDict

from alpha_arena.data import (
    RAW_DATA_DIR,
)
from alpha_arena.data.helpers.tushare_helper import (
    TuShareHelper,
    TuShareResult,
)
from alpha_arena.utils import get_logger

logger = get_logger(__name__)


def _stock_daily(
    code: str,
    start_date: str,
    end_date: str,
    storage_format: str = "csv",
    file_path: Path | None = None,
):
    """Ingest daily stock data for a given stock code and date range."""
    if storage_format not in ["csv", "parquet"]:
        raise ValueError("Unsupported storage format. Use 'csv' or 'parquet'.")
    if file_path is None:
        file_path = RAW_DATA_DIR / f"{code}_daily_{start_date}_{end_date}.{storage_format}"

    with TuShareHelper() as helper:
        daily_result = helper.daily(ts_code=code, start_date=start_date, end_date=end_date)
        if storage_format == "csv":
            daily_result.save_to_csv(file_path)
        else:
            daily_result.save_to_parquet(file_path)

# 生成每年每个月的月初月末日期范围对列表
def _get_adjustment_date_pairs(start_year: str, end_year: str):
    """Generate a list of date ranges for each month start end pair, which can be used to query the stock list for each month."""
    get_adjustment_date_pairs = []
    for year in range(int(start_year), int(end_year) + 1):
        if year == int(start_year):
            start_date = f"{year - 1}1101"
        else:
            start_date = f"{year}0101"
        end_date = f"{year}1231"
        get_adjustment_date_pairs.append((pd.to_datetime(start_date), pd.to_datetime(end_date)))
    return get_adjustment_date_pairs


def _get_csi300_stocks(
    helper: TuShareHelper,
    date_pair: tuple[pd.Timestamp, pd.Timestamp],
    pre_last_pair_stocks: tuple[pd.Timestamp, list[str]] | None = None,

):
    """Ingest csi300 stock list for a given date range.
    
    tushare api index_weight 的用法是按月查询的，所以我们生成每个月的月初月末日期范围对，来查询每个月的股票列表。
    然后取每个月的第一个交易日的股票列表作为该月的股票列表。
    这个方法会造成最多一个月的误差，因为指数调整的生效日可能不是月初，
    虽然不准确，但是不会造成信息泄露，随意这个误差在我们这个项目中是可以接受的。
    """
    HS_300_CODE = '399300.SZ'
    start_date, end_date = date_pair

    csi300_stocks_result = helper.index_weight(
        index_code=HS_300_CODE,
        start_date=start_date.strftime('%Y%m%d'),
        end_date=end_date.strftime('%Y%m%d'),
    )
    df = csi300_stocks_result.data
    df = df.sort_values('trade_date').reset_index(drop=True)
    result = OrderedDict()
    if not df.empty:
        trade_dates = df['trade_date'].unique()
        for i in range(len(trade_dates)):
            trade_date = trade_dates[i]
            if i == 0 and pre_last_pair_stocks:
                _pre_start_date, pre_last_pair_stocks = pre_last_pair_stocks
                pre_start_date = _pre_start_date + pd.Timedelta(days=1)
                pre_end_date = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
                result[(pre_start_date, pre_end_date)] = pre_last_pair_stocks
            post_trade_date = trade_dates[i + 1] if i + 1 < len(trade_dates) else None
            stocks_on_date = df[df['trade_date'] == trade_date]['con_code'].tolist()
            pair_end_date = pd.Timestamp(post_trade_date) - pd.Timedelta(days=1) if post_trade_date else end_date
            result[(pd.Timestamp(trade_date), pair_end_date)] = stocks_on_date
    result_pre_last_pair_stocks = None
    if result:
        result_pre_last_pair_stocks = (list(result.keys())[-1][0], list(result.values())[-1])
    return result, result_pre_last_pair_stocks


def _check_date_ranges(ranges):
    """
    检查时间区间是否连续无间断、无重叠
    
    ranges: List[(start_timestamp, end_timestamp)]
    
    返回:
        is_valid: bool
        issues: list[str]
    """
    if not ranges:
        return False, ["Empty ranges"]

    # 按 start 排序
    ranges = sorted(ranges, key=lambda x: x[0])

    issues = []

    for i in range(len(ranges) - 1):
        curr_start, curr_end = ranges[i]
        next_start, next_end = ranges[i + 1]

        # 确保是 Timestamp
        curr_end = pd.to_datetime(curr_end)
        next_start = pd.to_datetime(next_start)

        # 理想连续：next_start == curr_end + 1 day
        expected_next_start = curr_end + pd.Timedelta(days=1)

        if next_start > expected_next_start:
            issues.append(
                f"GAP between {curr_end.date()} and {next_start.date()}"
            )

        elif next_start < expected_next_start:
            issues.append(
                f"OVERLAP between {curr_end.date()} and {next_start.date()}"
            )

    is_valid = len(issues) == 0
    return is_valid, issues


def _csi300_stocks_by_date_range(
    helper: TuShareHelper,
    start_year: str,
    end_year: str,
):
    """Ingest csi300 stock list for a given date range."""
    date_pairs = _get_adjustment_date_pairs(start_year, end_year)
    logger.info("Generated adjustment date pairs", count=len(date_pairs))
    csi300_stocks_by_date_pairs = OrderedDict()
    pre_last_pair_stocks = None
    for date_pair in date_pairs:
        stock_dicts, pre_last_pair_stocks = _get_csi300_stocks(helper, date_pair, pre_last_pair_stocks)
        csi300_stocks_by_date_pairs.update(stock_dicts)
    
    check_date_ranges_result, issues = _check_date_ranges(list(csi300_stocks_by_date_pairs.keys()))
    if not check_date_ranges_result:
        logger.warning("Date range issues detected", issues=issues)
    else:
        logger.info("Date ranges are continuous and non-overlapping")

    stock_date_ranges = {}
    for date_pair, stocks in csi300_stocks_by_date_pairs.items():
        for stock in stocks:
            if stock not in stock_date_ranges:
                stock_date_ranges[stock] = []
            stock_date_ranges[stock].append(date_pair)
            if stock == '000001.SZ':
                logger.info(
                    "Tracked sample stock in csi300",
                    stock=stock,
                    start_date=str(date_pair[0]),
                    end_date=str(date_pair[1]),
                )
    return stock_date_ranges


def _check_stock_in_csi300(stock_date_ranges, stock_code, date):
    """Check if a given stock code is in the csi300 index on a given date."""
    if stock_code not in stock_date_ranges:
        return False
    d = pd.to_datetime(date)
    for start_date, end_date in stock_date_ranges[stock_code]:
        if start_date <= d <= end_date:
            return True
    return False


def _csi300_stocks(
    start_year: str,
    end_year: str,
    storage_format: str = "csv",
    file_path: Path | None = None,
):
    if storage_format not in ["csv", "parquet"]:
        raise ValueError("Unsupported storage format. Use 'csv' or 'parquet'.")
    if file_path is None:
        if not RAW_DATA_DIR.exists():
            RAW_DATA_DIR.mkdir(parents=True)
        file_path = RAW_DATA_DIR / f"csi300_stocks_{start_year}_{end_year}.{storage_format}"

    all_fds = []
    with TuShareHelper() as helper:
        stock_date_ranges = _csi300_stocks_by_date_range(helper, start_year, end_year)
        all_stocks = list(stock_date_ranges.keys())
        logger.info(
            "Collected csi300 universe",
            start_year=start_year,
            end_year=end_year,
            total_unique_stocks=len(all_stocks),
        )
        for i, stock_code in enumerate(all_stocks):
            start_date = f'{start_year}0101'
            end_date = f'{end_year}1231'
            daily_result = helper.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
            df = daily_result.data
            df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume',
            }, inplace=True)
            df['in_csi300'] = df['date'].apply(lambda x: _check_stock_in_csi300(stock_date_ranges, stock_code, x))
            all_fds.append(df)
            logger.info("Processed stock daily data", index=i, stock_code=stock_code, records=len(df))
            time.sleep(0.5)  # 避免请求过快

    if not all_fds:
        raise ValueError("No data was fetched for any stock. Please check the date range and stock codes.")
    
    final_df = pd.concat(all_fds, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["ts_code", "date"])
    final_df = final_df.sort_values(["ts_code", "date"]).reset_index(drop=True)
    if storage_format == "csv":
        final_df.to_csv(file_path, index=False)
    else:
        final_df.to_parquet(file_path, index=False)
