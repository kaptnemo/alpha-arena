import typer

from pathlib import Path
from alpha_arena.data.ingest_tushare import (
    _stock_daily,
    _index_stocks,
)

app = typer.Typer(help="Data ingestion utilities for AMC-LSTM project.")


@app.command()
def stock_daily(
    code: str = typer.Argument(..., help="Stock code (e.g., 'sh.000001')"),
    start_date: str = typer.Argument(..., help="Start date (YYYYMMDD)"),
    end_date: str = typer.Argument(..., help="End date (YYYYMMDD)"),
    storage_format: str = typer.Option("csv", "--storage-format", "-s", help="Storage format (csv or parquet)"),
    file_path: Path = typer.Option(None, "--file-path", "-f", help="Output file path"),
):
    _stock_daily(code, start_date, end_date, storage_format, file_path)


@app.command()
def index_stocks(
    start_year: str = typer.Argument(..., help="Start year (YYYY) for index stock list"),
    end_year: str = typer.Argument(..., help="End year (YYYY) for index stock list"),
    index_name: str = typer.Option(
        "csi300",
        "--index-name",
        "-i",
        help="Index name or TuShare index code, e.g. csi300, csi500, csi1000, sse50, 000905.SH",
    ),
    storage_format: str = typer.Option("csv", "--storage-format", "-s", help="Storage format (csv or parquet)"),
    file_path: Path = typer.Option(None, "--file-path", "-f", help="Output file path"),
):
    _index_stocks(start_year, end_year, storage_format, file_path, index_name=index_name)


def main():
    app()
