from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_arena.data import ingest_tushare


class _FakeResult:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data


class _FakeTuShareHelper:
    def __init__(self, calls: dict[str, str]) -> None:
        self.calls = calls

    def __enter__(self) -> "_FakeTuShareHelper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def index_weight(self, index_code: str, start_date: str, end_date: str) -> _FakeResult:
        self.calls["index_code"] = index_code
        self.calls["start_date"] = start_date
        self.calls["end_date"] = end_date
        return _FakeResult(
            pd.DataFrame(
                {
                    "trade_date": ["2023-01-03"],
                    "con_code": ["000001.SZ"],
                }
            )
        )

    def daily(self, ts_code: str, start_date: str, end_date: str) -> _FakeResult:
        self.calls["ts_code"] = ts_code
        return _FakeResult(
            pd.DataFrame(
                {
                    "ts_code": [ts_code],
                    "trade_date": ["20230103"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.8],
                    "close": [10.2],
                    "pre_close": [10.0],
                    "change": [0.2],
                    "pct_chg": [2.0],
                    "vol": [1000],
                    "amount": [10000],
                }
            )
        )


def test_resolve_index_config_supports_alias_and_tushare_code() -> None:
    assert ingest_tushare._resolve_index_config("csi500") == ("csi500", "000905.SH")
    assert ingest_tushare._resolve_index_config("000905.SH") == ("csi500", "000905.SH")


def test__index_stocks_uses_selected_index_name_for_output_and_membership_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, str] = {}

    monkeypatch.setattr(ingest_tushare, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(ingest_tushare, "TuShareHelper", lambda: _FakeTuShareHelper(calls))
    monkeypatch.setattr(ingest_tushare.time, "sleep", lambda _: None)

    ingest_tushare._index_stocks("2023", "2023", storage_format="csv", index_name="csi500")

    output_path = tmp_path / "csi500_stocks_2023_2023.csv"
    assert output_path.exists()
    assert calls["index_code"] == "000905.SH"

    df = pd.read_csv(output_path)
    assert "in_csi500" in df.columns
    assert "in_csi300" not in df.columns
    assert df["in_csi500"].tolist() == [True]
