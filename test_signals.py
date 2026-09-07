"""Offline unit tests for the signal logic in get_stock_data.py.

These tests build synthetic DataFrames and exercise the pure computation
functions directly, so they run with no network access.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

import get_stock_data as g


def _daily_df(n=30, open_=100.0, hl_gap=10.0, close_=102.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": [open_] * n, "High": [open_ + hl_gap] * n,
         "Low": [open_] * n, "Close": [close_] * n},
        index=idx,
    )


# --- SMA helper -----------------------------------------------------------
def test_sma_basic():
    s = pd.Series(range(1, 11), dtype=float)  # 1..10
    out = g.sma(s, 3)
    assert len(out) == 10
    assert np.isnan(out[0]) and np.isnan(out[1])   # warmup
    assert out[2] == 2.0                            # mean(1,2,3)
    assert out[-1] == 9.0                           # mean(8,9,10)


def test_sma_short_series():
    out = g.sma(pd.Series([1.0, 2.0]), 5)
    assert len(out) == 2


# --- Zone calculation -----------------------------------------------------
def test_calculate_zones():
    df = _daily_df(open_=100.0, hl_gap=10.0, close_=102.0)  # ADR = 10
    z = g.calculate_zones(df, ma_len1=10, ma_len2=5, multiplier1=0.2, multiplier2=0.2)
    assert z is not None
    assert z["AD"] == pytest.approx(102.0)
    assert z["BD"] == pytest.approx(98.0)
    assert z["CD"] == pytest.approx(102.0)
    assert z["DD"] == pytest.approx(98.0)


def test_calculate_zones_insufficient_data():
    df = _daily_df(n=5)  # fewer than ma_len1=10
    assert g.calculate_zones(df) is None


def test_calculate_zones_monthly():
    idx = pd.to_datetime([f"2024-03-{i:02d}" for i in range(1, 13)])
    df = pd.DataFrame(
        {"Open": [100] * 12, "High": [105] * 12, "Low": [95] * 12,
         "Close": [104] * 12, "Volume": [10] * 12},
        index=idx,
    )
    z = g.calculate_zones_monthly(df)
    assert z is not None
    assert z["AD"] == pytest.approx(102.0)   # month_open 100 + ADR 10 * 0.2
    assert z["BD"] == pytest.approx(98.0)


# --- Volume conditions ----------------------------------------------------
def test_volume_conditions():
    # 21-period SMA has a 21-bar warmup, so place the volume spike after it.
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    close = [101] * 30
    close[25] = 103                          # up-day after warmup
    vol = [5000] * 30
    vol[25] = 100000
    df = pd.DataFrame({"Open": [100] * 30, "Close": close, "Volume": vol,
                       "High": [105] * 30, "Low": [95] * 30}, index=idx)
    volu, vold = g.calculate_volume_conditions(df)
    assert bool(volu.iloc[25]) is True       # huge up-day after SMA warmup
    assert bool(volu.iloc[0]) is False       # within warmup -> NaN -> False
    assert (vold == False).all()             # no down-days with high volume


# --- 15d / 1d signal generation -------------------------------------------
def test_generate_signals15():
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    close = [101] * 30
    close[25] = 103
    vol = [5000] * 30
    vol[25] = 100000
    df = pd.DataFrame({"Open": [100] * 30, "Close": close, "Volume": vol,
                       "High": [105] * 30, "Low": [95] * 30}, index=idx)
    zones = {"AD": 102, "BD": 98, "CD": 102, "DD": 98}
    volu, vold = g.calculate_volume_conditions(df)
    buy, sell = g.generate_signals15(df, zones, volu, vold)
    assert bool(buy.iloc[25]) is True     # close 103 > AD 102 and high-volume up-day
    assert bool(sell.iloc[25]) is False


def test_generate_signalsd_yesterday_close():
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"Open": [100, 100], "Close": [103, 95],
                       "High": [105, 100], "Low": [95, 90]}, index=idx)
    zones = {"AD": 101, "BD": 99, "CD": 101, "DD": 99}
    volu = pd.Series([True, True], index=idx)
    vold = pd.Series([True, True], index=idx)
    b, s, ycb, ycs = g.generate_signalsd(df, zones, volu, vold)
    assert bool(ycb.iloc[0]) is True     # yesterday (row 0) close 103 > AD 101
    assert bool(ycs.iloc[0]) is False    # 103 not < BD 99


# --- collect_signals (filter routing) -------------------------------------
def _current_month_index():
    cm = datetime.now().strftime("%Y-%m")
    idx = pd.to_datetime([f"{cm}-01", f"{cm}-02", f"{cm}-03"])
    df = pd.DataFrame({"Close": [100, 105, 110], "Open": [99, 104, 109],
                       "High": [106, 107, 112], "Low": [98, 103, 108]}, index=idx)
    return idx, df


def test_collect_signals_cm():
    idx, df1d1y = _current_month_index()
    buy_1d = pd.Series([True, False, True], index=idx)
    sell_1d = pd.Series([False, False, False], index=idx)
    ycb = pd.Series([False, False, False], index=idx)
    ycs = pd.Series([False, False, False], index=idx)
    zones_m = {"AD": 101, "BD": 99, "CD": 102, "DD": 98}

    signals = g.collect_signals(
        "TEST.NS", ["CM"],
        pd.DataFrame(index=idx), df1d1y,
        pd.Series(index=idx), pd.Series(index=idx),
        buy_1d, sell_1d, ycb, ycs,
        {"AD": 0, "BD": 0, "CD": 0, "DD": 0}, zones_m,
    )
    buy_signals = [s for s in signals if s.get("signal_type") == "1d Buy Signal (CM)"]
    assert len(buy_signals) == 2
    assert buy_signals[0]["symbol"] == "TEST.NS"
    assert buy_signals[0]["close_price"] == 100
    assert buy_signals[0]["zones"]["AD"] == 101


def test_collect_signals_invalid_filter():
    idx = pd.to_datetime(["2024-01-01"])
    df = pd.DataFrame({"Close": [100]}, index=idx)
    signals = g.collect_signals("TEST.NS", ["XX"],
                                df, df,
                                pd.Series(index=idx), pd.Series(index=idx),
                                pd.Series(index=idx), pd.Series(index=idx),
                                pd.Series(index=idx), pd.Series(index=idx),
                                {}, {})
    assert signals[0]["signal_type"] == "Invalid filter"


def test_get_signals_empty_symbol():
    assert g.get_signals("   ", ["CM"])[0]["signal_type"] == "Error"
