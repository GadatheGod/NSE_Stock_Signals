# NSE Triggers V1

A PyQt5 desktop application that scans NSE (Indian) stocks for buy/sell signals
based on daily ADR (Average Daily Range) bands and volume, with live 15-minute
candlestick charts.

## Features

- Symbol lists: Nifty 50 / Nifty N50 / 100 / 200 / 500 and a full `TotalList`.
- Signal types:
  - **15m signals** (current day / previous days) using daily ADR bands + volume.
  - **1d signals** (this month / last month / 2 months ago) using monthly ADR bands.
  - **Yesterday-close** conditions (close > AD / close < BD).
- Combined **Buy** / **Sell** tabs that keep only symbols meeting the cross
  filter (present in both monthly and daily views, with a buy but not a sell).
- Double-click any row to open a live 15-minute candlestick chart (auto-refresh
  every 15 min).
- Export all six tables to Excel.
- Search/filter across every table, dark/light theme, and cache management.

## Requirements

- Python 3.11+ (tested on 3.14).
- Internet access at runtime (data is fetched from Yahoo Finance via `yfinance`).

## Installation

```bash
# System-wide install (Linux, PEP 668 externally-managed Python):
python3 -m pip install --break-system-packages -r requirements.txt
```

Or in a virtual environment:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python NSETriggersV1_MainM.py
```

## Running

```bash
python3 NSETriggersV1_MainM.py
```

## Usage

1. Pick a symbol list (radio buttons 1–6) or click **Select File** for your own.
2. Choose process options (monthly / daily) and the date/month filter.
3. Click **Process**. Results populate the six tables and a progress bar.
4. Double-click a symbol to view its candlestick chart.
5. Use the search box to filter tables, **Settings** to manage cache/theme,
   **Export** to save Excel, **Clear** to reset tables.

## Project layout

| File | Purpose |
|------|---------|
| `NSETriggersV1_MainM.py` | Main GUI window, threading, export, search, settings. |
| `NSETriggersV1M.ui` | Qt Designer layout for the main window. |
| `get_stock_data.py` | Data fetching (yfinance), ADR-zone & volume signal logic. |
| `chart.py` | `CandlestickChartWindow` — live 15m candlestick viewer. |
| `nifty50.txt` … `TotalList.txt` | Comma/newline-separated symbol lists. |
| `cached/` | joblib cache of fetched data (auto-managed; safe to delete). |

## Notes

- Signal math uses pandas rolling-mean Simple Moving Averages (no TA-Lib / C
  dependency).
- The `cached/` folder stores fetched data to speed up repeated runs; use
  **Settings → Clear Data Cache** (or **Clear Memory**) to regenerate it.

## Tests

```bash
python3 -m pytest test_signals.py -v
```

The tests run offline against synthetic DataFrames (no network required).
