import yfinance as yf
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from joblib import Memory
import os
from dateutil.relativedelta import relativedelta

# Cache directories are resolved relative to this file so the app works on
# any machine / OS (was previously hardcoded to a Windows path).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
daily_cache_dir = os.path.join(BASE_DIR, "cached", "daily")
monthly_cache_dir = os.path.join(BASE_DIR, "cached", "monthly")



# Create Memory instances for each.
daily_memory = Memory(daily_cache_dir, verbose=10)
monthly_memory = Memory(monthly_cache_dir, verbose=10)


def reset_cache_if_needed(memory, cache_dir, period='daily'):
    """
    Clears the cache if the current period (day or month) is different
    from the last time the cache was cleared.
    """
    last_cleared_file = os.path.join(cache_dir, "last_cleared.txt")
    now = datetime.now()
    
    if period == 'daily':
        current_period = now.strftime("%Y-%m-%d")
    elif period == 'monthly':
        current_period = now.strftime("%Y-%m")
    else:
        raise ValueError("Invalid period specified for cache reset.")
    
    if os.path.exists(last_cleared_file):
        with open(last_cleared_file, "r") as f:
            last_cleared = f.read().strip()
        if last_cleared == current_period:
            return  # Cache is up-to-date.
    
    memory.clear()
    with open(last_cleared_file, "w") as f:
        f.write(current_period)

# Reset caches at the start of the script.
reset_cache_if_needed(daily_memory, daily_cache_dir, period='daily')
reset_cache_if_needed(monthly_memory, monthly_cache_dir, period='monthly')


def sma(values, period):
    """Simple moving average (TA-Lib SMA replacement) using pandas rolling mean.

    Returns a numpy array with NaN for the first ``period - 1`` entries, matching
    TA-Lib's warmup behaviour.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < period:
        return np.full(arr.size, np.nan)
    return pd.Series(arr).rolling(window=period, min_periods=period).mean().to_numpy()


# --- Define Data Fetching Functions ---

def fetch_data15m1d(symbol, timeframe='15m', period='5d'):
    df15m1d = yf.download(symbol, interval=timeframe, period=period, auto_adjust=False, progress=False)
    if df15m1d.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df15m1d.dropna().to_json()

@daily_memory.cache
def fetch_data1d1y(symbol, timeframe='1d', period='1y'):
    df1d1y = yf.download(symbol, interval=timeframe, period=period, auto_adjust=False, progress=False)
    if df1d1y.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df1d1y.dropna().to_json()

@daily_memory.cache
def fetch_data1d1m(symbol, timeframe='1d', period='1mo'):
    df1d1m = yf.download(symbol, interval=timeframe, period=period, auto_adjust=False, progress=False)
    if df1d1m.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df1d1m.dropna().to_json()

@monthly_memory.cache
def fetch_data1m1y(symbol, timeframe='1mo', period='1y'):
    df1m1y = yf.download(symbol, interval=timeframe, period=period, auto_adjust=False, progress=False)
    if df1m1y.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return df1m1y.dropna().to_json()

def parse_json_data(json_data, symbol):
    data_dict = json.loads(json_data)
    timestamps = [int(ts) for ts in data_dict[f"('Open', '{symbol}')"].keys()]
    df = pd.DataFrame({
        "Open": [data_dict[f"('Open', '{symbol}')"].get(str(ts), np.nan) for ts in timestamps],
        "High": [data_dict[f"('High', '{symbol}')"].get(str(ts), np.nan) for ts in timestamps],
        "Low": [data_dict[f"('Low', '{symbol}')"].get(str(ts), np.nan) for ts in timestamps],
        "Close": [data_dict[f"('Close', '{symbol}')"].get(str(ts), np.nan) for ts in timestamps],
        "Volume": [data_dict[f"('Volume', '{symbol}')"].get(str(ts), np.nan) for ts in timestamps]
    })
    df.index = pd.to_datetime(timestamps, unit='ms').tz_localize('UTC').tz_convert('Asia/Kolkata')
    return df

def calculate_zones(daily_df, ma_len1=10, ma_len2=5, multiplier1=0.2, multiplier2=0.2):
    day_open = float(daily_df['Open'].iloc[-1])
    adr10 = sma(daily_df['High'] - daily_df['Low'], ma_len1)
    adr5 = sma(daily_df['High'] - daily_df['Low'], ma_len2)
    if adr10 is None or adr5 is None or len(adr10) < ma_len1 or len(adr5) < ma_len2:
        print("Error: ADR values are empty or insufficient data")
        return None
    if np.isnan(adr10[-1]) or np.isnan(adr5[-1]):
        print("Error: ADR values are NaN (insufficient warmup data)")
        return None
    adr10_last = float(adr10[-1])
    adr5_last = float(adr5[-1])
    return {
        'AD': day_open + adr10_last * multiplier1,
        'BD': day_open - adr10_last * multiplier1,
        'CD': day_open + adr5_last * multiplier2,
        'DD': day_open - adr5_last * multiplier2,
    }

def calculate_zones_monthly(monthly_df, ma_len1=10, ma_len2=5, multiplier1=0.2, multiplier2=0.2):
    if monthly_df.empty:
        print("❌ No monthly data available.")
        return None

    # ✅ Ensure DataFrame is sorted by index (date) in ascending order
    monthly_df = monthly_df.sort_index()

    # ✅ Get the latest available month and year
    latest_date = monthly_df.index[-1]  # Most recent date in the DataFrame
    target_year = latest_date.year
    target_month = latest_date.month

    # ✅ Filter data for the latest month and year
    monthly_filtered = monthly_df[
        (monthly_df.index.month == target_month) & (monthly_df.index.year == target_year)
    ]

    if monthly_filtered.empty:
        print(f"❌ No data available for {target_year}-{target_month:02d}.")
        return None

    # ✅ Get the first available row in the target month
    first_row = monthly_filtered.iloc[0]
    first_date = first_row.name.strftime('%Y-%m-%d')  # Extract YYYY-MM-DD format
    month_open = float(first_row['Open'])

    # ✅ Compute ADR values
    adr10 = sma(monthly_df['High'] - monthly_df['Low'], ma_len1)
    adr5 = sma(monthly_df['High'] - monthly_df['Low'], ma_len2)

    if adr10 is None or adr5 is None or len(adr10) < ma_len1 or len(adr5) < ma_len2:
        print("❌ Monthly ADR values are empty or insufficient data.")
        return None

    if np.isnan(adr10[-1]) or np.isnan(adr5[-1]):
        print("❌ Monthly ADR values are NaN (insufficient warmup data).")
        return None

    adr10_last = float(adr10[-1])
    adr5_last = float(adr5[-1])

    # ✅ Debugging Output
    #print("🔍 Checking Monthly DataFrame:")
    #print(monthly_filtered.head())  # Show filtered data for the target month

    #print(f"📅 First Available Date in {target_year}-{target_month:02d}: {first_date}")
    #print(f"📌 Month Open: {month_open}")
    #print(f"📊 ADR10: {adr10_last}, ADR5: {adr5_last}")
    #print(f"✅ AD: {round(month_open + adr10_last * multiplier1, 2)}")
    #print(f"✅ BD: {round(month_open - adr10_last * multiplier1, 2)}")
    #print(f"✅ CD: {round(month_open + adr5_last * multiplier2, 2)}")
    #print(f"✅ DD: {round(month_open - adr5_last * multiplier2, 2)}")

    return {
        'AD': round(month_open + adr10_last * multiplier1, 2),
        'BD': round(month_open - adr10_last * multiplier1, 2),
        'CD': round(month_open + adr5_last * multiplier2, 2),
        'DD': round(month_open - adr5_last * multiplier2, 2),
    }




def calculate_volume_conditions(df):
    avg_vol = sma(df['Volume'], 21)
    volu1_15m = (df['Volume'] > avg_vol * 1.5) & (df['Close'] > df['Open'])
    vold1_15m = (df['Volume'] > avg_vol * 1.5) & (df['Close'] < df['Open'])
    return volu1_15m, vold1_15m

def calculate_volume_conditions_monthly(monthly_df):
    avg_volm = sma(monthly_df['Volume'], 21)
    monthly_volu1 = (monthly_df['Volume'] > avg_volm * 1.5) & (monthly_df['Close'] > monthly_df['Open'])
    monthly_vold1 = (monthly_df['Volume'] > avg_volm * 1.5) & (monthly_df['Close'] < monthly_df['Open'])
    return monthly_volu1, monthly_vold1

def generate_signalsd(df, zones, monthly_volu1, monthly_vold1):
    buy_signal = (df['Close'] > zones['AD']) & monthly_volu1
    sell_signal = (df['Close'] < zones['BD']) & monthly_vold1

    # Initialize yesterday close signals
    yesterday_close_buy_signal = pd.Series(False, index=df.index)
    yesterday_close_sell_signal = pd.Series(False, index=df.index)

    if df.empty or 'Close' not in df.columns:
        #print("❌ DataFrame is empty or missing 'Close' column.")
        return buy_signal, sell_signal, yesterday_close_buy_signal, yesterday_close_sell_signal

    last_available_date = df.index[-1]  # Get the last available date in DataFrame
    yesterday_date = df.index[df.index < last_available_date][-1] if len(df.index[df.index < last_available_date]) > 0 else None


    # ✅ Debug: Print available dates
    #print(f"📅 Available Dates in DataFrame: {df.index[-5:]}")  
    #print(f"🔍 Checking for Yesterday's Date: {yesterday_date}")

    if yesterday_date is not None:
        # ✅ Use `.at[]` for scalar values, `.loc[]` for Series
        yesterday_close = df.at[yesterday_date, 'Close'] if isinstance(df.at[yesterday_date, 'Close'], (int, float)) else df.loc[yesterday_date, 'Close']
        #print(f"✅ Found Yesterday's Close: {yesterday_close}")

        yesterday_close_buy_signal.loc[yesterday_date] = yesterday_close > zones['AD']
        yesterday_close_sell_signal.loc[yesterday_date] = yesterday_close < zones['BD']

        #print(f"➡ Yesterday Close Buy Signal: {yesterday_close_buy_signal.loc[yesterday_date]}")
        #print(f"➡ Yesterday Close Sell Signal: {yesterday_close_sell_signal.loc[yesterday_date]}")
    else:
        print(f"❌ Yesterday's Date {yesterday_date} is missing in data.")

    return buy_signal, sell_signal, yesterday_close_buy_signal, yesterday_close_sell_signal


def generate_signals15(df15, zones, volu1_15m, vold1_15m):
    buy_signal15 = (df15['Close'] > zones['AD']) & volu1_15m
    sell_signal15 = (df15['Close'] < zones['BD']) & vold1_15m
    return buy_signal15, sell_signal15

def collect_signals(symbol, filters, parsed_df_15m1d, parsed_df_1d1y,
                    buy_signal_15m, sell_signal_15m, buy_signal_1d, sell_signal_1d,
                    yesterday_close_buy_signal, yesterday_close_sell_signal,  
                    zones_1d1m, zones_1m1y):
    signals = []
    
    for filter_type in filters:
        if filter_type == "CM":  # Current Month
            current_month = datetime.now().strftime('%Y-%m')
            filtered_df_1d1y = parsed_df_1d1y[parsed_df_1d1y.index.strftime('%Y-%m') == current_month]
            filtered_buy_signal_1d = buy_signal_1d[parsed_df_1d1y.index.strftime('%Y-%m') == current_month]
            filtered_sell_signal_1d = sell_signal_1d[parsed_df_1d1y.index.strftime('%Y-%m') == current_month]
            filtered_yesterday_buy_signal = yesterday_close_buy_signal[parsed_df_1d1y.index.strftime('%Y-%m') == current_month]
            filtered_yesterday_sell_signal = yesterday_close_sell_signal[parsed_df_1d1y.index.strftime('%Y-%m') == current_month]

            for i in range(len(filtered_df_1d1y)):
                zones_1m1y_formatted = {k: round(v, 2) for k, v in zones_1m1y.items()}
                
                if i < len(filtered_buy_signal_1d) and filtered_buy_signal_1d.iloc[i]:
                    signals.append({
                        'signal_type': '1d Buy Signal (CM)',
                        'symbol': symbol,
                        'timestamp': str(filtered_df_1d1y.index[i]),
                        'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                        'zones': zones_1m1y_formatted
                    })
                
                if i < len(filtered_sell_signal_1d) and filtered_sell_signal_1d.iloc[i]:
                    signals.append({
                        'signal_type': '1d Sell Signal (CM)',
                        'symbol': symbol,
                        'timestamp': str(filtered_df_1d1y.index[i]),
                        'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                        'zones': zones_1m1y_formatted
                    })
                
                # Adding yesterday's close signal conditions
                if i < len(filtered_yesterday_buy_signal) and filtered_yesterday_buy_signal.iloc[i]:
                    signals.append({
                        'signal_type': 'Yesterday Close > AD (CM)',
                        'symbol': symbol,
                        'timestamp': str(filtered_df_1d1y.index[i]),
                        'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                        'zones': zones_1m1y_formatted
                    })
                
                if i < len(filtered_yesterday_sell_signal) and filtered_yesterday_sell_signal.iloc[i]:
                    signals.append({
                        'signal_type': 'Yesterday Close < BD (CM)',
                        'symbol': symbol,
                        'timestamp': str(filtered_df_1d1y.index[i]),
                        'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                        'zones': zones_1m1y_formatted
                    })


        elif filter_type == "LM":  # Last Month 
                    last_month = (datetime.now().replace(day=1) - timedelta(days=1)).strftime('%Y-%m')
                    filtered_df_1d1y = parsed_df_1d1y[parsed_df_1d1y.index.strftime('%Y-%m') == last_month]
                    filtered_buy_signal_1d = buy_signal_1d[parsed_df_1d1y.index.strftime('%Y-%m') == last_month]
                    filtered_sell_signal_1d = sell_signal_1d[parsed_df_1d1y.index.strftime('%Y-%m') == last_month]
                    filtered_yesterday_buy_signal = yesterday_close_buy_signal[parsed_df_1d1y.index.strftime('%Y-%m') == last_month]
                    filtered_yesterday_sell_signal = yesterday_close_sell_signal[parsed_df_1d1y.index.strftime('%Y-%m') == last_month]

                    for i in range(len(filtered_df_1d1y)):
                        zones_1m1y_formatted = {k: round(v, 2) for k, v in zones_1m1y.items()}
                        
                        if i < len(filtered_buy_signal_1d) and filtered_buy_signal_1d.iloc[i]:
                            signals.append({
                                'signal_type': '1d Buy Signal (LM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })
                        
                        if i < len(filtered_sell_signal_1d) and filtered_sell_signal_1d.iloc[i]:
                            signals.append({
                                'signal_type': '1d Sell Signal (LM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })
                        
                        # Adding yesterday's close signal conditions
                        if i < len(filtered_yesterday_buy_signal) and filtered_yesterday_buy_signal.iloc[i]:
                            signals.append({
                                'signal_type': 'Yesterday Close > AD (LM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })
                        
                        if i < len(filtered_yesterday_sell_signal) and filtered_yesterday_sell_signal.iloc[i]:
                            signals.append({
                                'signal_type': 'Yesterday Close < BD (LM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })


        elif filter_type == "LLM":  # Last Last Month (This Month - 2)
                    two_months_ago = datetime.now().replace(day=1) - relativedelta(months=2)
                    two_months_ago = two_months_ago.strftime('%Y-%m')
                    filtered_df_1d1y = parsed_df_1d1y[parsed_df_1d1y.index.strftime('%Y-%m') == two_months_ago]
                    if filtered_df_1d1y.empty:
                        continue
                    filtered_buy_signal_1d = buy_signal_1d[parsed_df_1d1y.index.strftime('%Y-%m') == two_months_ago]
                    filtered_sell_signal_1d = sell_signal_1d[parsed_df_1d1y.index.strftime('%Y-%m') == two_months_ago]
                    filtered_yesterday_buy_signal = yesterday_close_buy_signal[parsed_df_1d1y.index.strftime('%Y-%m') == two_months_ago]
                    filtered_yesterday_sell_signal = yesterday_close_sell_signal[parsed_df_1d1y.index.strftime('%Y-%m') == two_months_ago]

                    for i in range(len(filtered_df_1d1y)):
                        zones_1m1y_formatted = {k: round(v, 2) for k, v in zones_1m1y.items()}
                                
                        if i < len(filtered_buy_signal_1d) and filtered_buy_signal_1d.iloc[i]:
                            signals.append({
                                'signal_type': '1d Buy Signal (LLM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })
                                
                        if i < len(filtered_sell_signal_1d) and filtered_sell_signal_1d.iloc[i]:
                            signals.append({
                                'signal_type': '1d Sell Signal (LLM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })
                                
                        # Adding yesterday's close signal conditions
                        if i < len(filtered_yesterday_buy_signal) and filtered_yesterday_buy_signal.iloc[i]:
                            signals.append({
                                'signal_type': 'Yesterday Close > AD (LLM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })
                                
                        if i < len(filtered_yesterday_sell_signal) and filtered_yesterday_sell_signal.iloc[i]:
                            signals.append({
                                'signal_type': 'Yesterday Close < BD (LLM)',
                                'symbol': symbol,
                                'timestamp': str(filtered_df_1d1y.index[i]),
                                'close_price': round(filtered_df_1d1y['Close'].iloc[i], 2),
                                'zones': zones_1m1y_formatted
                            })




        elif filter_type.startswith("CD"):  # Current Day or Previous Days
            if filter_type == "CD":
                target_date = datetime.now().strftime('%Y-%m-%d')
            else:
                days_back = int(filter_type.split('-')[1])
                target_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            filtered_df_15m1d = parsed_df_15m1d[parsed_df_15m1d.index.strftime('%Y-%m-%d') == target_date]
            filtered_buy_signal_15m = buy_signal_15m[parsed_df_15m1d.index.strftime('%Y-%m-%d') == target_date]
            filtered_sell_signal_15m = sell_signal_15m[parsed_df_15m1d.index.strftime('%Y-%m-%d') == target_date]

            for i in range(len(filtered_df_15m1d)):
                zones_1d1m_formatted = {k: round(v, 2) for k, v in zones_1d1m.items()}
                
                if i < len(filtered_buy_signal_15m) and filtered_buy_signal_15m.iloc[i]:
                    signals.append({
                        'signal_type': f'15m Buy Signal ({filter_type})',
                        'symbol': symbol,
                        'timestamp': str(filtered_df_15m1d.index[i]),
                        'close_price': round(filtered_df_15m1d['Close'].iloc[i], 2),
                        'zones': zones_1d1m_formatted
                    })
                
                if i < len(filtered_sell_signal_15m) and filtered_sell_signal_15m.iloc[i]:
                    signals.append({
                        'signal_type': f'15m Sell Signal ({filter_type})',
                        'symbol': symbol,
                        'timestamp': str(filtered_df_15m1d.index[i]),
                        'close_price': round(filtered_df_15m1d['Close'].iloc[i], 2),
                        'zones': zones_1d1m_formatted
                    })
        else:
            signals.append({
                'signal_type': 'Invalid filter',
                'symbol': symbol,
                'message': f"Invalid filter type: {filter_type}"
            })
    
    return signals


def get_signals(symbol, filters):
    symbol = (symbol or "").strip()
    if not symbol:
        return [{"signal_type": "Error", "message": "Empty symbol"}]

    try:
        data_json_15m1d = fetch_data15m1d(symbol)
        data_json_1d1m = fetch_data1d1m(symbol)
        data_json_1m1y = fetch_data1m1y(symbol)
        data_json_1d1y = fetch_data1d1y(symbol)

        parsed_df_15m1d = parse_json_data(data_json_15m1d, symbol)
        parsed_df_1d1m = parse_json_data(data_json_1d1m, symbol)
        parsed_df_1m1y = parse_json_data(data_json_1m1y, symbol)
        parsed_df_1d1y = parse_json_data(data_json_1d1y, symbol)

        zones_1d1m = calculate_zones(parsed_df_1d1m)
        zones_1m1y = calculate_zones_monthly(parsed_df_1m1y)
        if zones_1m1y is None:
            print(f"Insufficient monthly data to calculate zones for symbol: {symbol}")
            return [{"signal_type": "Error", "message": f"Insufficient monthly data for symbol {symbol}"}]

        volu1_15m, vold1_15m = calculate_volume_conditions(parsed_df_15m1d)
        monthly_volu1, monthly_vold1 = calculate_volume_conditions_monthly(parsed_df_1d1y)

        buy_signal_15m, sell_signal_15m = generate_signals15(parsed_df_15m1d, zones_1d1m, volu1_15m, vold1_15m)
        buy_signal_1d, sell_signal_1d, yesterday_close_buy_signal, yesterday_close_sell_signal = generate_signalsd(parsed_df_1d1y, zones_1m1y, monthly_volu1, monthly_vold1)


        signals = collect_signals(symbol, filters, parsed_df_15m1d, parsed_df_1d1y,buy_signal_15m, sell_signal_15m, buy_signal_1d, sell_signal_1d,yesterday_close_buy_signal, yesterday_close_sell_signal,zones_1d1m, zones_1m1y)

        return signals

    except Exception as e:
        return [{"signal_type": "Error", "message": f"{symbol}: {e}"}]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python get_stock_data.py <symbol> <filter1> <filter2> ...")
        sys.exit(1)
    symbol = sys.argv[1]
    filters = sys.argv[2:]
    signals = get_signals(symbol, filters)
    #
    # print(json.dumps(signals, indent=2))
