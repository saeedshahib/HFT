import os
import django
from decimal import Decimal

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from api_manager.bybit_api import Bybit


def ema(data, window):
    return data.ewm(span=window, adjust=False).mean()


def bollinger_bands(data, window, no_of_std):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * no_of_std)
    lower_band = rolling_mean - (rolling_std * no_of_std)
    bands = pd.DataFrame({
        'Middle Band': rolling_mean,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    })
    return bands


def awesome_oscillator(high, low):
    median_price = (high + low) / 2
    ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
    return ao


def generate_signals(df, short_ema_window, bb_window, bb_std, ao_short_window, ao_long_window):
    df['3 EMA'] = ema(df['Close'], short_ema_window)
    bollinger = bollinger_bands(df['Close'], bb_window, bb_std)
    df['Bollinger Middle Band'] = bollinger['Middle Band']
    df['Bollinger Upper Band'] = bollinger['Upper Band']
    df['Bollinger Lower Band'] = bollinger['Lower Band']
    df['AO'] = awesome_oscillator(df['High'], df['Low'])
    previous_row = None
    for index, row in df.iterrows():
        if previous_row is not None:
            if (row['3 EMA'] > row['Bollinger Middle Band'] and previous_row['3 EMA'] <
                    previous_row['Bollinger Middle Band'] and row['AO'] > previous_row['AO']):
                print("buy", row['ts'])
            if (row['3 EMA'] < row['Bollinger Middle Band'] and previous_row['3 EMA'] >
                    previous_row['Bollinger Middle Band'] and row['AO'] < previous_row['AO']):
                print("sell", row['ts'])
        previous_row = row
    signals = pd.DataFrame(index=df.index)
    signals['Buy'] = (df['3 EMA'] > df['Bollinger Middle Band']) & (df['AO'] > 0)
    signals['Sell'] = (df['3 EMA'] < df['Bollinger Middle Band']) & (df['AO'] < 0)

    # Adding trading hours condition
    # trading_hours = df.index.to_series().between_time('03:00', '12:00')
    # signals['Buy'] = signals['Buy'] & trading_hours
    # signals['Sell'] = signals['Sell'] & trading_hours

    return signals


def check_trending_market(df, threshold=0.005):
    price_diff = df['Close'].pct_change()
    volatility = price_diff.rolling(window=20).std()
    trending = volatility > threshold
    return trending


def apply_take_profit(df, signals, take_profit_pips=10, ao_change_col='AO'):
    df['Pips'] = df['Close'].diff()
    take_profit_points = take_profit_pips / 10000  # Assuming a standard pip value

    df['AO Change'] = df[ao_change_col].diff()

    buy_tp = signals['Buy'] & ((df['Pips'].cumsum() >= take_profit_points) | (df['AO Change'] < 0))
    sell_tp = signals['Sell'] & ((-df['Pips'].cumsum() >= take_profit_points) | (df['AO Change'] > 0))

    signals['Buy'] = buy_tp
    signals['Sell'] = sell_tp

    return signals


# Example usage
bybit = Bybit()
recent_candles = bybit.get_recent_candle(symbol="TRBUSDT")
high = []
low = []
close = []
ts = []
for candle in reversed(recent_candles):
    close.append(float(candle[4]))
    high.append(float(candle[2]))
    low.append(float(candle[3]))
    ts.append(str(candle[0]))

data = {
    'Close': close,
    'High': high,
    'Low': low,
    'ts': ts
}
index = pd.date_range(start='2024-01-01 03:00', periods=len(close), freq='h')
df = pd.DataFrame(data, index=index)

short_ema_window = 3
bb_window = 20
bb_std = 3
ao_short_window = 5
ao_long_window = 34

signals = generate_signals(df, short_ema_window, bb_window, bb_std, ao_short_window, ao_long_window)
buy_signals = signals[signals['Buy'] == True]
sell_signals = signals[signals['Sell'] == True]
# trending_market = check_trending_market(df)
# signals['Buy'] = signals['Buy'] & trending_market
# signals['Sell'] = signals['Sell'] & trending_market


# signals_with_tp = apply_take_profit(df, signals)
# print(signals_with_tp)

# # Plot Closing Prices, Bollinger Bands, and EMA
# fig, axs = plt.subplots(3, figsize=(15, 12))
# axs[0].plot(df.index, df['Close'], label='Close Price')
# axs[0].plot(df.index, df['3 EMA'], label='3 EMA', linestyle='--')
# axs[0].plot(df.index, df['Bollinger Middle Band'], label='Bollinger Middle Band', linestyle='--')
# axs[0].fill_between(df.index, df['Bollinger Upper Band'], df['Bollinger Lower Band'], color='gray', alpha=0.3)
# axs[0].scatter(buy_signals.index, df.loc[buy_signals.index]['Close'], marker='^', color='g', label='Buy Signal', s=100)
# axs[0].scatter(sell_signals.index, df.loc[sell_signals.index]['Close'], marker='v', color='r', label='Sell Signal', s=100)
# axs[0].set_title('Closing Prices, Bollinger Bands, and EMA')
# axs[0].legend()
#
# # Plot Awesome Oscillator
# axs[1].bar(df.index, df['AO'], label='Awesome Oscillator', color=(df['AO'] > 0).map({True: 'g', False: 'r'}))
# axs[1].axhline(0, color='black', linewidth=0.5)
# axs[1].set_title('Awesome Oscillator')
# axs[1].legend()
#
# # Plot Buy and Sell signals
# axs[2].plot(df.index, df['Close'], label='Close Price')
# axs[2].scatter(buy_signals.index, df.loc[buy_signals.index]['Close'], marker='^', color='g', label='Buy Signal', s=100)
# axs[2].scatter(sell_signals.index, df.loc[sell_signals.index]['Close'], marker='v', color='r', label='Sell Signal', s=100)
# axs[2].set_title('Buy and Sell Signals')
# axs[2].legend()
#
# plt.tight_layout()
# plt.show()
