import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from decimal import Decimal
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from api_manager.bybit_api import Bybit
from trading.indicator_manager import strategy_manager
# Example dataframe
# df = pd.read_csv('your_data.csv')  # Load your data here
# df should have columns: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
bybit = Bybit()
batch_period = 1000 * 60 * 60
end_time = int(time.time())
start_time = end_time - batch_period
print(start_time)
print(end_time)
recent_candles = bybit.get_recent_candle(symbol="TRBUSDT", start=start_time, end=end_time)
df = strategy_manager.generate_df(recent_candles)

# Apply custom indicator
df = strategy_manager.custom_indicator(df)

# Example strategy: Buy when Top_Swing is detected and Sell when Bottom_Swing is detected
# Example strategy: Buy when Top_Swing is detected and Sell when Bottom_Swing is detected
df['Signal'] = np.where(df['Top_Swing'].notna(), 1, np.where(df['Bottom_Swing'].notna(), -1, 0))
df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')

# Backtest parameters
initial_cash = Decimal('10000')
tp_level = Decimal('0.02')  # 2% take profit
sl_level = Decimal('0.01')  # 1% stop loss

# Metrics
trades = []
win_count = Decimal('0')
loss_count = 0
tp_reached = 0
sl_reached = 0

# Backtest loop
df['Cash'] = initial_cash
df['Holdings'] = 0
df['Total'] = initial_cash

for i in range(1, len(df)):
    if df['Position'].iloc[i] == 1 and df['Position'].iloc[i-1] != 1:  # Buy
        entry_price = df['Close'].iloc[i]
        for j in range(i+1, len(df)):
            if df['High'].iloc[j] >= entry_price * (1 + tp_level):
                trades.append(tp_level * entry_price)
                win_count += 1
                tp_reached += 1
                break
            elif df['Low'].iloc[j] <= entry_price * (1 - sl_level):
                trades.append(-sl_level * entry_price)
                loss_count += 1
                sl_reached += 1
                break
        else:
            trades.append(df['Close'].iloc[-1] - entry_price)  # If no TP/SL reached
    elif df['Position'].iloc[i] == -1 and df['Position'].iloc[i-1] != -1:  # Sell
        entry_price = df['Close'].iloc[i]
        for j in range(i+1, len(df)):
            if df['Low'].iloc[j] <= entry_price * (1 - tp_level):
                trades.append(tp_level * entry_price)
                win_count += 1
                tp_reached += 1
                break
            elif df['High'].iloc[j] >= entry_price * (1 + sl_level):
                trades.append(-sl_level * entry_price)
                loss_count += 1
                sl_reached += 1
                break
        else:
            trades.append(entry_price - df['Close'].iloc[-1])  # If no TP/SL reached

# Calculate metrics
total_trades = len(trades)
winrate = win_count / total_trades if total_trades > 0 else 0
pnl = sum(trades)

print(f"Total Trades: {total_trades}")
print(f"Win Rate: {winrate * 100:.2f}%")
print(f"PnL: {pnl:.2f}")
print(f"Take Profit Reached: {tp_reached}")
print(f"Stop Loss Reached: {sl_reached}")

