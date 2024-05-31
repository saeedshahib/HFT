import os
import time

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from api_manager.bybit_api import Bybit
from trading.indicator_manager import strategy_manager, macd_and_rsi_manager, arg_manager


# Example usage
def run_backtest():
    bybit = Bybit()
    batch_period = 1000 * 60
    end_time = int(time.time())
    start_time = end_time - batch_period
    for i in range(20):
        print(start_time)
        print(end_time)
        recent_candles = bybit.get_recent_candle(symbol="1000PEPEUSDT", start=start_time, end=end_time)
        df = strategy_manager.generate_df(recent_candles)

        strategy_manager.generate_signals(df)
        # buy_signals = signals[signals['Buy'] == True]
        # sell_signals = signals[signals['Sell'] == True]

        start_time = start_time - batch_period
        end_time = end_time - batch_period
        print()


def run_backtest_macd_rsi():
    bybit = Bybit()
    batch_period = 1000 * 60
    end_time = int(time.time())
    start_time = end_time - batch_period
    for i in range(20):
        print(start_time)
        print(end_time)
        recent_candles = bybit.get_recent_candle(symbol="1000PEPEUSDT", start=start_time, end=end_time)
        data = strategy_manager.generate_df(recent_candles)

        macd_and_rsi_manager.run_backtest(data)
        # buy_signals = signals[signals['Buy'] == True]
        # sell_signals = signals[signals['Sell'] == True]

        start_time = start_time - batch_period
        end_time = end_time - batch_period
        print()
    print(macd_and_rsi_manager.pnl)


def run_backtest_arg_manager():
    bybit = Bybit()
    batch_period = 1000 * 60 * 240
    end_time = int(time.time())
    start_time = end_time - batch_period
    print(start_time)
    print(end_time)
    recent_candles = bybit.get_recent_candle(symbol="BTCUSDT", start=start_time, end=end_time)
    data = strategy_manager.generate_df(recent_candles)
    arg_manager.run_backtest(data)
    arg_manager.create_csv_from_signals()


if __name__ == '__main__':
    run_backtest_arg_manager()
    # run_backtest_macd_rsi()
    # print("tp_count: ", strategy_manager.tp_count)
    # print("sl_count: ", strategy_manager.sl_count)
    # print("pnl: ", strategy_manager.pnl)
