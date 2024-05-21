import os
import time

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from api_manager.bybit_api import Bybit
from trading.indicator_manager import strategy_manager


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


if __name__ == '__main__':
    run_backtest()
    print("tp_count: ", strategy_manager.tp_count)
    print("sl_count: ", strategy_manager.sl_count)
    print("pnl: ", strategy_manager.pnl)
