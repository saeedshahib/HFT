import os
import time

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from api_manager.bybit_api import Bybit
from trading.indicator_manager import trading_view_indicator


def run_backtest_arg_manager():
    bybit = Bybit()
    interval = '240'
    batch_period = 1000 * 60 * int(interval)
    end_time = int(time.time())
    start_time = end_time - batch_period
    print(start_time)
    print(end_time)
    for i in range(20):
        recent_candles = bybit.get_recent_candle(symbol="BTCUSDT", start=start_time, end=end_time, interval=interval)
        data = trading_view_indicator.generate_df(recent_candles)
        trading_view_indicator.run_backtest(df=data)
        start_time = start_time - batch_period
        end_time = end_time - batch_period
        print()


if __name__ == '__main__':
    run_backtest_arg_manager()