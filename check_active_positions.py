import os
import traceback

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

import time

from trading.models import Position

if __name__ == '__main__':
    while True:
        try:
            last_three_candles = Position.check_active_strategies_and_open_position()
            Position.check_open_positions(last_three_candles)
            time.sleep(1)
        except Exception as ve:
            traceback.print_exc()
