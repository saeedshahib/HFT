import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

import time

from trading.models import Position

if __name__ == '__main__':
    while True:
        try:
            Position.check_active_strategies_and_open_position()
            Position.check_open_positions()
            time.sleep(5)
        except Exception as ve:
            print(ve)
