import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

import time

from trading.models import *
from utils.redis import global_redis_instance

if __name__ == '__main__':
    while True:
        markets = Strategy.objects.filter(active=True).values_list('market', flat=True)
        for market in markets:
            try:
                market = Market.objects.get(id=market)
                exchange_obj = market.get_exchange_object()
                # balance = exchange_obj.get_balance(currency=market.second_currency.symbol)
                balance = '1000'
                print("balance: ", balance)
                global_redis_instance.set(name=f'{market.second_currency.symbol}_{market.exchange}_balance',
                                          value=balance)
            except Exception as e:
                print(e)
        time.sleep(60)
