import os
import django
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from pybit.unified_trading import WebSocket
from time import sleep

from trading.models import *
from utils.redis import *

ws = WebSocket(
    testnet=False,
    channel_type="linear",
)

TIME_WINDOW = 5
sensitivities = dict(Strategy.objects.filter(active=True).values_list('market__symbol', 'sensitivity'))


def add_ticker(symbol, price, timestamp):
    name = get_queue_name(symbol)
    global_redis_instance.zadd(name, {f'{{"price": {price}, "timestamp": {timestamp}}}': timestamp})
    cutoff = timestamp - TIME_WINDOW * 1000
    global_redis_instance.zremrangebyscore(name, '-inf', cutoff)
    change_percent = get_change_percent(symbol=symbol)
    print(timestamp)
    if abs(change_percent) > Decimal(sensitivities[symbol]):
        print(change_percent)
        strategy = Strategy.objects.get(market__symbol=symbol, active=True,
                                        strategy_type=Strategy.Type.WebsocketChange.value)
        if Position.objects.filter(strategy=strategy, status__in=[Position.Status.OPEN.value,
                                                                  Position.Status.Initiated.value]).exists():
            return
        if change_percent > strategy.sensitivity:
            Position.objects.create(strategy=strategy, side=Position.Side.Long.value)
        elif change_percent < strategy.sensitivity:
            Position.objects.create(strategy=strategy, side=Position.Side.Short.value)
        print(change_percent, symbol, timestamp)


def handle_message(message):
    try:
        symbol = message['data']['symbol']
        price = message['data']['indexPrice']
        timestamp = message['ts']
        add_ticker(symbol=symbol, price=price, timestamp=timestamp)
    except Exception as ve:
        print(ve)


symbols = list(Strategy.objects.filter(active=True).values_list('market__symbol', flat=True))


ws.ticker_stream(symbols, handle_message)

while True:
    sleep(1)
