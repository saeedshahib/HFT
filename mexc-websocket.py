import os
import time
import traceback
import json
from decimal import Decimal

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from django.conf import settings
from pymexc import spot, futures
from utils.redis import global_redis_instance
from trading.models import ArbitragePosition, Market, Asset, Currency

api_key = settings.MEXC_API_KEY
api_secret = settings.MEXC_API_SECRET


def handle_order_book_message(message):
    timestamp = int(message['t']) // 1000
    symbol = message['s']
    ask_price = message['d']['asks'][0]['p']
    bid_price = message['d']['bids'][0]['p']
    data = json.dumps(dict(ask_price=ask_price, bid_price=bid_price, timestamp=timestamp))
    # handle websocket message
    global_redis_instance.set(name=f'{symbol}_price_spot_mexc', value=data)
    try:
        mexc_market = Market.objects.get(symbol=symbol, exchange=Market.Exchange.MEXC.value)
        open_positions = ArbitragePosition.objects.filter(source_market=mexc_market,
                                                          status=ArbitragePosition.ArbitrageStatus.Open.value)
        for position in open_positions:
            position.check_and_close_position(reached_price=bid_price)
    except Exception as ve:
        print(traceback.print_exc())


ws_spot_client = spot.WebSocket(api_key=api_key, api_secret=api_secret)


def subscribe():
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'AVAXUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'XRPUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'WAVESUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'OPUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'FTTUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'MATICUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'APEUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'JASMYUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'LUNAUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'LUNCUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'SHIBUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'FTMUSDC', 5)
    ws_spot_client.limit_depth_stream(handle_order_book_message, 'CELUSDC', 5)


subscribe()


while True:
    try:
        print(json.loads(global_redis_instance.get(name=f'XRPUSDC_price_spot_mexc'))['ask_price'])
        time.sleep(5)
    except TypeError:
        pass
