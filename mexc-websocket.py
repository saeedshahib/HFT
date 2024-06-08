import os
import time
import traceback

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from django.conf import settings
from pymexc import spot, futures
from utils.redis import global_redis_instance
from trading.models import ArbitragePosition, Market

api_key = settings.MEXC_API_KEY
api_secret = settings.MEXC_API_SECRET


def handle_order_book_message(message):
    symbol = message['s']
    ask_price = message['d']['asks'][0]['p']
    bid_price = message['d']['bids'][0]['p']
    print(symbol, ask_price)
    # handle websocket message
    global_redis_instance.set(name=f'{symbol}_ask_spot_mexc', value=ask_price)
    global_redis_instance.set(name=f'{symbol}_bid_spot_mexc', value=bid_price)
    try:
        mexc_market = Market.objects.get(symbol=symbol, exchange=Market.Exchange.MEXC.value)
        open_positions = ArbitragePosition.objects.filter(source_market=mexc_market,
                                                          status=ArbitragePosition.ArbitrageStatus.Open.value)
        for position in open_positions:
            position.check_and_close_position(reached_price=bid_price)
    except Exception as ve:
        print(traceback.print_exc())


def handle_order_update_message(message):
    #  Todo change arbitrage status to open if an order gets filled (check status it should be finalized)
    symbol = message['s']
    data = message['d']
    unique_id = data['c']

    ArbitragePosition.update_status_based_on_websocket_payload(order_id=unique_id, symbol=symbol)
    print(message)


def handle_assets_message(message):
    #  Todo save your assets to db and update them here
    print(message)


ws_spot_client = spot.WebSocket(api_key=api_key, api_secret=api_secret)

ws_spot_client.limit_depth_stream(handle_order_book_message, 'MOVRUSDT', 5)
ws_spot_client.limit_depth_stream(handle_order_book_message, 'HIGHUSDT', 5)
ws_spot_client.limit_depth_stream(handle_order_book_message, 'TRUUSDT', 5)
ws_spot_client.limit_depth_stream(handle_order_book_message, 'BTCUSDT', 5)
ws_spot_client.limit_depth_stream(handle_order_book_message, 'ETHUSDT', 5)
ws_spot_client.limit_depth_stream(handle_order_book_message, 'XRPUSDT', 5)
ws_spot_client.account_orders(handle_order_update_message)
ws_spot_client.account_update(handle_assets_message)

while True:
    time.sleep(0.1)
