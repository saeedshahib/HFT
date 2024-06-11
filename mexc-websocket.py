import os
import time
import traceback
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
    symbol = message['s']
    ask_price = message['d']['asks'][0]['p']
    bid_price = message['d']['bids'][0]['p']
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
    try:
        print(message)
        symbol = message['s']
        data = message['d']
        unique_id = int(data['c'])
        status = int(data['s'])
        filled_amount = Decimal(data['cv'])
        avg_price = Decimal(str(data['ap']))
        if status in [2, 4, 5]:
            ArbitragePosition.update_status_based_on_websocket_payload(order_id=unique_id,
                                                                       filled_amount=filled_amount,
                                                                       avg_price=avg_price)
    except Exception:
        print(traceback.print_exc())


def handle_assets_message(message):
    try:
        print(message)
        data = message['d']
        currency = Currency.objects.get(symbol=data['a'])
        value = Decimal(data['f'])
        asset, _ = Asset.objects.update_or_create(currency=currency, exchange=Market.Exchange.MEXC.value,
                                                  defaults={'value': value})
    except Exception:
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
    ws_spot_client.account_orders(handle_order_update_message)
    ws_spot_client.account_update(handle_assets_message)


while True:
    try:
        print(global_redis_instance.get(name=f'XRPUSDC_ask_spot_mexc'))
        time.sleep(5)
        subscribe()
    except:
        print(traceback.print_exc())
