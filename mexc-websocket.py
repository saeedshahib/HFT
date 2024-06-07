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


def handle_message(message):
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



# SPOT V3

ws_spot_client = spot.WebSocket(api_key=api_key, api_secret=api_secret)

# create websocket connection to public channel (spot@public.deals.v3.api@BTCUSDT)
# all messages will be handled by function `handle_message`
# ws_spot_client.deals_stream(handle_message, "BTCUSDT")
ws_spot_client.limit_depth_stream(handle_message, 'MOVRUSDT', 5)
ws_spot_client.limit_depth_stream(handle_message, 'HIGHUSDT', 5)
ws_spot_client.limit_depth_stream(handle_message, 'TRUUSDT', 5)
ws_spot_client.limit_depth_stream(handle_message, 'BTCUSDT', 5)
ws_spot_client.limit_depth_stream(handle_message, 'ETHUSDT', 5)
ws_spot_client.limit_depth_stream(handle_message, 'XRPUSDT', 5)
# ws_spot_client.deals_stream(handle_message, "ETHUSDT")
# ws_spot_client.deals_stream(handle_message, "AVAXUSDT")

while True:
    time.sleep(0.1)
