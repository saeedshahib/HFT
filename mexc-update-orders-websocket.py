import os
import time
import traceback
from decimal import Decimal

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from django.conf import settings
from pymexc import spot
from trading.models import ArbitragePosition, Market, Asset, Currency

api_key = settings.MEXC_API_KEY
api_secret = settings.MEXC_API_SECRET


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
        currency, _ = Currency.objects.get_or_create(symbol=data['a'], defaults=dict(precision=8))
        value = Decimal(data['f'])
        asset, _ = Asset.objects.update_or_create(currency=currency, exchange=Market.Exchange.MEXC.value,
                                                  defaults={'value': value})
    except Exception:
        print(traceback.print_exc())


ws_spot_client = spot.WebSocket(api_key=api_key, api_secret=api_secret)


def subscribe():
    ws_spot_client.account_orders(handle_order_update_message)
    ws_spot_client.account_update(handle_assets_message)


subscribe()


while True:
    time.sleep(5)
