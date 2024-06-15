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
from trading.models import ArbitragePosition, Order
from api_manager.mexc_api import MEXCSpot

api_key = settings.MEXC_API_KEY
api_secret = settings.MEXC_API_SECRET
mexc_spot = MEXCSpot()


def get_and_update_orders():
    try:
        pending_arbitrages = ArbitragePosition.objects.filter(status__in=
                                                              [ArbitragePosition.ArbitrageStatus.Pending.value,
                                                               ArbitragePosition.ArbitrageStatus.Open.value,
                                                               ArbitragePosition.ArbitrageStatus.CloseRequested])
        order_ids = list(pending_arbitrages.values_list('open_order', 'close_order'))
        for order_ids_tuple in order_ids:
            for order_id in order_ids_tuple:
                if order_id is None:
                    continue
                order = Order.objects.get(id=order_id)
                if order.status in [Order.Status.PENDING.value, Order.Status.PARTIALLY_FIELD.value]:
                    print(order.market.symbol, order_id)
                    data = mexc_spot.order_details(symbol=order.market.symbol, order_id=order_id)
                    print(data)
                    if data['status'] in ['FILLED', 'PARTIALLY_CANCELED', 'CANCELED']:
                        ArbitragePosition.update_status_based_on_websocket_payload(
                            order_id=order_id,
                            filled_amount=Decimal(str(data['executedQty'])),
                            avg_price=Decimal(str(data['price']))
                        )

    except Exception:
        print(traceback.print_exc())


while True:
    get_and_update_orders()
    time.sleep(10)
