import os
import time
import traceback
from decimal import Decimal

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from trading.models import Market, Currency

currencies = dict(
    OP=2,
    FTT=4,
    MATIC=5,
    APE=3,
    JASMY=6,
    LUNA=4,
    LUNC=5,
    SHIB=10,
    FTM=7,
    CEL=4
)
usdt = Currency.objects.get(symbol='USDT')
usdc = Currency.objects.get(symbol='USDC')

for key, value in currencies.items():
    currency = Currency.objects.create(symbol=key, precision=value)
    Market.objects.create(exchange=Market.Exchange.MEXC.value, symbol=f'{key}USDC', first_currency=currency,
                          second_currency=usdc, market_type=Market.Type.Spot.value)
    Market.objects.create(exchange=Market.Exchange.Binance.value, symbol=f'{key}USDT', first_currency=currency,
                         second_currency=usdt, market_type=Market.Type.Spot.value)
