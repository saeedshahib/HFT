import os
import time
import traceback
from decimal import Decimal

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from trading.models import Market, Currency, Asset

usdc_currencies = dict(
    AVAX=2,
    XRP=2,
    WAVES=2,
    OP=2,
    FTT=4,
    MATIC=5,
    APE=3,
    JASMY=6,
    LUNA=4,
    LUNC=5,
    SHIB=10,
    FTM=7,
)
usdt_currencies = dict(
    CEL=4,
    CVX=3,
    GFT=6,
    AR=3,
    TAO=2,
    BNX=4,
    CTK=4,
    SUI=4,
)

usdt, _ = Currency.objects.get_or_create(symbol='USDT', defaults=dict(precision=8))
usdc, _ = Currency.objects.get_or_create(symbol='USDC', defaults=dict(precision=8))
Asset.objects.get_or_create(currency=usdc, exchange=Market.Exchange.MEXC.value, defaults=dict(value=Decimal('40')))
Asset.objects.get_or_create(currency=usdt, exchange=Market.Exchange.MEXC.value, defaults=dict(value=Decimal('40')))


def create_usdc_markets():
    for key, value in usdc_currencies.items():
        currency, _ = Currency.objects.get_or_create(symbol=key, precision=value)
        Market.objects.get_or_create(exchange=Market.Exchange.MEXC.value, symbol=f'{key}USDC', first_currency=currency,
                                     second_currency=usdc, market_type=Market.Type.Spot.value)
        Market.objects.get_or_create(exchange=Market.Exchange.Binance.value, symbol=f'{key}USDT', first_currency=currency,
                                     second_currency=usdt, market_type=Market.Type.Spot.value)


def create_usdt_markets():
    for key, value in usdt_currencies.items():
        currency, _ = Currency.objects.get_or_create(symbol=key, precision=value)
        Market.objects.get_or_create(exchange=Market.Exchange.MEXC.value, symbol=f'{key}USDT',
                                     first_currency=currency,
                                     second_currency=usdt, market_type=Market.Type.Spot.value)
        Market.objects.get_or_create(exchange=Market.Exchange.Binance.value, symbol=f'{key}USDT',
                                     first_currency=currency,
                                     second_currency=usdt, market_type=Market.Type.Spot.value)

if __name__ == '__main__':
    create_usdt_markets()
