import os
from decimal import Decimal
import time

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
django.setup()

from binance import ThreadedWebsocketManager

from utils.redis import global_redis_instance
from trading.models import ArbitragePosition, Market


def main():
    twm = ThreadedWebsocketManager()
    # start is required to initialise its internal loop
    twm.start()

    def handle_socket_message(msg):
        symbol = msg['data']['s']
        first_currency_symbol = str(symbol).replace('USDT', '')
        binance_price = Decimal(msg['data']['p'])
        # print(f"message type: {msg['e']}")
        mexc_ask_price = Decimal(global_redis_instance.get(name=f'{first_currency_symbol}USDC_ask_spot_mexc'))
        difference = (binance_price - mexc_ask_price) / mexc_ask_price
        # mexc_bid_price = Decimal(global_redis_instance.get(name=f'{symbol}_bid_spot_mexc'))

        if difference >= Decimal('0.002'):
            print(difference)
            print(f"arbitrage found in {symbol}, binance price is "
                  f"{binance_price} and mexc ask price is {mexc_ask_price}")
            mexc_market = Market.objects.get(first_currency__symbol=first_currency_symbol,
                                             second_currency__symbol="USDC",
                                             exchange=Market.Exchange.MEXC.value)
            binance_market = Market.objects.get(symbol=symbol, exchange=Market.Exchange.Binance.value)
            ArbitragePosition.open_position_if_not_open(source_price=mexc_ask_price, source_market=mexc_market,
                                                        target_price=binance_price, target_market=binance_market)

    # or a multiplex socket can be started like this
    # see Binance docs for stream names
    streams = ['avaxusdt@aggTrade', 'btcusdt@aggTrade', 'ethusdt@aggTrade',
               'xrpusdt@aggTrade', 'wavesusdt@aggTrade']
    twm.start_multiplex_socket(callback=handle_socket_message, streams=streams)

    twm.join()


if __name__ == "__main__":
    main()
