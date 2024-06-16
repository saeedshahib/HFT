import os
import traceback
import json
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
        try:
            symbol = msg['data']['s']
            first_currency_symbol = str(symbol).replace('USDT', '')
            binance_ask_price = Decimal(msg['data']['a'])
            binance_bid_price = Decimal(msg['data']['b'])
            # print(f"message type: {msg['e']}")
            mexc_price_data = json.loads(global_redis_instance.get(name=f'{first_currency_symbol}USDC_price_spot_mexc'))
            mexc_ask_price = Decimal(mexc_price_data['ask_price'])
            mexc_bid_price = Decimal(mexc_price_data['bid_price'])
            spread = (mexc_ask_price - mexc_bid_price) / mexc_bid_price
            profit = Decimal('0.001')
            commission = Decimal('0.002')
            difference = (binance_ask_price - mexc_ask_price) / mexc_ask_price
            # mexc_bid_price = Decimal(global_redis_instance.get(name=f'{symbol}_bid_spot_mexc'))

            if difference >= spread + commission + profit:
                print(difference, spread)
                print(f"arbitrage found in {symbol}, binance price is "
                      f"{binance_ask_price} and mexc ask price is {mexc_ask_price}")
                source_price = mexc_ask_price
                target_price = binance_bid_price * (1 - spread)
                if target_price <= source_price * (1 + commission):
                    raise Exception(f"not profitable arbitrage!, target price is {target_price} and source_price is "
                                    f"{source_price} and with commission is {source_price * (1 + commission)}")
                mexc_market = Market.objects.get(first_currency__symbol=first_currency_symbol,
                                                 second_currency__symbol="USDC",
                                                 exchange=Market.Exchange.MEXC.value)
                binance_market = Market.objects.get(symbol=symbol, exchange=Market.Exchange.Binance.value)
                ArbitragePosition.open_position_if_not_open(source_price=source_price, source_market=mexc_market,
                                                            target_price=target_price, target_market=binance_market)
        except Exception as e:
            print(traceback.format_exc())

    # or a multiplex socket can be started like this
    # see Binance docs for stream names
    websocket_type = "bookTicker"
    streams = [f'avaxusdt@{websocket_type}', f'xrpusdt@{websocket_type}', f'wavesusdt@{websocket_type}',
               f'opusdt@{websocket_type}', f'fttusdt@{websocket_type}', f'maticusdt@{websocket_type}',
               f'apeusdt@{websocket_type}', f'jasmyusdt@{websocket_type}', f'lunausdt@{websocket_type}',
               f'luncusdt@{websocket_type}', f'shibusdt@{websocket_type}', f'ftmusdt@{websocket_type}',
               f'celusdt@{websocket_type}', ]
    twm.start_multiplex_socket(callback=handle_socket_message, streams=streams)

    twm.join()


if __name__ == "__main__":
    main()
