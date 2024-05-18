import time

from .api_manager_interface import APIManagerInterface
from pybit.unified_trading import HTTP

from django.conf import settings


class Bybit(APIManagerInterface):
    def __init__(self):
        demo = True
        api_key = settings.BYBIT_API_KEY if demo is False else settings.BYBIT_API_KEY_TESTNET
        api_secret = settings.BYBIT_API_SECRET if demo is False else settings.BYBIT_API_SECRET_TESTNET
        self.session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
            demo=demo
        )

    def get_recent_candle(self, symbol, start=None, end=None):
        return self.session.get_kline(category='linear', symbol=symbol, interval='1',
                                      start=start*1000, end=end*1000, limit=1000)['result']['list']

    def get_balance(self, currency):
        return self.session.get_coin_balance(accountType='UNIFIED', coin=currency)['result']['balance']['walletBalance']

    def place_market_order(self, symbol, side, amount, order_type, unique_id, leverage=0, take_profit=None,
                           stop_loss=None):
        # order_type = 'Market'
        # side = 'Buy' or 'Sell'
        data = dict(category='linear', symbol=symbol, side=side, orderType=order_type,
                    qty=str(amount), orderLinkId=str(unique_id), isLeverage=leverage,
                    takeProfit=str(take_profit), stopLoss=str(stop_loss))
        print(data)
        return self.session.place_order(**data)

    def get_order_details(self, unique_id):
        return self.session.get_order_history(category='linear', orderLinkId=str(unique_id))['result']['list'][0]

    def get_position_details(self, symbol):
        return self.session.get_positions(category='linear', symbol=symbol)

    def get_order_book(self, symbol):
        return self.session.get_orderbook(category="linear", symbol=symbol)

    def close_position(self, symbol, side):
        return self.session.place_order(category='linear', symbol=symbol, qty='0', reduceOnly=True,
                                        closeOnTrigger=True, orderType='Market', side=side)

    def set_sl_tp(self, symbol, tp_price, sl_price):
        return self.session.set_trading_stop(category='linear', symbol=symbol, orderType='Market',
                                             takeProfit=str(tp_price), stopLoss=str(sl_price))
