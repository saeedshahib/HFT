import hashlib
import hmac
import json
import time
import urllib.parse
from .api_manager_interface import APIManagerInterface

import requests
import datetime

from django.conf import settings
from pymexc import spot


class MEXCFutures(APIManagerInterface):
    base_url = 'https://contract.mexc.com/'
    api_key = settings.MEXC_API_KEY
    api_secret = settings.MEXC_API_SECRET

    def generate_headers(self, http_method,  params, api_key, api_secret):
        # Get current time for 'Request-Time' header
        request_time = int(time.time() * 1000)  # In milliseconds

        # Create the parameter string based on the HTTP method
        if http_method in ['GET', 'DELETE']:
            # Sort the parameters by key and join them with '&'
            param_str = '&'.join(
                [f"{urllib.parse.quote_plus(k)}={urllib.parse.quote_plus(str(v))}" for k, v in sorted(params.items())]
            ) if params else ""
        elif http_method == 'POST':
            # Convert the parameters to a JSON string
            param_str = json.dumps(params) if params else ""
        else:
            raise ValueError("Unsupported HTTP method")

        # Create the target string to be signed
        target_str = api_key + str(request_time) + param_str

        # Generate HMAC SHA256 signature
        signature = hmac.new(api_secret.encode('utf-8'), target_str.encode('utf-8'), hashlib.sha256).hexdigest()

        # Create the headers dictionary
        headers = {
            'ApiKey': api_key,
            'Request-Time': str(request_time),
            'Signature': signature,
            'Content-Type': 'application/json',
        }

        return headers

    def get_recent_candle(self, symbol):
        start = int(time.time()) - 120
        http_method = 'GET'
        headers = self.generate_headers(http_method=http_method, params={},
                                        api_key=self.api_key, api_secret=self.api_secret)
        endpoint = f'api/v1/contract/kline/{symbol}?start={start}'
        response = requests.get(self.base_url + endpoint, headers=headers)
        return response.json()['data']

    def get_balance(self, currency):
        http_method = 'GET'
        endpoint = f'api/v1/private/account/asset/{currency}'
        headers = self.generate_headers(http_method=http_method, params={},
                                        api_key=self.api_key, api_secret=self.api_secret)
        response = requests.get(self.base_url + endpoint, headers=headers)
        return response.json()

    def place_market_order(self, symbol, side, amount, order_type, unique_id, price=None, take_profit=None,
                           stop_loss=None):
        http_method = 'POST'
        endpoint = f'api/v1/private/order/submit'
        params = {
            'symbol': symbol,
            'price': str(price),
            'vol': str(amount),
            'side': side,
            'type': order_type,
            'externalOid': unique_id,
            'openType': 1
        }
        headers = self.generate_headers(http_method=http_method, params=params,
                                        api_key=self.api_key, api_secret=self.api_secret)
        response = requests.post(self.base_url + endpoint, headers=headers, json=params)
        return response.json()

    def get_order_details(self, unique_id, symbol=None):
        http_method = 'GET'
        endpoint = f'api/v1/private/order/external/{symbol}/{unique_id}'
        headers = self.generate_headers(http_method=http_method, params={},
                                        api_key=self.api_key, api_secret=self.api_secret)
        response = requests.get(self.base_url + endpoint, headers=headers)
        return response.json()

    def get_position_details(self, symbol):
        http_method = 'GET'
        endpoint = f'api/v1/private/position/open_positions?symbol={symbol}'
        params = {
            'symbol': symbol,
        }
        headers = self.generate_headers(http_method=http_method, params=params,
                                        api_key=self.api_key, api_secret=self.api_secret)
        response = requests.get(self.base_url + endpoint, headers=headers)
        return response.json()['data']

    def get_order_book(self, symbol):
        http_method = 'GET'
        endpoint = f'api/v1/contract/depth/{symbol}?limit=10'
        headers = self.generate_headers(http_method=http_method, params={},
                                        api_key=self.api_key, api_secret=self.api_secret)
        response = requests.get(self.base_url + endpoint, headers=headers)
        return response.json()

    def close_position(self, symbol, side):
        raise NotImplementedError

    def set_sl_tp(self, symbol, tp_price, sl_price):
        raise NotImplementedError

    def place_immediate_or_cancel_order(self, symbol, side, amount, order_type, unique_id, price):
        raise NotImplementedError


class MEXCSpot(APIManagerInterface):
    api_key = settings.MEXC_API_KEY
    api_secret = settings.MEXC_API_SECRET
    spot_obj = spot.HTTP(api_key=api_key, api_secret=api_secret)

    def place_immediate_or_cancel_order(self, symbol, side, amount, order_type, unique_id, price):
        self.spot_obj.new_order(symbol=symbol, side=str(side).upper(), order_type=str(order_type).upper(),
                                quantity=amount, price=price, new_client_order_id=unique_id)

    def place_market_order(self, symbol, side, amount, order_type, unique_id, take_profit=None, stop_loss=None):
        self.spot_obj.new_order(symbol=symbol, side=str(side).upper(), order_type=str(order_type).upper(),
                                quantity=amount, new_client_order_id=str(unique_id))

    def cancel_order(self, symbol, order_id):
        self.spot_obj.cancel_order(symbol=symbol, new_client_order_id=order_id, orig_client_order_id=order_id)

    def order_details(self, symbol, order_id):
        return self.spot_obj.query_order(symbol=symbol, orig_client_order_id=order_id)
