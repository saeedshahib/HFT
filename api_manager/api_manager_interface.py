import hashlib
import hmac
import json
import time
import urllib.parse
import abc


class APIManagerInterface(metaclass=abc.ABCMeta):
    base_url = ''
    api_key = ''
    api_secret = ''

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
        raise NotImplementedError

    def get_balance(self, currency):
        raise NotImplementedError

    def place_market_order(self, symbol, side, amount, order_type, unique_id, take_profit=None, stop_loss=None):
        raise NotImplementedError

    def get_order_details(self, unique_id):
        raise NotImplementedError

    def get_position_details(self, symbol):
        raise NotImplementedError

    def get_order_book(self, symbol):
        raise NotImplementedError

    def close_position(self, symbol, side):
        raise NotImplementedError

    def set_sl_tp(self, symbol, tp_price, sl_price):
        raise NotImplementedError
