

class MEXC:
    def __init__(self, api_key, api_secret):
        pass

    def get_recent_candle(self):
        pass

    def get_balance(self, wallet, symbol=None):
        pass

    def place_market_order(self, symbol, side, amount, market_type, unique_id):
        pass

    def get_order_details(self, unique_id):
        pass

    def get_position_details(self, symbol):
        pass

    def get_order_book(self, symbol):
        pass

    def close_position(self, symbol):
        pass