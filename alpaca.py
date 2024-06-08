import os
import time

import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from alpha_vantage.foreignexchange import ForeignExchange
import pandas as pd

# api_key = '1FT1CQXHAZMT6W45'
# fx = ForeignExchange(key=api_key)
#
# # Get intraday data for EUR/USD
# data, _ = fx.get_currency_exchange_intraday(from_symbol='EUR', to_symbol='USD', interval='1min', outputsize='full')
#
# # Convert to DataFrame
# df = pd.DataFrame.from_dict(data, orient='index')
# df.index = pd.to_datetime(df.index)
# df = df.astype(float)
# print(df.head())

import yfinance as yf

# Get historical data for EUR/USD
data = yf.download('EURUSD=X', interval='1m', period='5d')
print(data.head())

