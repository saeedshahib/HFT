import json
from decimal import Decimal

import redis

global_redis_instance = redis.Redis(host='localhost', port=6379, decode_responses=True)


def get_queue_name(symbol):
    return f'{symbol}_bybit_queue'


def get_first(symbol):
    result = global_redis_instance.zrange(get_queue_name(symbol=symbol), 0, 0)
    return result[0] if result else None


def get_last(symbol):
    result = global_redis_instance.zrange(get_queue_name(symbol=symbol), -1, -1)
    return result[0] if result else None


def get_change_percent(symbol):
    first_price = Decimal(str(json.loads(get_first(symbol))['ask_price']))
    last_price = Decimal(str(json.loads(get_last(symbol))['ask_price']))
    change_percent = (last_price - first_price) / first_price * 100
    return change_percent


def get_price(symbol, side):
    from trading.models import Position
    if side == Position.Side.Long.value:
        return Decimal(str(json.loads(get_last(symbol))['bid_price']))
    elif side == Position.Side.Short.value:
        return Decimal(str(json.loads(get_first(symbol))['ask_price']))
    else:
        raise Exception('Invalid side')


def get_balance(symbol, exchange):
    print(global_redis_instance.get(name=f'{symbol}_{exchange}_balance'))
    return Decimal(str(global_redis_instance.get(name=f'{symbol}_{exchange}_balance')))
