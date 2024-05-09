import math
from decimal import Decimal


def truncate(amount, decimals, get_ceil=False):
    if decimals is None:
        return amount
    if decimals < 0:
        multiplier = 10 ** (-decimals)
        return (Decimal(str(amount)) // multiplier) * multiplier
    multiplier = 10 ** decimals
    if get_ceil:
        return Decimal(str(math.ceil(amount * multiplier) / multiplier))
    else:
        return Decimal(int(amount * multiplier)) / multiplier
