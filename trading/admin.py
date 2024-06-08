from django.contrib import admin

from utils.admin import BaseAdminWithActionButtons
from trading.models import *

# Register your models here.


@admin.register(Currency)
class CurrencyAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in Currency._meta.fields]


@admin.register(Market)
class MarketAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in Market._meta.fields]


@admin.register(Order)
class OrderAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in Order._meta.fields]


@admin.register(Trade)
class TradeAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in Trade._meta.fields]


@admin.register(Strategy)
class StrategyAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in Strategy._meta.fields]


@admin.register(Position)
class PositionAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in Position._meta.fields]


@admin.register(ArbitragePosition)
class ArbitragePositionAdmin(BaseAdminWithActionButtons):
    list_display = [field.name for field in ArbitragePosition._meta.fields]
