from decimal import Decimal

from django.db import models

from HFT.models import BaseModel
from api_manager.manager import MEXC
# Create your models here.


class Account(BaseModel):
    class Exchange(models.TextChoices):
        MEXC = "MEXC"
        BingX = "BingX"
        Kucoin = "Kucoin"

    exchange = models.CharField(max_length=50, choices=Exchange.choices)
    api_key = models.CharField(max_length=50)
    api_secret = models.CharField(max_length=50)


class Market(BaseModel):
    class Type(models.TextChoices):
        Spot = "Spot"
        Margin = "Margin"
        Futures = "Futures"

    exchange = models.CharField(max_length=50, choices=Account.Exchange.choices)
    symbol = models.CharField(max_length=50)
    market_type = models.CharField(max_length=50, choices=Type.choices)

    def get_exchange_object(self):
        if self.exchange == Account.Exchange.MEXC:
            account = Account.objects.get(exchange=self.exchange)
            return MEXC(api_key=account.api_key, api_secret=account.api_secret)
        else:
            raise NotImplementedError


class Order(BaseModel):
    class Side(models.TextChoices):
        BUY = "BUY"
        SELL = "SELL"

    class OrderType(models.TextChoices):
        LIMIT = "LIMIT"
        MARKET = "MARKET"

    class Status(models.TextChoices):
        PENDING = "Pending"
        FILLED = "Filled"
        PARTIALLY_FIELD = "Partially Field"
        PARTIALLY_FILLED_DONE = "Partially Filled Done"
        CANCELLED = "Cancelled"

    market = models.ForeignKey("Market", on_delete=models.SET_NULL, null=True)
    position = models.ForeignKey("Position", on_delete=models.SET_NULL, null=True)
    symbol = models.CharField(max_length=50)
    order_type = models.CharField(max_length=50, choices=OrderType.choices)
    side = models.CharField(max_length=50, choices=Side.choices)
    amount = models.DecimalField(max_digits=32, decimal_places=16)
    filled_amount = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    price = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    average_price = models.DecimalField(max_digits=32, decimal_places=16)
    status = models.CharField(max_length=50, choices=Status.choices)
    take_profit_price = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    stop_loss_price = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.pk:
            self.symbol = self.market.symbol
            super().save(*args, **kwargs)
            self.place_order()
        else:
            super().save(*args, **kwargs)

    def place_order(self):
        exchange_obj = self.market.get_exchange_object()
        exchange_obj.place_market_order(symbol=self.symbol, amount=self.amount, side=self.side, unique_id=self.id,
                                        market_type=self.market.market_type)
        self.update_status()

    def update_status(self):
        exchange_obj = self.market.get_exchange_object()
        order_details = exchange_obj.get_order_details(unique_id=self.id)
        self.filled_amount = order_details['filled_amount']
        self.average_price = order_details['price']
        self.status = order_details['status']
        self.save(update_fields=['filled_amount', 'average_price', 'status', 'updated_at'])


class Trade(BaseModel):
    account = models.ForeignKey("Account", on_delete=models.SET_NULL, null=True)
    market = models.ForeignKey("Market", on_delete=models.SET_NULL, null=True)
    order = models.ForeignKey("Order", on_delete=models.SET_NULL, null=True)
    symbol = models.CharField(max_length=50)
    amount = models.DecimalField(max_digits=32, decimal_places=16)
    price = models.DecimalField(max_digits=32, decimal_places=16)
    fee_amount = models.DecimalField(max_digits=32, decimal_places=16)
    side = models.CharField(max_length=50, choices=Order.Side.choices)


class Strategy(BaseModel):
    class Type(models.TextChoices):
        RecentCandle = "Recent Candle"

    market = models.ForeignKey("Market", on_delete=models.SET_NULL, null=True)
    strategy_type = models.CharField(max_length=50, choices=Type.choices)
    active = models.BooleanField(default=False)
    leverage = models.DecimalField(max_digits=32, decimal_places=16)
    order_size_from_basket = models.DecimalField(max_digits=32, decimal_places=16)
    take_profit = models.DecimalField(max_digits=32, decimal_places=16)
    stop_loss = models.DecimalField(max_digits=32, decimal_places=16)


class Position(BaseModel):
    class Status(models.TextChoices):
        Initiated = "Initiated"
        OPEN = "Open"
        CLOSED = "Closed"

    strategy = models.ForeignKey("Strategy", on_delete=models.SET_NULL, null=True)
    market = models.ForeignKey("Market", on_delete=models.SET_NULL, null=True)
    symbol = models.CharField(max_length=50)
    amount = models.DecimalField(max_digits=32, decimal_places=16)
    unrealized_usdt_pnl = models.DecimalField(max_digits=32, decimal_places=16)
    realized_usdt_pnl = models.DecimalField(max_digits=32, decimal_places=16)
    average_entry_price = models.DecimalField(max_digits=32, decimal_places=16)
    take_profit_price = models.DecimalField(max_digits=32, decimal_places=16)
    stop_loss_price = models.DecimalField(max_digits=32, decimal_places=16)
    status = models.CharField(max_length=50, choices=Status.choices, default=Status.Initiated.value)

    @staticmethod
    def check_active_strategies_and_open_position():
        strategies = Strategy.objects.filter(active=True)
        for strategy in strategies:
            if strategy.strategy_type == Strategy.Type.RecentCandle:
                if Position.objects.filter(market=strategy.market, status__in=[Position.Status.Initiated.value,
                                                                               Position.Status.OPEN.value]).exists():
                    continue
                Position.objects.create(strategy=strategy)
            else:
                raise NotImplementedError

    @staticmethod
    def check_open_positions():
        positions = Position.objects.filter(status=Position.Status.OPEN.value)
        for position in positions:
            try:
                position.check_and_manage_position()
            except Exception as ve:
                print(ve)

    def save(self, *args, **kwargs):
        if not self.pk:
            self.market = self.strategy.market
            self.symbol = self.market.symbol
            super().save(*args, *kwargs)
            if self.strategy == Strategy.Type.RecentCandle:
                self.check_and_open_position_based_on_recent_candle()
            else:
                raise NotImplementedError
        else:
            super().save(*args, *kwargs)

    def check_and_open_position_based_on_recent_candle(self):
        exchange_obj = self.market.get_exchange_object()
        recent_candle = exchange_obj.get_recent_candle()
        balance = exchange_obj.get_balance(wallet=self.market.market_type, symbol=self.symbol)['usdt_amount']
        amount_to_open = Decimal(str(balance)) * self.strategy.order_size_from_basket
        if recent_candle['open'] < recent_candle['close']:
            self.open_long_position(amount=amount_to_open)
        elif recent_candle['open'] > recent_candle['close']:
            self.open_short_position(amount=amount_to_open)
        else:
            return

    def check_and_manage_position(self):
        if self.strategy == Strategy.Type.RecentCandle.value:
            self.manage_recent_candle_strategy()
        else:
            raise NotImplementedError

    def manage_recent_candle_strategy(self):
        if self.status == Position.Status.OPEN.value:
            self.update_status()
            self.close_position_if_tp_or_sl_reached()
        else:
            raise Exception("Position is not Open")

    def open_long_position(self, amount):
        self.open_position()
        Order.objects.create(market=self.market, position=self, amount=amount,
                             side=Order.Side.BUY.value, order_type=Order.OrderType.MARKET.value, take_profit=self.take_profit_price)
        self.update_status()

    def open_short_position(self, amount):
        self.open_position()
        Order.objects.create(market=self.market, position=self, amount=amount,
                             side=Order.Side.SELL.value, order_type=Order.OrderType.MARKET.value)
        self.update_status()

    def open_position(self):
        self.status = Position.Status.OPEN.value
        self.save(update_fields=['status', 'updated_at'])

    def update_status(self):
        exchange_obj = self.market.get_exchange_object()
        position_details = exchange_obj.get_position_details(symbol=self.symbol)
        self.update_fields_based_on_position(position_details=position_details)

    def update_fields_based_on_position(self, position_details):
        raise NotImplementedError

    def close_position_if_tp_or_sl_reached(self):
        exchange_obj = self.market.get_exchange_object()
        order_book = exchange_obj.get_order_book(symbol=self.symbol)
        if self.amount > 0:
            best_bid_price = order_book['bids'][-1]['price']
            if best_bid_price >= self.take_profit_price:
                self.close_position()
            elif best_bid_price <= self.stop_loss_price:
                self.close_position()
        elif self.amount < 0:
            best_ask_price = order_book['asks'][-1]['price']
            if best_ask_price <= self.take_profit_price:
                self.close_position()
            elif best_ask_price >= self.stop_loss_price:
                self.close_position()

    def close_position(self):
        exchange_obj = self.market.get_exchange_object()
        exchange_obj.close_position(symbol=self.symbol)
        self.update_status()


