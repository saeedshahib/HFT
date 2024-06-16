import time
import traceback
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from django.db import models

from utils.models import BaseModel
from utils.bases import *
from api_manager.bybit_api import Bybit
from api_manager.mexc_api import MEXCFutures, MEXCSpot
from utils.redis import *

from .indicator_manager import strategy_manager
# Create your models here.


class Currency(BaseModel):
    symbol = models.CharField(max_length=255)
    precision = models.IntegerField(default=8)

    def __str__(self):
        return self.symbol


class Market(BaseModel):
    class Exchange(models.TextChoices):
        MEXC = "MEXC"
        BingX = "BingX"
        Kucoin = "Kucoin"
        Bybit = "Bybit"
        Binance = "Binance"

    class Type(models.TextChoices):
        Spot = "Spot"
        Margin = "Margin"
        Futures = "Futures"

    exchange = models.CharField(max_length=50, choices=Exchange.choices)
    first_currency = models.ForeignKey('Currency', on_delete=models.SET_NULL, null=True,
                                       related_name='first_currency')
    second_currency = models.ForeignKey('Currency', on_delete=models.SET_NULL, null=True,
                                        related_name='second_currency')
    symbol = models.CharField(max_length=50)
    market_type = models.CharField(max_length=50, choices=Type.choices)
    fee = models.DecimalField(max_digits=20, decimal_places=8, default=Decimal('0.001'))

    def __str__(self):
        return f'{self.symbol}_{self.exchange}_{self.market_type}'

    def get_exchange_object(self):
        if self.exchange == Market.Exchange.MEXC.value:
            if self.market_type == Market.Type.Futures.value:
                return MEXCFutures()
            elif self.market_type == Market.Type.Spot.value:
                return MEXCSpot()
        elif self.exchange == Market.Exchange.Bybit.value:
            return Bybit()
        else:
            raise NotImplementedError


class Order(BaseModel):
    class Side(models.TextChoices):
        BUY = "Buy"
        SELL = "Sell"

    class OrderType(models.TextChoices):
        LIMIT = "Limit"
        MARKET = "Market"
        ImmediateOrCancel = "IMMEDIATE_OR_CANCEL"

    class Status(models.TextChoices):
        PENDING = "New"
        FILLED = "Filled"
        PARTIALLY_FIELD = "PartiallyFilled"
        PARTIALLY_FILLED_CANCELLED = "PartiallyFilledCanceled"
        CANCELLED = "Cancelled"

    market = models.ForeignKey("Market", on_delete=models.SET_NULL, null=True)
    position = models.ForeignKey("Position", on_delete=models.SET_NULL, null=True, blank=True)
    symbol = models.CharField(max_length=50)
    order_type = models.CharField(max_length=50, choices=OrderType.choices)
    side = models.CharField(max_length=50, choices=Side.choices)
    amount = models.DecimalField(max_digits=32, decimal_places=16)
    filled_amount = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    price = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    average_price = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    status = models.CharField(max_length=50, choices=Status.choices, default=Status.PENDING.value)
    take_profit_price = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    stop_loss_price = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)

    def save(self, *args, **kwargs):
        if not self.pk:
            self.symbol = self.market.symbol
            self.amount = truncate(self.amount, self.market.first_currency.precision)
            super().save(*args, **kwargs)
        else:
            super().save(*args, **kwargs)

    def execute(self):
        if self.order_type == Order.OrderType.MARKET.value:
            self.place_order()
        elif self.order_type in [Order.OrderType.ImmediateOrCancel.value,
                                 Order.OrderType.LIMIT.value]:
            self.place_limit_order()
        else:
            raise NotImplementedError

    def place_order(self):
        exchange_obj = self.market.get_exchange_object()
        response = exchange_obj.place_market_order(symbol=self.symbol,
                                                   amount=self.amount,
                                                   side=self.side, unique_id=self.id,
                                                   order_type=self.order_type,
                                                   take_profit=self.take_profit_price,
                                                   stop_loss=self.stop_loss_price)
        print(response)
        # self.update_status()

    def place_limit_order(self):
        exchange_obj = self.market.get_exchange_object()
        response = exchange_obj.place_immediate_or_cancel_order(symbol=self.symbol, unique_id=self.id,
                                                                order_type=self.order_type, amount=self.amount,
                                                                side=self.side, price=self.price)
        return response

    def cancel_order(self):
        exchange_obj = self.market.get_exchange_object()
        exchange_obj.cancel_order(symbol=self.symbol, order_id=self.id)

    def update_status(self):
        exchange_obj = self.market.get_exchange_object()
        order_details = exchange_obj.get_order_details(unique_id=self.id)
        self.filled_amount = order_details['cumExecQty']
        self.average_price = order_details['avgPrice']
        self.status = order_details['orderStatus']
        self.save(update_fields=['filled_amount', 'average_price', 'status', 'updated_at'])

    def update_filled_amount_and_state(self, filled_amount, avg_price=None):
        if avg_price is not None:
            self.average_price = avg_price
        self.filled_amount = filled_amount
        if self.filled_amount == 0:
            self.status = self.Status.CANCELLED.value
        elif self.amount == self.filled_amount:
            self.status = self.Status.FILLED.value
        elif self.amount > self.filled_amount:
            self.status = self.Status.PARTIALLY_FILLED_CANCELLED.value
        else:
            raise Exception("amount must be equal or greater than filled amount")
        self.save(update_fields=['filled_amount', 'average_price', 'status', 'updated_at'])


class Trade(BaseModel):
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
        WebsocketChange = "Websocket Change"
        BollingerAwesome = "Bollinger Awesome"

    market = models.ForeignKey("Market", on_delete=models.SET_NULL, null=True)
    strategy_type = models.CharField(max_length=50, choices=Type.choices)
    active = models.BooleanField(default=False)
    leverage = models.DecimalField(max_digits=32, decimal_places=16)
    order_size_from_basket = models.DecimalField(max_digits=32, decimal_places=16)
    take_profit = models.DecimalField(max_digits=32, decimal_places=16)
    stop_loss = models.DecimalField(max_digits=32, decimal_places=16)
    sensitivity = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)


class Position(BaseModel):
    class Status(models.TextChoices):
        Initiated = "Initiated"
        OPEN = "Open"
        CLOSED = "Closed"

    class Side(models.TextChoices):
        Long = "Long"
        Short = "Short"

    class VolatilityTooLow(Exception):
        pass

    trace = models.TextField(null=True, blank=True)
    strategy = models.ForeignKey("Strategy", on_delete=models.SET_NULL, null=True)
    market = models.ForeignKey("Market", on_delete=models.SET_NULL, null=True)
    symbol = models.CharField(max_length=50)
    amount = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    value = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    unrealized_usdt_pnl = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    realized_usdt_pnl = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    average_entry_price = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    take_profit_price = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    stop_loss_price = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    break_even_price = models.DecimalField(max_digits=32, decimal_places=16, null=True)
    status = models.CharField(max_length=50, choices=Status.choices, default=Status.Initiated.value)
    side = models.CharField(max_length=50, choices=Side.choices, null=True, blank=True)

    def add_trace(self, log):
        print(log)
        if self.trace is None:
            self.trace = log
        else:
            self.trace += log
        self.save(update_fields=['trace', 'updated_at'])

    @staticmethod
    def check_active_strategies_and_open_position():
        strategies = Strategy.objects.filter(active=True)
        for strategy in strategies:
            try:
                if strategy.strategy_type == Strategy.Type.RecentCandle:
                    if Position.objects.filter(market=strategy.market, status=Position.Status.OPEN.value).exists():
                        continue
                    initiated_position: Position = Position.objects.filter(market=strategy.market,
                                                                           status=Position.Status.Initiated.value).last()
                    if initiated_position is not None:
                        initiated_position.check_and_open()
                    else:
                        Position.objects.create(strategy=strategy)
                elif strategy.strategy_type == Strategy.Type.BollingerAwesome.value:
                    return Position.check_and_open_position_based_on_bollinger_awesome()
            except Position.VolatilityTooLow:
                continue
            except Exception as e:
                print(e)

    @staticmethod
    def check_open_positions(last_three_candles):
        positions = Position.objects.filter(status=Position.Status.OPEN.value)
        for position in positions:
            try:
                position.check_and_manage_position(last_three_candles)
            except Exception as ve:
                print(traceback.format_exc())

    @staticmethod
    def check_and_open_position_based_on_bollinger_awesome():
        strategy: Strategy = Strategy.objects.filter(active=True,
                                                     strategy_type=Strategy.Type.BollingerAwesome.value).last()
        batch_period = 100 * 60
        # end_time = (datetime.now(tz=timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=1)).timestamp()
        end_time = int(time.time())
        start_time = end_time - batch_period
        exchange_obj = strategy.market.get_exchange_object()
        recent_candles = exchange_obj.get_recent_candle(symbol=strategy.market.symbol, start=start_time, end=end_time)
        df, strategy = strategy_manager.check_active_strategy_and_generate_df(strategy=strategy,
                                                                              recent_candles=recent_candles)
        last_three_candles = df.tail(3)
        previous_row = last_three_candles.iloc[0]
        current_row = last_three_candles.iloc[1]
        if Position.objects.filter(strategy=strategy, status__in=[Position.Status.OPEN.value,
                                                                  Position.Status.Initiated.value]).exists():
            return last_three_candles
        if strategy_manager.check_open_long(df, current_row, previous_row):
            print(last_three_candles)
            Position.objects.create(strategy=strategy, side=Position.Side.Long.value)
        if strategy_manager.check_open_short(df, current_row, previous_row):
            print(last_three_candles)
            Position.objects.create(strategy=strategy, side=Position.Side.Short.value)
        return last_three_candles

    def save(self, *args, **kwargs):
        if not self.pk:
            self.market = self.strategy.market
            self.symbol = self.market.symbol
            super().save(*args, **kwargs)
            self.check_and_open()
        else:
            super().save(*args, **kwargs)

    def check_and_open(self):
        if self.strategy.strategy_type == Strategy.Type.RecentCandle.value:
            self.check_and_open_position_based_on_recent_candle()
        elif self.strategy.strategy_type == Strategy.Type.WebsocketChange.value:
            self.open_position_based_on_websocket_change()
        elif self.strategy.strategy_type == Strategy.Type.BollingerAwesome.value:
            self.open_position_based_on_websocket_change()

    def check_and_open_position_based_on_recent_candle(self):
        if self.status == Position.Status.OPEN.value:
            self.add_trace("position is open!!!")
            return
        exchange_obj = self.market.get_exchange_object()
        recent_candles = exchange_obj.get_recent_candle(symbol=self.symbol)
        volatility = self.calculate_volatility(recent_candles)
        balance = Decimal(str(exchange_obj.get_balance(currency=self.market.second_currency.symbol)))
        order_book = exchange_obj.get_order_book(symbol=self.symbol)
        value_to_open = Decimal(str(balance)) * self.strategy.leverage
        self.add_trace(f"high, low: {recent_candles[-1][1]}, {recent_candles[0][4]}")
        if recent_candles[-1][1] < recent_candles[0][4]:
            best_bid_price = Decimal(str(order_book['result']['b'][0][0]))
            self.calculate_sl_tp(price=best_bid_price, side=Order.Side.BUY.value)
            self.check_volatility(volatility=volatility)
            amount_to_open = (value_to_open / best_bid_price) * Decimal('0.9')
            self.open_long_position(amount=amount_to_open)
        elif recent_candles[-1][1] > recent_candles[0][4]:
            best_ask_price = Decimal(str(order_book['result']['a'][0][0]))
            amount_to_open = (value_to_open / best_ask_price) * Decimal('0.9')
            self.calculate_sl_tp(price=best_ask_price, side=Order.Side.SELL.value)
            self.check_volatility(volatility=volatility)
            self.open_short_position(amount=amount_to_open)
        else:
            return

    def check_and_manage_position(self, last_three_candles):
        if self.strategy.strategy_type in [Strategy.Type.RecentCandle.value,
                                           Strategy.Type.WebsocketChange.value]:
            self.manage_recent_candle_strategy()
        elif self.strategy.strategy_type == Strategy.Type.BollingerAwesome.value:
            self.manage_bollinger_awesome_strategy(last_three_candles)

    def manage_recent_candle_strategy(self):
        if self.status == Position.Status.OPEN.value:
            self.update_status()
            self.close_position_if_tp_or_sl_reached()
        else:
            raise Exception("Position is not Open")

    def open_long_position(self, amount):
        self.add_trace("open_long_position")
        self.open_position(side=Position.Side.Long.value)
        Order.objects.create(market=self.market, position=self, amount=amount,
                             side=Order.Side.BUY.value, order_type=Order.OrderType.MARKET.value,
                             take_profit_price=self.take_profit_price, stop_loss_price=self.stop_loss_price)
        self.update_status()

    def open_short_position(self, amount):
        self.add_trace("open_short_position")
        self.open_position(side=Position.Side.Short.value)
        Order.objects.create(market=self.market, position=self, amount=amount,
                             side=Order.Side.SELL.value, order_type=Order.OrderType.MARKET.value,
                             take_profit_price=self.take_profit_price, stop_loss_price=self.stop_loss_price)
        self.update_status()

    def open_position(self, side):
        self.status = Position.Status.OPEN.value
        self.side = side
        self.save(update_fields=['status', 'side', 'updated_at'])

    def update_status(self):
        exchange_obj = self.market.get_exchange_object()
        position_details = exchange_obj.get_position_details(symbol=self.symbol)
        self.update_fields_based_on_position(position_details=position_details)

    def update_fields_based_on_position(self, position_details):
        """
        {'retCode': 0,
         'retMsg': 'OK',
         'result': {'nextPageCursor': 'BTCUSDT%2C1715191462644%2C0',
          'category': 'linear',
          'list': [{'symbol': 'BTCUSDT',
            'leverage': '100',
            'autoAddMargin': 0,
            'avgPrice': '0',
            'liqPrice': '',
            'riskLimitValue': '2000000',
            'takeProfit': '',
            'positionValue': '',
            'isReduceOnly': False,
            'tpslMode': 'Full',
            'riskId': 1,
            'trailingStop': '0',
            'unrealisedPnl': '',
            'markPrice': '62595.9',
            'adlRankIndicator': 0,
            'cumRealisedPnl': '-0.1469502',
            'positionMM': '0',
            'createdTime': '1715111020486',
            'positionIdx': 0,
            'positionIM': '0',
            'seq': 173265968624,
            'updatedTime': '1715191462644',
            'side': '',
            'bustPrice': '',
            'positionBalance': '0',
            'leverageSysUpdatedTime': '',
            'curRealisedPnl': '0',
            'size': '0',
            'positionStatus': 'Normal',
            'mmrSysUpdatedTime': '',
            'stopLoss': '',
            'tradeMode': 0,
            'sessionAvgPrice': ''}]},
         'retExtInfo': {},
         'time': 1715191641476}
        """
        data = position_details['result']['list'][0]
        self.amount = Decimal(data['size']) if data['size'] != '0' else self.amount
        self.value = Decimal(data['positionValue']) if data['positionValue'] not in ['0', ''] else self.value
        self.unrealized_usdt_pnl = Decimal(data['curRealisedPnl']) if data['curRealisedPnl'] not in ['0', ''] else (
            self.unrealized_usdt_pnl)
        self.average_entry_price = Decimal(data['avgPrice']) if data['avgPrice'] not in ['0', ''] else self.average_entry_price
        if data['positionValue'] in ['0', '']:
            self.status = Position.Status.CLOSED.value
        self.save(update_fields=['amount', 'value', 'status', 'unrealized_usdt_pnl',
                                 'average_entry_price', 'updated_at'])

    def close_position_if_tp_or_sl_reached(self):
        if self.status == Position.Status.CLOSED.value:
            return
        exchange_obj = self.market.get_exchange_object()
        order_book = exchange_obj.get_order_book(symbol=self.symbol)
        print("check tp sl")
        if self.side == Position.Side.Long.value:
            best_bid_price = Decimal(str(order_book['result']['b'][0][0]))
            if best_bid_price >= self.take_profit_price:
                self.add_trace("tp reached, close position")
                self.close_position(side=Order.Side.SELL.value)
            elif best_bid_price <= self.stop_loss_price:
                self.add_trace("sl reached, close position")
                self.close_position(side=Order.Side.SELL.value)
            elif best_bid_price >= self.break_even_price * (1 + 2 * self.market.fee):
                self.set_stop_loss_to_break_even()
        elif self.side == Position.Side.Short.value:
            best_ask_price = Decimal(str(order_book['result']['a'][0][0]))
            if best_ask_price <= self.take_profit_price:
                self.add_trace("tp reached, close position")
                self.close_position(side=Order.Side.BUY.value)
            elif best_ask_price >= self.stop_loss_price:
                self.add_trace("sl reached, close position")
                self.close_position(side=Order.Side.BUY.value)
            if best_ask_price <= self.break_even_price * (1 - 2 * self.market.fee):
                self.set_stop_loss_to_break_even()

    def close_position(self, side):
        exchange_obj = self.market.get_exchange_object()
        exchange_obj.close_position(symbol=self.symbol, side=side)
        self.update_status()

    def calculate_sl_tp(self, price, side):
        total_fee = 2 * self.market.fee
        if side == Order.Side.BUY.value:
            profit_change = (self.strategy.take_profit / self.strategy.leverage) / 100
            take_profit_price = price * (1 + profit_change + total_fee)
            loss_change = (self.strategy.stop_loss / self.strategy.leverage) / 100
            stop_loss_price = price * (1 - loss_change + total_fee)
            self.take_profit_price, self.stop_loss_price = take_profit_price, stop_loss_price
            self.break_even_price = price * (1 + 3 * self.market.fee)
            self.save(update_fields=['take_profit_price', 'stop_loss_price', 'break_even_price', 'updated_at'])
            return self.take_profit_price, self.stop_loss_price
        elif side == Order.Side.SELL.value:
            profit_change = (self.strategy.take_profit / self.strategy.leverage) / 100
            take_profit_price = price * (1 - profit_change - total_fee)
            loss_change = (self.strategy.stop_loss / self.strategy.leverage) / 100
            stop_loss_price = price * (1 + loss_change - total_fee)
            self.take_profit_price, self.stop_loss_price = take_profit_price, stop_loss_price
            self.break_even_price = price * (1 - 3 * self.market.fee)
            self.save(update_fields=['take_profit_price', 'stop_loss_price', 'break_even_price', 'updated_at'])
            return self.take_profit_price, self.stop_loss_price
        else:
            raise Exception('Invalid side')

    @staticmethod
    def calculate_volatility(recent_candles):
        sorted_candles = sorted(recent_candles, key=lambda x: x[0])

        max_high_price = max([Decimal(value[2]) for value in sorted_candles])
        min_low_price = min([Decimal(value[3]) for value in sorted_candles])

        change = ((max_high_price - min_low_price) / min_low_price)

        return change

    def check_volatility(self, volatility):
        if self.strategy.sensitivity > volatility:
            raise Position.VolatilityTooLow()
        self.add_trace(f"volatility: {volatility}, sensitivity: {self.strategy.sensitivity}")

    def set_stop_loss_to_break_even(self):
        if self.stop_loss_price == self.break_even_price:
            return
        self.add_trace("break even reached, set lower stop loss")
        exchange_obj = self.market.get_exchange_object()
        if self.side == Position.Side.Long.value:
            self.take_profit_price = self.take_profit_price + abs(self.take_profit_price - self.break_even_price)
        else:
            self.take_profit_price = self.take_profit_price - abs(self.take_profit_price - self.break_even_price)
        exchange_obj.set_sl_tp(symbol=self.symbol, sl_price=self.break_even_price, tp_price=self.take_profit_price)
        self.stop_loss_price = self.break_even_price
        self.save(update_fields=['stop_loss_price', 'take_profit_price', 'updated_at'])

    def open_position_based_on_websocket_change(self):
        balance = get_balance(symbol=self.market.second_currency.symbol, exchange=self.market.exchange)
        print("balance", balance)
        value_to_open = Decimal(str(balance)) * self.strategy.leverage
        print("value_to_open: ", value_to_open)
        if self.side == Position.Side.Long.value:
            price = get_price(symbol=self.symbol, side=Position.Side.Long.value)
            print("price long: ", price)
            amount_to_open = (value_to_open / price) * Decimal('0.9')
            self.calculate_sl_tp(price=price, side=Order.Side.BUY.value)
            self.open_long_position(amount=amount_to_open)
        elif self.side == Position.Side.Short.value:
            price = get_price(symbol=self.symbol, side=Position.Side.Short.value)
            print("price short: ", price)
            amount_to_open = (value_to_open / price) * Decimal('0.9')
            self.calculate_sl_tp(price=price, side=Order.Side.SELL.value)
            self.open_short_position(amount=amount_to_open)

    def manage_bollinger_awesome_strategy(self, last_three_candles):
        if self.status == Position.Status.OPEN.value:
            self.update_status()
            # current_candle = last_three_candles.iloc[-1]
            # if self.side == Position.Side.Long.value:
            #     current_price = get_price(symbol=self.symbol, side=Position.Side.Long.value)
            #     if ((Decimal(str(current_price)) >=
            #             Decimal(str(current_candle['Bollinger Upper Band'])) * Decimal('0.999')) and
            #             Decimal(str(current_price)) >= self.break_even_price):
            #         self.add_trace("tp reached, close position")
            #         self.close_position(side=Order.Side.SELL.value)
            #     elif current_price <= self.stop_loss_price:
            #         self.add_trace("sl reached, close position")
            #         self.close_position(side=Order.Side.SELL.value)
            # elif self.side == Position.Side.Short.value:
            #     current_price = get_price(symbol=self.symbol, side=Position.Side.Long.value)
            #     if ((Decimal(str(current_price)) <=
            #             Decimal(str(current_candle['Bollinger Lower Band'])) * Decimal('1.001')) and
            #             Decimal(str(current_price)) <= self.break_even_price):
            #         self.add_trace("tp reached, close position")
            #         self.close_position(side=Order.Side.BUY.value)
            #     elif current_price >= self.stop_loss_price:
            #         self.add_trace("sl reached, close position")
            #         self.close_position(side=Order.Side.BUY.value)
        else:
            raise Exception("Position is not Open")


class ArbitragePosition(BaseModel):
    class ArbitrageStatus(models.TextChoices):
        Pending = "Pending"
        Cancelled = "Cancelled"
        Open = 'Open'
        CloseRequested = "Close requested"
        ClosedWithTP = 'Closed with tp'
        ClosedWithSL = 'Closed with sl'

    open_order = models.ForeignKey(Order, on_delete=models.SET_NULL, null=True, related_name='open_order', blank=True)
    close_order = models.ForeignKey(Order, on_delete=models.SET_NULL, null=True, related_name='close_order', blank=True)
    source_market = models.ForeignKey(Market, on_delete=models.SET_NULL, null=True, related_name='source_market')
    source_price = models.DecimalField(max_digits=32, decimal_places=16)
    target_market = models.ForeignKey(Market, on_delete=models.SET_NULL, null=True, related_name='target_market')
    target_price = models.DecimalField(max_digits=32, decimal_places=16)
    closed_price = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    pnl = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    pnl_percent = models.DecimalField(max_digits=32, decimal_places=16, null=True, blank=True)
    status = models.CharField(choices=ArbitrageStatus.choices, default=ArbitrageStatus.Pending.value, max_length=32)

    def save(self, *args, **kwargs):
        if not self.pk:
            usdt_asset = Asset.objects.get(currency=self.source_market.second_currency,
                                           exchange=self.source_market.exchange).value
            if usdt_asset <= 5:
                raise Exception("Not enough USD")
            amount_to_buy = (usdt_asset / self.source_price) * Decimal('0.99')
            order = Order.objects.create(market=self.source_market, amount=amount_to_buy,
                                         side=Order.Side.BUY.value, order_type=Order.OrderType.ImmediateOrCancel.value,
                                         price=self.source_price)
            self.open_order = order
            super().save(*args, **kwargs)
            order.execute()
        else:
            super().save(*args, **kwargs)

    def check_and_close_position(self, reached_price):
        reached_price = Decimal(str(reached_price))
        if reached_price <= self.source_price * Decimal('0.99'):
            if self.close_order.status in [Order.Status.PENDING.value, Order.Status.PARTIALLY_FIELD.value]:
                self.close_order.cancel_order()
            print("go close position")
            self.status = self.ArbitrageStatus.CloseRequested.value
            order = Order.objects.create(market=self.source_market, amount=self.open_order.filled_amount,
                                         price=self.source_price * Decimal('0.95'),
                                         side=Order.Side.SELL.value, order_type=Order.OrderType.ImmediateOrCancel.value)
            self.close_order = order
            order.execute()
            self.save(update_fields=['status', 'close_order', 'updated_at'])


    @staticmethod
    def open_position_if_not_open(source_price, target_price, source_market, target_market):
        if ArbitragePosition.objects.filter(status__in=[ArbitragePosition.ArbitrageStatus.Pending.value,
                                                        ArbitragePosition.ArbitrageStatus.Open.value,
                                                        ArbitragePosition.ArbitrageStatus.CloseRequested]).exists():
            return
        ArbitragePosition.objects.create(source_market=source_market, target_market=target_market,
                                         status=ArbitragePosition.ArbitrageStatus.Pending.value,
                                         source_price=source_price * Decimal('1.005'),
                                         target_price=target_price * Decimal('0.995'))

    @staticmethod
    def update_status_based_on_websocket_payload(order_id, filled_amount, avg_price=None):
        order = Order.objects.get(id=order_id)
        order.update_filled_amount_and_state(filled_amount=filled_amount, avg_price=avg_price)
        if order.side == Order.Side.BUY.value:
            arbitrage_position = ArbitragePosition.objects.get(open_order=order)
            if order.status == Order.Status.CANCELLED.value:
                arbitrage_position.status = ArbitragePosition.ArbitrageStatus.Cancelled.value
                arbitrage_position.save(update_fields=['status', 'updated_at'])
            else:
                tp_order = Order.objects.create(market=arbitrage_position.source_market,
                                                amount=arbitrage_position.open_order.filled_amount,
                                                side=Order.Side.SELL.value,
                                                order_type=Order.OrderType.LIMIT.value,
                                                price=arbitrage_position.target_price)
                arbitrage_position.close_order = tp_order
                tp_order.execute()
                arbitrage_position.status = ArbitragePosition.ArbitrageStatus.Open.value
                arbitrage_position.save(update_fields=['status', 'close_order', 'updated_at'])
        else:
            if order.filled_amount > 0:
                arbitrage_position = ArbitragePosition.objects.get(close_order=order)
                arbitrage_position.closed_price = avg_price
                arbitrage_position.pnl_percent = ((arbitrage_position.closed_price - arbitrage_position.source_price) /
                                                  arbitrage_position.source_price)
                arbitrage_position.pnl = order.filled_amount * order.average_price * arbitrage_position.pnl_percent
                if arbitrage_position.pnl > 0:
                    arbitrage_position.status = ArbitragePosition.ArbitrageStatus.ClosedWithTP.value
                else:
                    arbitrage_position.status = ArbitragePosition.ArbitrageStatus.ClosedWithSL.value
                arbitrage_position.save(update_fields=['closed_price', 'status', 'pnl', 'pnl_percent', 'updated_at'])


class Asset(BaseModel):
    currency = models.ForeignKey(Currency, on_delete=models.SET_NULL, null=True)
    exchange = models.CharField(max_length=32, choices=Market.Exchange.choices)
    value = models.DecimalField(max_digits=32, decimal_places=16)
