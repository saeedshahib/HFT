from datetime import datetime
from decimal import Decimal
import pandas as pd


class StrategyManager:
    def __init__(self):
        self.short_ema_window = 3
        self.bb_window = 20
        self.bb_std = 3
        self.ao_short_window = 5
        self.ao_long_window = 34

    def generate_df(self, recent_candles):
        high = []
        low = []
        close = []
        ts = []
        date = []
        for candle in reversed(recent_candles):
            close.append(Decimal(str((candle[4]))))
            high.append(Decimal(str((candle[2]))))
            low.append(Decimal(str((candle[3]))))
            ts.append(str(candle[0]))
            date.append(datetime.fromtimestamp(int(candle[0]) // 1000))
        data = {
            'Close': close,
            'High': high,
            'Low': low,
            'ts': ts,
            'date': date
        }
        index = pd.date_range(start='2024-01-01 03:00', periods=len(close), freq='h')
        df = pd.DataFrame(data, index=index)
        return df

    def apply_take_profit(self, df, signals, take_profit_pips=10, ao_change_col='AO'):
        df['Pips'] = df['Close'].diff()
        take_profit_points = take_profit_pips / 10000  # Assuming a standard pip value

        df['AO Change'] = df[ao_change_col].diff()

        buy_tp = signals['Buy'] & ((df['Pips'].cumsum() >= take_profit_points) | (df['AO Change'] < 0))
        sell_tp = signals['Sell'] & ((-df['Pips'].cumsum() >= take_profit_points) | (df['AO Change'] > 0))

        signals['Buy'] = buy_tp
        signals['Sell'] = sell_tp

        return signals

    def ema(self, data, window):
        return data.ewm(span=window, adjust=False).mean()

    def bollinger_bands(self, data, window, no_of_std):
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * no_of_std)
        lower_band = rolling_mean - (rolling_std * no_of_std)
        bands = pd.DataFrame({
            'Middle Band': rolling_mean,
            'Upper Band': upper_band,
            'Lower Band': lower_band
        })
        return bands

    def awesome_oscillator(self, high, low, short_window, long_window):
        median_price = (high + low) / 2
        ao = median_price.rolling(window=short_window).mean() - median_price.rolling(window=long_window).mean()
        return ao

    def generate_signals(self, df):
        df['3 EMA'] = self.ema(df['Close'], self.short_ema_window)
        bollinger = self.bollinger_bands(df['Close'], self.bb_window, self.bb_std)
        df['Bollinger Middle Band'] = bollinger['Middle Band']
        df['Bollinger Upper Band'] = bollinger['Upper Band']
        df['Bollinger Lower Band'] = bollinger['Lower Band']
        df['AO'] = self.awesome_oscillator(df['High'], df['Low'], self.ao_short_window, self.ao_long_window)
        previous_row = None
        position = None
        side = None
        sum_pnl = 0
        tp_count = 0
        sl_count = 0
        for index, row in df.iterrows():
            if previous_row is not None:
                if position is None:
                    if (row['3 EMA'] > row['Bollinger Middle Band'] and previous_row['3 EMA'] <
                            previous_row['Bollinger Middle Band'] and row['AO'] > previous_row['AO']):
                        # print("buy", row['date'])
                        position = row
                        side = "Long"
                    if (row['3 EMA'] < row['Bollinger Middle Band'] and previous_row['3 EMA'] >
                            previous_row['Bollinger Middle Band'] and row['AO'] < previous_row['AO']):
                        # print("sell", row['date'])
                        position = row
                        side = "Short"
                else:
                    open = Decimal(str(position['Close']))
                    # position, sl_count, sum_pnl, tp_count = check_tp_sl_fixed(open, position, row, side, sl_count, sum_pnl,
                    #                                                           tp_count)
                    position, sl_count, sum_pnl, tp_count = self.check_tp_sl_bollinger(open, position, row, side,
                                                                                       sl_count, sum_pnl, tp_count)
            previous_row = row
        print(tp_count)
        print(sl_count)
        print(sum_pnl)
        signals = pd.DataFrame(index=df.index)
        signals['Buy'] = (df['3 EMA'] > df['Bollinger Middle Band']) & (df['AO'] > 0)
        signals['Sell'] = (df['3 EMA'] < df['Bollinger Middle Band']) & (df['AO'] < 0)

        # Adding trading hours condition
        # trading_hours = df.index.to_series().between_time('03:00', '12:00')
        # signals['Buy'] = signals['Buy'] & trading_hours
        # signals['Sell'] = signals['Sell'] & trading_hours

        return signals

    def check_active_strategy_and_generate_df(self, strategy, recent_candles):
        df = self.generate_df(recent_candles)
        df['3 EMA'] = self.ema(df['Close'], self.short_ema_window)
        bollinger = self.bollinger_bands(df['Close'], self.bb_window, self.bb_std)
        df['Bollinger Middle Band'] = bollinger['Middle Band']
        df['Bollinger Upper Band'] = bollinger['Upper Band']
        df['Bollinger Lower Band'] = bollinger['Lower Band']
        df['AO'] = self.awesome_oscillator(df['High'], df['Low'], self.ao_short_window, self.ao_long_window)
        return df, strategy

    def check_tp_sl_fixed(self, open, position, row, side, sl_count, sum_pnl, tp_count):
        if side == "Long":
            if row['High'] >= open * Decimal('1.02'):
                print("tp reached long")
                tp_count += 1
                sum_pnl += ((row['High'] - open) / open * 100) - Decimal('0.3')
                position = None
            elif row['Low'] <= open * Decimal('0.99'):
                print("sl reached long")
                sl_count += 1
                sum_pnl += ((row['Low'] - open) / open * 100) - Decimal('0.3')
                position = None
            print(sum_pnl)
        else:
            if row['Low'] <= open * Decimal('0.98'):
                print("tp reached short")
                tp_count += 1
                sum_pnl += abs((row['Low'] - open) / open * 100) - Decimal('0.3')
                position = None
            elif row['High'] >= open * Decimal('1.01'):
                print("sl reached short")
                sl_count += 1
                sum_pnl += ((row['High'] - open) / open * 100) - Decimal('0.3')
                position = None
            print(sum_pnl)
        return position, sl_count, sum_pnl, tp_count

    def check_tp_sl_bollinger(self, open, position, row, side, sl_count, sum_pnl, tp_count):
        if side == "Long":
            if row['High'] >= position['Bollinger Upper Band']:
                # print("tp reached long")
                tp_count += 1
                sum_pnl += ((row['High'] - open) / open * 100) - Decimal('0.3')
                position = None
            elif row['Low'] <= open * Decimal('0.98'):
                # print("sl reached long")
                sl_count += 1
                sum_pnl += ((row['Low'] - open) / open * 100) - Decimal('0.3')
                position = None
        else:
            if row['Low'] <= position['Bollinger Lower Band']:
                # print("tp reached short")
                tp_count += 1
                sum_pnl += abs((row['Low'] - open) / open * 100) - Decimal('0.3')
                position = None
            elif row['High'] >= open * Decimal('1.02'):
                # print("sl reached short")
                sl_count += 1
                sum_pnl += ((row['High'] - open) / open * 100) - Decimal('0.3')
                position = None
        return position, sl_count, sum_pnl, tp_count

    def check_trending_market(self, df, threshold=0.005):
        price_diff = df['Close'].pct_change()
        volatility = price_diff.rolling(window=20).std()
        trending = volatility > threshold
        return trending

strategy_manager = StrategyManager()
