from datetime import datetime
from decimal import Decimal
import pandas as pd
import numpy as np
import ta


class StrategyManager:
    def __init__(self):
        self.short_ema_window = 5
        self.bb_window = 30
        self.bb_std = 3
        self.ao_short_window = 5
        self.ao_long_window = 34
        self.pnl = 0
        self.tp_count = 0
        self.sl_count = 0

    def generate_df(self, recent_candles):
        high = []
        low = []
        close = []
        ts = []
        date = []
        volume = []
        for candle in reversed(recent_candles):
            close.append(float(str((candle[4]))))
            high.append(float(str((candle[2]))))
            low.append(float(str((candle[3]))))
            ts.append(str(int(candle[0]) // 1000))
            date.append(datetime.fromtimestamp(int(candle[0]) // 1000))
            volume.append(float(str(candle[6])))
        data = {
            'Close': close,
            'High': high,
            'Low': low,
            'ts': ts,
            'date': date,
            'Volume': volume,
        }
        # index = pd.date_range(start='2024-01-01 03:00', periods=len(close), freq='h')
        df = pd.DataFrame(data)
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
            current_df = df.iloc[:index].copy()
            if previous_row is not None:
                if position is None:
                    if self.check_open_long(current_df, row, previous_row):
                        # print("buy", row['date'])
                        position = row
                        side = "Long"
                    elif self.check_open_short(current_df, row, previous_row):
                        # print("sell", row['date'])
                        position = row
                        side = "Short"
                else:
                    open = Decimal(str(position['Close']))
                    # position, sl_count, sum_pnl, tp_count = check_tp_sl_fixed(open, position, row, side, sl_count, sum_pnl,
                    #                                                           tp_count)
                    position, sl_count, sum_pnl, tp_count = self.check_tp_sl_fixed(open, position, row, side,
                                                                                       sl_count, sum_pnl, tp_count)
            previous_row = row
        print(tp_count)
        print(sl_count)
        print(sum_pnl)
        self.pnl += sum_pnl
        self.tp_count += tp_count
        self.sl_count += sl_count
        # signals = pd.DataFrame(index=df.index)
        # signals['Buy'] = (df['3 EMA'] > df['Bollinger Middle Band']) & (df['AO'] > 0)
        # signals['Sell'] = (df['3 EMA'] < df['Bollinger Middle Band']) & (df['AO'] < 0)

        # Adding trading hours condition
        # trading_hours = df.index.to_series().between_time('03:00', '12:00')
        # signals['Buy'] = signals['Buy'] & trading_hours
        # signals['Sell'] = signals['Sell'] & trading_hours

        # return signals

    def check_open_long(self, df, current_row, previous_row):
        if (current_row['3 EMA'] > current_row['Bollinger Middle Band'] and previous_row['3 EMA'] <
                previous_row['Bollinger Middle Band'] and current_row['AO'] > previous_row['AO']):
            if self.check_trending_market(df=df)[0]:
                return True
        return False

    def check_open_short(self, df, current_row, previous_row):
        if (current_row['3 EMA'] < current_row['Bollinger Middle Band'] and previous_row['3 EMA'] >
                previous_row['Bollinger Middle Band'] and current_row['AO'] < previous_row['AO']):
            if self.check_trending_market(df=df)[1]:
                return True
        return False

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
            if row['High'] >= open * Decimal('1.01'):
                # print("tp reached long")
                # print(position)
                tp_count += 1
                tp = Decimal('0.8')
                sum_pnl += tp
                position = None
            elif row['Low'] <= open * Decimal('0.99'):
                # print("sl reached long")
                # print(position)
                # print(row)
                sl_count += 1
                sl = Decimal('-1.2')
                sum_pnl += sl
                position = None
            # print(sum_pnl)
        else:
            if row['Low'] <= open * Decimal('0.99'):
                # print("tp reached short")
                # print(position)
                tp_count += 1
                tp = Decimal('0.8')
                sum_pnl += tp
                position = None
            elif row['High'] >= open * Decimal('1.01'):
                # print("sl reached short")
                # print(position)
                # print(row)
                sl_count += 1
                sl = Decimal('-1.2')
                sum_pnl += sl
                position = None
            # print(sum_pnl)
        return position, sl_count, sum_pnl, tp_count

    def check_tp_sl_bollinger(self, open, position, row, side, sl_count, sum_pnl, tp_count):
        if side == "Long":
            if Decimal(str(row['High'])) >= Decimal(str(position['Bollinger Upper Band'])):
                # print("tp reached long")
                tp_count += 1
                profit = ((row['High'] - open) / open * 100) - Decimal('0.2')
                # print("profit long: ", profit)
                sum_pnl += profit
                position = None
            elif Decimal(str(row['Low'])) <= open * Decimal('0.98'):
                # print("sl reached long")
                sl_count += 1
                loss = ((row['Low'] - open) / open * 100) - Decimal('0.2')
                # print("loss long: ", loss)
                sum_pnl += loss
                position = None
        else:
            if Decimal(str(row['Low'])) <= Decimal(str(position['Bollinger Lower Band'])):
                # print("tp reached short")
                tp_count += 1
                profit = ((open - (row['Low'])) / open * 100) - Decimal('0.2')
                # print("profit short: ", profit)
                sum_pnl += profit
                position = None
            elif Decimal(str(row['High'])) >= open * Decimal('1.02'):
                # print("sl reached short")
                sl_count += 1
                loss = ((open - (row['High'])) / open * 100) - Decimal('0.2')
                # print("loss short: ", loss)
                sum_pnl += loss
                position = None
        return position, sl_count, sum_pnl, tp_count

    def check_trending_market(self, df, short_window=20, long_window=50, threshold=0.5):
        """
            Checks whether the market is trending based on moving averages.

            Parameters:
            df (pd.DataFrame): DataFrame containing market data with at least a 'Close' column.
            short_window (int): The window size for the short-term moving average.
            long_window (int): The window size for the long-term moving average.
            threshold (float): The threshold proportion of days the short-term MA must be above/below the long-term MA to be considered trending.

            Returns:
            bool: True if the market is trending, False otherwise.
            """
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain a 'Close' column")

        df['Short_MA'] = df['Close'].rolling(window=short_window).mean()
        df['Long_MA'] = df['Close'].rolling(window=long_window).mean()

        df = df.dropna()  # Drop rows with NaN values due to rolling window
        trending_up = (df['Short_MA'] > df['Long_MA']).sum() / len(df) > threshold
        trending_down = (df['Short_MA'] < df['Long_MA']).sum() / len(df) > threshold
        return trending_up, trending_down

    def ichimoku_cloud(self, data, period1=9, period2=26, period3=52):
        """
        Calculate Ichimoku Cloud indicators and determine the trend.

        :param data: DataFrame with columns ['High', 'Low', 'Close']
        :param period1: Period for Tenkan-sen (default is 9)
        :param period2: Period for Kijun-sen (default is 26)
        :param period3: Period for Senkou Span B (default is 52)
        :return: DataFrame with Ichimoku Cloud components and trend indication
        """

        # Calculate the Tenkan-sen (Conversion Line)
        high_9 = data['High'].rolling(window=period1).max()
        low_9 = data['Low'].rolling(window=period1).min()
        tenkan_sen = (high_9 + low_9) / 2

        # Calculate the Kijun-sen (Base Line)
        high_26 = data['High'].rolling(window=period2).max()
        low_26 = data['Low'].rolling(window=period2).min()
        kijun_sen = (high_26 + low_26) / 2

        # Calculate the Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period2)

        # Calculate the Senkou Span B (Leading Span B)
        high_52 = data['High'].rolling(window=period3).max()
        low_52 = data['Low'].rolling(window=period3).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(period2)

        # Calculate the Chikou Span (Lagging Span)
        chikou_span = data['Close'].shift(-period2)

        # Determine the trend
        trend = []
        for i in range(len(data)):
            if str(senkou_span_a.iloc[i]) == 'nan':
                continue
            if str(senkou_span_b.iloc[i]) == 'nan':
                continue
            if data['Close'].iloc[i] > senkou_span_a.iloc[i] and data['Close'].iloc[i] > senkou_span_b.iloc[i]:
                if tenkan_sen.iloc[i] > kijun_sen.iloc[i] and tenkan_sen.iloc[i] > senkou_span_a.iloc[i] and tenkan_sen.iloc[i] > senkou_span_b.iloc[i]:
                    trend.append('Uptrend')
                else:
                    trend.append('Consolidation/Neutral')
            elif data['Close'].iloc[i] < senkou_span_a.iloc[i] and data['Close'].iloc[i] < senkou_span_b.iloc[i]:
                if tenkan_sen.iloc[i] < kijun_sen.iloc[i] and tenkan_sen.iloc[i] < senkou_span_a.iloc[i] and tenkan_sen.iloc[i] < senkou_span_b.iloc[i]:
                    trend.append('Downtrend')
                else:
                    trend.append('Consolidation/Neutral')
            else:
                trend.append('Consolidation/Neutral')

        # Combine all components into a DataFrame
        # ichimoku_df = pd.DataFrame({
        #     'Tenkan-sen': tenkan_sen,
        #     'Kijun-sen': kijun_sen,
        #     'Senkou Span A': senkou_span_a,
        #     'Senkou Span B': senkou_span_b,
        #     'Chikou Span': chikou_span,
        #     'Close': data['Close'],
        #     'Trend': trend
        # })

        return trend[-1] == 'Uptrend', trend[-1] == 'Downtrend'

    def calculate_pivots(self, df, length):
        df['High_Max'] = df['High'].rolling(window=length).max()
        df['Low_Min'] = df['Low'].rolling(window=length).min()
        df['Top_Swing'] = np.where(df['High'] > df['High_Max'].shift(1), df['High'], np.nan)
        df['Bottom_Swing'] = np.where(df['Low'] < df['Low_Min'].shift(1), df['Low'], np.nan)
        df['Top_Swing'].fillna(method='ffill', inplace=True)
        df['Bottom_Swing'].fillna(method='ffill', inplace=True)
        return df

    def calculate_order_blocks(self, df, sensitivity):
        df['Order_Block_High'] = df['High'].rolling(window=sensitivity).max()
        df['Order_Block_Low'] = df['Low'].rolling(window=sensitivity).min()
        return df

    def calculate_fvg(self, df):
        df['FVG_Up'] = np.where((df['Low'] > df['High'].shift(1)), df['Low'], np.nan)
        df['FVG_Down'] = np.where((df['High'] < df['Low'].shift(1)), df['High'], np.nan)
        df['FVG_Up'].fillna(method='ffill', inplace=True)
        df['FVG_Down'].fillna(method='ffill', inplace=True)
        return df

    def custom_indicator(self, df, pivot_length=25, ob_sensitivity=10):
        df = self.calculate_pivots(df, pivot_length)
        df = self.calculate_order_blocks(df, ob_sensitivity)
        df = self.calculate_fvg(df)
        return df

    def run_backtest(self, df):
        raise NotImplementedError

class MacdAndRSIManager(StrategyManager):
    def calculate_macd(self, data, short_window=12, long_window=26, signal_window=9):
        data['EMA12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
        return data

    def calculate_rsi(self, data, window=14):
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        RS = gain / loss
        data['RSI'] = 100 - (100 / (1 + RS))
        return data

    def is_trending(self, data, window=14, threshold=0.02):
        data['Rolling_Max'] = data['Close'].rolling(window).max()
        data['Rolling_Min'] = data['Close'].rolling(window).min()
        data['Trend_Strength'] = (data['Rolling_Max'] - data['Rolling_Min']) / data['Rolling_Min']
        data['Trending'] = np.where(data['Trend_Strength'] > threshold, 1, 0)
        return data

    def generate_signals(self, data):
        data['Buy_Signal'] = np.where((data['MACD'] > data['Signal_Line']) & (data['RSI'] > 30) & (data['RSI'] < 70) & (
                    data['Trending'] == 1), 1, 0)
        data['Sell_Signal'] = np.where(
            (data['MACD'] < data['Signal_Line']) & (data['RSI'] > 30) & (data['RSI'] < 70) & (data['Trending'] == 1),
            -1, 0)
        return data

    def backtest_strategy(self, data, initial_balance=10000, tp_pct=Decimal('0.01'), sl_pct=Decimal('0.005')):
        balance = initial_balance
        position = 0  # Positive for long, negative for short
        entry_price = 0
        tp_count = 0
        sl_count = 0
        win_trades = 0
        total_trades = 0

        for i in range(len(data)):
            if data['Buy_Signal'].iloc[i] == 1 and position == 0:
                position = balance / data['Close'].iloc[i]
                entry_price = data['Close'].iloc[i]
                balance = 0
                total_trades += 1
            elif data['Sell_Signal'].iloc[i] == -1 and position == 0:
                position = -balance / data['Close'].iloc[i]
                entry_price = data['Close'].iloc[i]
                balance = 0
                total_trades += 1
            elif position > 0:  # Long position
                if data['Close'].iloc[i] >= entry_price * (1 + tp_pct):
                    balance = position * data['Close'].iloc[i] * Decimal('0.998')
                    position = 0
                    tp_count += 1
                    win_trades += 1
                elif data['Close'].iloc[i] <= entry_price * (1 - sl_pct):
                    balance = position * data['Close'].iloc[i] * Decimal('0.998')
                    position = 0
                    sl_count += 1
            elif position < 0:  # Short position
                if data['Close'].iloc[i] <= entry_price * (1 - tp_pct):
                    balance = -position * data['Close'].iloc[i] * Decimal('0.998')
                    position = 0
                    tp_count += 1
                    win_trades += 1
                elif data['Close'].iloc[i] >= entry_price * (1 + sl_pct):
                    balance = -position * data['Close'].iloc[i] * Decimal('0.998')
                    position = 0
                    sl_count += 1

        # If still holding a position, close it at the end
        if position > 0:
            balance = position * data['Close'].iloc[-1]
        elif position < 0:
            balance = -position * data['Close'].iloc[-1]
        # if balance == 0:
        #     balance = initial_balance
        pnl = balance - initial_balance
        self.pnl += pnl
        win_rate = (win_trades / total_trades) * 100 if total_trades > 0 else 0
        return balance, tp_count, sl_count, win_rate

    def run_backtest(self, data):
        df = pd.DataFrame(data)

        # Apply the strategy
        df = self.calculate_macd(df)
        df = self.calculate_rsi(df)
        df = self.is_trending(df)
        df = self.generate_signals(df)

        # Backtest the strategy
        final_balance, tp_count, sl_count, win_rate = self.backtest_strategy(df)
        print(f"Final balance: {final_balance}")
        print(f"Take profit count: {tp_count}")
        print(f"Stop loss count: {sl_count}")
        print(f"Win rate: {win_rate}%")


class ARGIndicator(StrategyManager):
    def __init__(self):
        super().__init__()
        self.first_ma_window = 14
        self.second_ma_window = 50
        self.third_ma_window = 200
        self.signal_threshold = 7
        self.opposite_trend_threshold = Decimal('0.001')
        self.trend_threshold = Decimal('0.002')
        self.change_exposure()
        columns = ['Close', 'High', 'Low', 'ts', 'date', f'SMA_{self.first_ma_window}',
                   f'SMA_{self.second_ma_window}', f'SMA_{self.third_ma_window}', 'first', 'second']
        self.signals = pd.DataFrame(columns=columns)

    def sma(self, df, window=14):
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        return df

    def generate_signals(self, df):
        for index, candle in df.iterrows():
            if index < 7 + self.third_ma_window:
                continue
            if self.first_exposure == self.second_exposure:
                reached = self.check_ema_reached(df=df, index=index, window_size=self.third_ma_window)
                if not reached:
                    reached = self.check_ema_reached(df=df, index=index, window_size=self.second_ma_window)
                    if not reached:
                        reached = self.check_ema_reached(df=df, index=index, window_size=self.first_ma_window)
                if reached is True:
                    self.add_signal(row=candle)
                    # print(f"got signal, timestamp: {candle['ts']}, ema_size: {self.exposure_ema_window} - "
                    #       f"first: {self.first_exposure}, second: {self.second_exposure}")
            done = self.check_if_price_crosses_green_area(df=df, index=index)
            if done is True:
                self.add_signal(row=candle)
                # print(f"finished exposure, candle: {candle}")

    def check_signal_threshold_crosses(self, df, window_size):
        for index, candle in df.iterrows():
            if candle['Low'] < candle[f'SMA_{window_size}'] < candle['High']:
                return True
        return False

    def check_ema_reached(self, df, index, window_size):
        candle = df.iloc[index]
        if self.check_signal_threshold_crosses(df.iloc[index - self.signal_threshold:index],
                                               window_size=window_size) is True:
            return False
        previous_candle = df.iloc[index - 1]
        if candle['Low'] < candle[f'SMA_{window_size}'] < candle['High']:
            if previous_candle['Low'] > candle[f'SMA_{window_size}']:
                self.change_exposure(first=1, second=-1, candle=candle, window_size=window_size)
                return True
            elif previous_candle['High'] < candle[f'SMA_{window_size}']:
                self.change_exposure(first=-1, second=1, candle=candle, window_size=window_size)
                return True
        return False

    def check_if_price_crosses_green_area(self, df, index):
        try:
            df.iloc[index + 1]
        except IndexError:
            return False
        candle = df.iloc[index]
        if self.first_exposure > 0:
            if ((candle['Close'] < Decimal(str(self.exposure_candle[f'SMA_{self.exposure_ema_window}'])) *
                    (Decimal('1') - self.opposite_trend_threshold)) or
                    candle['Close'] > Decimal(str(self.exposure_candle[f'SMA_{self.exposure_ema_window}'])) * (Decimal('1') + self.trend_threshold)):
                self.change_exposure()
                return True
        if self.second_exposure > 0:
            if ((candle['Close'] > Decimal(str(self.exposure_candle[f'SMA_{self.exposure_ema_window}'])) *
                 (Decimal('1') + self.opposite_trend_threshold)) or
                    candle['Close'] < Decimal(str(self.exposure_candle[f'SMA_{self.exposure_ema_window}'])) * (Decimal('1') - self.trend_threshold)):
                self.change_exposure()
                return True
        return False

    def change_exposure(self, first=0, second=0, candle=None, window_size=None):
        self.first_exposure = first
        self.second_exposure = second
        self.exposure_candle = candle
        self.exposure_ema_window = window_size

    def run_backtest(self, data):
        df = self.sma(df=data, window=self.first_ma_window)
        df = self.sma(df=data, window=self.second_ma_window)
        df = self.sma(df=data, window=self.third_ma_window)
        self.generate_signals(df)

    def add_signal(self, row):
        print(row)
        row['first'] = self.first_exposure
        row['second'] = self.second_exposure
        self.signals.loc[len(self.signals.index)] = row

    def create_csv_from_signals(self):
        csv_file_path = 'signals.csv'
        self.signals.to_csv(csv_file_path, index=False)


class TradingViewIndicator(StrategyManager):

    def hull_moving_average(self, df, window):
        wma_half = ta.trend.sma_indicator(df, window // 2)
        wma_full = ta.trend.sma_indicator(df, window)
        hma = ta.trend.sma_indicator(2 * wma_half - wma_full, int(window ** 0.5))
        return hma

    def generate_df(self, recent_candles):
        df = super().generate_df(recent_candles)
        df['EMA10'] = ta.trend.ema_indicator(df['Close'], window=10)
        df['SMA10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['EMA30'] = ta.trend.ema_indicator(df['Close'], window=30)
        df['SMA30'] = ta.trend.sma_indicator(df['Close'], window=30)
        df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA100'] = ta.trend.ema_indicator(df['Close'], window=100)
        df['SMA100'] = ta.trend.sma_indicator(df['Close'], window=100)
        df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
        df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['Ichimoku'] = ta.trend.ichimoku_base_line(df['High'], df['Low'])
        df['VWMA20'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'],
                                                               window=20)
        df['HMA9'] = self.hull_moving_average(df['Close'], 9)

        # Calculate oscillators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['StochK'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        df['AO'] = ta.momentum.awesome_oscillator(df['High'], df['Low'])
        # df['Momentum'] = ta.momentum.momentum(df['Close'], window=10)
        df['Momentum'] = df['Close'].diff(periods=10)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['StochRSI'] = ta.momentum.stochrsi_k(df['Close'], window=14, smooth1=3, smooth2=3)
        df['WilliamsR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
        df['BullBearPower'] = df['Close'] - ta.trend.ema_indicator(df['Close'], window=13)
        df['UltimateOsc'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'], window1=7, window2=14,
                                                            window3=28)
        return df

    def generate_signals(self, row):
        signals = []
        # Moving Averages
        if row['Close'] > row['EMA10']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['SMA10']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['EMA20']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['SMA20']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['EMA30']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['SMA30']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['EMA50']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['SMA50']:
            signals.append('Sell')
        else:
            signals.append('Buy')

        if row['Close'] > row['EMA100']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['SMA100']:
            signals.append('Sell')
        else:
            signals.append('Buy')

        if row['Close'] > row['EMA200']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['SMA200']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['Ichimoku']:
            signals.append('Buy')
        else:
            signals.append('Neutral')

        if row['Close'] > row['VWMA20']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['Close'] > row['HMA9']:
            signals.append('Buy')
        else:
            signals.append('Sell')

        # Oscillators
        if row['RSI'] > 50:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['StochK'] > 80:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['CCI'] > 100:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['ADX'] > 25:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['AO'] > 0:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['Momentum'] > 100:
            signals.append('Sell')
        else:
            signals.append('Neutral')

        if row['MACD'] > 0:
            signals.append('Buy')
        else:
            signals.append('Sell')

        if row['StochRSI'] > 80:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['WilliamsR'] > -20:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['BullBearPower'] > 0:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        if row['UltimateOsc'] > 50:
            signals.append('Neutral')
        else:
            signals.append('Neutral')

        return signals

    def decide_position(self, signals):
        buy_signals = signals.count('Buy')
        sell_signals = signals.count('Sell')
        neutral_signals = signals.count('Neutral')
        threshold = 0.85
        neutral_threshold = 0.5
        if (buy_signals / (buy_signals + sell_signals) >= threshold and
                buy_signals / (buy_signals + sell_signals + neutral_signals) >= neutral_threshold):
            print("long")
            print(buy_signals / (buy_signals + sell_signals))
            print(buy_signals / (buy_signals + sell_signals + neutral_signals))
            return 'Long'
        elif (sell_signals / (buy_signals + sell_signals) >= threshold and
                sell_signals / (buy_signals + sell_signals + neutral_signals) >= neutral_threshold):
            print("short")
            print(sell_signals / (buy_signals + sell_signals))
            print(sell_signals / (buy_signals + sell_signals + neutral_signals))
            return 'Short'
        else:
            return 'Neutral'

    def run_backtest(self, df):
        df['Signals'] = df.apply(self.generate_signals, axis=1)
        df['Position'] = df['Signals'].apply(self.decide_position)
        initial_balance = 100000
        balance = initial_balance
        position = None
        entry_price = 0
        tp = 0.04  # Take profit percentage
        sl = 0.02  # Stop loss percentage
        wins = 0
        losses = 0
        for i in range(1, len(df)):
            if position == 'Long':
                if (df['High'].iloc[i] >= entry_price * (1 + tp)) or (df['Low'].iloc[i] <= entry_price * (1 - sl)):
                    if df['High'].iloc[i] >= entry_price * (1 + tp):
                        wins += 1
                        balance += balance * (df['High'].iloc[i] / entry_price - 1) * float('0.998')
                    else:
                        losses += 1
                        balance += balance * (df['Low'].iloc[i] / entry_price - 1) * float('0.998')
                    position = None
            elif position == 'Short':
                if (df['Low'].iloc[i] <= entry_price * (1 - tp)) or (df['High'].iloc[i] >= entry_price * (1 + sl)):
                    if df['Low'].iloc[i] <= entry_price * (1 - tp):
                        wins += 1
                        balance += balance * (entry_price / df['Low'].iloc[i] - 1) * float('0.998')
                    else:
                        losses += 1
                        balance += balance * (entry_price / df['High'].iloc[i] - 1) * float('0.998')
                    position = None
            else:
                if df['Position'].iloc[i] == 'Long':
                    print("long date: ", df['date'].iloc[i], df['Close'].iloc[i])
                    position = 'Long'
                    entry_price = df['Close'].iloc[i]
                elif df['Position'].iloc[i] == 'Short':
                    print("short date: ", df['date'].iloc[i], df['Close'].iloc[i])
                    position = 'Short'
                    entry_price = df['Close'].iloc[i]
        winrate = wins / (wins + losses) if (wins + losses) > 0 else 0
        # pnl = (balance - initial_balance) / initial_balance
        balance = (1 + (wins * tp) - (losses * sl) - ((wins + losses) * float('0.002'))) * balance
        pnl = (wins * tp) - (losses * sl) - ((wins + losses) * float('0.002'))
        print("total trades : ", wins + losses)
        print("commissions: ", ((wins + losses) * float('0.002')))
        print(f'Winrate: {winrate:.2f}')
        print(f'PnL: {pnl:.2f}')


strategy_manager = StrategyManager()
macd_and_rsi_manager = MacdAndRSIManager()
arg_manager = ARGIndicator()
trading_view_indicator = TradingViewIndicator()
