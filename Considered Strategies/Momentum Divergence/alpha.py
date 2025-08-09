from AlgorithmImports import *
from hurst_oracle import hurst_lssd
from trendCalculator import get_trend
from macd_oracle import get_macd_score
from bollinger_oracle import get_bollinger_buy_and_short
from rsi_oracle import get_rsi_buy_short
from datetime import timedelta
from collections import deque
import numpy as np

"""
As mentioned in the IQP report, this alpha model is inspired by and applies the trend-following logic of Charles Hacker's trend-following algorithm.
https://www.quantconnect.com/league/19294/2025-q3/trend-following/p1
"""

class adapted_alpha(AlphaModel):

    class bollinger_holder:
       def __init__(self, lower, middle, upper, price):
           self.lower = lower
           self.middle = middle
           self.upper = upper
           self.price = price
   
    class macd_holder:
       def __init__(self, fast, slow, signal, macd, hist):
           self.fast = fast
           self.slow = slow
           self.signal = signal
           self.macd = macd
           self.hist = hist

    def __init__(self, algo):
        self.algo = algo
        self._symbols = set()


        # region Indicator Parameters

        # MACD Parameters
        self.macd_params = {'cross_check_length': 35, 'macd_above_below_length': 28, 'long_macd_threshold': 0.25, 
                            'short_macd_threshold': -0.25}
        self.macd_window_size = 35

        # Bollinger Band Parameters
        self.Bollinger_window_size = 25
        self.long_threshold = 1
        self.bollinger_params = {'long_threshold': self.long_threshold, 'short_threshold': self.long_threshold}

        # EMA parameters
        self.ema_rolling_window_length = 250
        self.derivative_threshold = .005

        # RSI Parameters 
        self.rsi_rolling_window_length = 30

        # ADX parameters
        self.adx_rolling_window_length = 30
        self.adx_threshold = 30

        # OBV parameters
        self.obv_rolling_window_length = 150
        self.obv_threshold = .5          

        # Hurst parameters
        self.hurst_rolling_window_length = 100
        self.hurst_mom_threshold = 0.55
        self.hurst_rev_threshold = 0.50
    
        # ATR parameters / Stop Loss
        self.atr_stop_multiplier = 3
        self.atr_stop_multiplier_pullback = 1.5

        # Price Rolling Window
        self.price_rolling_window_length = 30
        
        # Establish Indicator Dicts
        self.MACDs = {}
        self.macd_consolidators = {}
        self.MACDs_rolling_windows = {}
        self.Bollingers = {}
        self.bollinger_consolidators = {}
        self.Bollingers_rolling_windows = {}
        self.EMA_50s = {}
        self.EMA50s_rolling_windows = {}
        self.ema50_consolidators = {}
        self.EMA_200s = {}
        self.EMA200s_rolling_windows = {}
        self.ema200_consolidators = {}
        self.daily_RSIs = {} 
        self.hourly_RSIs = {}
        self.rsi_consolidators = {}
        self.RSIs_rolling_windows = {}
        self.ADXs = {}
        self.adx_consolidators = {}
        self.ADXs_rolling_windows = {}
        self.OBVs = {}
        self.obv_consolidators = {}
        self.OBVs_rolling_windows = {}
        self.ATRs = {}
        self.atr_consolidators = {}
        self.Hursts_rolling_windows = {}
        self.Hursts= {}

        self.peak_prices = {}

        self.price_rolling_windows = {}

        # Portfolio Management
        self.port_bias = 500
        self.trend_follow_bias = 700
        self.entry_scores = {}
        self.look_for_entries = {}
        self.insight_types = {}
        self.insight_expiry = 7
        self.insight_expiry_sell = 4
        self.insight_expiry_tf= 14
        self.insight_expiry_tf_sell= 7

        # Trend Parameters
        self.trend_order = 5
        self.K_order = 2
        self.trend_history_size = 30

        self.rsi_trend_order = 5
        self.rsi_K_order = 2

        self.macd_trend_order = 5
        self.macd_K_order = 2

        self.obv_trend_order = 2
        self.obv_K_order = 2

        # endregion

    def on_securities_changed(self, algo, changes):

        for security in changes.removed_securities:
            if security.Symbol in self._symbols:
                self._symbols.remove(security.Symbol)
        
        for security in changes.added_securities:
            symbol = security.Symbol
            self._symbols.add(symbol)

            # region Initialize Indicators and Rolling Windows
            self.price_rolling_windows[symbol] = RollingWindow[float](self.price_rolling_window_length)

            self.MACDs[symbol] = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.EXPONENTIAL)
            self.macd_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.MACDs[symbol], self.macd_consolidators[symbol])
            self.MACDs_rolling_windows[symbol] = deque(maxlen=self.macd_window_size)

            self.Bollingers[symbol] = BollingerBands(20, 2, MovingAverageType.SIMPLE)
            self.bollinger_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.Bollingers[symbol], self.bollinger_consolidators[symbol])
            self.Bollingers_rolling_windows[symbol] = deque(maxlen=self.Bollinger_window_size)
   
            self.hourly_RSIs[symbol] = algo.rsi(symbol, 14, Resolution.Hour)
            self.daily_RSIs[symbol] = RelativeStrengthIndex(14)
            self.rsi_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.daily_RSIs[symbol], self.rsi_consolidators[symbol])
            self.RSIs_rolling_windows[symbol] = RollingWindow[float](self.rsi_rolling_window_length)

            self.EMA_200s[symbol] = ExponentialMovingAverage(200)
            self.ema200_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.EMA_200s[symbol], self.ema200_consolidators[symbol])
            self.EMA200s_rolling_windows[symbol] = RollingWindow[float](self.ema_rolling_window_length)
            
            self.EMA_50s[symbol] = ExponentialMovingAverage(50)
            self.ema50_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.EMA_50s[symbol], self.ema50_consolidators[symbol])
            self.EMA50s_rolling_windows[symbol] = RollingWindow[float](self.ema_rolling_window_length)

            self.ADXs[symbol] = AverageDirectionalIndex(14)
            self.adx_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.ADXs[symbol], self.adx_consolidators[symbol])
            self.ADXs_rolling_windows[symbol] = RollingWindow[float](self.adx_rolling_window_length)

            self.OBVs[symbol] = OnBalanceVolume()
            self.obv_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.OBVs[symbol], self.obv_consolidators[symbol])
            self.OBVs_rolling_windows[symbol] = RollingWindow[float](self.obv_rolling_window_length)

            self.Hursts_rolling_windows[symbol] = RollingWindow[float](self.hurst_rolling_window_length)

            self.ATRs[symbol] = AverageTrueRange(14)
            self.atr_consolidators[symbol] = TradeBarConsolidator(timedelta(days=1))
            algo.register_indicator(symbol, self.ATRs[symbol], self.atr_consolidators[symbol])

            self.Hursts[symbol] = None


            # endregion

            hourly_history = algo.History[TradeBar](symbol, self.ema_rolling_window_length * 3, Resolution.HOUR)
            daily_history = algo.History[TradeBar](symbol, self.ema_rolling_window_length * 3, Resolution.DAILY)

            for bar in hourly_history:
                self.hourly_RSIs[symbol].Update(bar.EndTime, bar.Close)
                if self.hourly_RSIs[symbol].is_ready:
                    self.RSIs_rolling_windows[symbol].add(self.hourly_RSIs[symbol].current.value)
                
            for bar in daily_history:
                self.price_rolling_windows[symbol].add(bar.Close)
                if self.price_rolling_windows[symbol].count > 1:
                    self.Hursts_rolling_windows[symbol].add(np.log(bar.Close / self.price_rolling_windows[symbol][1]))
                macd = self.MACDs[symbol]
                macd.Update(bar.EndTime, bar.Close)
                new_macd_holder = self.macd_holder(macd.fast.current.value, macd.slow.current.value, macd.signal.current.value, macd.current.value, macd.histogram.current.value)
                self.MACDs_rolling_windows[symbol].append(new_macd_holder)

                bb = self.Bollingers[symbol]
                bb.Update(bar.EndTime, bar.Close)
                new_bol_holder = self.bollinger_holder(bb.lower_band.current.value, bb.middle_band.current.value, bb.upper_band.current.value, bar.Close)
                self.Bollingers_rolling_windows[symbol].append(new_bol_holder)

                self.daily_RSIs[symbol].Update(bar.EndTime, bar.Close)

                self.EMA_50s[symbol].Update(bar.EndTime, bar.Close)
                if self.EMA_50s[symbol].is_ready:
                    self.EMA50s_rolling_windows[symbol].add(self.EMA_50s[symbol].current.value)
                self.EMA_200s[symbol].Update(bar.EndTime, bar.Close)
                if self.EMA_200s[symbol].is_ready:
                    self.EMA200s_rolling_windows[symbol].add(self.EMA_200s[symbol].current.value)
                
                self.ADXs[symbol].Update(bar)
                self.ADXs_rolling_windows[symbol].add(self.ADXs[symbol].current.value)

                self.OBVs[symbol].Update(bar)
                self.OBVs_rolling_windows[symbol].add(self.OBVs[symbol].current.value)

                self.ATRs[symbol].Update(bar)

    def Update(self, algo, data):
        insights = []

        for symbol in self._symbols:

            if not data.ContainsKey(symbol) or data[symbol] is None:
                # no data
                continue

            # region Update Indicators

            if data[symbol].EndTime.hour == 10 and data[symbol].EndTime.minute == 0:
                
                self.price_rolling_windows[symbol].add(data[symbol].Close)
                self.Hursts_rolling_windows[symbol].add(np.log(data[symbol].Close / self.price_rolling_windows[symbol][1]))
    
                bb = self.Bollingers[symbol]
                self.Bollingers_rolling_windows[symbol].append(self.bollinger_holder(bb.lower_band.current.value, bb.middle_band.current.value, bb.upper_band.current.value, data[symbol].price))
                
                macd = self.MACDs[symbol]
                self.MACDs_rolling_windows[symbol].append(self.macd_holder(macd.fast.current.value, macd.slow.current.value, macd.signal.current.value, macd.current.value, macd.histogram.current.value))

                self.RSIs_rolling_windows[symbol].add(self.daily_RSIs[symbol].current.value)
                self.EMA50s_rolling_windows[symbol].add(self.EMA_50s[symbol].current.value)
                self.EMA200s_rolling_windows[symbol].add(self.EMA_200s[symbol].current.value)
                self.OBVs_rolling_windows[symbol].add(self.OBVs[symbol].current.value)
                self.ADXs_rolling_windows[symbol].add(self.ADXs[symbol].current.value)

                hurst_input = [r for r in (self.Hursts_rolling_windows[symbol])]
                hurst_input.reverse
                self.Hursts[symbol] = hurst_lssd(hurst_input)

            # endregion
            price_trend = get_trend(self.price_rolling_windows[symbol], self.trend_order, self.K_order, self.RSIs_rolling_windows[symbol], 0.002)
            obv_trend = get_trend(self.OBVs_rolling_windows[symbol], self.obv_trend_order, self.obv_K_order, None)
        
            hurst = self.Hursts[symbol]
            hurst_trending = hurst >= self.hurst_mom_threshold
            hurst_mean_revert = hurst <= self.hurst_rev_threshold

            N = 5
            ema50s = [x for x in self.EMA50s_rolling_windows[symbol]][:N]
            ema200s = [x for x in self.EMA200s_rolling_windows[symbol]][:N]
            if len(ema50s) != len(ema200s):
                continue
            ema_bullish_for_while = np.all(ema50s > ema200s)
            ema_bearish_for_while = np.all(ema50s < ema200s)

            obv_confirms_up   = obv_trend["net"] > self.obv_threshold
            obv_confirms_down = obv_trend["net"] < -self.obv_threshold

            current_adx = self.ADXs[symbol].current.value
            strong_trend = current_adx > self.adx_threshold
            weak_trend = current_adx < self.adx_threshold


            if price_trend['bull_pullback'] and ema_bullish_for_while and obv_confirms_up and hurst_trending and strong_trend:
                    p_score = max(price_trend['net'], 1)
                    obv_score = max(obv_trend['net'], 1)
                    adx_score = min(current_adx / self.adx_threshold, 1.0)
                    hurst_score = min(abs(hurst - 0.5) * 2, 1.0)
                    score = (p_score * obv_score * adx_score * hurst_score * 100 + self.port_bias)

                    open_orders = algo.Transactions.GetOpenOrders(symbol)
                    if not algo.Portfolio[symbol].Invested and len(open_orders) == 0:
                        if symbol not in self.look_for_entries or self.look_for_entries[symbol] == 0:
                            self.look_for_entries[symbol] = 1
                            self.entry_scores[symbol] = score
                            self.insight_types[symbol] = 'bull_pullback'
        
                    # true “buy the dip” in a strong uptrend
                
            elif price_trend['bull_reversal'] and ema_bearish_for_while and obv_confirms_up and hurst_mean_revert and weak_trend:                
                    # reversal right after a downtrend has lost steam
                    p_score = max(price_trend['net'], 1)
                    obv_score = max(obv_trend['net'], 1)
                    adx_score = (self.adx_threshold - current_adx) / self.adx_threshold
                    hurst_score = min(abs(hurst - 0.5) * 2, 1.0)
                    score = (p_score * obv_score * adx_score * hurst_score * 100 + self.port_bias)

                    open_orders = algo.Transactions.GetOpenOrders(symbol)
                    if not algo.Portfolio[symbol].Invested and len(open_orders) == 0:
                        if symbol not in self.look_for_entries or self.look_for_entries[symbol] == 0:
                            self.look_for_entries[symbol] = 1
                            self.entry_scores[symbol] = score
                            self.insight_types[symbol] = 'bull_reversal'

            elif price_trend['bear_pullback'] and ema_bearish_for_while and obv_confirms_down and hurst_trending and strong_trend:
                    # true “sell the bounce” in a strong downtrend
                    p_score = max(-price_trend['net'], 1)
                    obv_score = max(-obv_trend['net'], 1)
                    adx_score = min(current_adx / self.adx_threshold, 1.0)
                    hurst_score = min(abs(hurst - 0.5) * 2, 1.0)
                    score = (0.8*p_score * obv_score * adx_score * hurst_score * 100 + self.port_bias)

                    open_orders = algo.Transactions.GetOpenOrders(symbol)
                    if not algo.Portfolio[symbol].Invested and len(open_orders) == 0:
                        if symbol not in self.look_for_entries or self.look_for_entries[symbol] == 0:
                            self.look_for_entries[symbol] = -1
                            self.entry_scores[symbol] = score
                            self.insight_types[symbol] = 'bear_pullback'
            
            elif price_trend['bear_reversal']and ema_bullish_for_while and obv_confirms_down and hurst_mean_revert and weak_trend:

                    p_score = max(-price_trend['net'], 1)
                    obv_score = max(-obv_trend['net'], 1)
                    adx_score = (self.adx_threshold - current_adx) / self.adx_threshold
                    adx_score = min(max(adx_score, 0.0), 1.0)
                    hurst_score = min(abs(hurst - 0.5) * 2, 1.0)
                    score = (0.8*p_score * obv_score * adx_score * hurst_score * 100 + self.port_bias)

                    # reversal right after an uptrend has broken down
                    open_orders = algo.Transactions.GetOpenOrders(symbol)
                    if not algo.Portfolio[symbol].Invested and len(open_orders) == 0:
                        if symbol not in self.look_for_entries or self.look_for_entries[symbol] == 0:
                            self.look_for_entries[symbol] = -1
                            self.entry_scores[symbol] = score
                            self.insight_types[symbol] = 'bear_reversal'

            else:
                # trend following
                rsi_trend = get_trend(self.RSIs_rolling_windows[symbol], self.rsi_trend_order, self.rsi_K_order, None)
                ema_trend = 0
                ema50s = [x for x in self.EMA50s_rolling_windows[symbol]]
                ema200s = [x for x in self.EMA200s_rolling_windows[symbol]]
                if len(ema50s) != len(ema200s):
                    continue
                for i in range(len(ema50s)):
                    if ema50s[i] > ema200s[i]:
                        ema_trend += 1

                bollinger_score_buy_short = get_bollinger_buy_and_short(algo, self.Bollingers_rolling_windows[symbol], 1, self.bollinger_params)
                macd_score = get_macd_score(self.MACDs_rolling_windows[symbol], 1, self.macd_params)  
                rsi_score = get_rsi_buy_short(price_trend['net'], rsi_trend['net'])

                ema50s.reverse()
                if len(ema50s) < 2:
                    derivative = 0
                else:
                    derivative = np.gradient(ema50s)[-1]/self.EMA50s_rolling_windows[symbol][0]
                
                if (ema_trend >= 210 and bollinger_score_buy_short == 1 and 
                    macd_score == 1 and rsi_score == 1 and
                    derivative > self.derivative_threshold and
                    self.ADXs[symbol].Current.Value > self.adx_threshold):

                    max_adx = max(self.ADXs_rolling_windows[symbol])
                    current_adx = self.ADXs[symbol].Current.Value
                    if current_adx >= max_adx * .95:
                        if obv_trend['net'] > self.obv_threshold:
                            open_orders = algo.Transactions.GetOpenOrders(symbol)
                            if not algo.Portfolio[symbol].Invested and len(open_orders) == 0:
                                if symbol not in self.look_for_entries or self.look_for_entries[symbol] == 0:
                                    self.look_for_entries[symbol] = 1
                                    self.entry_scores[symbol] = abs(int(derivative * self.ADXs[symbol].Current.Value * max(price_trend['net'], 1) * max(rsi_trend['net'], 1) * max(obv_trend['net'], 1) * 100 + self.trend_follow_bias))
                                    self.insight_types[symbol] = 'trend_follow'

                elif (bollinger_score_buy_short == 2 and macd_score == 2 and
                    rsi_score == 2 and derivative < -self.derivative_threshold and
                    self.ADXs[symbol].Current.Value > self.adx_threshold):
                    
                    min_adx = min(self.ADXs_rolling_windows[symbol])
                    current_adx = self.ADXs[symbol].Current.Value
                    if current_adx <= min_adx * 1.05:
                        if obv_trend['net'] < -self.obv_threshold:
                            open_orders = algo.Transactions.GetOpenOrders(symbol)
                            if not algo.Portfolio[symbol].Invested and len(open_orders) == 0:
                                self.look_for_entries[symbol] = -1
                                self.entry_scores[symbol] = abs(int(derivative * self.ADXs[symbol].Current.Value * max(price_trend['net'], 1) * max(rsi_trend['net'], 1) * max(obv_trend['net'], 1) * 100 + self.trend_follow_bias))
                                self.insight_types[symbol] = 'trend_follow'

        if len(self.look_for_entries.keys()) > 0:
            for key in self.look_for_entries:
                if self.look_for_entries[key] > 0:
                    self.look_for_entries[key] += 1
                    max_age = 20 if self.insight_types[key] != 'trend_follow' else 70
                    if self.look_for_entries[key] > max_age:
                        self.look_for_entries[key] = 0
                    else:
                        if self.insight_types[key] == 'trend_follow' and self.price_rolling_windows[key][0] > self.Bollingers[key].middle_band.current.value:
                            insight = Insight.price(key, timedelta(days=self.insight_expiry_tf), InsightDirection.Up, weight = self.entry_scores[key])
                            self.peak_prices[key] = data[key].price
                            insights.append(insight)
                            self.look_for_entries[key] = 0
                        else:
                            bb1 = self.Bollingers_rolling_windows[key][-1]
                            bb2 = self.Bollingers_rolling_windows[key][-2]
                            curr_price, prev_price = bb1.price, bb2.price
                            curr_mid, prev_mid = bb1.middle, bb2.middle

                            if self.insight_types[key] == 'bull_pullback' and curr_price > curr_mid:
                                    insight = Insight.price(key, timedelta(days = self.insight_expiry), InsightDirection.UP, weight = self.entry_scores[key], tag='bull_pullback')
                                    self.peak_prices[key] = data[key].price
                                    insights.append(insight)
                                    self.look_for_entries[key] = 0

                            elif self.insight_types[key] == 'bull_reversal' and prev_price <= prev_mid and curr_price > curr_mid:
                                    insight = Insight.price(key, timedelta(days = self.insight_expiry), InsightDirection.UP, weight = self.entry_scores[key], tag='bull_reversal')
                                    self.peak_prices[key] = data[key].price
                                    insights.append(insight)
                                    self.look_for_entries[key] = 0

                elif self.look_for_entries[key] < 0:
                    self.look_for_entries[key] -= 1
                    max_age = -20 if self.insight_types[key] != 'trend_follow' else -70
                    if self.look_for_entries[key] < max_age:
                        self.look_for_entries[key] = 0
                    else:
                        if self.insight_types[key] == 'trend_follow' and self.price_rolling_windows[key][0] < self.Bollingers[key].middle_band.current.value:
                            insight = Insight.price(key, timedelta(days=self.insight_expiry_tf_sell), InsightDirection.DOWN, weight = self.entry_scores[key])
                            self.peak_prices[key] = data[key].price
                            insights.append(insight)
                            self.look_for_entries[key] = 0
                        else:

                            bb1 = self.Bollingers_rolling_windows[key][-1]
                            bb2 = self.Bollingers_rolling_windows[key][-2]
                            curr_price, prev_price = bb1.price, bb2.price
                            curr_mid, prev_mid = bb1.middle, bb2.middle

                            if self.insight_types[key] == 'bear_pullback' and curr_price < curr_mid:
                                    insight = Insight.price(key, timedelta(days = self.insight_expiry_sell), InsightDirection.DOWN, weight = self.entry_scores[key], tag='bear_pullback')
                                    self.peak_prices[key] = data[key].price
                                    insights.append(insight)
                                    self.look_for_entries[key] = 0

                            elif self.insight_types[key] == 'bear_reversal' and prev_price >= prev_mid and curr_price < curr_mid:
                                    insight = Insight.price(key, timedelta(days = self.insight_expiry_sell), InsightDirection.DOWN, weight = self.entry_scores[key], tag='bear_reversal')
                                    self.peak_prices[key] = data[key].price
                                    insights.append(insight)
                                    self.look_for_entries[key] = 0
                
        added_insights = self.atr_trail_stop_loss(algo, data)
        for insight in added_insights:
            insights.append(insight)
                    
        return insights

    def atr_trail_stop_loss(self, algo, data):
        added_insights = []
        for key in algo.Portfolio.Keys:
            if key in self.peak_prices and self.peak_prices[key] != None:
                multiplier = self.atr_stop_multiplier if self.insight_types[key] not in ('bull_pullback', 'bear_pullback') else self.atr_stop_multiplier_pullback
                if algo.Portfolio[key].Quantity > 0:
                    if key in data and data[key] != None:
                        price = data[key].price
                    else:
                        price = self.price_rolling_windows[key][0]
                    if price > self.peak_prices[key]: 
                        self.peak_prices[key] = price
                    
                    if price < self.peak_prices[key] - multiplier * self.ATRs[key].current.value:
                        added_insights.append(Insight.price(key, timedelta(days=7), InsightDirection.Flat, weight = 1))
                        algo.Log("liquidating long " + str(key) + " price is: " + str(price) + " peak price is: " + str(self.peak_prices[key]) + " atr is: " + str(self.ATRs[key].Current.Value))
                        algo.Liquidate(key)
                        self.peak_prices[key] = None
                else:
                    if key in data and data[key] != None:
                        price = data[key].price
                    else:
                        price = self.price_rolling_windows[key][0]
                    if price < self.peak_prices[key]:
                        self.peak_prices[key] = price
                    if price > self.peak_prices[key] + multiplier * self.ATRs[key].current.value:
                        added_insights.append(Insight.price(key, timedelta(days=7), InsightDirection.Flat, weight = 1))
                        algo.Log("liquidating short " + str(key) + " price is: " + str(price) + " peak price is: " + str(self.peak_prices[key]) + " atr is: " + str(self.ATRs[key].Current.Value))
                        algo.Liquidate(key)
                        self.peak_prices[key] = None

        return added_insights


                
            
            
            


