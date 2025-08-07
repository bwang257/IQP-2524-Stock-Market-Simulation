# region imports
from AlgorithmImports import *
import numpy as np
# endregion

"""
Note that this is a slightly adapted version of the HMM-Regime Adaptive Multi-Alpha Model considered in the IQP report. 
Here, the QQQ ETF was segmented into 4 latent regimes using a HMM to check if the market regimes could be exploited through conditional strategies. 
"""
#Regime 0: Strong Bullish Trend
#Regime 1: Low Volatility Mean Reversion
#Regime 2: Strong bearish trend/ breakdown
#Regime 3: Volatile Bullish Recovery / Choppy Uptrend


class QQQRegimeAlphaModel(AlphaModel):

    def __init__(self, algo: QCAlgorithm):
        self.algo = algo
        self.qqq = Symbol.create("QQQ", SecurityType.EQUITY, Market.USA)
        self.assets = {}
        self.stop_losses = {}

        # self.algo.schedule.on(self.algo.date_rules.every_day(), self.algo.time_rules.every(timedelta(minutes= 30)), self.check_losses)
   
    def check_losses(self):
        if not self.algo.is_market_open:
            return

        for symbol, security in self.algo.securities.items():
            if security.invested:
                quantity = security.holdings.quantity
                if quantity > 0:
                    if symbol not in self.stop_losses:
                        self.stop_losses[symbol] = 0.98 * security.price
                        continue
                    if security.price < self.stop_losses[symbol]:
                        self.algo.liquidate(symbol)
                        self.algo.log(f"{symbol} liquidated due to stop loss: {self.stop_losses[symbol]}")
                        self.stop_losses.pop(symbol, None)
                elif quantity < 0:
                    if symbol not in self.stop_losses:
                        self.stop_losses[symbol] = 1.02 * security.price
                        continue
                    if security.price > self.stop_losses[symbol]:
                        self.algo.liquidate(symbol)
                        self.algo.log(f"{symbol} liquidated due to stop loss: {self.stop_losses[symbol]}")
                        self.stop_losses.pop(symbol, None)
            else:
                if symbol in self.stop_losses:
                    self.stop_losses.pop(symbol, None)


    def OnSecuritiesChanged(self, algo: QCAlgorithm, changes: SecurityChanges):

        for security in changes.added_securities:
            symbol = security.Symbol
            if symbol == self.qqq:
                continue
            self.assets[symbol] = AssetState(self.algo, symbol)
        
        for security in changes.removed_securities:
            symbol = security.Symbol
            if symbol in self.assets:
                self.algo.deregister_indicator(self.assets[symbol].vwap)
                self.algo.deregister_indicator(self.assets[symbol].atr)
                self.algo.deregister_indicator(self.assets[symbol].rsi)
                self.algo.deregister_indicator(self.assets[symbol].adx)
                self.algo.deregister_indicator(self.assets[symbol].bb)
                self.assets.pop(symbol, None)

    def Update(self, algo: QCAlgorithm, slice: Slice):

        insights = []
        vwap_devs = {}
        weights = {}

        for symbol, trade_bar in slice.bars.items():
            if symbol not in self.assets:
                continue
            asset_state = self.assets[symbol]

            if not asset_state.vwap.is_ready:
                continue
            vwap_dev = trade_bar.close / asset_state.vwap.current.value
            vwap_devs[symbol] = vwap_dev

        if len(vwap_devs) == 0:
            return insights
                
        vwap_devs_mean = sum(vwap_devs.values()) / len(vwap_devs.values())
        
        for symbol in list(vwap_devs.keys()):
            self.assets[symbol].vwap_devs.add(vwap_devs[symbol]-vwap_devs_mean)
        
        zscores = {}
        for symbol in list(vwap_devs.keys()):
            hist_vwap_devs = list(self.assets[symbol].vwap_devs)
            mu = np.mean(hist_vwap_devs)
            sigma = np.std(hist_vwap_devs)
            if sigma == 0:
                continue
            z = (self.assets[symbol].vwap_devs[0] - mu) / sigma
            zscores[symbol] = z
    
        for symbol, zscore in zscores.items():
            if not np.isnan(zscore):
                weight = zscore
            else:
                weight = 0
            weights[symbol] = weight

        final_weights = {}
        abs_weight = {key: abs(val) for key, val in weights.items()}
        weights_sum = sum(abs_weight.values())
        if weights_sum != 0:
            for symbol, weight in weights.items():
                final_weights[symbol] = weight/ weights_sum

            for symbol, final_weight in final_weights.items():
                if 5> weights[symbol] > 3:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.DOWN, weight=final_weight))
                    price = self.algo.portfolio[symbol].price
                elif 3> weights[symbol] > 0:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.UP, weight=final_weight))
                    price = self.algo.portfolio[symbol].price
                
                elif 0 > weights[symbol] > -2:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.DOWN, weight=final_weight))
                    price = self.algo.portfolio[symbol].price
                elif -2 > weights[symbol]:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.UP, weight=final_weight))
                    price = self.algo.portfolio[symbol].price
                
                    # stop_loss = 0.98 * price
                    # self.stop_losses[symbol] = stop_loss
                    # algo.log(f"Long Insight Emitted for {symbol} with z-score {weight * weights_sum} and  stop_loss {stop_loss}")

                # elif weight < 0:
                #     insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.DOWN, weight=weight))
                #     price = self.algo.portfolio[symbol].price
                #     stop_loss = 1.02 * price
                #     self.stop_losses[symbol] = stop_loss
                #     algo.log(f"Short Insight Emitted for {s

        return insights

    
class AssetState:

    def __init__(self, algo: QCAlgorithm, symbol: Symbol):
        self.algo = algo
        self.symbol = symbol

        self.vwap = self.algo.vwap(symbol, 10)
        self.atr = self.algo.atr(symbol, 7, MovingAverageType.WILDERS)
        self.adx = self.algo.adx(symbol, 7)
        self.bb = self.algo.bb(symbol, 10, 2)
        self.rsi = self.algo.rsi(symbol, 7, MovingAverageType.WILDERS)
        self.rsi_window = RollingWindow[float](2)
        self.vwap_devs = RollingWindow[float](7)

        #Load Indicator With Data
        self.algo.indicator_history(self.vwap, self.symbol, timedelta(days=10), Resolution.DAILY)
        self.algo.indicator_history(self.atr, self.symbol, timedelta(days=7), Resolution.DAILY)
        self.algo.indicator_history(self.rsi, self.symbol, timedelta(days=7), Resolution.DAILY)
        self.algo.indicator_history(self.adx, self.symbol, timedelta(days=7), Resolution.DAILY)
        self.algo.indicator_history(self.bb, self.symbol, timedelta(days=10), Resolution.DAILY)
    
    def ready(self):
        return self.vwap.is_ready and self.atr.is_ready and self.rsi.is_ready and self.adx.is_ready and self.bb.is_ready and self.rsi.is_ready and self.vwap_devs.is_ready
