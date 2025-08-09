# region imports
from AlgorithmImports import *
import numpy as np
from collections import deque
import pandas as pd
from scipy.stats import zscore
from datetime import timedelta
from collections import deque, defaultdict
import math
# endregion


class HurstExponentAlgorithm(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2024, 1, 1)
        self.set_end_date(2025, 1, 1) 
        self.set_cash(1_000_000)
        self.set_benchmark(self.add_equity("SPY").Symbol)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0
        self.settings.rebalance_portfolio_on_security_changes = False
        self.settings.rebalance_portfolio_on_insight_changes = False
        self.day = -1
        self.set_warm_up(timedelta(60))

        self.universe_settings.asynchronous = True
        self.add_alpha(MomentumReversionAlphaModel(self))

        self.set_portfolio_construction(InsightWeightingPortfolioConstructionModel())
        
        self.add_risk_management(MaximumDrawdownPercentPerSecurity(0.02))

        self.set_execution(ImmediateExecutionModel())

class MomentumReversionAlphaModel(AlphaModel):
    def __init__(self, algo: QCAlgorithm):
        
        self.algo = algo
        self.indicators = {}
        self.windows = {}
        self.positions = {}
        self.historical_ratios = {}
        self.insight_period = timedelta(days= 2)

        self.tickers = [
            # Core Market Indexes
            "SPY", "QQQ", "DIA", "VTI", "IWM", "VOO", "IVV",

            # Leveraged Indexes (for regime/volatility sensitivity)
            "TQQQ", "SQQQ",

            # Sector ETFs (broader coverage)
            "XLF", "XLE", "XLK", "XLY", "XLV", "XLP", "XLB", "XLU", "XLI",

            # Fixed Income
            "TLT", "IEF", "AGG", "LQD", "HYG",

            # Commodities and FX
            "GLD", "UUP",

            # International / Emerging Markets
            "EEM", "EFA",

            # Volatility (optional, use with caution)
            "VXX", "UVXY"
        ]


        self.symbols = [self.algo.add_equity(ticker, Resolution.HOUR).Symbol for ticker in self.tickers]
        for symbol in self.symbols:
            self.initialize_symbol(algo, symbol)


    def initialize_symbol(self, algo, symbol):

        self.indicators[symbol] = {
            "ema20": algo.ema(symbol, 20, Resolution.DAILY),
            # "ema65": algo.ema(symbol, 65, Resolution.DAILY),
            "hurst": algo.he(symbol, 40, 20, Resolution.DAILY),
            "hurst2": algo.he(symbol, 40, 20, Resolution.HOUR),
            "vwap": algo.vwap(symbol, 30, Resolution.HOUR),
            "obv": algo.obv(symbol, Resolution.HOUR)
        }

        self.windows[symbol] = {
            "hurst": RollingWindow[float](20),
            "obv": RollingWindow[float](10)
        }

        self.historical_ratios[symbol] = {
            "mom": RollingWindow[float](20),
            "rev": RollingWindow[float](20)
        }

        indicators = self.indicators[symbol]
        windows = self.windows[symbol]
        historical_ratios = self.historical_ratios[symbol]
        

    def indicators_ready(self, symbol):
        if symbol not in self.indicators:
            self.algo.debug(f"This shouldn't happen: {symbol} not in self.indicators")
            return False
        for indicator in self.indicators[symbol].values():
            if not indicator.is_ready:
                return False
        
        return True
    
    def windows_ready(self, symbol):
        if symbol not in self.windows:
            self.algo.debug(f"This shouldn't happen: {symbol} not in self.windows")
            return False

        for window in self.windows[symbol].values():
            if not window.is_ready:
                return False
        return True
        
    def update_windows(self, symbol, bar):
        if not self.indicators_ready(symbol):
            return

        indicators = self.indicators[symbol]
        windows = self.windows[symbol]

        windows["hurst"].add(indicators["hurst"].current.value)
        windows["obv"].add(indicators["obv"].current.value)

    def update(self, algorithm: QCAlgorithm, slice: Slice) -> List[Insight]:
        insights = []
        mom_list = {}
        rev_list = {}

        for symbol, trade_bar in slice.bars.items():

            if symbol not in self.symbols:
                continue
            if symbol not in self.indicators:
                self.initialize_symbol(algorithm, symbol)

            self.update_windows(symbol, trade_bar)

            if not self.windows_ready(symbol):
                continue

            indicators = self.indicators[symbol]
            windows = self.windows[symbol]

            hursts = [h for h in windows["hurst"]]
            hursts.reverse()
            hurst_derivative = np.gradient(hursts)
            hurst_deriv1 = hurst_derivative[-1]
            hurst_deriv2 = np.gradient(hurst_derivative)[-1]

            obvs = [o for o in windows["obv"]]
            obvs.reverse()
            obv_derivative = np.gradient(obvs)
            obv_deriv1 = obv_derivative[-1]
            obv_deriv2 = np.gradient(obv_derivative)[-1]   

            """
            Below is the layered derivative threshold logic removed to allow for more trade signals.
            Not discussed in the IQP report, the logic here actually uses a 20-day and 60-day EMA. 
            """
            # if self.indicators_ready(symbol):
            #     self.update_windows(symbol, bar)
            #     if self.windows_ready(symbol):
            #         hursts = [h for h in windows["hurst"]]
            #         hursts.reverse()
            #         hurst_derivative = np.gradient(hursts)
            #         hurst_deriv1 = hurst_derivative[-1]
            #         hurst_deriv2 = np.gradient(hurst_derivative)[-1]

            #         if hurst_deriv1 > 0 and hurst_deriv2 > 0 and indicators["hurst"].current.value > 0.5:

            #             if bar.Close == 0: continue

            #             ema20_ratio = bar.close / indicators["ema20"].current.value
            #             ema65_ratio = bar.close / indicators["ema65"].current.value
            #             vwap_ratio = bar.close / indicators["vwap"].current.value
            #             mom_ratio = 0.3 * vwap_ratio + 0.5 * ema20_ratio + 0.2 * ema65_ratio
            #             historical_ratios["mom"].add(mom_ratio)
                    
            #         elif hurst_deriv1 < 0 and hurst_deriv2 < 0 and indicators["hurst"].current.value < 0.5:

            #             if bar.Close == 0: continue

            #             ema20_ratio = bar.close / indicators["ema20"].current.value
            #             ema65_ratio = bar.close / indicators["ema65"].current.value
            #             vwap_ratio = bar.close / indicators["vwap"].current.value
            #             rev_ratio = 0.3 * vwap_ratio + 0.5 * ema20_ratio + 0.2 * ema65_ratio
            #             historical_ratios["rev"].add(rev_ratio)

            if indicators["hurst2"].current.value > 0.6 and indicators["hurst"].current.value > 0.6:

                if trade_bar.close == 0: continue

                ema20_ratio = trade_bar.close - indicators["ema20"].current.value
                vwap_ratio = trade_bar.close - indicators["vwap"].current.value
                mom_ratio = 0.8 * vwap_ratio + 0.2 * ema20_ratio

                if mom_ratio > 0 and obv_deriv1 > 0:
                    mom_list[symbol] = mom_ratio
                elif mom_ratio < 0 and obv_deriv1 < 0:
                    mom_list[symbol] = mom_ratio
            
            elif indicators["hurst2"].current.value < 0.4 and indicators["hurst"].current.value < 0.4:

                if trade_bar.close == 0: continue

                ema20_ratio = trade_bar.close - indicators["ema20"].current.value 
                vwap_ratio = trade_bar.close - indicators["vwap"].current.value
                rev_ratio = 0.8 * vwap_ratio + 0.2 * ema20_ratio

                if rev_ratio > 0 and obv_deriv2 < 0:
                    rev_list[symbol] = rev_ratio
                elif rev_ratio < 0 and obv_deriv2 < 0:
                    rev_list[symbol] = rev_ratio

        mom_zscores = {}
        for symbol, mom_ratio in mom_list.items():
            ratios = self.historical_ratios[symbol]["mom"]
            ratios.add(mom_ratio)
            if ratios.count > 5:
                mean_ratio = np.mean([r for r in ratios])
                std_ratio = np.std([r for r in ratios])
                z = (ratios[0] - mean_ratio) / (std_ratio + 1e-6)

                if algorithm.portfolio[symbol].quantity != 0:

                    if  z > 2 and algorithm.portfolio[symbol].quantity < 0:
                        algorithm.liquidate(symbol)

                    elif  z  < -2 and algorithm.portfolio[symbol].quantity > 0:
                        algorithm.liquidate(symbol)
                    
                    continue

                mom_zscores[symbol] = z

        
        rev_zscores = {}
        for symbol, rev_ratio in rev_list.items():

            ratios = self.historical_ratios[symbol]["rev"]
            ratios.add(rev_ratio)

            if ratios.count > 5:
                mean_ratio = np.mean([r for r in ratios])
                std_ratio = np.std([r for r in ratios])
                z = (ratios[0] - mean_ratio) / (std_ratio + 1e-6)

                if algorithm.portfolio[symbol].quantity != 0:

                    if  z < 0 and algorithm.portfolio[symbol].quantity < 0:
                        algorithm.liquidate(symbol)

                    elif  z  > 0 and algorithm.portfolio[symbol].quantity > 0:
                        algorithm.liquidate(symbol)
                    
                    continue

                rev_zscores[symbol] = z


        # create insights to long / short the asset
        if len(mom_zscores.keys()) > 0:
            mom_weights = {}
            for symbol, zscore in mom_zscores.items():
                if not np.isnan(zscore) and abs(zscore)  >1.5:
                    weight = zscore
                else:
                    weight = 0
                mom_weights[symbol] = weight

            abs_weight = {key: abs(val) for key, val in mom_weights.items()}
            weights_sum = sum(abs_weight.values())
            if weights_sum != 0:
                for symbol, weight in mom_weights.items():
                    mom_weights[symbol] = weight/ weights_sum

            for symbol, weight in mom_weights.items():
                if weight > 0:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.UP, weight=weight))
                elif weight < 0:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.DOWN, weight=weight))
        
        if len(rev_zscores.keys()) > 0:
            rev_weights = {}
            for symbol, zscore in rev_zscores.items():
                if not np.isnan(zscore) and abs(zscore)>1.5:
                    weight = zscore
                else:
                    weight = 0
                rev_weights[symbol] = weight

            abs_weight = {key: abs(val) for key, val in rev_weights.items()}
            weights_sum = sum(abs_weight.values())
            if weights_sum != 0:
                for symbol, weight in rev_weights.items():
                    rev_weights[symbol] = weight/ weights_sum
            
            for symbol, weight in rev_weights.items():
                if weight > 0:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.DOWN, weight=weight))
                elif weight < 0:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.UP, weight=weight))

        return insights
