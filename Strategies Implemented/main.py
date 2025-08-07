# region imports
from AlgorithmImports import *
from PortfolioConstructor import MLP_PortfolioConstructionModel
# endregion


class VWAP_Reversion_Algo(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2025, 6, 27) 
        self.set_cash(1_000_000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0
        self.settings.rebalance_portfolio_on_security_changes = False
        self.settings.rebalance_portfolio_on_insight_changes = False
        self.day = -1
        self.set_warm_up(timedelta(days = 30))

        """
        Removed Simple Regime Detection using SPY ETF
        """
        self.spy = self.add_equity("SPY", Resolution.DAILY, Market.USA).symbol
        # self.spy_ema21 = self.ema(self.spy, 21, Resolution.DAILY)
        # self.prev_ema21 = RollingWindow[float](2)
        # self.slope_sma = SimpleMovingAverage(4)
        # self.current_regime = "Neutral"
        self.set_benchmark(self.spy)

        self.universe_settings.asynchronous = True
        self.universe_settings.leverage = 3
        self.add_universe_selection(FundamentalUniverseSelectionModel(self.fundamental_filter_function))
        
        self.set_alpha(VwapReversion(self))

        """
        This algorithm uses the MLP Portfolio Construction Model featured on QuantConnect. The only modification was an increase in postion exposure by a factor of 2.5.
        """

        self.set_portfolio_construction(MLP_PortfolioConstructionModel(algorithm=self, rebalance=timedelta(1)))

        self.add_risk_management(NullRiskManagementModel())

        self.set_execution(ImmediateExecutionModel())


    def fundamental_filter_function(self, fundamental: List[Fundamental]):
        filtered = [f for f in fundamental if f.symbol.value != "AMC" and f.has_fundamental_data and not np.isnan(f.dollar_volume) and not np.isnan(f.price) and f.price > 5]
        sorted_by_dollar_volume = sorted(filtered, key=lambda f: f.dollar_volume, reverse=True)
        return [f.symbol for f in sorted_by_dollar_volume[0:100]]


    """
    Removed Simple Regime Detection Logic
    """
    # def on_data(self, slice):

    #     if self.is_warming_up or not self.spy_ema21.is_ready:
    #         return 

    #     self.prev_ema21.add(self.spy_ema21.current.value)
    #     if self.prev_ema21.is_ready:
    #         slope = self.prev_ema21[0] - self.prev_ema21[1]
    #         self.slope_sma.update(self.time, slope)

    #     else: return 

    #     regime = self.get_regime()
    #     self.current_regime = regime

    # def get_regime(self):
    #     if not self.slope_sma.is_ready:
    #         return "Neutral"
        
    #     slope =  self.slope_sma.current.value
    #     thresh = 0.1
    #     if slope > thresh:
    #         return "Bull"
    #     elif slope < -thresh:
    #         return "Bear"
    #     return "Neutral"


class VwapReversion(AlphaModel):

    def __init__(self, algo: VWAP_Reversion_Algo):
        self.algo = algo
        self.VWAP_inds = {}
        self.day = -1
        self.historical_VwapReversion_by_symbol = {}
        self.stop_losses = {}

        self.algo.schedule.on(self.algo.date_rules.every_day(), self.algo.time_rules.every(timedelta(minutes= 30)), self.check_losses)
   
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

    def on_securities_changed(self, algo: QCAlgorithm, changes: SecurityChanges) -> None:
        # Register each security in the universe 
        for security in changes.added_securities:
            symbol = security.symbol
            history = algo.history(symbol, 25, Resolution.DAILY)
            if not history.empty:
                vwap = VolumeWeightedAveragePriceIndicator(21)
                for bar in history.itertuples():
                    bar = TradeBar(bar.Index[1], symbol, float(bar.open), float(bar.high), float(bar.low), float(bar.close), float(bar.volume))
                    vwap.update(bar)
                
                algo.register_indicator(symbol, vwap)   
                self.VWAP_inds[symbol] = vwap
                assert self.VWAP_inds[symbol] == vwap
            else:
                algo.debug(f"History pull for {symbol} empty")
                vwap = algo.vwap(symbol, 21)
                self.VWAP_inds[symbol] = vwap
                assert self.VWAP_inds[symbol] == vwap

            self.historical_VwapReversion_by_symbol[symbol] = RollingWindow[float](21)
                
        for security in changes.removed_securities:
            symbol = security.symbol
            if symbol in self.VWAP_inds:
                vwap = self.VWAP_inds[symbol]
                algo.deregister_indicator(vwap)
            self.VWAP_inds.pop(symbol, None)
            self.historical_VwapReversion_by_symbol.pop(symbol, None)

    def update(self, algo: QCAlgorithm, data: Slice) -> List[Insight]: 

        if data.bars.count == 0:
            return []

        # Daily Insights
        if self.day == algo.time.day:  
            return []
        self.day = algo.time.day

        vwap_dev = {}
        symbols = list(data.bars.keys())
        if "SPY" in symbols:
            symbols.remove("SPY")

        for symbol in symbols:
            if symbol not in self.VWAP_inds or not self.VWAP_inds[symbol].is_ready or symbol not in self.historical_VwapReversion_by_symbol:
                symbols.remove(symbol)
                if symbol in symbols:
                    algo.debug("Remove not working")
                continue

            close = data.bars[symbol].close
            vwap = self.VWAP_inds[symbol].current.value
            vwap_dev[symbol] = vwap / close

        vwap_dev_mean = sum(vwap_dev.values()) / len(vwap_dev.values())

        for symbol in symbols:
            if symbol not in vwap_dev:
                symbols.remove(symbol)
                continue
            self.historical_VwapReversion_by_symbol[symbol].add(vwap_dev[symbol] - vwap_dev_mean)
        
        zscores = {}
        for symbol in symbols:
            if symbol not in self.historical_VwapReversion_by_symbol or self.historical_VwapReversion_by_symbol[symbol].count < 2:
                continue
            
            hist_vwap_devs = list(self.historical_VwapReversion_by_symbol[symbol])
            mu = np.mean(hist_vwap_devs)
            sigma = np.std(hist_vwap_devs)
            if sigma == 0:
                continue
            z = (self.historical_VwapReversion_by_symbol[symbol][0] - mu) / sigma
            zscores[symbol] = z
        
        insights = []
        weights = {}

        """
        Removed Simple Regime Detection Logic
        """
        # if self.algo.current_regime == "Neutral":
        #     return []
    
        for symbol, zscore in zscores.items():
            if not np.isnan(zscore):
                weight = zscore
            else:
                weight = 0
            weights[symbol] = weight
        

        abs_weight = {key: abs(val) for key, val in weights.items()}
        weights_sum = sum(abs_weight.values())
        if weights_sum != 0:
            for symbol, weight in weights.items():
                weights[symbol] = weight/ weights_sum

            for symbol, weight in weights.items():
                if weight > 0:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.UP, weight=weight))
                    price = self.algo.portfolio[symbol].price
                    stop_loss = 0.98 * price
                    self.stop_losses[symbol] = stop_loss
                    algo.log(f"Long Insight Emitted for {symbol} with stop_loss {stop_loss}")

                elif weight < 0:
                    insights.append(Insight.price(symbol, Expiry.END_OF_DAY, InsightDirection.DOWN, weight=weight))
                    price = self.algo.portfolio[symbol].price
                    stop_loss = 1.02 * price
                    self.stop_losses[symbol] = stop_loss
                    algo.log(f"Short Insight Emitted for {symbol} with stop_loss {stop_loss}")

        return insights
