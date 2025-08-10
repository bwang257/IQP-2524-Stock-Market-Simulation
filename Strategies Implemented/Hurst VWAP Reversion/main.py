# region imports
from AlgorithmImports import *
from hurst_oracle import hurst_lssd
from PortfolioConstructor import MLP_PortfolioConstructionModel
# endregion


class IQP_TrendFollow(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_end_date(2025, 8, 30) 
        self.set_cash(1_000_000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0
        self.settings.rebalance_portfolio_on_security_changes = False
        self.settings.rebalance_portfolio_on_insight_changes = False
        self.day = -1
        self.set_warm_up(timedelta(days = 30))

        self.spy = self.add_equity("SPY", Resolution.DAILY, Market.USA).symbol
        self.set_benchmark(self.spy)

        self.universe_settings.asynchronous = True
        self.universe_settings.leverage = 3
        self.add_universe_selection(FundamentalUniverseSelectionModel(self.fundamental_filter_function))
        
        self.set_alpha(VwapReversion(self))

        self.set_portfolio_construction(MLP_PortfolioConstructionModel(algorithm=self, rebalance=timedelta(5)))
        self.add_risk_management(NullRiskManagementModel())
        self.set_execution(ImmediateExecutionModel())


    def fundamental_filter_function(self, fundamental: List[Fundamental]):
        filtered = [f for f in fundamental if f.symbol.value != "AMC" and f.has_fundamental_data and not np.isnan(f.dollar_volume) and not np.isnan(f.price) and f.price > 5]
        sorted_by_dollar_volume = sorted(filtered, key=lambda f: f.dollar_volume, reverse=True)
        return [f.symbol for f in sorted_by_dollar_volume[0:100]]


class VwapReversion(AlphaModel):

    def __init__(self, algo: IQP_TrendFollow):
        self.algo = algo
        self.VWAP_inds = {}
        self.day = -1
        self.historical_VwapReversion_by_symbol = {}
        self.stop_losses = {}
        self.prev_prices = {}
        self.return_window = {}

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
        for security in changes.added_securities:
            symbol = security.symbol
            self.return_window[symbol] = RollingWindow[float](20)
            history = algo.history(symbol, 25, Resolution.DAILY)
            if not history.empty:
                vwap = VolumeWeightedAveragePriceIndicator(21)
                for bar in history.itertuples():
                    bar = TradeBar(bar.Index[1], symbol, float(bar.open), float(bar.high), float(bar.low), float(bar.close), float(bar.volume))
                    vwap.update(bar)
                    if symbol in self.prev_prices and self.prev_prices[symbol] is not None:
                        self.return_window[symbol].add(np.log(bar.close / self.prev_prices[symbol]))
                    self.prev_prices[symbol] = bar.close
                
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
            self.return_window.pop(symbol, None)
            self.prev_prices.pop(symbol, None)
            self.historical_VwapReversion_by_symbol.pop(symbol, None)

    def update(self, algo: QCAlgorithm, data: Slice) -> List[Insight]: 

        if data.bars.count == 0:
            return []

        # Daily Insights
        if self.day == algo.time.day:  
            return []
        self.day = algo.time.day

        # Calculate Price Deviations from VWAP
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
            self.return_window[symbol].add(np.log(close / self.prev_prices[symbol]))
            returns = [r for r in self.return_window[symbol]]
            returns.reverse()
            hurst = hurst_lssd(returns)
            self.prev_prices[symbol] = close
            vwap = self.VWAP_inds[symbol].current.value
            if hurst < 0.5:
                vwap_dev[symbol] = vwap / close

        if len(vwap_dev.values()) == 0:
            return
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
