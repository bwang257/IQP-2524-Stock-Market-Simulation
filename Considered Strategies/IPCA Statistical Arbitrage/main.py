# region imports
from AlgorithmImports import *
from IPCA import IPCA
from RollingDictionary import MyRollingDict

import numpy as np
import pandas as pd
from scipy import stats
import math
import datetime
from enum import Enum
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.linear_model import TheilSenRegressor
# endregion

class IPCA_CNN_LSTM_StatArb(QCAlgorithm):

    def initialize(self):

        #algorithm settings
        self.set_start_date(2023, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(100000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        # US Equity trades cost $0.005/share with a $1 minimum fee and a 0.5% maximum fee.
        self.fee_per_share = 0.005
        self.min_fee = 1
        self.max_fee = 0.005

        self.frequency = 5
        self.close_buffer = 7

        #universe settings
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        self.universe_settings.resolution = Resolution.MINUTE
        self.universe_settings.leverage = 2
        
        schedule_symbol = Symbol.create("SPY", SecurityType.EQUITY, Market.USA)
        self.universe_settings.schedule.on(self.date_rules.every_day()) #daily universe selection
        self.add_universe(self.universe_selection_function)

        #IPCA
        self.IPCA_history_lookback = 126 #6 Months
        self.IPCA_history_buffer = 21 #1 Month

        self.ipca = IPCA(algorithm = self, n_factors=1, intercept=False, max_iter=10000,
                iter_tol=10e-6, alpha=0.05, l1_ratio=0.1, n_jobs=1,
                backend="loky")

        self.ipca_last_update = None
        self.ipca_initialized = False

        self.ipca_characteristics = ["LogMarketCap", "LogTotalAssets",
         "ProfitMargin", "ROE", "Debt_Equity_Ratio", "Earnings_Yield",
          "Sales_Yield", "Mom1M", "Mom12_2", "LongMom36_13", "SUV", 
          "AverageTurnover", "W52H", "Realized_Volatility", "Book_To_Market"]

        self.ipca_feature_config = {}

        self.r2_history = []
        self.r2_lookback = 6
        self.r2_std_buffer = 2

        #Load Residual Data
        self.set_warm_up(timedelta(days=5))

        #Stored Data
        self.assets = { }
        self.order_tracker = OrderTracker(self)
        self.to_be_liquidated = []

        #Pull Latest Daily 
        self.schedule.on(self.date_rules.every_day(), self.time_rules.before_market_open(schedule_symbol, 30), self.generate_predictions)

    def on_data(self, slice: Slice):
        
        if self.is_warming_up:
            self.debug("On data Warmup Called")
            self.update_intraday_data(slice)
            return 

        if self.close_to_market_close():
            self.update_intraday_data(slice)

        self.update_intraday_data(slice)

        self.execute_portfolio_rebalancing(slice)

    def update_intraday_data(self, slice):
        
        for symbol, trade_bar in slice.bars.items():
                if symbol not in self.assets:
                    self.debug("Received data from symbol not in self.assets")
                    continue

                current_day = self.time.date()
                asset_state = self.assets[symbol]

                if asset_state.open_price_time is None or asset_state.open_price_time.date() != current_day:
                    asset_state.open_price_time = self.time
                    asset_state.open_price_today = trade_bar.open
                
                    if asset_state.open_price_today is None or asset_state.open_price_today <=0:
                        self.debug(f"{symbol}: invalid open price")
                        return

                close = trade_bar.close
                should_update = asset_state.should_update(self.time)

                if should_update and asset_state.prediction_time == self.time.date(): 
                    if asset_state.last_update is None or asset_state.last_update.date() != self.time.date():
                        #Upon initialization and if data is the first bar on market open
                        asset_state.last_price = close
                        asset_state.last_update = self.time
                        asset_state.should_trade = False

                    if asset_state.last_update.date() == self.time.date():
                        #After the first bar of data
                        intraday_return = ((close - asset_state.open_price_today) / asset_state.open_price_today) - 1
                        
                        time_elapsed_seconds = (self.time - asset_state.open_price_time).total_seconds()

                        expected_return = asset_state.predicted_return * time_elapsed_seconds / (390 * 60)

                        residual = intraday_return - expected_return
                        asset_state.residuals.add(residual)

                        if asset_state.residuals.is_ready:
                            #If residuals are filled, calculate OU and rolling std
                            self.calculate_ou_parameters(symbol)
                            self.calculate_rolling_std(symbol)
                            asset_state.should_trade = True
                        
                    else:
                        self.debug("This shouldn't happen. Bar of data should be added")
                        asset_state.should_trade = False
                        continue
        
                else: 
                    asset_state.should_trade = False
        return

    def close_to_market_close(self):

        minute = self.time.minute
        hour = self.time.hour
        day = self.time.day
        month = self.time.month
        year = self.time.year

        curr_time = datetime.datetime(year, month, day, hour, minute, 0)
        market_close = datetime.datetime(year, month, day, 16, 0, 0)

        if curr_time > market_close:
            self.debug(f"Current time {curr_time} is after market_close")
            return False

        difference = market_close - curr_time
        return difference.total_seconds() < (self.close_buffer * 60)

    def calculate_trading_fees(self, symbol, shares):
        if abs(shares) < 1:
            return 0.0
            
        price = self.securities[symbol].price
        if price <= 0:
            return 0.0
            
        trade_value = abs(shares) * price
        
        # Calculate fee based on shares
        fee_by_shares = abs(shares) * self.fee_per_share
        
        # Apply minimum and maximum constraints
        fee = max(fee_by_shares, self.min_fee)
        max_fee = trade_value * self.max_fee  # 1% maximum
        fee = min(fee, max_fee)
        
        return fee

    def calculate_portfolio_weights(self, slice):

        pending_liquidations = set()
        for order_info in self.order_tracker.pending_orders.values():
            if order_info["type"] == OrderClass.LIQUIDATION:
                pending_liquidations.add(order_info["symbol"])
      
        signals = {}

        for symbol, trade_bar in slice.bars.items():
            if symbol not in self.assets:
                self.debug("Received data for a symbol not in self.assets")
                return
            
            #skip liquidated assets
            if symbol in pending_liquidations:
                continue

            asset_state = self.assets[symbol]

            if not asset_state.should_trade:
                continue

            if (asset_state.residuals.count == 0 or asset_state.rolling_std is None or asset_state.OU_mu is None or asset_state.OU_theta is None):
                continue

            residual = asset_state.residuals[0]
            z_score = (residual - asset_state.OU_mu) / asset_state.rolling_std
            
            if abs(z_score) > 1.5: 
                # Signal strength combines z-score magnitude and mean reversion speed
                theta = asset_state.OU_theta
                signal= z_score * theta if theta > 0 else z_score

                if asset_state.OU_half_life is not None and asset_state.OU_half_life > 0:
                    # Prefer faster mean reversion (shorter half-life)
                    effective_half_life = min(asset_state.OU_half_life, 48.0) #capped at 4 hours (48*5)
                    half_life_scalar = np.clip(24 / effective_half_life, 0.5, 2.0)  # 2-hour target
                    signal *= half_life_scalar
               
                    signals[symbol] = signal
        
        if not signals:
            return {}

        #normalize weights
        total_signal = sum(abs(s) for s in signals.values())
        if total_signal == 0:
            return {}

        weights = {symbol: signal/total_signal for symbol, signal in signals.items()}

        #Concentration limit
        max_weight = 0.15
        final_weights = {}
        for symbol, weight in weights.items():
            adjusted_weight = np.sign(weight) * min(abs(weight), max_weight)
            final_weights[symbol] = -adjusted_weight
        
        return final_weights

    def calculate_target_positions(self, portfolio_weights):

        target_positions = {}
        
        # Portfolio metrics
        portfolio_value = self.portfolio.total_portfolio_value
        cash = self.portfolio.cash
        margin_remaining = self.portfolio.margin_remaining
        
        target_gross_leverage = 2.0
        max_margin_utilization = 0.85
        max_position_weight = 0.15
        
        # Calculate available capital
        available_margin = margin_remaining * max_margin_utilization
        
        for symbol, weight in portfolio_weights.items():
            if symbol not in self.assets:
                continue
            
            asset_state = self.assets[symbol]
            price = asset_state.last_price

            if price is None or price <= 0:
                self.debug(f"Invalid last price for {symbol}: {price}")
                price = self.portfolio[symbol].price

            max_dollar_value = portfolio_value * max_position_weight
            dollar_value = abs(weight) * portfolio_value * target_gross_leverage

            target_dollar_value = min(dollar_value, max_dollar_value)

            if weight > 0:
                buying_power = self.portfolio.get_buying_power(symbol, OrderDirection.BUY)
            else:
                buying_power = self.portfolio.get_buying_power(symbol, OrderDirection.SELL)

            #Volatility scaling
            #Target 20% daily volatility, #limit scaling vetween 0.5 and 2.0
            if asset_state.rolling_std is not None:
                daily_vol = asset_state.rolling_std * np.sqrt(390/5)
                vol_target = 0.20
                vol_scalar = vol_target / daily_vol if daily_vol > 0 else 1.0
                vol_scalar = np.clip(vol_scalar, 0.3, 3.0)  
                
                adjusted_dollar_value = target_dollar_value * vol_scalar
            else:
                adjusted_dollar_value = target_dollar_value

            max_position_value = min(target_dollar_value, buying_power * 0.95) #95% buffer
            target_shares = int(max_position_value / price)
       
            if weight < 0:
                target_shares = -target_shares
        
        # Minimum position size filter
        min_position_value = 100  # Minimum $100 position
        if abs(target_shares * price) >= min_position_value:
            target_positions[symbol] = target_shares
        else:
            self.debug(f"Position too small for {symbol}: ${abs(target_shares * price):.0f}")
        
        return target_positions



    def should_liquidate_position(self, symbol):

        if symbol not in self.assets:
            self.debug("Should Liquidate Positions called for an asset not in self.assets")

        asset_state = self.assets[symbol]
        current_position = asset_state.current_position
        
        if current_position == 0:
            return False, None
        
        if asset_state.entry_time is None:
            self.debug(f"{symbol} has position but no entry time but a position != 0")
            return False, None

        # Time-based exit
        position_age_minutes = (self.time - asset_state.entry_time).total_seconds() / 60
        
        if asset_state.OU_half_life is not None and asset_state.OU_half_life > 0:
            half_life_minutes = 5 * asset_state.OU_half_life
            max_hold_minutes = 2 * half_life_minutes
        else:
            max_hold_minutes = 300  # Default 5 hours

        if position_age_minutes > max_hold_minutes:
            return True, "Time exit - beyond max hold period"

        if asset_state.residuals.count > 0 and asset_state.rolling_std is not None and asset_state.OU_mu is not None:
        
            current_residual = asset_state.residuals[0]
            z_score = (current_residual - asset_state.OU_mu) / asset_state.rolling_std

            # Signal reversal check
            if (asset_state.entry_z_score is not None and  np.sign(z_score) != np.sign(asset_state.entry_z_score) and abs(z_score) < 0.75):
                return True, "Signal reversal - mean reversion achieved"
            
            # Stop loss - signal getting worse
            if abs(z_score) > 3.0:
                return True, "Stop loss - signal deteriorating"
            
            # Profit taking - signal weak
            if abs(z_score) < 0.25:
                return True, "Profit taking - signal weak/mean reversion"
        
        return False, None

    def execute_portfolio_rebalancing(self, slice):
        
        #Check liquidations first
        liquidations = []

        for symbol, trade_bar in slice.bars.items():
            should_exit, reason = self.should_liquidate_position(symbol)
            if should_exit:
                current_position = self.assets[symbol].current_position
                if current_position != 0:
                    order_ticket = self.market_order(symbol, -current_position)
                    if order_ticket and order_ticket.order_id:
                        self.order_tracker.add_order(order_ticket, OrderClass.LIQUIDATION, symbol, -current_position)

                        liquidations.append(f"{symbol}: {reason}")
        
                    else:
                        self.debug(f"Failed to place liquidation order for {symbol}")
        
        for symbol in self.to_be_liquidated:
            current_position = self.assets[symbol].current_position
            quantity = self.portfolio[symbol].quantity
            if current_position != quantity:
                self.debug(f"Current position of {current_position} for {symbol} does not match quantity {quantity}")
            if current_position != 0:
                order_ticket = self.market_order(symbol, -current_position)
                if order_ticket and order_ticket.order_id:
                    self.order_tracker.add_order(order_ticket, OrderClass.LIQUIDATION, symbol, -current_position)
                    liquidations.append(f"{symbol}: {reason}")
    
                else:
                    self.debug(f"Failed to place liquidation order for {symbol}")

        if liquidations:
            self.log(f"Liquidation orders submitted: {', '.join(liquidations)}")

        #Calculate new portfolio weights
        portfolio_weights = self.calculate_portfolio_weights(slice)
        
        if not portfolio_weights:
            self.debug("No valid signals for rebalancing")
            return
        
        #Calculate target positions
        target_positions = self.calculate_target_positions(portfolio_weights)
        
        # Execute trades
        rebalance_orders = []
        for symbol, target_shares in target_positions.items():

            if symbol not in self.assets:
                self.debug("Target positions contains asset not in self.assets")
                continue

            current_shares = self.portfolio[symbol].quantity
            required_shares = target_shares - current_shares
            
            if abs(required_shares) < 1:
                continue

            fee = self.calculate_trading_fees(symbol, required_shares)
            position_value = abs(required_shares) * self.securities[symbol].price
            
            if fee / position_value > 0.01:
                self.debug(f"Skipping trade for {symbol} due to high fee ratio: {fee/position_value:.3f}")
                continue

            #Place market order
            order_ticket = self.market_order(symbol, required_shares)
            
            if order_ticket and order_ticket.order_id:
                self.order_tracker.add_order(order_ticket, OrderClass.REBALANCE, symbol, required_shares)
                rebalance_orders.append(f"{symbol}: {required_shares}")
  
            else:
                self.debug(f"Failed to place rebalance order for {symbol}")
        
        if rebalance_orders:
            self.log(f"Rebalance orders submitted: {', '.join(rebalance_orders)}")

    def on_order_event(self, order_event: OrderEvent):

        symbol = order_event.symbol

        if symbol not in self.assets:
            self.debug("Received order_event for symbol not in self.assets")

        if order_event.status == OrderStatus.FILLED:
            self._handle_filled_order(order_event)

        elif order_event.status == OrderStatus.INVALID or order_event.status == OrderStatus.CANCELED:
            self._handle_failed_order(order_event)
        
        elif order_event.status == OrderStatus.PARTIALLY_FILLED:
            self._handle_partial_fill(order_event)      

    def _handle_filled_order(self, order_event):

        order_id = order_event.order_id
        symbol = order_event.symbol
        asset_state = self.assets[symbol]

        previous_position = asset_state.current_position
        current_holdings = self.portfolio[symbol].quantity
        asset_state.current_position = current_holdings

        #Was it a liquidation order
        was_liquidation = self.order_tracker.is_liquidation_order(order_id)
        
        if was_liquidation:
            self.log(f"Liquidation completed for {symbol}: Position changed from {previous_position:+.0f} to {current_holdings:+.0f}")

            if symbol in self.to_be_liquidated:
                self.to_be_liquidated.remove(symbol)
                self.assets.pop(symbol, None)
            
            elif abs(current_holdings) < 0.01:
                asset_state.reset_position_tracking()

        else:
            #If rebalance order
            if self._is_new_position_or_direction_change(previous_position, current_holdings):
                asset_state.entry_time = self.time
                asset_state.last_trade_time = self.time
                
                # Calculate entry z-score
                if (asset_state.residuals.count > 0 and asset_state.rolling_std is not None and asset_state.OU_mu is not None):
                    current_residual = asset_state.residuals[0]
                    asset_state.entry_z_score = (current_residual - asset_state.OU_mu) / asset_state.rolling_std
                else:
                    self.debug("This should not happen. Entry z-score on rebalancing not defined")
                    asset_state.entry_z_score = 0.0
                
                self.debug(f"New position established for {symbol}: {current_holdings:+.0f} shares, entry z-score: {asset_state.entry_z_score:.3f}")

        self.order_tracker.remove_order(order_id)
        
        self.log(f"Order filled: {symbol} - {order_event.fill_quantity:+.0f} shares at ${order_event.fill_price:.2f}, Total position: {current_holdings:+.0f} shares")
        return 

    def _handle_partial_fill(self, order_event):
        #Will not occur during backtesting but I need to figure out if I should cancel it or have some other way to deal with this
        symbol = order_event.symbol
        order_id = order_event.order_id
        asset_state = self.assets[symbol]

        asset_state.current_position = self.portfolio[symbol].quantity
        order_info = self.order_tracker.pending_orders.get(order_id)

        if not order_info:
            self.debug("Received a partially filled order for an order not logged")
            return
        
        total_filled = order_event.fill_quantity
        expected_quantity = order_info["expected_quantity"]
        remaining_quantity = expected_quantity - total_filled

        self.debug(f"Partial fill: {symbol} {total_filled}/{expected_quantity} filled, {remaining_quantity} remaining")
        
        if self.order_tracker.is_liquidation_order(order_id) and abs(remaining_quantity) > 0.01:
            self.liquidate(symbol)
            if abs(self.portfolio[symbol].quantity) > 0.01:
                self.debug(f"Liquidation incomplete for {symbol} even after liquidation attempt, canceling remainder")
                order_ticket = self.transactions.get_order_ticket(order_id)
                if order_ticket:
                    order_ticket.cancel()
                
            else:
                self.debug(f"Received incomplete liqudiation for {symbol}. Liquidation attempt success, canceling order")
                if symbol in self.to_be_liquidated:
                    self.to_be_liquidated.remove(symbol)
                    self.assets.pop(symbol, None)
                else:
                    asset_state.reset_position_tracking()
                self.log(f"Liquidation completed for {symbol}")
            
        elif self.order_tracker.is_rebalance_order(order_id):
            #THIS NEEDS TO BE FIXED
            order_ticket = self.transactions.get_order_ticket(order_id)
            if order_ticket:
                order_ticket.cancel()
        
    def _handle_failed_order(self, order_event):

        order_id = order_event.order_id
        symbol = order_event.symbol
        asset_state = self.assets[symbol]
        
        was_liquidation = self.order_tracker.is_liquidation_order(order_id)
        
        if was_liquidation:
            self.liquidate(symbol)
            if abs(self.portfolio[symbol].quantity) > 0.01:
                self.log(f"WARNING: Liquidation order failed for {symbol} - Order ID: {order_id}, Status: {order_event.status}")
            else:
                if symbol in self.to_be_liquidated:
                    self.to_be_liquidated.remove(symbol)
                    self.assets.pop(symbol, None)
                else:
                    asset_state.reset_position_tracking()
                self.debug(f"Liquidation of {symbol} sucessful after failed liquidation order")
                self.log(f"Liquidation completed for {symbol}")
        else:
            self.debug(f"Rebalance order failed for {symbol} - Order ID: {order_id}, Status: {order_event.status}")
        
        self.order_tracker.remove_order(order_id)

    def _is_new_position_or_direction_change(self, previous_position, current_position):
        # New position from zero
        if abs(previous_position) < 0.01 and abs(current_position) > 0.01:
            return True
        
        # Direction change
        if previous_position * current_position < 0:
            return True
        
        return False

    def calculate_ou_parameters(self, symbol, dt=5.0):

        asset_state = self.assets[symbol]

        if not asset_state.residuals.is_ready:
            return
    
        #Convert to numpy array and reverse chronological order
        residual_array = np.array([asset_state.residuals[i] for i in range(asset_state.residuals.count)])
        residual_array = residual_array[::-1]
        
        #Calculate differences and lagged values for OU estimation

        X_t_minus_1 = residual_array[:-1] #X(t-1)
        dX = np.diff(residual_array) #X(t) - X(t-1)

        if len(X_t_minus_1) == 0 or len(dX) ==0:
            self.debug("Length of residuals X_t-1 and dX not sufficient for OU calculation")

        try:
            model = TheilSenRegressor()
            model.fit(X_t_minus_1.reshape(-1, 1), dX)
            slope = model.coef_[0]
            intercept = model.intercept_
        except Exception as e:
            self.debug(f"Theil-Sen failed for {symbol}, fallback to OLS: {e}")
            slope, intercept, *_ = stats.linregress(X_t_minus_1, dX)

        theta = -slope / dt
        mu = -intercept / slope if slope != 0 else 0
        half_life = math.log(2) / theta if theta > 0 else float("inf")
        asset_state.OU_theta = theta
        asset_state.OU_mu = mu
        asset_state.OU_half_life = half_life

        #OLS regression: dX = alpha + beta * X(t-1) + epsilon
        # theta = -beta/dt, mu = -alpha/beta, sigma = std(epsilon)/sqrt(dt)

        if np.ptp(X_t_minus_1) < 1e-8: #residuals are basically constant, max-min ~ 0
            self.debug(f"{symbol} OU regression skipped: no variation in residuals")
            return

        slope, intercept, r_value, p_value, std_err = stats.linregress(X_t_minus_1, dX)

        theta = -slope / dt #mean reversion speed
        mu = -intercept / slope if slope != 0 else 0 #long-term mean
        half_life = math.log(2) / theta if theta > 0 else float("inf")

        asset_state.OU_theta = theta
        asset_state.OU_mu= mu
        asset_state.OU_half_life = half_life

        return
    
    def calculate_rolling_std(self, symbol, lookback= None):

        asset_state = self.assets[symbol]

        if lookback is None:
            lookback = asset_state.residuals.count
        else:
            lookback = min(lookback, asset_state.residuals.count)

        if lookback >0:
            data = [asset_state.residuals[i] for i in range(lookback)]
            asset_state.rolling_std = np.std(data, ddof=1)
    
    def on_securities_changed(self, changes):
        
        for security in changes.removed_securities:
            if security.Symbol in self.assets:
                self.to_be_liquidated.append(security.Symbol)

        added_assets = []
        for security in changes.added_securities:
            if security.Symbol not in self.assets:
                self.assets[security.Symbol] = AssetState(self, security.Symbol)
                added_assets.append(security.Symbol)
        
        lookback = 1000 + self.IPCA_history_lookback +self.IPCA_history_buffer 
        self.pull_historical_data(added_assets, lookback)

    def generate_predictions(self):
        
        if len(self.assets) == 0:
            self.debug("No assets in self.assets")

        #Find the previous trading-day ordinal
        curr_ordinal = self.time.toordinal()
        pot_ordinal = curr_ordinal - 1
        prev_ordinal = None

        while True:
            dt =  datetime.date.fromordinal(pot_ordinal)
            td = self.trading_calendar.get_trading_day(dt)
            if td.business_day:
                prev_ordinal = pot_ordinal
                break
            pot_ordinal -=1

        to_pull = []
        for symbol, asset_state in self.assets.items():
            if prev_ordinal not in asset_state.fundamental_history.rolling_dict:
                to_pull.append(symbol)

        #Pull previous day daily data
        if to_pull:
            # self.debug(f"{curr_ordinal}: Pulling data from prior trading day ordinal {prev_ordinal} for {to_pull}")
            self.pull_historical_data(to_pull, 1)

        
        self.update_IPCA()
        if not self.ipca_initialized:
            self.debug("IPCA model not initialized. Skipping predictions")

        try:
            X_new, indices_new, symbols_list = self.build_IPCA_X_indices_latest(prev_ordinal)
            if X_new.shape[0] == 0:
                self.debug("No valid data for predictions")
                return

            predicted_returns = self.ipca.predict(X_new, indices_new, W=None, mean_factor=True, data_type="panel", label_ind=False)

            if len(predicted_returns) != len(symbols_list):
                self.debug(f"Length of predicted returns {len(predicted_returns)} does not match symbols length {len(symbols_list)}")
                return

            for symbol, pred_return in zip(symbols_list, predicted_returns):
                if symbol in self.assets:
                    asset_state = self.assets[symbol]
                    asset_state.predicted_return = pred_return
                    asset_state.prediction_time = self.time.date()
        
        except Exception as e:
            self.debug(f"Error in generate_predictions: {str(e)}")

        return

    def build_IPCA_X_indices_latest(self, date_ordinal):
        #Detect multicollinearity and apply PCA to high-coorelation groups
        #Drop one feature per highly correlated pair
        #Store the feature construction logic for prediction generation

        self.ipca_characteristics = ["LogMarketCap", "LogTotalAssets",
         "ProfitMargin", "ROE", "Debt_Equity_Ratio", "Earnings_Yield",
          "Sales_Yield", "Mom1M", "Mom12_2", "LongMom36_13", "SUV", 
          "AverageTurnover", "W52H", "Realized_Volatility", "Book_To_Market"]
 
        rows, symbols = [], []

        for symbol, asset_state in self.assets.items():
            if len(asset_state.IPCA_ready_data.rolling_dict) == 0:
                self.debug(f"{symbol} IPCA_ready_data empty")
                continue


            latest_date = max(asset_state.IPCA_ready_data.rolling_dict.keys())
            values = asset_state.IPCA_ready_data.rolling_dict[latest_date]

            if values is None:
                self.debug(f"No values associated with latest data for {symbol}")
                continue

            row = {c: values.get(c, np.nan) for c in self.ipca_characteristics}

            # Check if we have at least some valid data
            valid_values = sum(1 for v in row.values() if not np.isnan(v))
            if valid_values >= len(self.ipca_characteristics) * 0.5:  # At least 50% valid
                rows.append(row)
                symbols.append(symbol)
            
        if not rows:
            self.debug("No valid rows for prediction")
            raise ValueError("Rows empty in prediction")
            
        df= pd.DataFrame(rows, index = pd.Index(symbols, name="symbol"), columns = self.ipca_characteristics)
        config = self.ipca_feature_config
        df_combined = df.copy()

        # Apply PCA transforms
        for group, pca in zip(config["pca_groups"], config["pca_models"]):
            group_data = df_combined[group].dropna(axis=0)
            if group_data.shape[0] == df_combined.shape[0]:
                df_combined[f"PCA_{'_'.join(group)}"] = pca.transform(df_combined[group]).ravel()

        df_combined.drop(columns=config["dropped_features"], errors='ignore', inplace=True)

        for f in config["final_feature_names"]:
            if f not in df_combined.columns:
                df_combined[f] = np.nan

        df_combined = df_combined[config["final_feature_names"]]
        means = pd.Series(config["means"])
        stds = pd.Series(config["stds"]).replace(0, 1)
        X = (df_combined - means) / stds

        X = X.dropna(axis=0)
        if X.empty:
            self.debug("All prediction rows dropped after standardization")
            return np.array([]), np.array([]), []

        #Find indices
        symbols_list = X.index.tolist()
        symbol_index_dict = {s:i for i,s in enumerate(symbols_list)}
        symbol_codes = [symbol_index_dict[s] for s in symbols_list]

        date_codes = [0] * len(symbols_list)
        indices = np.vstack([symbol_codes, date_codes]).T

        #symbols_list[i] corresponds to y_pred[i]
        return X.to_numpy(), indices, symbols_list

    def build_IPCA_X_y(self, for_training: bool):
        #builds a dataframe with multiIndex (symbol, date) and IPCA characteristics columns for IPCA Model Fitting
        #extracts matrix X, y, and indices

        rows, index_tuples = [], []

        for symbol, asset_state in self.assets.items():
            for date, values in asset_state.IPCA_ready_data.rolling_dict.items():
                if values is None:
                    continue
                row = {f: values.get(f, np.nan) for f in self.ipca_characteristics + ["Return"]}
                rows.append(row)
                index_tuples.append((symbol, date))

        if not rows:
            self.debug("No valid rows for IPCA matrix build")
            return np.array([]), np.array([]), []

        df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index_tuples, names=["symbol", "date"]))
 
        #Forward Fill Missing Data
        df = df.sort_index(level=["symbol", "date"])

        ffill_cols = ["LogMarketCap", "LogTotalAssets", "ProfitMargin", "ROE",
                     "Debt_Equity_Ratio", "Earnings_Yield", "Sales_Yield", "Book_To_Market"]

        for col in ffill_cols:
            if col in df.columns:
                df[col] = df.groupby(level="symbol")[col].ffill()

        #Debug: Display Number of np.nans per column
        nan_counts = df.isna().sum()
        for column, cnt in nan_counts.items():
            if cnt > 0:
                self.debug(f"Column: {column}: Count: {cnt}")

        #Shift returns forward
        df = df.reset_index()
        df = df.sort_values(['symbol', 'date'])
        df['Return'] = df.groupby('symbol')['Return'].shift(-1)

        #Remove the current current return and use forward return
        df = df.set_index(['symbol', 'date'])

        df = df.dropna(subset= ["Return"] + self.ipca_characteristics) #adjusts for shifted returns

        X_raw = df[self.ipca_characteristics]
        y = df["Return"].to_numpy()


        if for_training:
            #process the ipca_characteristics, drop/run PCA
            #construct indices, store the configuration
            #return X, y, indices
            X_processed, config, indices = self.pca1_processing(X_raw)
            self.ipca_feature_config = config
            return X_processed.to_numpy(), y, indices

        else:
            #process the ipca characteristics according the previous configuration
            #construct indices
            #return an alligned X, y, indices for scoring
            X_aligned, indices = self.apply_ipca_config(X_raw, self.ipca_feature_config)
            return X_aligned.to_numpy(), y, indices

    def pca1_processing(self, X_raw, threshold = 0.85):

        X = X_raw.copy()
        X = X.dropna(axis=0)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        used_features = set()
        pca_groups = []
        pca_models = []
        dropped_features = []

        for col in upper.columns:
            high_corr = [row for row in upper.index if upper.loc[row, col] > threshold and row not in used_features]
            if len(high_corr) >= 1:
                group = sorted(set([col] + high_corr))
                group_data = X[group].dropna()
                if group_data.shape[1] > 1:
                    pca = PCA(n_components=1)
                    X[f"PCA_{'_'.join(group)}"] = pca.fit_transform(group_data).ravel()
                    pca_groups.append(group)
                    pca_models.append(pca)
                    used_features.update(group)

        for group in pca_groups:
            for f in group:
                if f in X.columns:
                    dropped_features.append(f)
                    X.drop(columns=f, inplace=True)

        X_means = X.mean()
        X_stds = X.std(ddof=0).replace(0, 1)
        X = (X - X_means) / X_stds
       
        # Index mapping
        symbols = X.index.get_level_values("symbol").unique()
        dates = X.index.get_level_values("date").unique()
        symbol_index_dict = {s: i for i, s in enumerate(symbols)}
        date_index_dict = {d: i for i, d in enumerate(dates)}

        symbol_codes = X.index.get_level_values("symbol").map(symbol_index_dict).to_numpy()
        date_codes = X.index.get_level_values("date").map(date_index_dict).to_numpy()
        indices = np.vstack([symbol_codes, date_codes]).T.astype(int)

        config = {
            'final_feature_names': X.columns.tolist(),
            'dropped_features': dropped_features,
            'pca_groups': pca_groups,
            'pca_models': pca_models,
            'means': X_means.to_dict(),
            'stds': X_stds.to_dict()
        }

        return X, config, indices

    def apply_ipca_config(self, X_raw, config: dict):

        X = X_raw.copy()

        for f in config['dropped_features']:
            X.drop(columns=f, errors='ignore', inplace=True)

        for group, pca in zip(config['pca_groups'], config['pca_models']):
            group_cols = [col for col in group if col in X.columns]
            if len(group_cols) != len(group):
                continue
            X[f"PCA_{'_'.join(group)}"] = pca.transform(X[group_cols]).ravel()
            X.drop(columns=group_cols, inplace=True)


        # Align column order and pad with NaNs if needed
        for f in config['final_feature_names']:
            if f not in X.columns:
                X[f] = np.nan

        X_aligned = X[config['final_feature_names']]
        means = pd.Series(config['means'])
        stds = pd.Series(config['stds']).replace(0, 1)
        X_aligned = (X_aligned - means) / stds

        # Index alignment
        symbols = X_aligned.index.get_level_values("symbol").unique()
        dates = X_aligned.index.get_level_values("date").unique()
        symbol_index_dict = {s: i for i, s in enumerate(symbols)}
        date_index_dict = {d: i for i, d in enumerate(dates)}

        symbol_codes = X_aligned.index.get_level_values("symbol").map(symbol_index_dict).to_numpy()
        date_codes = X_aligned.index.get_level_values("date").map(date_index_dict).to_numpy()
        indices = np.vstack([symbol_codes, date_codes]).T.astype(int)

        return X_aligned, indices

    def pull_historical_data(self, list_of_symbols, total_lookback):

        if not list_of_symbols:
            return

        symbols = list_of_symbols
        
        #pull data sets
        try:
            historical_price_data = self.history(symbols, total_lookback, Resolution.DAILY) #updates at market close
            historical_fundamentals = self.history[Fundamental](symbols, total_lookback, Resolution.DAILY) #updates at midnight

        except Exception as e:
            self.debug(f"Error pulling historical data: {str(e)}")
            return

        if historical_price_data is not None and not historical_price_data.empty:
            for symbol in symbols:
                if symbol not in self.assets or symbol not in historical_price_data.index.get_level_values(0):
                    self.debug(f"{Symbol} not in self.assets or historical price data index")
                    continue
                    

                symbol_hist_df = historical_price_data.loc[symbol]
                asset = self.assets[symbol]
                
                for row in symbol_hist_df.itertuples():
                    timestamp = pd.to_datetime(row.Index)
                    date_ordinal = timestamp.toordinal()

                    close = getattr(row, 'close', np.nan)
                    high = getattr(row, 'high', np.nan)
                    volume = getattr(row, 'volume', np.nan)
                    open_price = getattr(row, 'open', np.nan)
                    if np.isnan(open_price):
                        self.debug(f"{symbol} has nan for open price")
                    
                    if np.isnan(close) or close <= 0:
                        self.debug(f"{symbol} close not defined in historical data pull")
                        continue
                    
                    #open-to-close return
                    daily_return = (close - open_price) / open_price

                    # Store price data
                    asset.fundamental_history.append(date_ordinal, "Close", close)
                    asset.fundamental_history.append(date_ordinal, "High", high)
                    asset.fundamental_history.append(date_ordinal, "Volume", volume)
                    asset.fundamental_history.append(date_ordinal, "Open", open_price)
                    asset.fundamental_history.append(date_ordinal, "Return", daily_return)
                    bruh = asset.fundamental_history.rolling_dict[date_ordinal]["Return"]
                    # self.debug(f"Daily Return: {bruh} when should be {daily_return} for {symbol}")


        if historical_fundamentals is not None:
            for row in historical_fundamentals:
                for symbol, fundamental in row.items():
                    if symbol not in self.assets:
                        continue

                    asset = self.assets[symbol]
                    date_ordinal = fundamental.time.toordinal()

                    shares_outstanding = fundamental.company_profile.shares_outstanding if fundamental.company_profile.shares_outstanding is not None else np.nan
                    asset.fundamental_history.append(date_ordinal, "SharesOutstanding", shares_outstanding)
                    asset.IPCA_ready_data.append(date_ordinal, "SharesOutstanding", shares_outstanding)

                    characteristics = self.calculate_characteristics(fundamental, asset, date_ordinal)

                    # Store in all data structures
                    for char_name, char_value in characteristics.items():
                        asset.fundamental_history.append(date_ordinal, char_name, char_value)
                        asset.IPCA_ready_data.append(date_ordinal, char_name, char_value)
                    
        self.debug(f"Historical data pulled for {len(symbols)} symbols over period {total_lookback}.")           

        return  

    def calculate_characteristics(self, fundamental, asset, date_ordinal):
        
        characteristics = {}

        # Market cap
        mcap = fundamental.market_cap
        characteristics["LogMarketCap"] = np.log(mcap) if mcap and mcap > 0 else np.nan
        
        # Total assets
        try:
            tassets = fundamental.financial_statements.balance_sheet.total_assets.value
            characteristics["LogTotalAssets"] = np.log(tassets) if tassets and tassets > 0 else np.nan
        except:
            characteristics["LogTotalAssets"] = np.nan
        
        # Financial ratios
        try:
            characteristics["ProfitMargin"] = fundamental.operation_ratios.normalized_net_profit_margin.value
        except:
            characteristics["ProfitMargin"] = np.nan
            
        try:
            characteristics["ROE"] = fundamental.operation_ratios.roe.value
        except:
            characteristics["ROE"] = np.nan
        
        try:
            characteristics["Debt_Equity_Ratio"] = fundamental.operation_ratios.total_debt_equity_ratio.value
        except:
            characteristics["Debt_Equity_Ratio"] = np.nan
        
        # Valuation ratios
        characteristics["Earnings_Yield"] = getattr(fundamental.valuation_ratios, 'earning_yield', np.nan)
        characteristics["Sales_Yield"] = getattr(fundamental.valuation_ratios, 'sales_yield', np.nan)
        
        # Book to market
        try:
            book_value = fundamental.valuation_ratios.book_value_per_share
            price = fundamental.price
            characteristics["Book_To_Market"] = book_value / price if book_value and price and price > 0 else np.nan
        except:
            characteristics["Book_To_Market"] = np.nan
        
        # Momentum and technical indicators
        characteristics["Mom1M"] = asset.fundamental_history.calculate_momentum(date_ordinal, 21, 0)
        characteristics["Mom12_2"] = asset.fundamental_history.calculate_momentum(date_ordinal, 252, 42)
        characteristics["LongMom36_13"] = asset.fundamental_history.calculate_momentum(date_ordinal, 756, 273)
        
        # #open-to-close return
        # try:
        #     open_price = asset.fundamental_history.get(date_ordinal, "Open")
        #     close_price = asset.fundamental_history.get(date_ordinal, "Close")
        #     if open_price is not None and close_price is not None and open_price > 0:
        #         characteristics["Return"] = (close_price - open_price) / open_price
        #     else:
        #         characteristics["Return"] = np.nan
        #         if open_price is None:
        #             self.debug(f"Open price for {asset.symbol} none")
        #         if close_price is None:
        #             self.debug(f"Close price for {asset.symbol} none")
        # except:
        #     characteristics["Return"] = np.nan

        # Volume-based indicators
        avt, suv = asset.fundamental_history.calculate_AVT_SUV(date_ordinal)
        characteristics["AverageTurnover"] = avt
        characteristics["SUV"] = suv
        
        # Other technical indicators
        characteristics["W52H"] = asset.fundamental_history.calculate_w52h(date_ordinal)
        characteristics["Realized_Volatility"] = asset.fundamental_history.calculate_realized_volatility(date_ordinal)

        return characteristics

    def universe_selection_function(self, fundamental):
        #selects universe based on liquidity and price floor
        #could group by market cap/sector to limit market cap and sector bias

        price_floor = 20
        universe_size = 20
        filtered =  [x for x in fundamental if x.has_fundamental_data and x.dollar_volume is not None and not np.isnan(x.dollar_volume) and x.price > price_floor]
        sorted_by_dollar_volume  = sorted(filtered, key = lambda c: c.dollar_volume, reverse = True)
        return [x.symbol for x in sorted_by_dollar_volume[:universe_size]]

    def check_sufficient_data_for_fitting(self):
        if len(self.assets) < 10:  # Minimum number of assets
            return False
            
        valid_assets = 0
        for symbol, asset_state in self.assets.items():
            if len(asset_state.IPCA_ready_data.rolling_dict) >= 126:  # Minimum historical points
                valid_assets += 1
                
        return valid_assets >= 10

    def update_IPCA(self):

        current_time = self.time
        retrain_due_to_time = (self.ipca_last_update is None) or (current_time - self.ipca_last_update.days >= 7) if self.ipca_last_update is not None else True
        retrain_due_to_drift = False

        if not self.check_sufficient_data_for_fitting():
            self.debug("Insufficient data for IPCA update")
            return

        try:
            if self.ipca_feature_config:
                X_score, y, indices_score = self.build_IPCA_X_y(for_training=False)
                
                if X_score.shape[0] == 0:
                    self.debug("No data for IPCA scoring")
                    return
                
                r2 = self.ipca.score(X_score, y, indices_score)
                self.log(f"Current IPCA R^2: {r2}")
                self.r2_history.append(r2)
                if len(self.r2_history) > self.r2_lookback:
                    self.r2_history.pop(0)

                if len(self.r2_history) >= 3:
                    mean_r2 = np.mean(self.r2_history)
                    std_r2 = np.std(self.r2_history)
                    dynamic_threshold = mean_r2 - self.r2_std_buffer * std_r2
                    if r2 < dynamic_threshold:
                            self.debug(f"Adaptive retrain: R² {r2:.4f} below dynamic threshold {dynamic_threshold:.4f}")
                            retrain_due_to_drift = True
                        
                elif r2 < 0.01:
                    self.debug(f"Adaptive retrain: R² {r2:.4f} below absolute threshold 0.01 (insufficient history)")
                    retrain_due_to_drift = True

                #Check eigenvector angular drift
                Gamma_old = self.ipca.Gamma.copy()
                Gamma_new, _ = self.ipca.project_factors(X_score, y, indices_score)
                angle_drift = eigenvector_angle_change(Gamma_old, Gamma_new)
                if angle_drift > 25:
                    self.debug(f"Adaptive retrain: Eigenvector drift {angle_drift:.2f}° exceeds threshold")
                    retrain_due_to_drift = True
                        
            if retrain_due_to_time or retrain_due_to_drift or not self.ipca_initialized:
                X_train, y_train, indices_train = self.build_IPCA_X_y(for_training=True)
                if X_train.shape[0] == 0:
                    self.debug("No data for IPCA training")
                    return

                self.ipca.fit(X_train, y_train, indices_train)
                self.ipca_last_update = current_time
                self.ipca_initialized = True
                self.debug(f"IPCA retrained on {current_time.date()} with {X_train.shape[0]} samples, {X_train.shape[1]} features")
                    
        except Exception as e:
            self.debug(f"Error updating IPCA: {str(e)}")

class AssetState:
    def __init__(self, algorithm, symbol, residual_window = 150, momentum_window = 30):
        self.symbol = symbol
        self.last_update = None
        self.last_price = None

        self.predicted_return = 0.0
        self.prediction_time = None

        self.open_price_today = None
        self.open_price_time = None
        self.momentum_history = RollingWindow[float](momentum_window)
        self.residuals = RollingWindow[float](residual_window)
        self.OU_theta = None
        self.OU_mu = None
        self.OU_half_life = None
        self.rolling_std = None

        self.should_trade = False

        #Position Tracking
        self.entry_z_score = None
        self.current_position = 0.0
        self.last_trade_time = None
        self.entry_time = None

        #Daily Resolution Data
        self.fundamental_history = MyRollingDict(algorithm, 1200) #for all characteristic calculations
        self.IPCA_ready_data = MyRollingDict(algorithm, 150) #for fitting

    def should_update(self, current_time, frequency_minutes = 5):
        #Returns if data should be added to the asset state
        if self.last_update is None:
            return True
        return (current_time - self.last_update).total_seconds() >= frequency_minutes * 60

    def reset_position_tracking(self):
        #Reset position tracking when position goes to 0 
        self.current_position = 0.0
        self.entry_time = None
        self.entry_z_score = None
        self.last_trade_time = None

class OrderClass(Enum):
    LIQUIDATION = "liquidation"
    REBALANCE = "rebalance"
    
class OrderTracker:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.pending_orders = {}  # order_id: order_info
        self.liquidation_orders = set()  # track liquidation order IDs
        self.rebalance_orders = set() # track rebalance order IDs
        
    def add_order(self, order_ticket, order_type, symbol, expected_quantity):
        if order_ticket and order_ticket.order_id:
            self.pending_orders[order_ticket.order_id] = {
                "symbol": symbol,
                "type": order_type,
                "expected_quantity": expected_quantity,
                "submitted_time": self.algorithm.time
            }
            if order_type == OrderClass.LIQUIDATION:
                self.liquidation_orders.add(order_ticket.order_id)

            if order_type == OrderClass.REBALANCE:
                self.rebalance_orders.add(order_ticket.order_id)
    
    def remove_order(self, order_id):
        self.pending_orders.pop(order_id, None)
        if order_id in self.liquidation_orders: 
            self.liquidation_orders.discard(order_id)
        
        if order_id in self.rebalance_orders:
            self.rebalance_orders.discard(order_id)
    
    def is_liquidation_order(self, order_id):
        return order_id in self.liquidation_orders

    def is_rebalance_order(self, order_id):
        return order_id in self.rebalance_orders

def eigenvector_angle_change(Gamma_old, Gamma_new):
    """
    Computes the largest angle in degrees between the columns of two factor loading matrices.
    Columns should be aligned (i.e., same ordering).
    """
    angles = []
    for i in range(min(Gamma_old.shape[1], Gamma_new.shape[1])):
        u = Gamma_old[:, i]
        v = Gamma_new[:, i]
        cos_angle = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angles.append(np.degrees(angle_rad))
    return max(angles)
