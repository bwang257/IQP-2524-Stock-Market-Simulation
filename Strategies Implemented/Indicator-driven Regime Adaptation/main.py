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

# region imports
from AlgorithmImports import *
# from alpha import *
from PortfolioConstructor import MLP_PortfolioConstructionModel
# endregion


class OptimizedRegimeAdaptiveAlgo(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_end_date(2025, 7, 11)
        self.set_cash(1_000_000)
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        self.settings.minimum_order_margin_portfolio_percentage = 0
        self.settings.rebalance_portfolio_on_security_changes = False
        self.settings.rebalance_portfolio_on_insight_changes = False
        
        self.set_warm_up(timedelta(days=5))

        # Core market assets for regime detection
        self._spy = self.add_equity("SPY", Resolution.MINUTE).symbol
        self._vix = self.add_equity("VIX", Resolution.MINUTE).symbol
        self._tlt = self.add_equity("TLT", Resolution.MINUTE).symbol
        self._qqq = self.add_equity("QQQ", Resolution.MINUTE).symbol
        self._iwm = self.add_equity("IWM", Resolution.MINUTE).symbol
        self._gld = self.add_equity("GLD", Resolution.MINUTE).symbol
        self.set_benchmark(self._spy)

        # Comprehensive indicator suite
        self._indicators = self.setup_comprehensive_indicators()
        
        # Advanced regime detection system
        self._regime_detector = AdvancedRegimeDetector(self)
        
        # Dynamic signal generation - FIX: Actually create the signal generator
        self._signal_generator = DynamicSignalGenerator(self)
        
        # Portfolio management
        self._portfolio_manager = AdaptivePortfolioManager(self)
        
        # Universe selection
        self.universe_settings.asynchronous = True
        self.universe_settings.leverage = 2.0
        self.add_universe_selection(FundamentalUniverseSelectionModel(self.fundamental_filter_function))
        
        # Alpha model
        self.set_alpha(RegimeAdaptiveAlphaModel(self))
        
        # Portfolio construction
        self.set_portfolio_construction(InsightWeightingPortfolioConstructionModel())
        
        # Risk management
        # self.add_risk_management(AdaptiveRiskManagementModel(self))
        
        # Execution
        self.set_execution(ImmediateExecutionModel())

        # Scheduling
        self.schedule.on(self.date_rules.every_day(), 
                        self.time_rules.every(timedelta(minutes=1)), 
                        self.update_regime_fast)
        
        self.schedule.on(self.date_rules.every_day(), 
                        self.time_rules.every(timedelta(minutes=5)), 
                        self.update_regime_comprehensive)
        
        self.schedule.on(self.date_rules.every_day(), 
                        self.time_rules.at(9, 30), 
                        self.market_open)

    def fundamental_filter_function(self, fundamental: List[Fundamental]) -> List[Symbol]:
        filtered = [f for f in fundamental if f.symbol.value != "AMC" and f.has_fundamental_data and not np.isnan(f.dollar_volume) and not np.isnan(f.price) and f.price > 5]
        sorted_by_dollar_volume = sorted(filtered, key=lambda f: f.dollar_volume, reverse=True)
        return [f.symbol for f in sorted_by_dollar_volume[0:25]]

    def rebalance_func(self, curr_time):
        """Rebalancing schedule based on regime"""
        current_regime = getattr(self._regime_detector, 'current_regime', 'NEUTRAL')
        
        if current_regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
            return curr_time.hour % 2 == 0  # Every 2 hours in trending markets
        elif current_regime in ['RANGING_LOW_VOL', 'RANGING_HIGH_VOL']:
            return curr_time.hour % 4 == 0  # Every 4 hours in ranging markets
        else:
            return curr_time.hour % 6 == 0  # Every 6 hours in neutral markets

    def setup_comprehensive_indicators(self):
        """Setup comprehensive indicator suite"""
        indicators = {}
        
        # Trend indicators
        indicators['spy_sma_10'] = self.sma(self._spy, 10, Resolution.MINUTE)
        indicators['spy_sma_20'] = self.sma(self._spy, 20, Resolution.MINUTE)
        indicators['spy_sma_50'] = self.sma(self._spy, 50, Resolution.MINUTE)
        indicators['spy_ema_12'] = self.ema(self._spy, 12, Resolution.MINUTE)
        indicators['spy_ema_26'] = self.ema(self._spy, 26, Resolution.MINUTE)
        indicators['spy_tema'] = self.tema(self._spy, 21, Resolution.MINUTE)
        
        # Momentum indicators
        indicators['spy_rsi'] = self.rsi(self._spy, 14, Resolution.MINUTE)
        indicators['spy_macd'] = self.macd(self._spy, 12, 26, 9, Resolution.MINUTE)
        indicators['spy_ppo'] = self.ppo(self._spy, 12, 26, 9, Resolution.MINUTE)
        indicators['spy_roc'] = self.roc(self._spy, 10, Resolution.MINUTE)
        indicators['spy_mom'] = self.mom(self._spy, 10, Resolution.MINUTE)
        indicators['spy_cci'] = self.cci(self._spy, 20, Resolution.MINUTE)
        indicators['spy_cmo'] = self.cmo(self._spy, 14, Resolution.MINUTE)
        indicators['spy_williams'] = self.wilr(self._spy, 14, Resolution.MINUTE)
        indicators['spy_stoch'] = self.sto(self._spy, 14, 3, Resolution.MINUTE) 
        indicators['spy_stoch_rsi'] = self.srsi(self._spy, 14, 14, 3, 3, Resolution.MINUTE)  
        indicators['spy_trix'] = self.trix(self._spy, 14, Resolution.MINUTE)
        
        # Volatility indicators
        indicators['spy_bb'] = self.bb(self._spy, 20, 2, Resolution.MINUTE)
        indicators['spy_kch'] = self.kch(self._spy, 20, 2, Resolution.MINUTE)
        indicators['spy_dch'] = self.dch(self._spy, 20, Resolution.MINUTE)
        indicators['spy_atr'] = self.atr(self._spy, 14, Resolution.MINUTE)
        indicators['spy_natr'] = self.natr(self._spy, 14, Resolution.MINUTE)
        indicators['spy_tr'] = self.tr(self._spy, Resolution.MINUTE)
        indicators['spy_std'] = self.std(self._spy, 20, Resolution.MINUTE)
        indicators['spy_variance'] = self.var(self._spy, 20, Resolution.MINUTE)
        
        # Volume indicators
        indicators['spy_obv'] = self.obv(self._spy, Resolution.MINUTE)
        indicators['spy_ad'] = self.ad(self._spy, Resolution.MINUTE)
        
        # Specialized indicators
        indicators['spy_adx'] = self.adx(self._spy, 14, Resolution.MINUTE)
        indicators['spy_aroon'] = self.aroon(self._spy, 14, Resolution.MINUTE)
        indicators['spy_mfi'] = self.mfi(self._spy, 14, Resolution.MINUTE)
        indicators['spy_chop'] = self.chop(self._spy, 14, Resolution.MINUTE)
        indicators['spy_supertrend'] = self.str(self._spy, 10, 3, Resolution.MINUTE)
        indicators['spy_ichimoku'] = self.ichimoku(self._spy, 9, 26, 17, 52, 26, 26, Resolution.MINUTE)
        indicators['spy_psar'] = self.psar(self._spy, 0.02, 0.2, Resolution.MINUTE)
        
        # Market context indicators
        indicators['vix_sma'] = self.sma(self._vix, 10, Resolution.MINUTE)
        indicators['vix_ema'] = self.ema(self._vix, 10, Resolution.MINUTE)
        indicators['vix_bb'] = self.bb(self._vix, 20, 2, Resolution.MINUTE)
        indicators['vix_rsi'] = self.rsi(self._vix, 14, Resolution.MINUTE)
        
        return indicators

    def market_open(self):
        """Market open routine"""
        self._regime_detector.daily_reset()
        self._signal_generator.daily_reset()
        self._portfolio_manager.daily_reset()

    def update_regime_fast(self):
        """Fast regime updates"""
        if not self._indicators['spy_rsi'].is_ready:
            return
        
        self._regime_detector.update_fast_regime()

    def update_regime_comprehensive(self):
        """Comprehensive regime analysis"""
        if not self._indicators['spy_adx'].is_ready:
            return
        
        self._regime_detector.update_comprehensive_regime()


class AdvancedRegimeDetector:
    """Advanced multi-factor regime detection system"""
    
    def __init__(self, algorithm):
        self._algo = algorithm
        self._current_regime = "NEUTRAL"
        self._regime_confidence = 0.5
        self._regime_history = deque(maxlen=100)
        
        # Regime definitions
        self._regimes = {
            'TRENDING_BULL': {'momentum_factor': 1.4, 'mean_reversion_factor': 0.2, 'volatility_target': 0.15},
            'TRENDING_BEAR': {'momentum_factor': 1.4, 'mean_reversion_factor': 0.3, 'volatility_target': 0.20},
            'RANGING_LOW_VOL': {'momentum_factor': 0.2, 'mean_reversion_factor': 1.4, 'volatility_target': 0.10},
            'RANGING_HIGH_VOL': {'momentum_factor': 0.3, 'mean_reversion_factor': 1.4, 'volatility_target': 0.25},
            'CRISIS': {'momentum_factor': 0.1, 'mean_reversion_factor': 0.2, 'volatility_target': 0.05},
            'NEUTRAL': {'momentum_factor': 0.65, 'mean_reversion_factor': 0.65, 'volatility_target': 0.18}
        }
        
        # Regime scoring
        self._regime_scores = {regime: deque(maxlen=20) for regime in self._regimes.keys()}
        
        # Market state tracking
        self._volatility_state = 'NORMAL'
        self._trend_state = 'NEUTRAL'
        self._momentum_state = 'NEUTRAL'

    @property
    def current_regime(self):
        return self._current_regime

    def daily_reset(self):
        """Daily regime reset"""
        self._regime_history.append(self._current_regime)

    def update_fast_regime(self):
        """Fast regime detection using key indicators"""
        indicators = self._algo._indicators
        
        # Crisis detection (highest priority)
        if self.detect_crisis_regime(indicators):
            self._current_regime = 'CRISIS'
            self._regime_confidence = 0.95
            return
        
        # Fast assessments
        volatility_signal = self.assess_volatility_fast(indicators)
        trend_signal = self.assess_trend_fast(indicators)
        momentum_signal = self.assess_momentum_fast(indicators)
        
        # Combine signals
        new_regime = self.combine_fast_signals(volatility_signal, trend_signal, momentum_signal)
        
        if new_regime != self._current_regime:
            self._current_regime = new_regime
            self._regime_confidence = 0.7

    def detect_crisis_regime(self, indicators):
        """Detect crisis/extreme market conditions using multiple indicators"""
        crisis_score = 0
        
        # VIX spike detection
        if indicators['vix_sma'].is_ready:
            vix_value = indicators['vix_sma'].current.value
            if vix_value > 35:
                crisis_score += 2
            elif vix_value > 28:
                crisis_score += 1
        
        # VIX Bollinger Band breakout
        if indicators['vix_bb'].is_ready:
            vix_current = self._algo.securities[self._algo._vix].price
            vix_bb_upper = indicators['vix_bb'].upper_band.current.value
            vix_bb_lower = indicators['vix_bb'].lower_band.current.value
            
            if vix_current > vix_bb_upper:
                crisis_score += 2  # VIX breaking above upper BB
        
        # Extreme SPY volatility
        if indicators['spy_atr'].is_ready and indicators['spy_bb'].is_ready:
            atr_value = indicators['spy_atr'].current.value
            bb_width = (indicators['spy_bb'].upper_band.current.value - 
                       indicators['spy_bb'].lower_band.current.value) / indicators['spy_bb'].middle_band.current.value
            
            if atr_value > 0.06:
                crisis_score += 2
            elif atr_value > 0.04:
                crisis_score += 1
                
            if bb_width > 0.10:
                crisis_score += 2
            elif bb_width > 0.08:
                crisis_score += 1
        
        # Extreme momentum divergence
        if indicators['spy_rsi'].is_ready and indicators['spy_cci'].is_ready:
            rsi_value = indicators['spy_rsi'].current.value
            cci_value = indicators['spy_cci'].current.value
            
            if rsi_value < 20 or rsi_value > 80:
                crisis_score += 1
            if abs(cci_value) > 200:
                crisis_score += 1
        
        return crisis_score >= 4  # Threshold for crisis detection

    def assess_volatility_fast(self, indicators):
        """Fast volatility assessment"""
        if not indicators['spy_atr'].is_ready:
            return 'NORMAL'
        
        atr_value = indicators['spy_atr'].current.value
        
        if atr_value > 0.04:
            return 'HIGH'
        elif atr_value > 0.025:
            return 'ELEVATED'
        else:
            return 'LOW'

    def assess_trend_fast(self, indicators):
        """Fast trend assessment"""
        if not indicators['spy_adx'].is_ready or not indicators['spy_sma_20'].is_ready:
            return 'NEUTRAL'
        
        adx_value = indicators['spy_adx'].current.value
        current_price = self._algo.securities[self._algo._spy].price
        sma_20 = indicators['spy_sma_20'].current.value
        
        # Use ADX components for directional bias
        plus_di = indicators['spy_adx'].positive_directional_index.current.value if indicators['spy_adx'].positive_directional_index.is_ready else 0
        minus_di = indicators['spy_adx'].negative_directional_index.current.value if indicators['spy_adx'].negative_directional_index.is_ready else 0
        
        if adx_value > 25:
            if plus_di > minus_di and current_price > sma_20:
                return 'STRONG_BULL'
            elif minus_di > plus_di and current_price < sma_20:
                return 'STRONG_BEAR'
            else:
                return 'STRONG_BULL' if current_price > sma_20 else 'STRONG_BEAR'
        elif adx_value > 15:
            if current_price > sma_20:
                return 'WEAK_BULL'
            else:
                return 'WEAK_BEAR'
        else:
            return 'RANGING'

    def assess_momentum_fast(self, indicators):
        """Fast momentum assessment using multiple indicators"""
        if not indicators['spy_rsi'].is_ready or not indicators['spy_macd'].is_ready:
            return 'NEUTRAL'
        
        rsi_value = indicators['spy_rsi'].current.value
        macd_histogram = indicators['spy_macd'].histogram.current.value
        
        momentum_score = 0
        
        # RSI momentum
        if rsi_value > 60:
            momentum_score += 1
        elif rsi_value < 40:
            momentum_score -= 1
        
        # MACD momentum
        if macd_histogram > 0:
            momentum_score += 1
        else:
            momentum_score -= 1
        
        # CCI confirmation
        if indicators['spy_cci'].is_ready:
            cci_value = indicators['spy_cci'].current.value
            if cci_value > 100:
                momentum_score += 0.5
            elif cci_value < -100:
                momentum_score -= 0.5
        
        # Williams %R confirmation
        if indicators['spy_williams'].is_ready:
            williams_value = indicators['spy_williams'].current.value
            if williams_value > -20:  # Overbought
                momentum_score -= 0.3
            elif williams_value < -80:  # Oversold
                momentum_score += 0.3
        
        # Stochastic RSI for additional confirmation
        if indicators['spy_stoch_rsi'].is_ready:
            stoch_rsi_k = indicators['spy_stoch_rsi'].k.current.value
            if stoch_rsi_k > 80:
                momentum_score -= 0.5  # Overbought
            elif stoch_rsi_k < 20:
                momentum_score += 0.5  # Oversold
        
        if momentum_score >= 1.5:
            return 'BULLISH'
        elif momentum_score <= -1.5:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def combine_fast_signals(self, volatility_signal, trend_signal, momentum_signal):
        """Combine fast signals into regime"""
        self._algo.log(f"volatility: {volatility_signal}, trend signal: {trend_signal}, momentum signal: {momentum_signal}")
        if trend_signal in ['STRONG_BULL', 'WEAK_BULL']:
            if volatility_signal == 'LOW' and momentum_signal == 'BULLISH':
                return 'TRENDING_BULL'
            elif volatility_signal == 'HIGH':
                return 'RANGING_HIGH_VOL'
            else:
                return 'NEUTRAL'
        elif trend_signal in ['STRONG_BEAR', 'WEAK_BEAR']:
            if momentum_signal == 'BEARISH':
                return 'TRENDING_BEAR'
            else:
                return 'RANGING_HIGH_VOL'
        elif trend_signal == 'RANGING':
            if volatility_signal == 'LOW':
                return 'RANGING_LOW_VOL'
            else:
                return 'RANGING_HIGH_VOL'
        else:
            return 'NEUTRAL'

    def update_comprehensive_regime(self):
        """Comprehensive regime analysis"""
        indicators = self._algo._indicators
        
        # Calculate regime scores
        for regime in self._regimes.keys():
            score = self.calculate_regime_score(regime, indicators)
            self._regime_scores[regime].append(score)
        
        # Find best regime
        regime_averages = {}
        for regime, scores in self._regime_scores.items():
            if len(scores) >= 5:
                regime_averages[regime] = np.mean(list(scores)[-5:])
        
        if regime_averages:
            best_regime = max(regime_averages, key=regime_averages.get)
            confidence = regime_averages[best_regime]
            
            if confidence > 0.6:
                self._current_regime = best_regime
                self._regime_confidence = min(confidence, 0.95)

    def calculate_regime_score(self, regime, indicators):
        """Calculate comprehensive regime score"""
        score = 0.0
        
        # Trend component (40% weight)
        trend_score = self.calculate_trend_score(regime, indicators)
        score += trend_score * 0.4
        
        # Volatility component (30% weight)
        volatility_score = self.calculate_volatility_score(regime, indicators)
        score += volatility_score * 0.3
        
        # Momentum component (20% weight)
        momentum_score = self.calculate_momentum_score(regime, indicators)
        score += momentum_score * 0.2
        
        # Market structure component (10% weight)
        structure_score = self.calculate_structure_score(regime, indicators)
        score += structure_score * 0.1
        
        return np.clip(score, 0, 1)

    def calculate_trend_score(self, regime, indicators):
        """Calculate trend component score using multiple indicators"""
        if not indicators['spy_adx'].is_ready:
            return 0.5
        
        adx_value = indicators['spy_adx'].current.value
        current_price = self._algo.securities[self._algo._spy].price
        total_score = 0.0
        
        # ADX trend strength (40% weight)
        if regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
            total_score += min(adx_value / 50, 1.0) * 0.4
        else:
            total_score += max(0, (40 - adx_value) / 40) * 0.4
        
        # Moving average alignment (30% weight)
        if indicators['spy_sma_10'].is_ready and indicators['spy_sma_20'].is_ready and indicators['spy_sma_50'].is_ready:
            sma_10 = indicators['spy_sma_10'].current.value
            sma_20 = indicators['spy_sma_20'].current.value
            sma_50 = indicators['spy_sma_50'].current.value
            
            # Check moving average alignment
            if regime == 'TRENDING_BULL':
                if current_price > sma_10 > sma_20 > sma_50:
                    total_score += 0.3
                elif current_price > sma_20:
                    total_score += 0.15
            elif regime == 'TRENDING_BEAR':
                if current_price < sma_10 < sma_20 < sma_50:
                    total_score += 0.3
                elif current_price < sma_20:
                    total_score += 0.15
            elif regime in ['RANGING_LOW_VOL', 'RANGING_HIGH_VOL']:
                # Price oscillating around moving averages
                if sma_10 * 0.98 < current_price < sma_10 * 1.02:
                    total_score += 0.2
        
        # Aroon trend confirmation (20% weight)
        if indicators['spy_aroon'].is_ready:
            aroon_up = indicators['spy_aroon'].aroon_up.current.value
            aroon_down = indicators['spy_aroon'].aroon_down.current.value
            
            if regime == 'TRENDING_BULL' and aroon_up > 70 and aroon_down < 30:
                total_score += 0.2
            elif regime == 'TRENDING_BEAR' and aroon_down > 70 and aroon_up < 30:
                total_score += 0.2
            elif regime in ['RANGING_LOW_VOL', 'RANGING_HIGH_VOL'] and aroon_up < 60 and aroon_down < 60:
                total_score += 0.15
        
        # Parabolic SAR trend confirmation (10% weight)
        if indicators['spy_psar'].is_ready:
            psar_value = indicators['spy_psar'].current.value
            
            if regime == 'TRENDING_BULL' and current_price > psar_value:
                total_score += 0.1
            elif regime == 'TRENDING_BEAR' and current_price < psar_value:
                total_score += 0.1
        
        return min(total_score, 1.0)

    def calculate_volatility_score(self, regime, indicators):
        """Calculate volatility component score using multiple indicators"""
        if not indicators['spy_atr'].is_ready:
            return 0.5
        
        atr_value = indicators['spy_atr'].current.value
        target_vol = self._regimes[regime]['volatility_target']
        total_score = 0.0
        
        # ATR volatility alignment (50% weight)
        vol_diff = abs(atr_value - target_vol)
        atr_score = max(0, 1 - vol_diff * 10)
        total_score += atr_score * 0.5
        
        # Choppiness Index alignment (30% weight)
        if indicators['spy_chop'].is_ready:
            chop_value = indicators['spy_chop'].current.value
            
            if regime in ['RANGING_LOW_VOL', 'RANGING_HIGH_VOL'] and chop_value > 61.8:
                total_score += 0.3  # High choppiness confirms ranging market
            elif regime in ['TRENDING_BULL', 'TRENDING_BEAR'] and chop_value < 38.2:
                total_score += 0.3  # Low choppiness confirms trending market
            elif 38.2 <= chop_value <= 61.8:
                total_score += 0.15  # Neutral choppiness
        
        # Bollinger Band width for volatility confirmation (20% weight)
        if indicators['spy_bb'].is_ready:
            bb_upper = indicators['spy_bb'].upper_band.current.value
            bb_lower = indicators['spy_bb'].lower_band.current.value
            bb_middle = indicators['spy_bb'].middle_band.current.value
            
            if bb_middle > 0:
                bb_width = (bb_upper - bb_lower) / bb_middle
                
                if regime in ['RANGING_HIGH_VOL', 'CRISIS'] and bb_width > 0.08:
                    total_score += 0.2
                elif regime in ['RANGING_LOW_VOL', 'TRENDING_BULL'] and bb_width < 0.04:
                    total_score += 0.2
                elif 0.04 <= bb_width <= 0.08:
                    total_score += 0.1
        
        return min(total_score, 1.0)

    def calculate_momentum_score(self, regime, indicators):
        """Calculate momentum component score using multiple indicators"""
        if not indicators['spy_rsi'].is_ready or not indicators['spy_macd'].is_ready:
            return 0.5
        
        rsi_value = indicators['spy_rsi'].current.value
        macd_signal = indicators['spy_macd'].signal.current.value
        macd_value = indicators['spy_macd'].current.value
        
        momentum_factor = self._regimes[regime]['momentum_factor']
        total_score = 0.0
        
        # RSI momentum alignment (25% weight)
        rsi_momentum = (rsi_value - 50) / 50
        if regime in ['TRENDING_BULL', 'RANGING_LOW_VOL']:
            total_score += max(0, rsi_momentum) * 0.25
        elif regime in ['TRENDING_BEAR']:
            total_score += max(0, -rsi_momentum) * 0.25
        
        # MACD momentum alignment (25% weight)
        macd_momentum = 1 if macd_value > macd_signal else -1
        if regime in ['TRENDING_BULL', 'RANGING_LOW_VOL'] and macd_momentum > 0:
            total_score += 0.25
        elif regime in ['TRENDING_BEAR'] and macd_momentum < 0:
            total_score += 0.25
        
        # CCI momentum (20% weight)
        if indicators['spy_cci'].is_ready:
            cci_value = indicators['spy_cci'].current.value
            cci_normalized = np.tanh(cci_value / 200)  # Normalize to -1 to 1
            if regime in ['TRENDING_BULL'] and cci_normalized > 0:
                total_score += cci_normalized * 0.2
            elif regime in ['TRENDING_BEAR'] and cci_normalized < 0:
                total_score += abs(cci_normalized) * 0.2
        
        # Williams %R confirmation (15% weight)
        if indicators['spy_williams'].is_ready:
            williams_value = indicators['spy_williams'].current.value
            williams_normalized = (williams_value + 50) / 50  # Convert to -1 to 1 scale
            if regime in ['TRENDING_BULL'] and williams_normalized > 0:
                total_score += williams_normalized * 0.15
            elif regime in ['TRENDING_BEAR'] and williams_normalized < 0:
                total_score += abs(williams_normalized) * 0.15
        
        # MFI volume-price momentum (15% weight)
        if indicators['spy_mfi'].is_ready:
            mfi_value = indicators['spy_mfi'].current.value
            mfi_momentum = (mfi_value - 50) / 50
            if regime in ['TRENDING_BULL'] and mfi_momentum > 0:
                total_score += mfi_momentum * 0.15
            elif regime in ['TRENDING_BEAR'] and mfi_momentum < 0:
                total_score += abs(mfi_momentum) * 0.15
        
        return min(total_score * momentum_factor, 1.0)

    def calculate_structure_score(self, regime, indicators):
        """Calculate market structure component score"""
        score = 0.0
        current_price = self._algo.securities[self._algo._spy].price
        
        # Bollinger Bands structure
        if indicators['spy_bb'].is_ready:
            bb_upper = indicators['spy_bb'].upper_band.current.value
            bb_lower = indicators['spy_bb'].lower_band.current.value
            bb_middle = indicators['spy_bb'].middle_band.current.value
            
            # Bollinger band position
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            
            if regime == 'TRENDING_BULL' and bb_position > 0.6:
                score += 0.4
            elif regime == 'TRENDING_BEAR' and bb_position < 0.4:
                score += 0.4
            elif regime in ['RANGING_LOW_VOL', 'RANGING_HIGH_VOL'] and 0.3 < bb_position < 0.7:
                score += 0.4
        
        # Ichimoku Cloud structure
        if indicators['spy_ichimoku'].is_ready:
            tenkan = indicators['spy_ichimoku'].tenkan.current.value
            kijun = indicators['spy_ichimoku'].kijun.current.value
            senkou_a = indicators['spy_ichimoku'].senkou_a.current.value
            senkou_b = indicators['spy_ichimoku'].senkou_b.current.value
            
            # Cloud analysis
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            
            # Price position relative to cloud
            if current_price > cloud_top:  # Above cloud
                if regime == 'TRENDING_BULL':
                    score += 0.3
            elif current_price < cloud_bottom:  # Below cloud
                if regime == 'TRENDING_BEAR':
                    score += 0.3
            else:  # Inside cloud
                if regime in ['RANGING_LOW_VOL', 'RANGING_HIGH_VOL', 'NEUTRAL']:
                    score += 0.2
            
            # Tenkan-Kijun cross
            if tenkan > kijun and regime in ['TRENDING_BULL', 'NEUTRAL']:
                score += 0.1
            elif tenkan < kijun and regime in ['TRENDING_BEAR', 'NEUTRAL']:
                score += 0.1
        
        # SuperTrend structure
        if indicators['spy_supertrend'].is_ready:
            supertrend_value = indicators['spy_supertrend'].current.value
            
            if regime == 'TRENDING_BULL' and current_price > supertrend_value:
                score += 0.2
            elif regime == 'TRENDING_BEAR' and current_price < supertrend_value:
                score += 0.2
        
        return min(score, 1.0)

    def get_regime_parameters(self):
        """Get current regime parameters"""
        return self._regimes.get(self._current_regime, self._regimes['NEUTRAL'])


class DynamicSignalGenerator:
    """Dynamic signal generation based on regime"""
    
    def __init__(self, algorithm):
        self._algo = algorithm
        self._signals = {}
        self._signal_history = deque(maxlen=50)

    def daily_reset(self):
        """Daily signal reset"""
        self._signals = {}


    def generate_signals(self, symbols):
        """Generate signals for given symbols"""
        regime_params = self._algo._regime_detector.get_regime_parameters()
        current_regime = self._algo._regime_detector._current_regime
        
        signals = {}
        
        for symbol in symbols:
            momentum_signal = self.generate_momentum_signal(symbol, regime_params)
            
            # Heavily weight momentum in trending markets
            if current_regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
                combined_signal = momentum_signal
            else:
                mean_reversion_signal = self.generate_mean_reversion_signal(symbol, regime_params)
                combined_signal = (momentum_signal * 0.9 + mean_reversion_signal * 0.3)
            
            # Less aggressive regime adjustments
            adjusted_signal = self.apply_light_regime_adjustments(combined_signal, current_regime)
            
            signals[symbol] = adjusted_signal
        
        return signals

    def apply_light_regime_adjustments(self, signal, regime):
        """Apply lighter regime-specific signal adjustments"""
        if regime == 'CRISIS':
            return signal * 0.7  # Less reduction in crisis
        elif regime == 'TRENDING_BULL':
            return signal * 1.1 if signal > 0 else signal * 0.9
        elif regime == 'TRENDING_BEAR':
            return signal * 1.1 if signal < 0 else signal * 0.9
        else:
            return signal

    def get_cross_asset_signals(self):
        """Calculate cross-asset signals manually (no ratio indicator exists)"""
        signals = {}
        
        if (self._algo._spy in self._algo.securities and 
            self._algo._tlt in self._algo.securities and
            self._algo._qqq in self._algo.securities and
            self._algo._iwm in self._algo.securities and
            self._algo._gld in self._algo.securities):
            
            spy_price = self._algo.securities[self._algo._spy].price
            tlt_price = self._algo.securities[self._algo._tlt].price
            qqq_price = self._algo.securities[self._algo._qqq].price
            iwm_price = self._algo.securities[self._algo._iwm].price
            gld_price = self._algo.securities[self._algo._gld].price
            
            # TLT/SPY ratio (bonds vs stocks)
            if spy_price > 0:
                signals['tlt_spy_ratio'] = tlt_price / spy_price
            
            # QQQ/IWM ratio (large cap tech vs small cap)
            if iwm_price > 0:
                signals['qqq_iwm_ratio'] = qqq_price / iwm_price
                
            # GLD/SPY ratio (gold vs stocks)
            if spy_price > 0:
                signals['gld_spy_ratio'] = gld_price / spy_price
        
        return signals

    def generate_momentum_signal(self, symbol, regime_params):
        """Generate momentum-based signal using multiple indicators"""
        if symbol not in self._algo.securities:
            return 0.0
        
        signal = 0.0
        
        # Create symbol-specific indicators if needed
        rsi_key = f"{symbol}_rsi"
        macd_key = f"{symbol}_macd"
        cci_key = f"{symbol}_cci"
        williams_key = f"{symbol}_williams"
        stoch_key = f"{symbol}_stoch"
        
        if rsi_key not in self._algo._indicators:
            self._algo._indicators[rsi_key] = self._algo.rsi(symbol, 14, Resolution.MINUTE)
        if macd_key not in self._algo._indicators:
            self._algo._indicators[macd_key] = self._algo.macd(symbol, 12, 26, 9, Resolution.MINUTE)
        if cci_key not in self._algo._indicators:
            self._algo._indicators[cci_key] = self._algo.cci(symbol, 20, Resolution.MINUTE)
        if williams_key not in self._algo._indicators:
            self._algo._indicators[williams_key] = self._algo.wilr(symbol, 14, Resolution.MINUTE)
        if stoch_key not in self._algo._indicators:
            self._algo._indicators[stoch_key] = self._algo.sto(symbol, 14, 3, Resolution.MINUTE)
        
        # Multi-indicator momentum signal
        momentum_signals = []
        
        # RSI momentum (25% weight)
        if self._algo._indicators[rsi_key].is_ready:
            rsi_value = self._algo._indicators[rsi_key].current.value
            rsi_signal = (rsi_value - 50) / 50
            momentum_signals.append(rsi_signal * 0.25)
        
        # MACD momentum (25% weight)
        if self._algo._indicators[macd_key].is_ready:
            macd_value = self._algo._indicators[macd_key].current.value
            macd_signal_value = self._algo._indicators[macd_key].signal.current.value
            macd_signal = 1 if macd_value > macd_signal_value else -1
            momentum_signals.append(macd_signal * 0.25)
        
        # CCI momentum (20% weight)
        if self._algo._indicators[cci_key].is_ready:
            cci_value = self._algo._indicators[cci_key].current.value
            cci_signal = np.tanh(cci_value / 200)  # Normalize to -1 to 1
            momentum_signals.append(cci_signal * 0.2)
        
        # Williams %R momentum (15% weight)
        if self._algo._indicators[williams_key].is_ready:
            williams_value = self._algo._indicators[williams_key].current.value
            williams_signal = (williams_value + 50) / 50  # Convert to -1 to 1
            momentum_signals.append(williams_signal * 0.15)
        
        # Stochastic momentum (15% weight)
        if self._algo._indicators[stoch_key].is_ready:
            stoch_k = self._algo._indicators[stoch_key].stoch_k.current.value
            stoch_signal = (stoch_k - 50) / 50
            momentum_signals.append(stoch_signal * 0.15)
        
        # Combine all momentum signals
        if momentum_signals:
            signal = sum(momentum_signals)
        
        return np.clip(signal, -1, 1)

    def generate_mean_reversion_signal(self, symbol, regime_params):
        """Generate mean reversion signal using multiple indicators"""
        if symbol not in self._algo.securities:
            return 0.0
        
        signal = 0.0
        current_price = self._algo.securities[symbol].price
        
        # Create symbol-specific indicators if needed
        bb_key = f"{symbol}_bb"
        rsi_key = f"{symbol}_rsi"
        williams_key = f"{symbol}_williams"
        stoch_rsi_key = f"{symbol}_stoch_rsi"
        mfi_key = f"{symbol}_mfi"
        
        if bb_key not in self._algo._indicators:
            self._algo._indicators[bb_key] = self._algo.bb(symbol, 20, 2, Resolution.MINUTE)
        if rsi_key not in self._algo._indicators:
            self._algo._indicators[rsi_key] = self._algo.rsi(symbol, 14, Resolution.MINUTE)
        if williams_key not in self._algo._indicators:
            self._algo._indicators[williams_key] = self._algo.wilr(symbol, 14, Resolution.MINUTE)
        if stoch_rsi_key not in self._algo._indicators:
            self._algo._indicators[stoch_rsi_key] = self._algo.srsi(symbol, 14, 14, 3, 3, Resolution.MINUTE)
        if mfi_key not in self._algo._indicators:
            self._algo._indicators[mfi_key] = self._algo.mfi(symbol, 14, Resolution.MINUTE)
        
        reversion_signals = []
        
        # Bollinger Bands mean reversion (40% weight)
        if self._algo._indicators[bb_key].is_ready:
            bb_upper = self._algo._indicators[bb_key].upper_band.current.value
            bb_lower = self._algo._indicators[bb_key].lower_band.current.value
            bb_middle = self._algo._indicators[bb_key].middle_band.current.value
            
            if (bb_upper - bb_lower) > 0:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                
                if bb_position > 0.8:
                    reversion_signals.append(-0.6 * 0.4)  # Strong sell signal
                elif bb_position < 0.2:
                    reversion_signals.append(0.6 * 0.4)   # Strong buy signal
                else:
                    # Revert to middle
                    reversion_signals.append((0.5 - bb_position) * 0.8 * 0.4)
        
        # RSI mean reversion (25% weight)
        if self._algo._indicators[rsi_key].is_ready:
            rsi_value = self._algo._indicators[rsi_key].current.value
            
            if rsi_value > 75:
                reversion_signals.append(-0.7 * 0.25)  # Overbought
            elif rsi_value < 25:
                reversion_signals.append(0.7 * 0.25)   # Oversold
            elif rsi_value > 60:
                reversion_signals.append(-0.3 * 0.25)  # Mildly overbought
            elif rsi_value < 40:
                reversion_signals.append(0.3 * 0.25)   # Mildly oversold
        
        # Williams %R mean reversion (20% weight)
        if self._algo._indicators[williams_key].is_ready:
            williams_value = self._algo._indicators[williams_key].current.value
            
            if williams_value > -10:  # Severely overbought
                reversion_signals.append(-0.8 * 0.2)
            elif williams_value < -90:  # Severely oversold
                reversion_signals.append(0.8 * 0.2)
            elif williams_value > -20:  # Overbought
                reversion_signals.append(-0.4 * 0.2)
            elif williams_value < -80:  # Oversold
                reversion_signals.append(0.4 * 0.2)
        
        # Stochastic RSI mean reversion (10% weight)
        if self._algo._indicators[stoch_rsi_key].is_ready:
            stoch_rsi_k = self._algo._indicators[stoch_rsi_key].k.current.value
            
            if stoch_rsi_k > 85:
                reversion_signals.append(-0.6 * 0.1)
            elif stoch_rsi_k < 15:
                reversion_signals.append(0.6 * 0.1)
        
        # MFI mean reversion (5% weight)
        if self._algo._indicators[mfi_key].is_ready:
            mfi_value = self._algo._indicators[mfi_key].current.value
            
            if mfi_value > 80:
                reversion_signals.append(-0.5 * 0.05)
            elif mfi_value < 20:
                reversion_signals.append(0.5 * 0.05)
        
        # Combine all reversion signals
        if reversion_signals:
            signal = sum(reversion_signals)
        
        return np.clip(signal, -1, 1)

    def apply_regime_adjustments(self, signal, regime):
        """Apply regime-specific signal adjustments"""
        if regime == 'CRISIS':
            return signal * 0.3  # Reduce signal strength in crisis
        elif regime == 'TRENDING_BULL':
            return signal * 1.2 if signal > 0 else signal * 0.8  # Enhance bullish signals
        elif regime == 'TRENDING_BEAR':
            return signal * 1.2 if signal < 0 else signal * 0.8  # Enhance bearish signals
        elif regime in ['RANGING_LOW_VOL', 'RANGING_HIGH_VOL']:
            return signal * 0.9  # Slightly reduce signals in ranging markets
        else:
            return signal


class AdaptivePortfolioManager:
    """Adaptive portfolio management"""
    
    def __init__(self, algorithm):
        self._algo = algorithm
        self._position_sizes = {}
        self._max_positions = 25
        self._max_position_size = 0.15

    def daily_reset(self):
        """Daily portfolio reset"""
        pass

    def calculate_position_sizes(self, signals):
        """Calculate position sizes based on signals and regime"""
        regime_params = self._algo._regime_detector.get_regime_parameters()
        current_regime = self._algo._regime_detector._current_regime
        
        # Regime-based position sizing

        regime_multipliers = {
            'TRENDING_BULL': 1.5,
            'TRENDING_BEAR': 1.5,
            'RANGING_LOW_VOL': 1.3,
            'RANGING_HIGH_VOL': 0.9,
            'CRISIS': 0.6,
            'NEUTRAL': 0.9
        }
        
        base_multiplier = regime_multipliers.get(current_regime, 0.9)
        
        # Sort signals by strength
        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        position_sizes = {}
        total_allocated = 0.0
        
        for symbol, signal in sorted_signals[:self._max_positions]:
            if abs(signal) > 0.0225:  # Minimum signal threshold
                # Base position size
                base_size = min(abs(signal) * self._max_position_size, self._max_position_size)
                
                # Apply regime multiplier
                adjusted_size = base_size * base_multiplier
                
                # Ensure we don't over-allocate
                if total_allocated + adjusted_size <= 0.98:  # Keep 5% cash buffer
                    position_sizes[symbol] = adjusted_size * (1 if signal > 0 else -1)
                    total_allocated += adjusted_size
                else:
                    break
        
        return position_sizes
   

class RegimeAdaptiveAlphaModel(AlphaModel):
    """Regime adaptive alpha generation"""
    
    def __init__(self, algorithm):
        self._algorithm = algorithm
        self._symbols_data = {}

    def update(self, algorithm, data):
        """Generate alpha insights"""
        insights = []
        
        # Only generate insights if regime detection is ready and not warming up
        if algorithm.is_warming_up:
            return insights
            
        if not hasattr(self._algorithm._regime_detector, '_current_regime'):
            return insights
        
        # Get current universe
        securities = list(algorithm.securities.keys())
        
        # Filter out benchmark and regime detection symbols
        active_symbols = [s for s in securities
                         if s not in [self._algorithm._spy, self._algorithm._vix, self._algorithm._tlt, 
                                    self._algorithm._qqq, self._algorithm._iwm, self._algorithm._gld]
                         and algorithm.securities[s].has_data
                         and algorithm.securities[s].price > 0]
        
        if not active_symbols:
            return insights
        
        # Generate signals
        signals = self._algorithm._signal_generator.generate_signals(active_symbols)
        
        # Create insights from signals
        for symbol, signal in signals.items():
            if abs(signal) > 0.0225:  # Signal threshold
                direction = InsightDirection.UP if signal > 0 else InsightDirection.DOWN
                magnitude = min(abs(signal), 1.0)
                
                # Regime-based insight duration
                current_regime = self._algorithm._regime_detector._current_regime
                if current_regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
                    duration = timedelta(hours=4)
                else:
                    duration = timedelta(hours=2)
                
                insight = Insight.price(symbol, duration, direction, weight = magnitude)
                insights.append(insight)
        
        return insights


class AdaptiveRiskManagementModel(RiskManagementModel):
    """Adaptive risk management based on regime"""
    
    def __init__(self, algorithm):
        self._algorithm = algorithm
        self._max_portfolio_risk = 0.05  # 4% daily portfolio risk
        self._max_position_risk = 0.02  # 1.2% per position risk

    def manage_risk(self, algorithm, targets):
        """Manage portfolio risk"""
        risk_adjusted_targets = []
        
        current_regime = getattr(self._algorithm._regime_detector, '_current_regime', 'NEUTRAL')
        
        # Regime-based risk multipliers

        risk_multipliers = {
            'TRENDING_BULL': 1.2,
            'TRENDING_BEAR': 1.0,
            'RANGING_LOW_VOL': 1.0,
            'RANGING_HIGH_VOL': 0.8,
            'CRISIS': 0.6,
            'NEUTRAL': 0.9
        }
        
        risk_multiplier = risk_multipliers.get(current_regime, 0.9)
        
        # Check overall portfolio risk
        total_risk = sum(abs(target.quantity * algorithm.securities[target.symbol].price) 
                        for target in targets if target.symbol in algorithm.securities)
        
        portfolio_value = algorithm.portfolio.total_portfolio_value
        portfolio_risk = total_risk / portfolio_value if portfolio_value > 0 else 0
        
        # Scale down if portfolio risk is too high
        if portfolio_risk > self._max_portfolio_risk * risk_multiplier:
            scale_factor = (self._max_portfolio_risk * risk_multiplier) / portfolio_risk
            for target in targets:
                new_quantity = target.quantity * scale_factor
                risk_adjusted_targets.append(PortfolioTarget(target.symbol, new_quantity))
        else:
            risk_adjusted_targets = targets
        
        # Position-level risk management
        final_targets = []
        for target in risk_adjusted_targets:
            if target.symbol in algorithm.securities:
                position_value = abs(target.quantity * algorithm.securities[target.symbol].price)
                position_risk = position_value / portfolio_value if portfolio_value > 0 else 0
                
                if position_risk <= self._max_position_risk * risk_multiplier:
                    final_targets.append(target)
                else:
                    # Scale down position
                    max_quantity = (self._max_position_risk * risk_multiplier * portfolio_value) / algorithm.securities[target.symbol].price
                    scaled_quantity = max_quantity if target.quantity > 0 else -max_quantity
                    final_targets.append(PortfolioTarget(target.symbol, scaled_quantity))
        
        return final_targets



