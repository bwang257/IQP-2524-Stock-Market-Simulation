"""
Note that this is a slightly adapted version of the HMM-Regime Adaptive Multi-Alpha Model considered in the IQP report. 
Here, the QQQ ETF was segmented into 4 latent regimes using a HMM to check if the market regimes could be exploited through conditional strategies. 
"""

# region imports
from AlgorithmImports import *
import joblib
import io
# endregion

class HMMRegimeAlgo(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2025, 1, 1)
        self.set_end_date(2025, 6, 30)
        self.set_cash(1000000)  # Set Strategy Cash

        #---QQQ Regime Detection--
        self.qqq = Symbol.create("QQQ", SecurityType.EQUITY, Market.USA)
        self.add_equity(self.qqq)
        self.vixy = Symbol.create("VIXY", SecurityType.EQUITY, Market.USA)

        self.add_chart(Chart("QQQ Regime"))

        if self.object_store.contains_key("hmm_model") and self.object_store.contains_key("hmm_scaler"):
            try:
                model_bytes = self.object_store.read_bytes("hmm_model")
                scaler_bytes = self.object_store.read_bytes("hmm_scaler")

                # If byte arrays are lists, convert to bytes
                if isinstance(model_bytes, list):
                    model_bytes = bytes(model_bytes)
                if isinstance(scaler_bytes, list):
                    scaler_bytes = bytes(scaler_bytes)

                # Load from in-memory bytes
                self.model = joblib.load(io.BytesIO(model_bytes))
                self.scaler = joblib.load(io.BytesIO(scaler_bytes))

            except Exception as e:
                self.debug(f"Failed to load model or scaler: {e}")
        else:
            self.debug("Model or scaler not found in ObjectStore.")


        #--Algorithm Settings--
        
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.data_normalization_mode = DataNormalizationMode.RAW
        # self.set_universe_selection(QQQConstituentsUniverseSelectionModel(self.universe_settings))
        self.today = -1


    def on_data(self, slice: Slice):

        if self.today == self.time.day:
            return 
        self.today = self.time.day


        features =  self.compute_features(self.qqq, self.vixy)
        if features:
            X = self.scaler.transform([features])  # Use same scaler as in training
            regime = self.model.predict(X)[0]
            self.plot("QQQ Regime", "Regime", regime)

            #Regime 0: Strong Bullish Trend
            #Regime 1: Low Volatility Mean Reversion
            #Regime 2: Strong bearish trend/ breakdown
            #Regime 3: Volatile Bullish Recovery / Choppy Uptrend


            self.debug(f"Predicted regime: {regime}")

            if regime == 0:
                self.set_holdings(self.qqq, 1.0)
                self.debug(f"Longing QQQ given current regime {regime} on {self.time.date()}")
                self.log(f"Longing QQQ given current regime {regime} on {self.time.date()}")
            elif regime == 1:
                self.liquidate(self.qqq)

            elif regime == 2:
                self.set_holdings(self.qqq, -1)
                self.debug(f"Shorting QQQ given current regime {regime} on {self.time.date()}")
                self.log(f"Shorting QQQ given current regime {regime} on {self.time.date()}")

    
    def compute_features(self, etf_symbol, vixy_symbol):

        #Pull Data
        history_lookback = 14
        history = self.history([etf_symbol, vixy_symbol], history_lookback, Resolution.DAILY)
        if history.empty or etf_symbol not in history.index.get_level_values(0):
            return None
        
        qqq = history.loc[etf_symbol].copy()
        vix = history.loc[vixy_symbol]["close"].rename("VIX")
        df = qqq.merge(vix, left_index=True, right_index=True, how="left")


        # Pre-allocate indicator containers
        gap_pct = ((df["open"] - df["close"].shift(1)) / df["close"].shift(1)).tolist()
        volume_change = df["volume"].pct_change().tolist()

        # Indicators
        bb = BollingerBands(10, 2)
        macd = MovingAverageConvergenceDivergence(5, 10, 5)
        rsi = RelativeStrengthIndex(7)
        adx = AverageDirectionalIndex(7)

        bb_width_vals = [np.nan] * len(df)
        bb_pos_vals = [np.nan] * len(df)
        macd_vals = [np.nan] * len(df)
        rsi_vals = [np.nan] * len(df)
        adx_vals = [np.nan] * len(df)

        for i, (time, row) in enumerate(df.iterrows()):
            close = row["close"]
            bb.Update(time, close)
            macd.Update(time, close)
            rsi.Update(time, close)
            adx.Update(TradeBar(time, etf_symbol, row["open"], row["high"], row["low"], close, row["volume"]))

            if bb.IsReady:
                bb_width_vals[i] = bb.band_width.Current.Value
                bb_pos_vals[i] = (close - bb.LowerBand.Current.Value) / (bb.UpperBand.Current.Value - bb.LowerBand.Current.Value)

            if macd.IsReady:
                macd_vals[i] = macd.Current.Value

            if rsi.IsReady:
                rsi_vals[i] = rsi.Current.Value

            if adx.IsReady:
                adx_vals[i] = adx.Current.Value
        
        df["bb_width"] = bb_width_vals
        df["bb_pos"] = bb_pos_vals
        df["macd_diff"] = macd_vals
        df["rsi"] = rsi_vals
        df["adx"] = adx_vals

        df["gap_pct"] = gap_pct
        df["volume_change"] = volume_change
        df["bb_width"] = bb_width_vals
        df["bb_pos"] = bb_pos_vals
        df["macd_diff"] = macd_vals
        df["rsi"] = rsi_vals
        df["adx"] = adx_vals

        df.dropna(inplace=True)

        if df.empty:
            return None

        latest = df.iloc[-1]
        features = [
            latest["gap_pct"],
            latest["bb_pos"],
            latest["volume_change"],
            latest["VIX"],
            latest["macd_diff"],
            latest["rsi"],
            latest["bb_width"],
            latest["adx"]
        ]

        return features
