# region imports
from AlgorithmImports import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from io import BytesIO
from TradingEnv import TradingEnv
import numpy as np
# endregion

class RLTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.set_start_date(2023, 10, 1)
        self.set_end_date(2024, 6, 30)
        self.set_cash(100000)
        self.symbol = self.add_equity("SPY", Resolution.MINUTE).Symbol

        # Indicators
        self.rsi = RelativeStrengthIndex(14)
        self.roc = RateOfChange(14)
        self.bb = BollingerBands(20, 2, MovingAverageType.SIMPLE)
        self.adx = AverageDirectionalIndex(14)
        self.obv = OnBalanceVolume()
        self.cmf = ChaikinMoneyFlow(self.symbol, 20)
        self.rs_vol = RogersSatchellVolatility(self.symbol, 14)
        self.spin = SpinningTop()
        self.ha = HeikinAshi()
        self.atr = AverageTrueRange(14, MovingAverageType.WILDERS)
        self.macd = MovingAverageConvergenceDivergence(12, 26, 9, MovingAverageType.EXPONENTIAL)
        self.doji = Doji()

        # Consolidator
        self.consolidator = TradeBarConsolidator(timedelta(minutes=15))
        self.consolidator.data_consolidated += self.OnNewBar
        self.subscription_manager.add_consolidator(self.symbol, self.consolidator)

        # Register indicators with 15-minute bars
        indicators = [self.rsi, self.roc, self.bb, self.adx, self.obv, self.cmf,
                      self.rs_vol, self.spin, self.ha, self.atr, self.macd, self.doji]
        for ind in indicators:
            self.RegisterIndicator(self.symbol, ind, self.consolidator)

        self.set_warm_up(timedelta(days=5), Resolution.MINUTE)
        self.indicators_ready = False
        self.model = None
        self.env = None
        self.LoadModel()

    def OnNewBar(self, sender, bar):
        self.Step()

    def Step(self):
        obs = self.GetObservation()
        if obs is None:
            return
        action, _ = self.model.predict(obs, deterministic=True)
        self.ExecuteAction(action)

    def GetObservation(self):
        if not self.indicators_ready and self.rsi.IsReady:
            self.indicators_ready = True

        if not self.indicators_ready:
            return None

        price = self.Securities[self.symbol].Price
        bb = self.bb
        zscore = (price - bb.MiddleBand.Current.Value) / (bb.UpperBand.Current.Value - bb.LowerBand.Current.Value) if bb.UpperBand.Current.Value != bb.LowerBand.Current.Value else 0

        features = np.array([
            price,
            self.rsi.Current.Value,
            self.roc.Current.Value,
            zscore,
            self.adx.Current.Value,
            self.obv.Current.Value,
            self.cmf.Current.Value,
            self.rs_vol.Current.Value,
            self.spin.Current.Value,
        ])

        return features.reshape(1, -1)

    def ExecuteAction(self, action):
        self.debug("Execute action called")
        if action == 1 and not self.Portfolio[self.symbol].Invested:
            self.SetHoldings(self.symbol, 1.0)
        elif action == 0 and self.Portfolio[self.symbol].Invested:
            self.Liquidate(self.symbol)
        elif action == 2 and not self.Portfolio[self.symbol].Invested:
            self.SetHoldings(self.symbol, -1.0)
        self.debug("Execute action called done")

    def LoadModel(self):
        def get_objstore_bytes(key):
            b = self.ObjectStore.ReadBytes(key)
            if isinstance(b, list):
                b = bytes(b)
            elif hasattr(b, '__iter__') and not isinstance(b, (bytes, bytearray)):
                b = bytes([int(x) for x in b])
            return b

        try:
            self.Debug("Loading PPO model...")
            model_bytes = get_objstore_bytes("ppo_trading_agent")
            vec_bytes = get_objstore_bytes("ppo_vec_normalize")

            with open("/tmp/agent.zip", "wb") as f:
                f.write(model_bytes)
            with open("/tmp/vec.pkl", "wb") as f:
                f.write(vec_bytes)

            dummy_data = pd.DataFrame(np.zeros((10, 9)), columns=[
                'close', 'RSI', 'ROC', 'BB_zscore', 'ADX', 'OBV', 'CMF', 'RSVol', 'SpinningTop'
            ])
            dummy_env = DummyVecEnv([lambda: TradingEnv(dummy_data)])

            self.env = VecNormalize.load("/tmp/vec.pkl", dummy_env)
            self.env.training = False
            self.env.norm_reward = False

            assert self.env.observation_space.shape == (9,), f"Loaded VecNormalize expects shape {self.env.observation_space.shape}, expected (9,)"

            self.model = PPO.load("/tmp/agent.zip", env=self.env)
            self.Debug("Model loaded successfully.")
        except Exception as e:
            self.Debug(f"Model load failed: {e}")



