# region imports
from AlgorithmImports import *
import gym
from gym import spaces
import numpy as np
# endregion

class TradingEnv(gym.Env):
    def __init__(self, df, transaction_cost=0.001, risk_penalty=0.3):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(df)
        self.current_step = 1
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty

        self.position = 0  # -1 short, 0 flat, 1 long
        self.last_position = 0
        self.entry_price = 0
        self.cash = 10000
        self.equity = 10000

        self.action_space = spaces.Discrete(3)  # 0 = flat, 1 = long, 2 = short

        # 13 features — must match your DataFrame
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self):
        self.current_step = 1
        self.position = 0
        self.last_position = 0
        self.entry_price = 0
        self.cash = 10000
        self.equity = 10000
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        done = False
        current_row = self.df.iloc[self.current_step]
        next_price = self.df.iloc[self.current_step + 1]["close"]
        current_price = current_row["close"]
        volatility = current_row["RSVol"]
        rsi = current_row["RSI"]
        bb_z = current_row["BB_zscore"]

        # Block long on overbought RSI
        if rsi > 80 and action == 1:
            action = 0  # force flat

        # === Position & Reward Logic ===
        new_position = {0: 0, 1: 1, 2: -1}[action]
        pnl = (next_price - current_price) * self.position
        turnover_penalty = 0.0001 * abs(new_position - self.last_position)
        risk_penalty = self.risk_penalty * volatility

        reward = (pnl - turnover_penalty - risk_penalty) * 100

        self.last_position = self.position
        self.position = new_position
        self.equity += pnl
        self.current_step += 1

        done = self.current_step >= self.n_steps - 2
        obs = self.df.iloc[self.current_step].values.astype(np.float32)

        info = {
            "equity": self.equity,
            "pnl": pnl,
            "volatility": volatility,
            "turnover_penalty": turnover_penalty
        }

        return obs, reward, done, info

