# region imports
from AlgorithmImports import *
# endregion
class MyRollingDict:
    def __init__(self, algorithm, maxlen):
        self.algorithm = algorithm
        self.rolling_dict = OrderedDict()
        self.maxlen = maxlen

    def append(self, date_ordinal, characteristic, value):
        if date_ordinal in self.rolling_dict:
            self.rolling_dict.move_to_end(date_ordinal)  # Update ordering
        
        if date_ordinal not in self.rolling_dict:
            self.rolling_dict[date_ordinal] = {}

        self.rolling_dict[date_ordinal][characteristic] = value
        
        if len(self.rolling_dict) > self.maxlen:
            self.rolling_dict.popitem(last=False)  # Remove oldest entry

    def get_previous_close(self, date_ordinal):

        dates = sorted(self.rolling_dict.keys())

        try: 
            idx = dates.index(date_ordinal)
        except ValueError:
            self.algorithm.debug("Current date ordinal in get previous close method not found")
            return np.nan

        target_index = idx - 1
        if target_index < 0:
            return 0

        target_day = dates[target_index]
        previous_close = self.rolling_dict.get(target_day, {}).get("Close", np.nan)
        if np.isnan(previous_close):
            self.algorithm.debug("Close not found when it should have been")

        return previous_close

    def calculate_momentum(self, date_ordinal, lookback_start_days, lookback_end_days):

        if lookback_end_days >= lookback_start_days:
            return np.nan

        dates = sorted(self.rolling_dict.keys())

        try: 
            idx = dates.index(date_ordinal)
        except ValueError:
            self.algorithm.debug("Current day not found in calculate momentum calculation")
            return np.nan

        start_idx = idx - lookback_start_days
        end_idx = idx - lookback_end_days

        if start_idx < 0 or end_idx < 0:
            return np.nan
        
        start_day = dates[start_idx]
        end_day = dates[end_idx]

        close_start = self.rolling_dict.get(start_day, {}).get("Close", np.nan)
        close_end = self.rolling_dict.get(end_day, {}).get("Close", np.nan)

        if np.isnan(close_start) or np.isnan(close_end) or close_start <= 0 or close_end <=0:
            return np.nan
        return (close_end / close_start) - 1



    def calculate_AVT_SUV(self, date_ordinal, period = 21):
        
        dates = sorted(self.rolling_dict.keys())
        
        try:
            current_idx = dates.index(date_ordinal)
        except ValueError:
            self.algorithm.debug("Current date not found in SUV AVT calculation")
            return np.nan, np.nan

        curr_volume = self.rolling_dict.get(date_ordinal, {}).get("Volume", np.nan)
        curr_shares_outstanding = self.rolling_dict.get(date_ordinal, {}).get("SharesOutstanding", np.nan)
        
        if current_idx < period - 1:
            return np.nan, np.nan
        
        start_idx = max(0, current_idx - period + 1)
        end_idx = current_idx + 1  # +1 because we want to include current date
        
        volumes = []
        for i in range(start_idx, end_idx):
            date = dates[i]
            volume = self.rolling_dict.get(date, {}).get("Volume", np.nan)
            if not np.isnan(volume) and volume > 0:
                volumes.append(volume)
        
        if len(volumes) < period * 0.7:  # Need at least 70% of data points
            return np.nan, np.nan
        
        av_volume = np.mean(volumes)
        std_volume = np.std(volumes, ddof=1) if len(volumes) > 1 else 0
        
        # Calculate AVT
        avt = np.nan
        if np.isnan(curr_shares_outstanding):
            self.algorithm.debug("Current Shares Outstanding not defined")

        if not np.isnan(curr_shares_outstanding) and curr_shares_outstanding > 0 and av_volume > 0:
            avt = av_volume / curr_shares_outstanding
        
        # Calculate SUV
        suv = np.nan
        if not np.isnan(curr_volume) and std_volume > 0:
            suv = (curr_volume - av_volume) / std_volume

        return avt, suv
    
    def calculate_w52h(self, date_ordinal, period =252):
        
        dates = sorted(self.rolling_dict.keys())

        try:
            current_idx = dates.index(date_ordinal)
        except ValueError:
            return np.nan
        
        curr_close = self.rolling_dict.get(date_ordinal, {}).get("Close", np.nan)
        
        if np.isnan(curr_close) or current_idx < period - 1:
            return np.nan
        

        start_idx = max(0, current_idx - period + 1)
        end_idx = current_idx + 1
        
        
        highs = []
        for i in range(start_idx, end_idx):
            date = dates[i]
            high = self.rolling_dict.get(date, {}).get("High", np.nan)
            if not np.isnan(high) and high > 0:
                highs.append(high)
        
        if len(highs) < period * 0.7:  # Need at least 70% of data points
            return np.nan
        
        max_high = np.max(highs)
        if max_high > 0:
            return curr_close / max_high

        return np.nan

    def calculate_realized_volatility(self, date_ordinal, period = 20):

        dates = sorted(self.rolling_dict.keys())
        
        try:
            current_idx = dates.index(date_ordinal)
        except ValueError:
            return np.nan
        
        if current_idx < period - 1:
            return np.nan
        
        # Get returns
        start_idx = max(0, current_idx - period + 1)
        end_idx = current_idx + 1
        
        returns = []
        for i in range(start_idx, end_idx):
            date = dates[i]
            daily_return = self.rolling_dict.get(date, {}).get("Return", np.nan)
            if not np.isnan(daily_return):
                returns.append(daily_return)
        
        if len(returns) < period * 0.75:  # Need at least 75% of data points
            return np.nan
        
        if len(returns) >= 2:  # Need at least 2 points for std calculation
            std_returns = np.std(returns, ddof=1)
            # Annualize the volatility
            return std_returns * np.sqrt(252)
        
        return np.nan

