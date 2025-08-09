#region imports
from AlgorithmImports import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from collections import deque
from matplotlib.lines import Line2D
from datetime import timedelta
#endregion

'''
    Much of this code is sourced at the following link: https://raposa.trade/blog/higher-highs-lower-lows-and-calculating-price-trends-in-python/
'''

def getHigherLows(data: np.array, order, K):
  '''
  Finds consecutive higher lows in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive lows need to be higher.
  '''
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are higher than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] < lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getLowerHighs(data: np.array, order=5, K=2):
  '''
  Finds consecutive lower highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be lower.
  '''
  # Get highs
  high_idx = argrelextrema(data, np.greater, order=order)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are lower than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] > highs[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getHigherHighs(data: np.array, order, K):
  '''
  Finds consecutive higher highs in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive highs need to be higher.
  '''
  # Get highs
  high_idx = argrelextrema(data, np.greater, order = order)[0]
  highs = data[high_idx]
  # Ensure consecutive highs are higher than previous highs
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(high_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if highs[i] < highs[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def getLowerLows(data: np.array, order, K):
  '''
  Finds consecutive lower lows in price pattern.
  Must not be exceeded within the number of periods indicated by the width 
  parameter for the value to be confirmed.
  K determines how many consecutive lows need to be lower.
  '''
  # Get lows
  low_idx = argrelextrema(data, np.less, order=order)[0]
  lows = data[low_idx]
  # Ensure consecutive lows are lower than previous lows
  extrema = []
  ex_deque = deque(maxlen=K)
  for i, idx in enumerate(low_idx):
    if i == 0:
      ex_deque.append(idx)
      continue
    if lows[i] > lows[i-1]:
      ex_deque.clear()

    ex_deque.append(idx)
    if len(ex_deque) == K:
      extrema.append(ex_deque.copy())

  return extrema

def fit_trendline(pattern_pairs, prices):

  if not pattern_pairs:
    return None, None
  last = pattern_pairs[-1]
  if len(last) < 2:
    return None, None
  i1, i2 = last[-2], last[-1]
  x = np.array([i1, i2], dtype=float)
  y = np.array([prices[i1], prices[i2]], dtype=float)
  m, b = np.polyfit(x, y, 1)
  return m, b


def get_trend(close_data, order, K, osc_data, break_tolerance: float = 0.002):

    #Reverse rolling windows
    close_data = [x for x in close_data]
    close_data.reverse()
  
    # data set to dataframe empty
    data = pd.DataFrame()
    data['Close'] = close_data
    close = data['Close'].values

    hh = getHigherHighs(close, order, K)
    hl = getHigherLows(close, order, K)
    ll = getLowerLows(close, order, K)
    lh = getLowerHighs(close, order, K)

    # format for tuples inside patterns: [type, first price index, second price index, first price, second price]
    patterns = []
    for i1, i2 in hh: patterns.append(('hh', i1, i2, close[i1], close[i2]))
    for i1, i2 in hl: patterns.append(('hl', i1, i2, close[i1], close[i2]))
    for i1, i2 in ll: patterns.append(('ll', i1, i2, close[i1], close[i2]))
    for i1, i2 in lh: patterns.append(('lh', i1, i2, close[i1], close[i2]))

    # sort by the second date (newest first)
    patterns.sort(key=lambda x: x[2], reverse=True)

    # net momentum
    total_up = sum(p[4] - p[3] for p in patterns if p[0] in ('hh', 'hl'))
    total_down = sum(p[4] - p[3] for p in patterns if p[0] in ('ll', 'lh'))
    net_swing = (total_up + total_down) / abs(close_data[-1])

    # # last bullish swing (higher lows)
    # bull_swing = 0.0
    # hl_patterns = [p for p in patterns if p[0] == 'hl']
    # if len(hl_patterns) >= 2:
    #     _, _, _, close1, close2 = hl_patterns[1]
    #     bull_swing = (close2 - close1) / close1 if close1 != 0 else 0.0
      
    # # last bearish swing (higher highs)
    # bear_swing = 0.0
    # hh_patterns = [p for p in patterns if p[0] == 'hh']
    # if len(hh_patterns) >= 2:
    #     _, _, _, close1, close2 = hh_patterns[1]
    #     bear_swing = (close2 - close1) / close1 if close1 != 0 else 0.0


    m_up, b_up = fit_trendline(hl, close)
    m_down, b_down = fit_trendline(lh, close)
    t = len(close)-1
    curr_price = close[t]
    break_up = (m_up is not None and curr_price < (m_up*t + b_up)*(1 - break_tolerance))
    break_down = (m_down is not None and curr_price > (m_down*t + b_down)*(1 - break_tolerance))

    divergences = {
      'bear_pullback': False,
      'bear_reversal': False,
      'bull_pullback': False,
      'bull_reversal': False
    }

    if osc_data is not None:
      osc_data = [x for x in osc_data]
      osc_data.reverse()
      # helper to grab last two indices
      def last_pair(pairs):
          return pairs[-1] if pairs and len(pairs[-1])>=2 else None

      # Bearish divergence on HH
      hh_pair = last_pair(hh)
      if hh_pair:
          i1, i2 = hh_pair[-2], hh_pair[-1]
          if close[i2] > close[i1] and osc_data[i2] < osc_data[i1]:
              if break_up:
                  divergences['bear_reversal'] = True
              else:
                  divergences['bear_pullback'] = True

      # Bullish divergence on LL
      ll_pair = last_pair(ll)
      if ll_pair:
          i1, i2 = ll_pair[-2], ll_pair[-1]
          if close[i2] < close[i1] and osc_data[i2] > osc_data[i1]:
              if break_down:
                  divergences['bull_reversal'] = True
              else:
                  divergences['bull_pullback'] = True
                  
    return {
        'net': net_swing,
        **divergences
    }




