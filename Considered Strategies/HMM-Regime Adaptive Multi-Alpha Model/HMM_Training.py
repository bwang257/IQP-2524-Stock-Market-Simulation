import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD

"""
Note that the following code was used to train and analyze the regime predictions of an HMM trained on the XLE ETF in a Jupyter Notebook file.
Summary statistics for the features were then used for backtest validation of the multi-alpha strategy. 
"""

def fetch_etf_data(ticker: str, start_date='2017-01-01', end_date='2025-6-1'):
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
    df['VIX'] = vix
    return df.dropna()

def engineer_features(df):

    df['gap_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    close_series = pd.Series(df['Close'].values, index=df.index)
    bb = BollingerBands(close=close_series, window=10)
    df['bb_width'] = bb.bollinger_wband()
    df['bb_pos'] = (close_series - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df['volume_change'] = df['Volume'].pct_change()
    df['macd_diff'] = MACD(close_series, 10, 5, 5).macd_diff()
    df['rsi'] = RSIIndicator(close_series, 7).rsi()
    df['adx'] = ADXIndicator(df['High'], df['Low'], close_series, 7).adx()

    return df.dropna()


def detect_regimes(df, features, n_regimes=4):

    X = StandardScaler().fit_transform(df[features])
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=200, random_state=42)
    model.fit(X)
    df['regime'] = model.predict(X)

    return df, X

def plot_time_series(df, ticker):
    plt.figure(figsize=(14, 5))
    for r in sorted(df['regime'].unique()):
        mask = df['regime'] == r
        plt.plot(df.index[mask], df['Close'][mask], '.', label=f"Regime {r}", alpha=0.5)
    plt.title(f"{ticker} Regimes Based on Indicators (HMM)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()


def plot_tsne(df, X, ticker):
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    plt.figure(figsize=(10, 7))
    for r in sorted(df['regime'].unique()):
        mask = df['regime'] == r
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=f"Regime {r}", alpha=0.6)
    plt.title(f"t-SNE Regime Visualization ({ticker})")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.legend()
    plt.grid()
    plt.show()

def analyze_etf(ticker):
    df = fetch_etf_data(ticker)
    df = engineer_features(df)
    features = ['gap_pct', 'bb_pos', 'volume_change', 'VIX', 'macd_diff', 'rsi', 'bb_width', 'adx']
    df, X = detect_regimes(df, features)
    plot_time_series(df, ticker)
    plot_tsne(df, X, ticker)
    return df


xle = analyze_etf("XLE")

features = ['gap_pct', 'bb_pos', 'volume_change', 'VIX', 'macd_diff', 'rsi', 'bb_width', 'adx']
summary_df = xle.groupby('regime')[features].describe().round(4)
summary_df = summary_df.transpose().reset_index() # Reset index to make MultiIndex levels into columns
summary_df = summary_df.rename(columns={'level_0': 'Feature', 'level_1': 'Statistic'}) # Rename the new columns
summary_df.columns = ['Feature', 'Statistic'] + [f'Regime {c}' for c in summary_df.columns[2:]] # Rename regime columns and combine with new labels
summary_df
