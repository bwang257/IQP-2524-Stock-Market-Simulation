# region imports
from AlgorithmImports import *
from scipy.signal import periodogram
# endregion


def hurst_lssd(ts, prior_mu=0.5, prior_sigma2=0.1, freq_max=None):
    """
    Estimate Hurst exponent via Bayesian log‑periodogram regression (LSSD framework).
    ts: array-like returns or prices
    prior_mu, prior_sigma2: prior mean and variance for H
    freq_max: upper frequency bound (Hz)
    Returns posterior mean Hurst
    """
    x = np.asarray(ts)
    N = len(x)
    if N < 10:
        return 0.5
    # Demean and compute periodogram
    y = x - x.mean()
    freqs, psd = periodogram(y, scaling='density')
    # Limit frequencies to positive non-zero
    valid = freqs > 0
    freqs = freqs[valid]
    psd = psd[valid]
    # Optionally cap high frequencies
    if freq_max is not None:
        mask = freqs <= freq_max
        freqs = freqs[mask]
        psd = psd[mask]
    # Log‑log regression: log(psd) ≈ intercept + slope*log(freq)
    X = np.log(freqs)
    Y = np.log(psd)
    slope, intercept = np.polyfit(X, Y, 1)
    # For fractional Gaussian noise, PSD ~ f^{-(2H-1)}, so slope ≈ -(2H-1)
    H_hat = (1 - slope) / 2
    # Estimate variance of slope
    resid = Y - (slope * X + intercept)
    var_slope = np.var(resid, ddof=1) / np.sum((X - X.mean())**2)
    # Map var_slope to var_H: var_H = var_slope / 4
    var_H = var_slope / 4
    # Bayesian update of H (Gaussian prior + Gaussian approx)
    prec0 = 1 / prior_sigma2
    prec1 = 1 / var_H
    mu_post = (prior_mu * prec0 + H_hat * prec1) / (prec0 + prec1)
    return float(mu_post)
