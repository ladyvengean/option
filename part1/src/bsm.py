# src/bsm.py
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

R_DEFAULT = 0.04     # risk-free rate (assumption)
Q_DEFAULT = 0.013    # dividend yield (assumption)

def bsm_price(S, K, T, sigma, r=R_DEFAULT, q=Q_DEFAULT, kind="call"):
    """European BSM price. SPY is American; this is an approximation (documented)."""
    S, K, T, sigma = map(float, (S, K, T, sigma))
    if T <= 0 or sigma <= 0 or K <= 0 or S <= 0:
        # intrinsic at/after expiry
        if kind == "call": return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if kind == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bsm_delta(S, K, T, sigma, r=R_DEFAULT, q=Q_DEFAULT, kind="call"):
    S, K, T, sigma = map(float, (S, K, T, sigma))
    if T <= 0 or sigma <= 0: 
        return float(S > K) if kind == "call" else -float(S < K)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    if kind == "call":
        return np.exp(-q*T)*norm.cdf(d1)
    return -np.exp(-q*T)*norm.cdf(-d1)

def bsm_implied_vol(price, S, K, T, r=R_DEFAULT, q=R_DEFAULT, kind="call"):
    """Invert BSM to IV (used for sanity checks)."""
    intrinsic = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    if price <= intrinsic + 1e-8 or T <= 0:
        return np.nan
    try:
        return brentq(lambda s: bsm_price(S, K, T, s, r, q, kind) - price,
                      1e-4, 5.0, maxiter=100)
    except ValueError:
        return np.nan

def put_call_parity_check(S, K, T, sigma, r=R_DEFAULT, q=Q_DEFAULT):
    c = bsm_price(S, K, T, sigma, r, q, "call")
    p = bsm_price(S, K, T, sigma, r, q, "put")
    lhs = c - p
    rhs = S*np.exp(-q*T) - K*np.exp(-r*T)
    return abs(lhs - rhs) < 1e-8