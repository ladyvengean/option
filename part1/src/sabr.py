# src/sabr.py
import numpy as np
from scipy.optimize import least_squares

BETA = 0.5  # common default

def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """Hagan 2002 lognormal SABR implied vol (approximation)."""
    K = np.asarray(K, dtype=float)
    if T <= 0 or alpha <= 0 or nu <= 0:
        return np.full_like(K, np.nan)
    eps = 1e-8
    K = np.maximum(K, eps)
    FK = F * K
    logFK = np.log(F / K)
    FK_b = FK ** ((1 - beta) / 2.0)
    z = (nu / alpha) * FK_b * logFK
    # handle ATM z=0
    z = np.where(np.abs(z) < 1e-8, 1e-8, z)
    x = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho))
    A = alpha / (FK_b * (1 + ((1-beta)**2/24)*logFK**2 + ((1-beta)**4/1920)*logFK**4))
    B = 1 + (((1-beta)**2/24)*alpha**2/FK**(1-beta)
             + 0.25*rho*beta*nu*alpha/FK_b
             + ((2-3*rho**2)/24)*nu**2) * T
    return A * (z / x) * B

def calibrate_smile(F, K, iv, T, beta=BETA):
    """Fit alpha, rho, nu to observed IVs at strikes K."""
    K = np.asarray(K, float); iv = np.asarray(iv, float)
    ok = np.isfinite(iv) & (iv > 0) & (iv < 5) & (K > 0)
    K, iv = K[ok], iv[ok]
    if len(K) < 4:
        return None
    atm_iv = np.interp(F, K, iv)
    x0 = np.array([atm_iv, 0.0, 0.3])
    bounds = ([1e-3, -0.999, 1e-3], [5.0, 0.999, 5.0])
    def resid(p):
        return sabr_iv(F, K, T, p[0], beta, p[1], p[2]) - iv
    try:
        r = least_squares(resid, x0, bounds=bounds, max_nfev=200)
    except Exception:
        return None
    if not r.success:
        return None
    return {"alpha": r.x[0], "rho": r.x[1], "nu": r.x[2],
            "rmse": float(np.sqrt(np.mean(r.fun**2))), "n": len(K)}