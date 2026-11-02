# src/heston_fast.py
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

_GL_U, _GL_W = np.polynomial.legendre.leggauss(64)
_GL_U = 50 * (_GL_U + 1) / 2
_GL_W = 50 * _GL_W / 2

def _cf(u, S, T, r, q, v0, kappa, theta, sigma, rho):
    u = u.astype(complex); i = 1j
    d = np.sqrt((rho*sigma*i*u - kappa)**2 + sigma**2*(i*u + u**2))
    g = (kappa - rho*sigma*i*u - d) / (kappa - rho*sigma*i*u + d)
    C = (r-q)*i*u*T + (kappa*theta/sigma**2)*((kappa-rho*sigma*i*u-d)*T - 2*np.log((1-g*np.exp(-d*T))/(1-g)))
    D = (kappa-rho*sigma*i*u-d)/sigma**2 * ((1-np.exp(-d*T))/(1-g*np.exp(-d*T)))
    return np.exp(C + D*v0 + i*u*np.log(S))

def heston_price(S, K, T, r, q, v0, kappa, theta, sigma, rho, kind="call"):
    if T <= 0:
        return max(S-K,0) if kind=="call" else max(K-S,0)
    logK = np.log(K); u = _GL_U; w = _GL_W
    phi1 = _cf(u - 1j, S, T, r, q, v0, kappa, theta, sigma, rho)
    P1 = 0.5 + (1/np.pi)*np.sum(w*np.real(np.exp(-1j*u*logK)*phi1/(1j*u*S*np.exp((r-q)*T))))
    phi2 = _cf(u, S, T, r, q, v0, kappa, theta, sigma, rho)
    P2 = 0.5 + (1/np.pi)*np.sum(w*np.real(np.exp(-1j*u*logK)*phi2/(1j*u)))
    call = S*np.exp(-q*T)*P1 - K*np.exp(-r*T)*P2
    return call if kind=="call" else call - S*np.exp(-q*T) + K*np.exp(-r*T)

def _bs(S,K,T,iv,r,q):
    d1 = (np.log(S/K)+(r-q+0.5*iv**2)*T)/(iv*np.sqrt(T)); d2 = d1 - iv*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def calibrate(S, strikes, ivs, T, r, q, is_call_mask):
    if len(strikes) < 8: return None
    obs = np.array([_bs(S,k,T,iv,r,q) if m else _bs(S,k,T,iv,r,q)-S*np.exp(-q*T)+k*np.exp(-r*T)
                    for k,iv,m in zip(strikes,ivs,is_call_mask)])
    def resid(p):
        v0,kap,th,sig,rho = p
        if not (v0>0 and kap>0 and th>0 and sig>0 and -0.999<rho<0.999):
            return np.ones(len(strikes))*1e3
        try:
            mod = np.array([heston_price(S,k,T,r,q,v0,kap,th,sig,rho,"call" if m else "put")
                            for k,m in zip(strikes,is_call_mask)])
        except Exception:
            return np.ones(len(strikes))*1e3
        return mod - obs
    x0 = [0.04, 2.0, 0.04, 0.5, -0.7]
    bounds = ([1e-3,0.1,1e-3,1e-2,-0.95],[1.0,10.0,1.0,3.0,0.95])
    try:
        r_ = least_squares(resid, x0, bounds=bounds, max_nfev=80)
    except Exception:
        return None
    if not r_.success: return None
    return {**dict(zip(["v0","kappa","theta","sigma","rho"], r_.x)),
            "rmse_price": float(np.sqrt(np.mean(r_.fun**2))), "n": len(strikes)}