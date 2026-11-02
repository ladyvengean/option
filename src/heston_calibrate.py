import os, time
import numpy as np
import pandas as pd
import duckdb
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss

CLEAN    = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
OUT      = os.path.join(CLEAN, "heston_estimates.parquet")
CKPT_DIR = os.path.join(CLEAN, "heston_ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)

STEP = 5          # set to 1 for every date
R    = 0.04
Q    = 0.0

_N, _w = leggauss(200)
UMAX = 200.0
U = 0.5 * UMAX * (_N + 1)
W = 0.5 * UMAX * _w

def heston_cf_j(u, S, T, v0, kappa, theta, xi, rho, j):
    uj, bj = (0.5, kappa - rho * xi) if j == 1 else (-0.5, kappa)
    rspi = rho * xi * u * 1j
    d = np.sqrt((rspi - bj) ** 2 - xi ** 2 * (2 * uj * u * 1j - u ** 2))
    g = (bj - rspi + d) / (bj - rspi - d)
    C = (R - Q) * u * 1j * T + (kappa * theta / xi ** 2) * (
        (bj - rspi + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = (bj - rspi + d) / xi ** 2 * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
    return np.exp(C + D * v0 + 1j * u * np.log(S))

def heston_calls(S, Ks, T, v0, kappa, theta, xi, rho):
    Ks = np.atleast_1d(Ks).astype(float)
    logK = np.log(Ks)
    P1 = np.empty(len(Ks)); P2 = np.empty(len(Ks))
    for j, arr in [(1, P1), (2, P2)]:
        phi = heston_cf_j(U, S, T, v0, kappa, theta, xi, rho, j)
        integ = np.real(np.exp(-1j * U[:, None] * logK[None, :]) * phi[:, None] / (1j * U[:, None]))
        arr[:] = 0.5 + (1.0 / np.pi) * (W @ integ)
    C = S * np.exp(-Q * T) * P1 - Ks * np.exp(-R * T) * P2
    intrinsic = np.maximum(S * np.exp(-Q * T) - Ks * np.exp(-R * T), 0.0)
    return np.maximum(C, intrinsic)

def bs_call(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-Q * T) - K * np.exp(-R * T), 0.0)
    d1 = (np.log(S / K) + (R - Q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-Q * T) * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)

def bs_vega(S, K, T, sigma):
    if T <= 0 or sigma <= 0:
        return 1e-8
    d1 = (np.log(S / K) + (R - Q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-Q * T) * norm.pdf(d1) * np.sqrt(T)

def invert_iv_from_call(price, S, K, T):
    intrinsic = max(S * np.exp(-Q * T) - K * np.exp(-R * T), 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-10:
        return np.nan
    try:
        f = lambda s: bs_call(S, K, T, s) - price
        if f(1e-4) > 0 or f(5.0) < 0:
            return np.nan
        return brentq(f, 1e-4, 5.0, xtol=1e-6, maxiter=100)
    except Exception:
        return np.nan

def calibrate(S, Ks, T, call_prices, ivs, weights):
    v0_0 = max(np.median(ivs) ** 2, 0.005)
    x0 = np.array([v0_0, 2.0, v0_0, 0.5, -0.5])
    lb = [1e-6, 0.01, 1e-6, 0.01, -0.99]
    ub = [4.0, 20.0, 4.0, 5.0, 0.99]
    def resid(p):
        v0, kappa, theta, xi, rho = p
        if v0 <= 0 or kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) >= 1:
            return np.full(len(Ks), 1e3)
        try:
            C = heston_calls(S, Ks, T, v0, kappa, theta, xi, rho)
        except Exception:
            return np.full(len(Ks), 1e3)
        return (C - call_prices) / weights
    try:
        res = least_squares(resid, x0, bounds=(lb, ub), method="trf",
                            max_nfev=250, ftol=1e-7, xtol=1e-7)
        return res.x if (np.all(np.isfinite(res.x)) and res.cost < 5.0) else None
    except Exception:
        return None

def process_date(date, df_day):
    out = []
    S_series = df_day["spy_close"].dropna()
    if len(S_series) == 0:
        return out
    S = float(S_series.iloc[0])

    for exp, grp in df_day.groupby("expiration"):
        grp = grp.dropna(subset=["strike", "tte_days", "implied_volatility", "mid"])
        if len(grp) < 6:
            continue
        T = float(grp["tte_days"].iloc[0]) / 365.0
        if T <= 0:
            continue
        F = S * np.exp((R - Q) * T)
        is_call = grp["type"].astype(str).str.lower() == "call"
        otm = grp[((grp["strike"] >= F) & is_call) | ((grp["strike"] < F) & ~is_call)]
        if len(otm) < 6:
            continue
        Ks = otm["strike"].values.astype(float)
        mkt = otm["mid"].values.astype(float)
        otm_call = otm["type"].astype(str).str.lower() == "call"
        call_eq = np.where(otm_call, mkt, mkt + S * np.exp(-Q * T) - Ks * np.exp(-R * T))
        ivs = otm["implied_volatility"].values.astype(float)
        w = np.array([max(bs_vega(S, k, T, iv), 1e-3) for k, iv in zip(Ks, ivs)])
        mask = np.isfinite(call_eq) & (call_eq > 0) & np.isfinite(ivs) & (ivs > 0) & np.isfinite(w)
        if mask.sum() < 6:
            continue
        params = calibrate(S, Ks[mask], T, call_eq[mask], ivs[mask], w[mask])
        if params is None:
            continue
        v0, kappa, theta, xi, rho = params
        uniq_K = np.unique(grp["strike"].values.astype(float))
        C_all = heston_calls(S, uniq_K, T, v0, kappa, theta, xi, rho)
        iv_map = {float(k_): invert_iv_from_call(float(c_), S, float(k_), T)
                  for k_, c_ in zip(uniq_K, C_all)}
        for _, r_ in grp.iterrows():
            hv = iv_map.get(float(r_["strike"]), np.nan)
            if not np.isfinite(hv):
                continue
            out.append({
                "date": date, "contract_id": r_["contract_id"],
                "expiration": exp, "strike": float(r_["strike"]),
                "type": r_["type"], "tte_days": int(r_["tte_days"]),
                "moneyness": float(r_["moneyness"]), "spy_close": S,
                "market_iv": float(r_["implied_volatility"]),
                "heston_iv": float(hv),
                "v0": float(v0), "kappa": float(kappa), "theta": float(theta),
                "xi": float(xi), "rho": float(rho),
            })
    return out

def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")
    all_rows = []
    for split in ["val", "test"]:
        f = os.path.join(CLEAN, f"{split}_features.parquet").replace("\\", "/")
        dates = con.execute(f"SELECT DISTINCT date FROM '{f}' ORDER BY date").df()["date"].tolist()
        dates = dates[::STEP]
        for i, d in enumerate(dates):
            t0 = time.time()
            df_day = con.execute(f"SELECT * FROM '{f}' WHERE date = '{d}'").df()
            if len(df_day) < 100:
                continue
            rows = process_date(d, df_day)
            all_rows.extend(rows)
            print(f"{split} {d} rows={len(rows)} t={time.time()-t0:.1f}s")
            if (i + 1) % 20 == 0 and all_rows:
                pd.DataFrame(all_rows).to_parquet(
                    os.path.join(CKPT_DIR, f"ckpt_{split}_{i+1}.parquet"), index=False)
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_parquet(OUT, index=False)
        print("saved:", OUT, df.shape)

if __name__ == "__main__":
    main()