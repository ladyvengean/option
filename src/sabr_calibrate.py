import os, duckdb, numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.optimize import least_squares
import lightgbm as lgb

CLEAN = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
OUT   = os.path.join(CLEAN, "sabr_estimates.parquet")
MODEL = os.path.join(CLEAN, "lgbm_iv.txt")

FEATS = ["implied_volatility","iv_lag1","iv_change","iv_ma_5","iv_std_5",
         "mid","rel_spread","bid_size","ask_size","volume","open_interest",
         "delta","gamma","theta","vega","rho","strike","tte_days","moneyness",
         "spy_ret_1d","spy_rv_20d","type","in_the_money"]

BETA = 0.5
STEP = 5  # sample every 5th date for SABR calibration speed

def sabr_iv(F, K, T, alpha, rho, nu, beta=BETA):
    if T <= 0 or alpha <= 0 or nu <= 0 or abs(rho) >= 1:
        return np.nan
    FK = F * K
    if FK <= 0:
        return np.nan
    logFK = np.log(F / K)
    FK_beta = FK ** ((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * logFK
    if abs(z) < 1e-8:
        x_z = 1.0
    else:
        x_z = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho)) / z
    A = alpha / (FK_beta * (1 + ((1-beta)**2/24)*logFK**2 + ((1-beta)**4/1920)*logFK**4))
    B = 1 + (((1-beta)**2/24)*(alpha**2/(FK**(1-beta))) +
             (rho*beta*nu*alpha)/(4*FK_beta) +
             ((2-3*rho**2)/24)*nu**2) * T
    return A * x_z * B

def calibrate_sabr(F, Ks, Ts, ivs):
    # initial guesses
    x0 = [np.nanmedian(ivs), -0.3, 0.5]
    def resid(p):
        a, r, n = p
        model = np.array([sabr_iv(F, K, T, a, r, n) for K, T in zip(Ks, Ts)])
        return model - ivs
    try:
        res = least_squares(resid, x0, bounds=([1e-4, -0.99, 1e-4], [5.0, 0.99, 5.0]), max_nfev=200)
        return res.x
    except Exception:
        return None

def process_date(date, df_day, model):
    # df_day: all contracts for this date with features + target_iv etc.
    # forward per expiry: use spy_close and r,q
    out_rows = []
    for exp, grp in df_day.groupby("expiration"):
        if len(grp) < 6:
            continue
        # use OTM options for calibration to avoid ITM noise
        cal = grp[(grp["moneyness"] > 0.9) & (grp["moneyness"] < 1.1)]
        if len(cal) < 5:
            cal = grp
        F = cal["spy_close"].iloc[0] * np.exp((0.04 - cal.get("q", 0.0).iloc[0] if "q" in cal else 0.0) * (cal["tte_days"].iloc[0]/365.0))
        # Actually q not in features; use 0 for forward approx or join rates_div. For simplicity assume q=0 here, but better join.
        # We'll use forward = spy_close * exp(0.04*T) as approximation; q small.
        T = cal["tte_days"].iloc[0] / 365.0
        if T <= 0: continue
        F = cal["spy_close"].iloc[0] * np.exp(0.04 * T)
        Ks = cal["strike"].values
        ivs = cal["implied_volatility"].values
        Ts = np.full_like(Ks, T, dtype=float)
        # filter valid
        mask = np.isfinite(ivs) & (ivs > 0) & (ivs < 5)
        if mask.sum() < 5: continue
        params = calibrate_sabr(F, Ks[mask], Ts[mask], ivs[mask])
        if params is None: continue
        alpha, rho, nu = params
        # compute SABR IV for all contracts in this expiry
        for _, row in grp.iterrows():
            K = row["strike"]; T_row = row["tte_days"]/365.0
            if T_row <= 0: continue
            F_row = row["spy_close"] * np.exp(0.04 * T_row)
            sabr_iv_val = sabr_iv(F_row, K, T_row, alpha, rho, nu)
            if not np.isfinite(sabr_iv_val): continue
            out_rows.append({
                "date": date, "contract_id": row["contract_id"], "expiration": exp,
                "strike": K, "type": row["type"], "tte_days": row["tte_days"],
                "moneyness": row["moneyness"], "spy_close": row["spy_close"],
                "market_iv": row["implied_volatility"], "sabr_iv_raw": sabr_iv_val,
                "alpha": alpha, "rho": rho, "nu": nu,
            })
    if not out_rows:
        return pd.DataFrame()
    sabr_df = pd.DataFrame(out_rows)
    # ML predictions for these contracts
    X = df_day[df_day["contract_id"].isin(sabr_df["contract_id"])].copy()
    for c in FEATS:
        X[c] = X[c].astype("category") if c == "type" else pd.to_numeric(X[c], errors="coerce").astype("float32")
    ml_pred = model.predict(X[FEATS], num_iteration=model.best_iteration)
    X["ml_iv"] = ml_pred
    sabr_df = sabr_df.merge(X[["contract_id","ml_iv"]], on="contract_id", how="left")
    # ATM shift: for each expiration, find strike closest to forward (use spy_close as proxy)
    for exp, grp in sabr_df.groupby("expiration"):
        atm_idx = (grp["strike"] - grp["spy_close"]).abs().idxmin()
        atm_row = grp.loc[atm_idx]
        sabr_atm = atm_row["sabr_iv_raw"]
        ml_atm = atm_row["ml_iv"]
        if pd.notna(sabr_atm) and pd.notna(ml_atm) and sabr_atm > 0:
            scale = ml_atm / sabr_atm
            grp = grp.copy()
            grp["sabr_iv"] = grp["sabr_iv_raw"] * scale
            sabr_df.loc[grp.index, "sabr_iv"] = grp["sabr_iv"]
    return sabr_df

def main():
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=4")
    model = lgb.Booster(model_file=MODEL)
    all_out = []
    for split in ["val", "test"]:
        f = os.path.join(CLEAN, f"{split}_features.parquet")
        dates = con.execute(f"SELECT DISTINCT date FROM '{f}' ORDER BY date").df()["date"].tolist()
        dates = dates[::STEP]  # sample
        for d in dates:
            df_day = con.execute(f"SELECT * FROM '{f}' WHERE date = '{d}'").df()
            if len(df_day) < 100: continue
            res = process_date(d, df_day, model)
            if len(res): all_out.append(res)
            print(f"{split} {d} done, rows={len(res)}")
    if all_out:
        out = pd.concat(all_out, ignore_index=True)
        out.to_parquet(OUT, index=False)
        print("saved:", OUT, out.shape)

if __name__ == "__main__":
    main()