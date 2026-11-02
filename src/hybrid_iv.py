import os, duckdb, numpy as np, pandas as pd, lightgbm as lgb

CLEAN = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
FEATS = ["implied_volatility","iv_lag1","iv_change","iv_ma_5","iv_std_5",
         "mid","rel_spread","bid_size","ask_size","volume","open_interest",
         "delta","gamma","theta","vega","rho","strike","tte_days","moneyness",
         "spy_ret_1d","spy_rv_20d","type","in_the_money"]

R, Q = 0.04, 0.0

con = duckdb.connect(); con.execute("SET threads=4"); con.execute("SET memory_limit='3GB'")
model = lgb.Booster(model_file=os.path.join(CLEAN, "lgbm_iv.txt"))

sabr = pd.read_parquet(os.path.join(CLEAN, "sabr_estimates.parquet"),
    columns=["date","contract_id","expiration","strike","tte_days",
             "sabr_iv_raw"])
heston = pd.read_parquet(os.path.join(CLEAN, "heston_estimates.parquet"),
    columns=["date","contract_id","heston_iv"])

def get_ml(split, dates):
    feat = os.path.join(CLEAN, f"{split}_features.parquet").replace("\\","/")
    dt = ",".join(f"DATE '{d}'" for d in dates)
    q = f"""SELECT date, contract_id, target_iv, implied_volatility, spy_close,
                   {",".join(FEATS)}
            FROM '{feat}' WHERE date IN ({dt})"""
    df = con.execute(q).df()
    for c in FEATS:
        df[c] = df[c].astype("category") if c=="type" else pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["ml_iv"] = model.predict(df[FEATS], num_iteration=model.best_iteration).astype("float64")
    return df[["date","contract_id","target_iv","spy_close","ml_iv"]]

def build(split):
    dates = sorted(set(sabr["date"]).union(heston["date"]))
    ml = get_ml(split, dates)
    df = ml.merge(sabr, on=["date","contract_id"], how="inner")
    df = df.merge(heston, on=["date","contract_id"], how="inner")
    df = df.rename(columns={"spy_close_x":"spy_close"}) if "spy_close_x" in df.columns else df
    df = df.dropna(subset=["target_iv","ml_iv","sabr_iv_raw","heston_iv"])
    df["T"] = df["tte_days"]/365.0
    df["F"] = df["spy_close"] * np.exp((R-Q) * df["T"])

    # Per (date, expiration): find ATM, compute shift
    df["dist_atm"] = (df["strike"] - df["F"]).abs()
    idx_atm = df.groupby(["date","expiration"])["dist_atm"].idxmin()
    atm = df.loc[idx_atm, ["date","expiration","ml_iv","sabr_iv_raw","heston_iv"]].rename(
        columns={"ml_iv":"ml_atm","sabr_iv_raw":"sabr_atm","heston_iv":"heston_atm"})
    df = df.merge(atm, on=["date","expiration"], how="left")

    df["sabr_hybrid"]   = df["sabr_iv_raw"] + (df["ml_atm"] - df["sabr_atm"])
    df["heston_hybrid"] = df["heston_iv"]   + (df["ml_atm"] - df["heston_atm"])

    out = os.path.join(CLEAN, f"hybrid_{split}.parquet")
    df[["date","contract_id","expiration","strike","tte_days","target_iv",
        "ml_iv","sabr_iv_raw","sabr_hybrid","heston_iv","heston_hybrid"]].to_parquet(out, index=False)
    print(f"saved {out} {df.shape}")

    for col in ["ml_iv","sabr_iv_raw","sabr_hybrid","heston_iv","heston_hybrid"]:
        sub = df.dropna(subset=[col, "target_iv"])
        e = sub["target_iv"] - sub[col]
        print(f"{split:5s} {col:15s} n={len(sub):>8,}  MAE={np.mean(np.abs(e)):.5f}  RMSE={np.sqrt(np.mean(e**2)):.5f}")

for s in ["val","test"]:
    build(s)