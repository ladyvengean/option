import os, gc, duckdb, numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import minimize

CLEAN = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
FEATS = ["implied_volatility","iv_lag1","iv_change","iv_ma_5","iv_std_5",
         "mid","rel_spread","bid_size","ask_size","volume","open_interest",
         "delta","gamma","theta","vega","rho","strike","tte_days","moneyness",
         "spy_ret_1d","spy_rv_20d","type","in_the_money"]

model = lgb.Booster(model_file=os.path.join(CLEAN, "lgbm_iv.txt"))
con = duckdb.connect(); con.execute("SET threads=4"); con.execute("SET memory_limit='3GB'")

def build_frame(split):
    sabr = pd.read_parquet(os.path.join(CLEAN, "sabr_estimates.parquet"),
                           columns=["date","contract_id","sabr_iv"])
    hest = pd.read_parquet(os.path.join(CLEAN, "heston_estimates.parquet"),
                           columns=["date","contract_id","heston_iv"])
    m = sabr.merge(hest, on=["date","contract_id"], how="inner")
    m["date"] = pd.to_datetime(m["date"]).dt.date

    feat = os.path.join(CLEAN, f"{split}_features.parquet").replace("\\","/")
    lo, hi = con.execute(f"SELECT MIN(date), MAX(date) FROM '{feat}'").fetchone()
    m = m[(m["date"] >= pd.Timestamp(lo).date()) & (m["date"] <= pd.Timestamp(hi).date())]

    key = os.path.join(CLEAN, f"_keys_{split}.parquet")
    m[["date","contract_id"]].to_parquet(key, index=False)

    sel = ", ".join("f."+c for c in FEATS)
    q = f"""SELECT f.date, f.contract_id, f.target_iv, f.implied_volatility, {sel}
            FROM '{feat}' f INNER JOIN '{key}' k
              ON f.date = k.date AND f.contract_id = k.contract_id
            WHERE f.target_iv IS NOT NULL"""
    df = con.execute(q).df()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.merge(m, on=["date","contract_id"], how="inner")

    for c in FEATS:
        df[c] = df[c].astype("category") if c=="type" else pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["ml_iv"]   = model.predict(df[FEATS], num_iteration=model.best_iteration).astype("float64")
    df["persist"] = df["implied_volatility"].astype("float64")
    for c in ["ml_iv","sabr_iv","heston_iv","target_iv","persist"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan).dropna(
        subset=["target_iv","ml_iv","sabr_iv","heston_iv","persist"])
    df = df[(df["sabr_iv"]>0)&(df["sabr_iv"]<5)&(df["heston_iv"]>0)&(df["heston_iv"]<5)]
    return df

def eval_all(df, w=None, tag=""):
    y = df["target_iv"].values
    base = {"persist": df["persist"].values,
            "ml":      df["ml_iv"].values,
            "sabr":    df["sabr_iv"].values,
            "heston":  df["heston_iv"].values}
    if w is not None:
        base["ensemble"] = w[0]*base["ml"]+w[1]*base["sabr"]+w[2]*base["heston"]
    for k,p in base.items():
        e = y - p
        print(f"{tag:5s} {k:10s} n={len(df):>8,}  MAE={np.mean(np.abs(e)):.5f}  RMSE={np.sqrt(np.mean(e**2)):.5f}")

print("building val...")
va = build_frame("val")
print("val rows:", len(va))

y = va["target_iv"].values
P = va[["ml_iv","sabr_iv","heston_iv"]].values

def loss(w):
    w = np.abs(w); w = w / w.sum()
    return float(np.mean(np.abs(y - P @ w)))

res = minimize(loss, x0=[0.5,0.3,0.2], method="Nelder-Mead",
               options={"xatol":1e-5,"fatol":1e-9,"maxiter":2000})
w_raw = np.abs(res.x); w = w_raw / w_raw.sum()
print("learned weights [ML, SABR, Heston]:", np.round(w,4))

eval_all(va, w, "val")
np.save(os.path.join(CLEAN, "ensemble_weights.npy"), w)

del va, P, y; gc.collect()

print("\nbuilding test...")
te = build_frame("test")
print("test rows:", len(te))
eval_all(te, w, "test")