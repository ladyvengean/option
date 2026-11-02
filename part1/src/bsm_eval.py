# src/bsm_eval.py
import os, numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from part1.src.bsm import bsm_price

CLEAN_DIR = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
MODEL     = os.path.join(CLEAN_DIR, "lgbm_iv.txt")
R, Q      = 0.04, 0.013
TTE_MAX   = 7
BATCH     = 200_000

FEATS = ["implied_volatility","iv_lag1","iv_change","iv_ma_5","iv_std_5",
         "mid","rel_spread","bid_size","ask_size","volume","open_interest",
         "delta","gamma","theta","vega","rho","strike","tte_days","moneyness",
         "spy_ret_1d","spy_rv_20d","type","in_the_money"]

model = lgb.Booster(model_file=MODEL)

def prep(df):
    for c in FEATS:
        df[c] = df[c].astype("category") if c == "type" else pd.to_numeric(df[c], errors="coerce").astype("float32")
    return df

def eval_split(name):
    f = os.path.join(CLEAN_DIR, f"{name}_features.parquet")
    pf = pq.ParquetFile(f)
    cols = FEATS + ["target_iv","spy_close","strike","tte_days","type","mid","bid","ask","expiration","date","contract_id"]

    abs_sum = sq_sum = 0.0
    n = 0
    inside = 0
    per_tte = {}

    for b in pf.iter_batches(batch_size=BATCH, columns=cols):
        df = prep(b.to_pandas())
        df = df[(df["target_iv"].notna()) & (df["tte_days"] >= 0) & (df["tte_days"] <= TTE_MAX)]
        if len(df) == 0: continue

        # predicted IV
        iv_hat = model.predict(df[FEATS], num_iteration=model.best_iteration).astype("float64")
        iv_hat = np.clip(iv_hat, 1e-3, 5.0)          # guard
        S  = df["spy_close"].astype("float64").values
        K  = df["strike"].astype("float64").values
        T  = (df["tte_days"].astype("float64").values + 1e-3) / 365.0   # avoid T=0
        T  = np.maximum(T, 1.0/365.0)
        kind = df["type"].astype(str).values

        # BSM theoretical prices (vectorised)
        d1 = (np.log(S/K) + (R - Q + 0.5*iv_hat**2)*T) / (iv_hat*np.sqrt(T))
        d2 = d1 - iv_hat*np.sqrt(T)
        from scipy.stats import norm
        call = S*np.exp(-Q*T)*norm.cdf(d1) - K*np.exp(-R*T)*norm.cdf(d2)
        put  = K*np.exp(-R*T)*norm.cdf(-d2) - S*np.exp(-Q*T)*norm.cdf(-d1)
        price_hat = np.where(kind == "call", call, put)

        y      = df["mid"].astype("float64").values
        bid    = df["bid"].astype("float64").values
        ask    = df["ask"].astype("float64").values
        ok     = np.isfinite(price_hat) & np.isfinite(y) & (y > 0)
        if ok.sum() == 0: continue

        err  = price_hat[ok] - y[ok]
        abs_sum += np.abs(err).sum()
        sq_sum  += (err**2).sum()
        n       += ok.sum()
        inside  += ((price_hat[ok] >= bid[ok]) & (price_hat[ok] <= ask[ok])).sum()

        # per TTE bucket
        tte_b = df["tte_days"].values[ok]
        for v, e in zip(tte_b, np.abs(err)):
            per_tte.setdefault(int(v), []).append(e)

    print(f"\n=== {name} (TTE 0-{TTE_MAX}) ===")
    print(f"n={n:,}  price MAE={abs_sum/n:.4f}  RMSE={np.sqrt(sq_sum/n):.4f}  inside_bidask={inside/n:.2%}")
    for k in sorted(per_tte):
        arr = np.array(per_tte[k])
        print(f"  TTE={k}: n={len(arr):>8,}  MAE={arr.mean():.4f}")

for s in ["val","test"]:
    eval_split(s)