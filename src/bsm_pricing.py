import os, numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
from scipy.stats import norm

CLEAN = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
FEATS = ["implied_volatility","iv_lag1","iv_change","iv_ma_5","iv_std_5",
         "mid","rel_spread","bid_size","ask_size","volume","open_interest",
         "delta","gamma","theta","vega","rho","strike","tte_days","moneyness",
         "spy_ret_1d","spy_rv_20d","type","in_the_money"]
R, Q = 0.04, 0.0
model = lgb.Booster(model_file=os.path.join(CLEAN, "lgbm_iv.txt"))

def bs_price(S, K, T, sigma, kind):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    out = np.where(kind == "call", np.maximum(S - K, 0.0), np.maximum(K - S, 0.0)).astype(float)
    ok = (T > 0) & (sigma > 0)
    if ok.any():
        d1 = (np.log(S[ok]/K[ok]) + (R - Q + 0.5*sigma[ok]**2)*T[ok])/(sigma[ok]*np.sqrt(T[ok]))
        d2 = d1 - sigma[ok]*np.sqrt(T[ok])
        c = S[ok]*np.exp(-Q*T[ok])*norm.cdf(d1) - K[ok]*np.exp(-R*T[ok])*norm.cdf(d2)
        p = K[ok]*np.exp(-R*T[ok])*norm.cdf(-d2) - S[ok]*np.exp(-Q*T[ok])*norm.cdf(-d1)
        out[ok] = np.where(kind[ok] == "call", c, p)
    return out

def eval_split(split, batch=200_000):
    f = os.path.join(CLEAN, f"{split}_features.parquet")
    pf = pq.ParquetFile(f)
    cols = FEATS + ["target_iv","mid","spy_close","tte_days","strike","type"]
    agg = {("call","ml"):[0.,0.,0], ("call","iv"):[0.,0.,0],
           ("put","ml"):[0.,0.,0],  ("put","iv"):[0.,0.,0]}
    for b in pf.iter_batches(batch_size=batch, columns=cols):
        df = b.to_pandas()
        df = df[df["target_iv"].notna() & df["mid"].notna() & (df["mid"] > 0)]
        if len(df) == 0: continue
        for c in FEATS:
            df[c] = df[c].astype("category") if c=="type" else pd.to_numeric(df[c], errors="coerce").astype("float32")
        df["ml_iv"] = model.predict(df[FEATS], num_iteration=model.best_iteration).astype("float64")
        df["T"]     = (df["tte_days"]/365.0).clip(lower=0)
        df["kind"]  = df["type"].astype(str).str.lower().str[0].map({"c":"call","p":"put"})
        for kind in ["call","put"]:
            m = (df["kind"] == kind).values
            if m.sum() == 0: continue
            s = df[m]
            p_ml = bs_price(s["spy_close"], s["strike"], s["T"], s["ml_iv"],       np.array([kind]*len(s)))
            p_iv = bs_price(s["spy_close"], s["strike"], s["T"], s["implied_volatility"], np.array([kind]*len(s)))
            for tag, p in [("ml", p_ml), ("iv", p_iv)]:
                e = p - s["mid"].values
                a = agg[(kind,tag)]
                a[0] += np.abs(e).sum(); a[1] += (e**2).sum(); a[2] += len(e)
    rows = []
    for (kind,tag), a in agg.items():
        n = a[2]
        if n == 0: continue
        rows.append({"split":split,"type":kind,"source":tag,"n":int(n),
                     "price_mae":a[0]/n,"price_rmse":np.sqrt(a[1]/n)})
    return pd.DataFrame(rows)

out = pd.concat([eval_split(s) for s in ["val","test"]], ignore_index=True)
pd.set_option("display.width", 200)
print(out.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
out.to_csv("audit_out/bsm_pricing_eval.csv", index=False)