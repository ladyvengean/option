# src/sabr_eval_v2.py
import os, duckdb, numpy as np, pandas as pd, sys
# sys.path.append(os.path.dirname(__file__))
from part1.src.sabr import sabr_iv, calibrate_smile

CLEAN_DIR = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
SPY_FILE  = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\underlying_prices.parquet"
TEMP_DIR  = r"D:\temp\duckdb"
R, Q      = 0.04, 0.013
TTE_MAX   = 30

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute(f"SET temp_directory='{TEMP_DIR}'")
con.execute("SET threads=4")

spy = SPY_FILE.replace("\\","/")
srcs = [os.path.join(CLEAN_DIR,f"options_{y}_clean.parquet").replace("\\","/") for y in (2024,2025)]

# OTM-only + delta band + good quotes + proper forward
q = f"""
WITH o AS (
    SELECT o.contract_id, CAST(o.date AS DATE) d, CAST(o.expiration AS DATE) e,
           o.strike, o.type, o.implied_volatility iv, o.delta, o.bid, o.ask,
           CASE WHEN o.type='call' THEN o.delta ELSE -o.delta END AS abs_delta,
           CAST(o.expiration AS DATE) - CAST(o.date AS DATE) AS tte_days
    FROM read_parquet({srcs}) o
    WHERE implied_volatility BETWEEN 0.02 AND 2.0
      AND bid > 0 AND ask > bid
      AND CAST(o.expiration AS DATE) > CAST(o.date AS DATE)
      AND CAST(o.expiration AS DATE) - CAST(o.date AS DATE) BETWEEN 1 AND {TTE_MAX}
      AND EXTRACT(DAY FROM CAST(o.date AS DATE)) <= 10
      AND ABS(o.delta) BETWEEN 0.05 AND 0.95
)
SELECT o.d date, o.e expiration, o.strike, o.type, o.iv,
       s.close spy_close, o.tte_days,
       s.close * EXP(({R} - {Q}) * (o.tte_days/365.0)) AS F
FROM o LEFT JOIN '{spy}' s ON o.d = CAST(s.date AS DATE)
WHERE s.close IS NOT NULL
  -- OTM-only: calls above forward, puts below forward
  AND ((o.type='call' AND o.strike >= s.close * EXP(({R}-{Q})*(o.tte_days/365.0)))
    OR (o.type='put'  AND o.strike <= s.close * EXP(({R}-{Q})*(o.tte_days/365.0))))
"""
df = con.execute(q).df()
print("rows:", len(df), "| days:", df["date"].nunique())

rows = []
for (d, e), g in df.groupby(["date","expiration"], sort=False):
    if len(g) < 8: continue
    F = float(g["F"].iloc[0])
    T = float(g["tte_days"].iloc[0]) / 365.0
    fit = calibrate_smile(F, g["strike"].values, g["iv"].values, T)
    if fit is None: continue
    fitted = sabr_iv(F, g["strike"].values, T, fit["alpha"], 0.5, fit["rho"], fit["nu"])
    rows.append({"date": d, "expiration": e, "n": fit["n"],
                 "rmse": fit["rmse"], "alpha": fit["alpha"],
                 "rho": fit["rho"], "nu": fit["nu"],
                 "max_err": float(np.nanmax(np.abs(fitted - g["iv"].values)))})

res = pd.DataFrame(rows)
print("\ncalibrated smiles:", len(res))
if len(res):
    print(res[["rmse","max_err","n"]].describe().round(5).to_string())
    print("\nworst 10 fits:")
    print(res.nlargest(10,"rmse")[["date","expiration","n","rmse","max_err"]].to_string(index=False))
    res.to_csv("audit_out/sabr_fit_v2.csv", index=False)
    print("saved: audit_out/sabr_fit_v2.csv")