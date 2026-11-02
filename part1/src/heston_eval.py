# src/heston_eval_v2.py
import os, duckdb, numpy as np, pandas as pd, sys
sys.path.append(os.path.dirname(__file__))

from part1.src.heston import calibrate

CLEAN_DIR = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
SPY_FILE  = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\underlying_prices.parquet"
TEMP_DIR  = r"D:\temp\duckdb"
R, Q      = 0.04, 0.013
TTE_MIN, TTE_MAX = 7, 21

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute(f"SET temp_directory='{TEMP_DIR}'")
con.execute("SET threads=4")
spy = SPY_FILE.replace("\\","/")
srcs = [os.path.join(CLEAN_DIR,f"options_{y}_clean.parquet").replace("\\","/") for y in (2024,2025)]

q = f"""
WITH o AS (
    SELECT CAST(o.date AS DATE) d, CAST(o.expiration AS DATE) e, o.strike, o.type,
           o.implied_volatility iv, o.delta,
           CAST(o.expiration AS DATE) - CAST(o.date AS DATE) AS tte_days
    FROM read_parquet({srcs}) o
    WHERE implied_volatility BETWEEN 0.02 AND 2.0
      AND bid > 0 AND ask > bid
      AND CAST(o.expiration AS DATE) > CAST(o.date AS DATE)
      AND CAST(o.expiration AS DATE) - CAST(o.date AS DATE) BETWEEN {TTE_MIN} AND {TTE_MAX}
      AND EXTRACT(DAY FROM CAST(o.date AS DATE)) <= 5
      AND ABS(o.delta) BETWEEN 0.10 AND 0.90
)
SELECT o.d date, o.e expiration, o.strike, o.type, o.iv,
       s.close spy_close, o.tte_days
FROM o LEFT JOIN '{spy}' s ON o.d = CAST(s.date AS DATE)
WHERE s.close IS NOT NULL
"""
df = con.execute(q).df()
print("rows:", len(df), "| days:", df["date"].nunique())

rows = []
for (d, e), g in df.groupby(["date","expiration"], sort=False):
    if len(g) < 8: continue
    S = float(g["spy_close"].iloc[0]); T = float(g["tte_days"].iloc[0])/365.0
    is_call = (g["type"].values == "call")
    fit = calibrate(S, g["strike"].values, g["iv"].values, T, R, Q, is_call)
    if fit is None: continue
    fit.update({"date":d, "expiration":e}); rows.append(fit)
    if len(rows) % 50 == 0: print(f"  calibrated {len(rows)}...")

res = pd.DataFrame(rows)
print("\ncalibrated:", len(res))
if len(res):
    print(res[["rmse_price","n"]].describe().round(5).to_string())
    print("\nworst 10:")
    print(res.nlargest(10,"rmse_price")[["date","expiration","n","rmse_price"]].to_string(index=False))
    res.to_csv("audit_out/heston_fit_v2.csv", index=False)
    print("saved: audit_out/heston_fit_v2.csv")