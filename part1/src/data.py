import os, functools, duckdb, pandas as pd, lightgbm as lgb

CLEAN_DIR = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
SPY_FILE  = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\underlying_prices.parquet"
MODEL_PATH = os.path.join(CLEAN_DIR, "lgbm_iv.txt")

FEATS = ["implied_volatility","iv_lag1","iv_change","iv_ma_5","iv_std_5",
         "mid","rel_spread","bid_size","ask_size","volume","open_interest",
         "delta","gamma","theta","vega","rho","strike","tte_days","moneyness",
         "spy_ret_1d","spy_rv_20d","type","in_the_money"]

@functools.lru_cache(maxsize=1)
def get_model(): return lgb.Booster(model_file=MODEL_PATH)

@functools.lru_cache(maxsize=1)
def get_con():
    c = duckdb.connect()
    c.execute("SET memory_limit='1GB'")
    return c

def parquet_path(split): 
    return os.path.join(CLEAN_DIR, f"{split}_features.parquet").replace("\\","/")

def list_contracts(split="test", limit=500):
    return get_con().execute(f"""
        SELECT contract_id, ANY_VALUE(type) type, ANY_VALUE(strike) strike,
               ANY_VALUE(expiration) expiration, MIN(date) d0, MAX(date) d1
        FROM '{parquet_path(split)}'
        GROUP BY contract_id LIMIT {limit}
    """).df()

def contract_history(contract_id, split="test"):
    cid = contract_id.replace("'","''")
    return get_con().execute(f"""
        SELECT date, implied_volatility, target_iv, mid, strike, type, tte_days
        FROM '{parquet_path(split)}'
        WHERE contract_id = '{cid}' ORDER BY date
    """).df()

def predict_next_iv(contract_id, split="test"):
    cid = contract_id.replace("'","''")
    df = get_con().execute(f"""
        SELECT * FROM '{parquet_path(split)}'
        WHERE contract_id = '{cid}' ORDER BY date DESC LIMIT 1
    """).df()
    if df.empty: return None
    for c in FEATS:
        df[c] = df[c].astype("category") if c=="type" else pd.to_numeric(df[c], errors="coerce").astype("float32")
    pred = float(get_model().predict(df[FEATS], num_iteration=get_model().best_iteration)[0])
    return {"row": df.iloc[0], "pred_iv": pred}

@functools.lru_cache(maxsize=1)
def metrics_df():
    return pd.DataFrame([
        {"model":"Persistence","split":"val","MAE":0.05441,"RMSE":0.21467},
        {"model":"Persistence","split":"test","MAE":0.06346,"RMSE":0.24341},
        {"model":"ML (LightGBM)","split":"val","MAE":0.04844,"RMSE":0.17103},
        {"model":"ML (LightGBM)","split":"test","MAE":0.05790,"RMSE":0.20105},
        {"model":"SABR (median fit)","split":"2024-25","MAE":0.0040,"RMSE":0.0040},
        {"model":"Heston (median fit)","split":"2024-25","MAE":0.0079,"RMSE":0.00995},
    ])