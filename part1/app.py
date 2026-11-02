# import streamlit as st
# import sys, os
# sys.path.append(os.path.dirname(__file__))
# from src.data import metrics_df

# st.set_page_config(page_title="SPY IV Project", layout="wide")
# st.title("SPY Implied Volatility Forecasting")
# st.caption("ML forecast → BSM pricing → SABR / Heston calibration")

# c1, c2, c3 = st.columns(3)
# c1.metric("Train", "2008–2022", "17.8M rows")
# c2.metric("Validation", "2023–2024", "4.1M rows")
# c3.metric("Test", "2025", "2.2M rows")

# st.subheader("Model comparison (out-of-sample)")
# st.dataframe(metrics_df(), use_container_width=True)

# st.info("Use the sidebar to open IV Forecast, BSM Pricing, SABR Smile,str Heston, and Evaluation pages.")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import duckdb
import os, sys, functools
import lightgbm as lgb
from scipy.stats import norm
from scipy.optimize import brentq

# Ensure src modules are importable
sys.path.append(os.path.dirname(__file__))
from part1.src.bsm import bsm_price
from part1.src.sabr import sabr_iv
from part1.src.heston import heston_price

st.set_page_config(page_title="SPY IV Project", layout="wide")
R, Q = 0.04, 0.013

CLEAN_DIR = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\clean"
SPY_FILE  = r"D:\Downloads\options-dataset-hist-main\options-dataset-hist-main\spy\underlying_prices.parquet"
MODEL_PATH = os.path.join(CLEAN_DIR, "lgbm_iv.txt")

FEATS = ["implied_volatility","iv_lag1","iv_change","iv_ma_5","iv_std_5",
         "mid","rel_spread","bid_size","ask_size","volume","open_interest",
         "delta","gamma","theta","vega","rho","strike","tte_days","moneyness",
         "spy_ret_1d","spy_rv_20d","type","in_the_money"]

# ---------- HELPERS ----------
@st.cache_resource
def get_model(): return lgb.Booster(model_file=MODEL_PATH)

@st.cache_resource
def get_con():
    c = duckdb.connect()
    c.execute("SET memory_limit='1GB'")
    return c

def parquet_path(split): return os.path.join(CLEAN_DIR, f"{split}_features.parquet").replace("\\","/")

@st.cache_data
def list_contracts(split="test", limit=500):
    return get_con().execute(f"""
        SELECT contract_id, ANY_VALUE(type) type, ANY_VALUE(strike) strike,
               ANY_VALUE(expiration) expiration, MIN(date) d0, MAX(date) d1
        FROM '{parquet_path(split)}'
        GROUP BY contract_id LIMIT {limit}
    """).df()

@st.cache_data
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

@st.cache_data
def load_csv_safe(path):
    if os.path.exists(path): return pd.read_csv(path)
    return pd.DataFrame()

# ---------- SIDEBAR NAVIGATION ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "IV Forecast", "BSM Pricing", "SABR Smile", "Heston Pricing", "Evaluation"])

# ---------- PAGES ----------

if page == "Home":
    st.title("SPY Implied Volatility Forecasting")
    st.caption("ML forecast → BSM pricing → SABR / Heston calibration")

    c1, c2, c3 = st.columns(3)
    c1.metric("Train", "2008–2022", "17.8M rows")
    c2.metric("Validation", "2023–2024", "4.1M rows")
    c3.metric("Test", "2025", "2.2M rows")

    st.subheader("Model comparison (out-of-sample)")
    metrics = pd.DataFrame([
        {"model":"Persistence","split":"val","MAE":0.05441,"RMSE":0.21467},
        {"model":"Persistence","split":"test","MAE":0.06346,"RMSE":0.24341},
        {"model":"ML (LightGBM)","split":"val","MAE":0.04844,"RMSE":0.17103},
        {"model":"ML (LightGBM)","split":"test","MAE":0.05790,"RMSE":0.20105},
        {"model":"SABR (median fit)","split":"2024-25","MAE":0.0040,"RMSE":0.0040},
        {"model":"Heston (median fit)","split":"2024-25","MAE":0.0079,"RMSE":0.00995},
    ])
    st.dataframe(metrics, use_container_width=True)

elif page == "IV Forecast":
    st.header("IV Forecast")
    split = st.sidebar.selectbox("Dataset Split", ["test", "val"])
    contracts_df = list_contracts(split)
    cid = st.selectbox("Select Contract", contracts_df["contract_id"].tolist())

    if cid:
        res = predict_next_iv(cid, split)
        hist = contract_history(cid, split)
        if res and not hist.empty:
            curr_iv = res['row']['implied_volatility']
            pred_iv = res['pred_iv']
            c1, c2, c3 = st.columns(3)
            c1.metric("Current IV", f"{curr_iv:.4f}")
            c2.metric("Predicted Next IV", f"{pred_iv:.4f}", delta=f"{pred_iv - curr_iv:.4f}")
            if pd.notna(res['row']['target_iv']):
                c3.metric("Actual Next IV", f"{res['row']['target_iv']:.4f}")
            
            st.subheader("Historical vs Target IV")
            fig = px.line(hist, x="date", y=["implied_volatility", "target_iv"], markers=True)
            st.plotly_chart(fig, use_container_width=True)

elif page == "BSM Pricing":
    st.header("BSM Pricing (ML IV)")
    split = st.sidebar.selectbox("Dataset Split", ["test", "val"], key="bsm_split")
    contracts_df = list_contracts(split)
    cid = st.selectbox("Select Contract", contracts_df["contract_id"].tolist(), key="bsm_cid")

    if cid:
        res = predict_next_iv(cid, split)
        if res:
            row = res['row']
            pred_iv = res['pred_iv']
            S, K = float(row['spy_close']), float(row['strike'])
            T = max(float(row['tte_days']) / 365.0, 1/365.0)
            kind, mid = str(row['type']), float(row['mid'])
            
            price_ml = bsm_price(S, K, T, pred_iv, r=R, q=Q, kind=kind)
            price_curr = bsm_price(S, K, T, float(row['implied_volatility']), r=R, q=Q, kind=kind)
            
            st.subheader("Pricing Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Market Mid Price", f"${mid:.2f}")
            c2.metric("BSM Price (ML IV)", f"${price_ml:.2f}", delta=f"{price_ml - mid:.2f}")
            c3.metric("BSM Price (Current IV)", f"${price_curr:.2f}")
            
            st.markdown("---")
            st.subheader("Inputs & Greeks")
            st.dataframe(pd.DataFrame([{
                "Spot (S)": S, "Strike (K)": K, "Days to Exp": row['tte_days'],
                "Type": kind, "Predicted IV": pred_iv, "Delta": row['delta'], 
                "Gamma": row['gamma'], "Theta": row['theta'], "Vega": row['vega']
            }]), use_container_width=True)

elif page == "SABR Smile":
    st.header("SABR Smile Reconstruction")
    fits = load_csv_safe("audit_out/sabr_fit_v2.csv")
    if not fits.empty:
        fits["tte"] = (pd.to_datetime(fits["expiration"]) - pd.to_datetime(fits["date"])).dt.days
        valid_dates = fits[fits["tte"] >= 7]["date"].unique()
        
        if len(valid_dates) > 0:
            sel_date = st.selectbox("Select Date", sorted(valid_dates, reverse=True))
            day_fits = fits[(fits["date"] == sel_date) & (fits["tte"] >= 7)]
            expiry = st.selectbox("Select Expiration (TTE >= 7d)", day_fits["expiration"].unique())
            fit = day_fits[day_fits["expiration"] == expiry].iloc[0]
            
            src = os.path.join(CLEAN_DIR, f"options_{sel_date[:4]}_clean.parquet").replace("\\","/")
            q = f"""SELECT strike, implied_volatility as iv, type, delta FROM '{src}'
                    WHERE CAST(date AS DATE) = DATE '{sel_date}' AND CAST(expiration AS DATE) = DATE '{expiry}' 
                    AND implied_volatility BETWEEN 0.02 AND 2.0 AND bid > 0 AND ask > bid AND ABS(delta) BETWEEN 0.05 AND 0.95"""
            obs = get_con().execute(q).df()

            if not obs.empty:
                F = float(get_con().execute(f"SELECT close FROM '{SPY_FILE.replace('\\','/')}' WHERE CAST(date AS DATE) = DATE '{sel_date}'").fetchone()[0])
                T = float(fit["tte"]) / 365.0
                obs["SABR_IV"] = sabr_iv(F, obs["strike"].values, T, fit["alpha"], 0.5, fit["rho"], fit["nu"])
                obs = obs.sort_values("strike")
                
                st.metric("SABR RMSE (on clean strikes)", f"{fit['rmse']:.5f}")
                st.dataframe(pd.DataFrame([fit[["alpha","rho","nu","n"]]]), use_container_width=True)
                st.plotly_chart(px.scatter(obs, x="strike", y=["iv", "SABR_IV"], title=f"Clean Smile: {sel_date} to {expiry}"), use_container_width=True)
            else:
                st.warning("No clean strikes found for this expiry.")
        else:
            st.warning("No smiles with TTE >= 7 days found.")
    else:
        st.error("SABR fits CSV not found.")

elif page == "Heston Pricing":
    st.header("Heston Calibration & Pricing")
    fits = load_csv_safe("audit_out/heston_fit_v2.csv")
    if not fits.empty:
        fits["tte"] = (pd.to_datetime(fits["expiration"]) - pd.to_datetime(fits["date"])).dt.days
        sel_date = st.selectbox("Select Date", sorted(fits["date"].unique(), reverse=True))
        day_fits = fits[fits["date"] == sel_date]
        expiry = st.selectbox("Select Expiration", day_fits["expiration"].unique())
        fit = day_fits[day_fits["expiration"] == expiry].iloc[0]

        src = os.path.join(CLEAN_DIR, f"options_{sel_date[:4]}_clean.parquet").replace("\\","/")
        q = f"""SELECT strike, implied_volatility as iv, type FROM '{src}'
                WHERE CAST(date AS DATE) = DATE '{sel_date}' AND CAST(expiration AS DATE) = DATE '{expiry}' 
                AND implied_volatility BETWEEN 0.02 AND 2.0 AND bid > 0 AND ask > bid AND ABS(delta) BETWEEN 0.10 AND 0.90"""
        obs = get_con().execute(q).df()

        def iv_from_price(p, S, K, T, kind):
            if p <= 0 or T <= 0: return np.nan
            def f(s):
                d1=(np.log(S/K)+(R-Q+0.5*s*s)*T)/(s*np.sqrt(T)); d2=d1-s*np.sqrt(T)
                return (S*np.exp(-Q*T)*norm.cdf(d1)-K*np.exp(-R*T)*norm.cdf(d2)-p) if kind=="call" else (K*np.exp(-R*T)*norm.cdf(-d2)-S*np.exp(-Q*T)*norm.cdf(-d1)-p)
            try: return brentq(f, 1e-3, 5.0, maxiter=80)
            except Exception: return np.nan

        if not obs.empty:
            S = float(get_con().execute(f"SELECT close FROM '{SPY_FILE.replace('\\','/')}' WHERE CAST(date AS DATE) = DATE '{sel_date}'").fetchone()[0])
            T = float(fit["tte"]) / 365.0
            obs["Heston_IV"] = [iv_from_price(heston_price(S,k,T,R,Q,fit["v0"],fit["kappa"],fit["theta"],fit["sigma"],fit["rho"],t), S,k,T,t) for k,t in zip(obs["strike"], obs["type"])]
            obs = obs.sort_values("strike")
            
            st.metric("Heston Price RMSE", f"{fit['rmse_price']:.4f}")
            st.dataframe(pd.DataFrame([fit[["v0","kappa","theta","sigma","rho","n"]]]), use_container_width=True)
            st.plotly_chart(px.scatter(obs, x="strike", y=["iv", "Heston_IV"], title=f"Heston IV Curve: {sel_date} to {expiry}"), use_container_width=True)
        else:
            st.warning("No clean strikes found for this expiry.")
    else:
        st.error("Heston fits CSV not found.")

elif page == "Evaluation":
    st.header("Model Evaluation Dashboard")
    st.markdown("Compare the out-of-sample performance of all model branches.")
    
    st.subheader("1. IV Forecast Accuracy (LightGBM vs Baseline)")
    iv_data = pd.DataFrame([
        {"Split": "Val (2023-24)", "Persistence": 0.0544, "ML (LightGBM)": 0.0484},
        {"Split": "Test (2025)", "Persistence": 0.0635, "ML (LightGBM)": 0.0579},
    ])
    fig_iv = px.bar(iv_data.melt(id_vars="Split"), x="Split", y="value", color="variable", barmode="group", labels={"value": "MAE (Decimal IV)"}, title="Next-Day IV MAE (Lower is Better)")
    st.plotly_chart(fig_iv, use_container_width=True)

    st.subheader("2. Volatility Surface Fit (SABR vs Heston)")
    col1, col2 = st.columns(2)
    sabr = load_csv_safe("audit_out/sabr_fit_v2.csv")
    hest = load_csv_safe("audit_out/heston_iv_errors.csv")
    
    with col1:
        if not sabr.empty:
            st.metric("SABR Median IV RMSE", f"{sabr['rmse'].median():.5f}")
            st.caption(f"Evaluated on {len(sabr)} smiles")
    with col2:
        if not hest.empty:
            st.metric("Heston Median IV RMSE", f"{hest['iv_rmse'].median():.5f}")
            st.caption(f"Evaluated on {len(hest)} smiles")

    st.markdown("---")
    st.markdown("""
    ### Summary
    * **ML** improves next-day IV prediction by ~10% over persistence.
    * **BSM** gives a realistic theoretical price (~60% inside bid-ask spread on test data).
    * **SABR** fits the smile extremely tightly but requires per-expiry calibration.
    * **Heston** provides a robust cross-expiry IV curve with excellent fit but slower calibration.
    """)