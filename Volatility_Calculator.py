import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math
import time

# --------------------------------------------
#  DARK MODE CSS OVERRIDE
# --------------------------------------------
st.set_page_config(page_title="Call Calendar Screener", layout="wide")
st.markdown(
    """
    <style>
    body, .css-18e3th9, .css-1gk4psh {
        background-color: #0E1117 !important;
        color: #FFFFFF !important;
    }
    .css-1offfwp {
        background-color: #0E1117 !important;
    }
    .css-1ex1afd tr, .css-1ex1afd td, .css-1ex1afd th {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------
#  UTILITY FUNCTIONS
# --------------------------------------------
def get_current_price(ticker_obj):
    """
    Returns today's close (or last available).
    """
    data = ticker_obj.history(period='1d')
    return data['Close'][0] if not data.empty else None

def pick_calendar_dates(expiration_strings):
    """
    1) Convert all expiration date strings to date objects and sort ascending.
    2) near_date: pick the earliest date < 30 days from now. 
       If none <30, pick the largest date >=30. 
    3) far_date: pick the next date in the list after near_date.
    Returns (near_date_str, far_date_str).
    
    If any step fails, raise ValueError.
    """
    today = datetime.today().date()
    exps = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in expiration_strings)
    
    # Calculate DTE for each
    dte_list = [(exp, (exp - today).days) for exp in exps if exp > today]
    if not dte_list:
        raise ValueError("No expirations in the future.")
    
    # Try to find the earliest date <30 days
    near_date = None
    candidates_ge_30 = [x for x in dte_list if x[1] < 30]
    if candidates_ge_30:
        # pick earliest among these
        near_date = candidates_ge_30[0][0]
    else:
        # all are < 30 days, pick the furthest one
        near_date = dte_list[-1][0]
    
    # now pick the next date strictly after near_date
    # find near_date index in exps
    idx = exps.index(near_date)
    if idx == len(exps) - 1:
        raise ValueError(f"No far expiration after near date={near_date}")
    
    far_date = exps[idx+1]
    
    return (near_date.strftime("%Y-%m-%d"), far_date.strftime("%Y-%m-%d"))

def compute_yang_zhang_volatility(history, window=30, trading_periods=252):
    """
    Yang-Zhang volatility (30-day window).
    """
    df = history.copy()
    if len(df) < window:
        return None
    
    log_ho = np.log(df['High'] / df['Open'])
    log_lo = np.log(df['Low']  / df['Open'])
    log_co = np.log(df['Close']/ df['Open'])
    
    log_oc = np.log(df['Open']/ df['Close'].shift(1))
    log_cc = np.log(df['Close']/ df['Close'].shift(1))
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    log_oc_sq = log_oc**2
    log_cc_sq = log_cc**2
    
    close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol  = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    rs_term   = rs.rolling(window=window).sum()         * (1.0 / (window - 1.0))
    
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    yz = (open_vol + k * close_vol + (1 - k) * rs_term).apply(np.sqrt) * np.sqrt(trading_periods)
    return yz.dropna().iloc[-1] if len(yz.dropna())>0 else None

def compute_calendar_recommendation(ticker):
    """
    1) Pick near_date (closest to ~30 days, or fallback) and the next far_date.
    2) Find near & far call at the same ATM strike (closest to current price).
    3) Compute net_debit = far_mid - near_mid.
    4) Return pass/fail checks + data for display.
    """
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."
        
        stock = yf.Ticker(ticker)
        if not stock.options:
            return f"Error: No options found for '{ticker}'."
        
        try:
            near_date_str, far_date_str = pick_calendar_dates(stock.options)
        except ValueError as e:
            return f"Error picking dates: {str(e)}"
        
        # Option chains
        near_chain = stock.option_chain(near_date_str)
        far_chain  = stock.option_chain(far_date_str)
        
        # Current stock price
        underlying_price = get_current_price(stock)
        if underlying_price is None:
            return f"Error: No underlying price for {ticker}."
        
        # ATM strike for near expiration
        near_calls = near_chain.calls
        if near_calls.empty:
            return f"Error: No calls found for near expiration {near_date_str}."
        
        call_diffs = (near_calls['strike'] - underlying_price).abs()
        near_call_idx = call_diffs.idxmin()
        atm_strike = near_calls.loc[near_call_idx, 'strike']
        
        # The far expiration's call at the same strike
        far_calls = far_chain.calls
        if far_calls.empty:
            return f"Error: No calls found for far expiration {far_date_str}."
        
        # Filter to the same strike
        same_strike_df = far_calls[far_calls['strike'] == atm_strike]
        if same_strike_df.empty:
            return f"Error: Far expiration {far_date_str} doesn't have strike {atm_strike}."
        
        far_call = same_strike_df.iloc[0]  # in case there's only one row
        
        # near leg data
        near_bid  = near_calls.loc[near_call_idx, 'bid']
        near_ask  = near_calls.loc[near_call_idx, 'ask']
        near_vol  = near_calls.loc[near_call_idx, 'volume']
        
        # far leg data
        far_bid   = far_call['bid']
        far_ask   = far_call['ask']
        far_vol   = far_call['volume']
        
        def mid(bid, ask):
            if pd.isna(bid) or pd.isna(ask):
                return None
            return (bid + ask)/2.0
        
        near_mid = mid(near_bid, near_ask)
        far_mid  = mid(far_bid,  far_ask)
        
        if near_mid is None or far_mid is None:
            return "Error: Missing bid/ask data for calls. Can't compute mid."
        
        net_debit = far_mid - near_mid
        
        # Additional checks: 
        # - Average daily volume (equity), 
        # - IV30>RV30, 
        # - negative term-structure slope? (optional).
        
        # Historical price data for 3 mo
        hist = stock.history(period="3mo")
        if len(hist) < 30:
            return "Error: Not enough historical data for IV/RV checks."
        
        # 30-day average equity volume
        avg_equity_vol = hist['Volume'].rolling(30).mean().dropna()
        if avg_equity_vol.empty:
            return "Error: Not enough data to compute 30-day average volume."
        eq_30day_avg = avg_equity_vol.iloc[-1]
        
        # Yang-Zhang realized vol
        rv30 = compute_yang_zhang_volatility(hist, window=30)
        if rv30 is None or rv30==0:
            return "Error: RV30 not computable."
        
        # A quick approach for IV30: 
        # We'll gather implied vol for all calls at all expirations, do a small interpolation around 30 DTE
        # (This is a simplified approach, you could refine it further.)
        # Gather all expiration dates, build day->IV
        # We'll pick the ATM call's IV from each chain (for simplicity).
        
        exps = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in stock.options)
        iv_map = {}
        today = datetime.today().date()
        
        for d in exps:
            chain = stock.option_chain(d.strftime("%Y-%m-%d"))
            calls = chain.calls
            if calls.empty:
                continue
            # ATM for that expiration
            diffs = (calls['strike'] - underlying_price).abs()
            idx = diffs.idxmin()
            if pd.isna(idx):
                continue
            iv_val = calls.loc[idx, 'impliedVolatility']
            if pd.isna(iv_val):
                continue
            dte = (d - today).days
            iv_map[dte] = iv_val
        
        if not iv_map:
            return "Error: No valid ATM call IV data for building IV30."
        
        # Build simple day->iv interpolation
        dtes = list(iv_map.keys())
        ivs  = [iv_map[d] for d in dtes]
        
        # Sort them
        sorted_pairs = sorted(zip(dtes, ivs), key=lambda x: x[0])
        dtes_sorted, ivs_sorted = zip(*sorted_pairs)
        dtes_arr = np.array(dtes_sorted)
        ivs_arr  = np.array(ivs_sorted)
        
        # We'll do a simple linear interpolation
        from scipy.interpolate import interp1d
        spline = interp1d(dtes_arr, ivs_arr, kind='linear', fill_value="extrapolate")
        
        def iv_estimate(dte):
            if dte < dtes_arr[0]:
                return ivs_arr[0]
            elif dte > dtes_arr[-1]:
                return ivs_arr[-1]
            return float(spline(dte))
        
        iv30 = iv_estimate(30)
        
        # IV30>RV30 pass
        iv30_rv30_pass = (iv30/rv30 >= 1.25)
        
        # Term structure slope: compare earliest DTE to 45 DTE
        first_dte = dtes_arr[0]
        if first_dte >= 45:
            slope_pass = False  # or interpret slope as 0? your call.
            slope_value = 0
        else:
            slope_value = (iv_estimate(45) - iv_estimate(first_dte)) / (45 - first_dte)
            # e.g. pass if slope is negative enough
            slope_pass = (slope_value <= -0.00406)
        
        # 1.5 million daily equity volume pass/fail
        avg_vol_pass = (eq_30day_avg >= 1_500_000)
        
        return {
            "ticker": ticker,
            "share_price": underlying_price,
            "near_expiration": near_date_str,
            "far_expiration": far_date_str,
            "calendar_strike": atm_strike,
            
            "near_call_bid": near_bid,
            "near_call_ask": near_ask,
            "near_call_vol": near_vol,
            
            "far_call_bid": far_bid,
            "far_call_ask": far_ask,
            "far_call_vol": far_vol,
            
            "net_debit": net_debit,
            
            # pass/fail checks
            "avg_volume_pass": avg_vol_pass,
            "iv30_rv30_pass": iv30_rv30_pass,
            "ts_slope_0_45_pass": slope_pass,
            "ts_slope_value": slope_value,  # might be interesting to see
        }

    except Exception as ex:
        return f"Error for {ticker}: {str(ex)}"


def main():
    st.title("Call Calendar Screener (~30 days → next expiration)")

    st.subheader("Ticker Cleaning Utility")
    st.write("Paste text containing tickers in quotes (e.g. `'AAPL', 'TSLA'`) and click **Clean Tickers**.")
    
    raw_tickers_text = st.text_area("Paste tickers with quotes here:", 
                                    value="'AMC', 'CART', 'CAVA', 'CPNG'")
    
    if st.button("Clean Tickers"):
        cleaned_text = (raw_tickers_text
                        .replace("'", "")
                        .replace('"', "")
                        .replace("[", "")
                        .replace("]", ""))
        st.write("**Cleaned Tickers** (comma-separated):")
        st.code(cleaned_text.strip())

    st.write("---")
    st.subheader("Run the Calendar Screener")

    tickers_input = st.text_input(
        "Tickers (comma‐separated, no quotes):",
        value="AAPL, TSLA"
    )

    if st.button("Run Screener"):
        tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers_list:
            st.error("No valid tickers provided.")
            return
        
        n = len(tickers_list)
        progress_bar = st.progress(0)
        
        results = []
        errors = []
        
        for i, ticker in enumerate(tickers_list):
            with st.spinner(f"Processing {ticker} ({i+1}/{n})..."):
                res = compute_calendar_recommendation(ticker)
                if isinstance(res, dict):
                    results.append(res)
                else:
                    errors.append(res)

            # update progress
            progress_val = int(((i + 1)/n)*100)
            progress_bar.progress(progress_val)
            
            if i < n-1:
                time.sleep(7)  # 7-second delay to avoid yfinance rate-limits

        # Build final table
        if results:
            rows = []
            for r in results:
                ticker      = r["ticker"]
                share_price = r["share_price"]
                near_exp    = r["near_expiration"]
                far_exp     = r["far_expiration"]
                strike      = r["calendar_strike"]
                
                near_call_ba = (f"{r['near_call_bid']:.2f} / {r['near_call_ask']:.2f}"
                                if not pd.isna(r['near_call_bid']) and not pd.isna(r['near_call_ask'])
                                else "N/A")
                far_call_ba  = (f"{r['far_call_bid']:.2f} / {r['far_call_ask']:.2f}"
                                if not pd.isna(r['far_call_bid']) and not pd.isna(r['far_call_ask'])
                                else "N/A")
                
                near_vol = r["near_call_vol"] if not pd.isna(r["near_call_vol"]) else 0
                far_vol  = r["far_call_vol"]  if not pd.isna(r["far_call_vol"])  else 0
                
                net_debit = r["net_debit"]
                
                # pass/fail flags
                vol_pass     = r["avg_volume_pass"]
                iv30rv30_pas = r["iv30_rv30_pass"]
                slope_pass   = r["ts_slope_0_45_pass"]
                
                # Simple recommendation logic
                if vol_pass and iv30rv30_pas and slope_pass:
                    recommendation = "Recommended"
                elif slope_pass and ((vol_pass and not iv30rv30_pas) or 
                                     (iv30rv30_pas and not vol_pass)):
                    recommendation = "Consider"
                else:
                    recommendation = "Avoid"
                
                row = {
                    "Ticker": ticker,
                    "Share Price": round(share_price, 2) if share_price else "N/A",
                    "Near Exp": near_exp,
                    "Far Exp":  far_exp,
                    "Strike":   strike,
                    
                    "Near Call B/A": near_call_ba,
                    "Near Call Vol": near_vol,
                    
                    "Far Call B/A":  far_call_ba,
                    "Far Call Vol":  far_vol,
                    
                    "Net Debit($)":  f"{net_debit:.2f}" if net_debit else "N/A",
                    "avg_volume":    "PASS" if vol_pass else "FAIL",
                    "iv30_rv30":     "PASS" if iv30rv30_pas else "FAIL",
                    "ts_slope":      "PASS" if slope_pass else "FAIL",
                    
                    "Recommendation": recommendation
                }
                rows.append(row)
            
            # sort by net debit ascending, for instance
            def sort_key(d):
                val_str = d["Net Debit($)"]
                try:
                    return float(val_str)
                except:
                    return 9999999  # 'N/A' goes last
            rows_sorted = sorted(rows, key=sort_key)
            
            df = pd.DataFrame(rows_sorted)
            st.subheader("Calendar Screener Results")
            st.dataframe(df, use_container_width=True)

        if errors:
            st.subheader("Errors")
            for e in errors:
                st.error(e)


if __name__ == "__main__":
    main()
