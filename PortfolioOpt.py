import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import io

st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Advanced Portfolio Optimizer")
st.markdown("Optimize your portfolio, apply constraints, and analyze your asset allocation.")

if "optimized" not in st.session_state:
    st.session_state.optimized = False

# --- HELPER FUNCTION: ASSET CLASSIFICATION ---
def classify_asset(ticker):
    """Attempts to categorize an asset based on Yahoo Finance metadata."""
    try:
        info = yf.Ticker(ticker).info
        q_type = info.get('quoteType', '').upper()
        country = info.get('country', 'Unknown').upper()
        category = info.get('category', '').upper()
        
        # Check for Cash/Bonds via Mutual Funds or ETFs
        if q_type in ['MUTUALFUND', 'ETF']:
            if 'BOND' in category or 'FIXED INCOME' in category: return 'Fixed Income'
            if 'MONEY MARKET' in category or 'CASH' in category: return 'Cash & Equivalents'
            if 'CANADA' in category or ticker.endswith('.TO'): return 'Canadian Equities'
            if 'FOREIGN' in category or 'EMERGING' in category or 'INTERNATIONAL' in category: return 'International Equities'
            return 'US Equities' # Fallback for US-listed funds
            
        # Check Equities by Country
        else:
            if country == 'CANADA' or ticker.endswith('.TO'): return 'Canadian Equities'
            if country == 'UNITED STATES': return 'US Equities'
            if country != 'UNKNOWN': return 'International Equities'
            
    except Exception:
        pass
    
    # Absolute Fallback based on ticker suffix
    if ticker.endswith('.TO'): return 'Canadian Equities'
    return 'Other'

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, GOOG, XIU.TO, XBB.TO")
benchmark_ticker = st.sidebar.text_input("Benchmark (for Alpha & Beta):", "SPY")

st.sidebar.header("2. Time Horizon")
time_range = st.sidebar.selectbox("Select Time Range", ("1 Year", "3 Years", "5 Years", "7 Years", "10 Years", "Custom Dates"), index=2)

end_date = pd.Timestamp.today()
if time_range == "1 Year": start_date = end_date - pd.DateOffset(years=1)
elif time_range == "3 Years": start_date = end_date - pd.DateOffset(years=3)
elif time_range == "5 Years": start_date = end_date - pd.DateOffset(years=5)
elif time_range == "7 Years": start_date = end_date - pd.DateOffset(years=7)
elif time_range == "10 Years": start_date = end_date - pd.DateOffset(years=10)
else:
    start_date = st.sidebar.date_input("Start Date", end_date - pd.DateOffset(years=5))
    end_date = st.sidebar.date_input("End Date", end_date)

st.sidebar.header("3. Optimization Strategy")
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))

# NEW: Position Sizing Constraint
max_weight_pct = st.sidebar.slider("Max Weight per Asset Constraint", min_value=10, max_value=100, value=100, step=5, format="%d%%")
max_w = max_weight_pct / 100.0

optimize_button = st.sidebar.button("Run Optimization", type="primary")

# --- MAIN APP LOGIC ---
if optimize_button:
    tickers = []
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'Ticker' in df.columns:
                tickers = df['Ticker'].dropna().astype(str).tolist()
        except Exception:
            st.error("Failed to read Excel file.")
            st.stop()
    else:
        tickers = [t.strip().upper() for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2:
        st.warning("Please provide at least two valid tickers.")
        st.stop()
        
    # Mathematical reality check for constraints
    if max_w < (1.0 / len(tickers)):
        st.error(f"Constraint mathematically impossible! You have {len(tickers)} assets. The max weight must be at least {np.ceil((1.0/len(tickers))*100)}%.")
        st.stop()
        
    bench_clean = benchmark_ticker.strip().upper()
    all_tickers = list(set(tickers + [bench_clean]))

    with st.spinner("Downloading pricing data and asset metadata..."):
        raw_data = yf.download(all_tickers, start=start_date, end=end_date)
        
        # Map Asset Classes (Dictionary Comprehension)
        st.session_state.asset_classes = {t: classify_asset(t) for t in tickers}
        
        if raw_data.empty:
            st.error("No data found.")
            st.stop()
            
        try: data = raw_data['Adj Close']
        except KeyError:
            try: data = raw_data['Close']
            except KeyError:
                st.error("Pricing columns not found.")
                st.stop()

        if isinstance(data, pd.Series): data = data.to_frame()
        data = data.dropna(axis=1, thresh=int(len(data)*0.8)).ffill().bfill()
        
        if bench_clean in data.columns:
            bench_data = data[bench_clean]
            port_data = data.drop(columns=[bench_clean], errors='ignore')
        else:
            bench_data = pd.Series(dtype=float)
            port_data = data

        if port_data.shape[1] < 2:
            st.error("Not enough valid asset data.")
            st.stop()

    with st.spinner(f"Applying constraints and calculating {opt_metric}..."):
        mu = expected_returns.mean_historical_return(port_data)
        S = risk_models.sample_cov(port_data)

        # APPLY THE USER CONSTRAINT HERE
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
        
        if "Max Sharpe" in opt_metric:
            raw_weights = ef.max_sharpe()
        else:
            raw_weights = ef.min_volatility()
            
        cleaned_weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()
        
        st.session_state.opt_target = "Max Sharpe" if "Max Sharpe" in opt_metric else "Min Volatility"
        st.session_state.mu = mu
        st.session_state.S = S
        st.session_state.cleaned_weights = cleaned_weights
        st.session_state.ret = ret
        st.session_state.vol = vol
        st.session_state.sharpe = sharpe
        st.session_state.asset_list = list(mu.index)
        st.session_state.daily_returns = port_data.pct_change().dropna()
        st.session_state.bench_returns = bench_data.pct_change().dropna() if not bench_data.empty else None
        st.session_state.optimized = True

# --- DASHBOARD VISUALS ---
if st.session_state.optimized:
    st.markdown("---")
    st.subheader(f"Interactive What-If Analysis ({st.session_state.opt_target} Baseline) 🎛️")
    
    adj_col1, adj_col2 = st.columns([1, 2])
    with adj_col1:
        adj_asset = st.selectbox("Select Asset to Adjust:", st.session_state.asset_list)
        orig_w = st.session_state.cleaned_weights.get(adj_asset, 0.0)
        new_w_pct = st.slider(f"Target Weight for {adj_asset}", min_value=0.0, max_value=100.0, value=float(orig_w*100), step=1.0, format="%.0f%%")
        new_w = new_w_pct / 100.0
    
    custom_weights = st.session_state.cleaned_weights.copy()
    for t in st.session_state.asset_list:
        if t not in custom_weights: custom_weights[t] = 0.0
            
    old_remaining = 1.0 - orig_w
    new_remaining = 1.0 - new_w
    
    for t in custom_weights:
        if t != adj_asset:
            if old_remaining > 0: custom_weights[t] = custom_weights[t] * (new_remaining / old_remaining)
            else: custom_weights[t] = new_remaining / (len(custom_weights) - 1)
    
    custom_weights[adj_asset] = new_w
    
    w_array = np.array([custom_weights[t] for t in st.session_state.asset_list])
    c_ret = np.dot(w_array, st.session_state.mu.values)
    c_vol = np.sqrt(np.dot(w_array.T, np.dot(st.session_state.S.values, w_array)))
    
    risk_free_rate = 0.02 
    c_sharpe = (c_ret - risk_free_rate) / c_vol
    
    port_daily = st.session_state.daily_returns.dot(w_array)
    downside_returns = port_daily[port_daily < 0]
    down_stdev = np.sqrt(252) * downside_returns.std()
    c_sortino = (c_ret - risk_free_rate) / down_stdev if down_stdev > 0 else 0

    c_beta, c_alpha = np.nan, np.nan
    if st.session_state.bench_returns is not None:
        aligned_data = pd.concat([port_daily, st.session_state.bench_returns], axis=1).dropna()
        if len(aligned_data) > 0:
            p_ret = aligned_data.iloc[:, 0]
            b_ret = aligned_data.iloc[:, 1]
            cov_matrix = np.cov(p_ret, b_ret)
            c_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            annual_bench_ret = b_ret.mean() * 252
            c_alpha = c_ret - (risk_free_rate + c_beta * (annual_bench_ret - risk_free_rate))

    st.markdown("### Risk & Return Metrics")
    # RESTORED EXPECTED RETURN TO THE TOP METRICS
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    
    m1.metric("Exp. Return", f"{c_ret*100:.2f}%")
    m2.metric("Sharpe Ratio", f"{c_sharpe:.2f}")
    m3.metric("Sortino Ratio", f"{c_sortino:.2f}")
    
    if not np.isnan(c_beta):
        m4.metric("Beta (β)", f"{c_beta:.2f}")
        m5.metric("Alpha (α)", f"{c_alpha*100:.2f}%")
    else:
        m4.metric("Beta (β)", "N/A")
        m5.metric("Alpha (α)", "N/A")
        
    m6.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%")

    st.markdown("---")
    
    # --- ASSET ALLOCATION & WEIGHTS SECTION ---
    st.markdown("### Portfolio Allocation")
    alloc_col1, alloc_col2, alloc_col3 = st.columns([1, 1, 1.5])

    # 1. Compile Asset Class Totals
    allocation_totals = {
        'Cash & Equivalents': 0.0,
        'Fixed Income': 0.0,
        'Canadian Equities': 0.0,
        'US Equities': 0.0,
        'International Equities': 0.0,
        'Other': 0.0
    }
    
    for t, w in custom_weights.items():
        a_class = st.session_state.asset_classes.get(t, 'Other')
        allocation_totals[a_class] += w
        
    alloc_df = pd.DataFrame(list(allocation_totals.items()), columns=['Category', 'Weight'])
    alloc_df = alloc_df[alloc_df['Weight'] > 0.001].sort_values(by='Weight', ascending=False).reset_index(drop=True)

    with alloc_col1:
        st.markdown("**By Asset Class**")
        display_alloc = alloc_df.copy()
        display_alloc['Weight'] = (display_alloc['Weight'] * 100).round(2).astype(str) + '%'
        st.dataframe(display_alloc, use_container_width=True)

    with alloc_col2:
        st.markdown("**By Individual Security**")
        weights_df = pd.DataFrame.from_dict(custom_weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
        display_df = weights_df.copy()
        display_df['Weight'] = (display_df['Weight'] * 100).round(2).astype(str) + '%'
        st.dataframe(display_df, use_container_width=True)
        
        csv = display_df.to_csv()
        st.download_button("📥 Download Weights", data=csv, file_name='portfolio_weights.csv', mime='text/csv')

    with alloc_col3:
        st.markdown("**Allocation Breakdown**")
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie(alloc_df['Weight'], labels=alloc_df['Category'], autopct='%1.1f%%', startangle=90, 
                   colors=sns.color_palette("pastel"))
        ax_pie.axis('equal') 
        fig_pie.patch.set_alpha(0.0) # Transparent background
        st.pyplot(fig_pie)

    st.markdown("---")
    
    # --- RESTORED CORRELATION MATRIX & EFFICIENT FRONTIER ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### Efficient Frontier")
        ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S, weight_bounds=(0, max_w))
        fig, ax = plt.subplots(figsize=(8, 6))
        plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
        
        star_label = f"{st.session_state.opt_target} Optimum"
        ax.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r", label=star_label)
        ax.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Custom Allocation")
        ax.set_title("")
        ax.legend()
        st.pyplot(fig)
        
    with chart_col2:
        st.markdown("### Asset Correlation Matrix")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        corr_matrix = st.session_state.daily_returns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax2, fmt=".2f", cbar=False)
        st.pyplot(fig2)
