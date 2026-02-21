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
st.markdown("Upload your portfolio or enter tickers manually to find your ideal allocation.")

if "optimized" not in st.session_state:
    st.session_state.optimized = False

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel File (Must have a 'Ticker' column)", type=["xlsx", "xls"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, GOOG")
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
# NEW: Dropdown for selecting the optimization objective
opt_metric = st.sidebar.selectbox(
    "Optimize For:", 
    ("Max Sharpe Ratio (Best Risk-Adjusted Return)", "Minimum Volatility (Lowest Risk)")
)

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
        
    bench_clean = benchmark_ticker.strip().upper()
    all_tickers = list(set(tickers + [bench_clean]))

    with st.spinner(f"Fetching data for {len(tickers)} assets + Benchmark..."):
        raw_data = yf.download(all_tickers, start=start_date, end=end_date)
        
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
            st.warning(f"Benchmark {bench_clean} data failed to load.")
            bench_data = pd.Series(dtype=float)
            port_data = data

        if port_data.shape[1] < 2:
            st.error("Not enough valid asset data to optimize.")
            st.stop()

    with st.spinner(f"Calculating {opt_metric}..."):
        mu = expected_returns.mean_historical_return(port_data)
        S = risk_models.sample_cov(port_data)

        ef = EfficientFrontier(mu, S)
        
        # NEW: Logic to branch based on user's selected dropdown metric
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
            if old_remaining > 0:
                custom_weights[t] = custom_weights[t] * (new_remaining / old_remaining)
            else:
                custom_weights[t] = new_remaining / (len(custom_weights) - 1)
    
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

    st.markdown("### Advanced Risk Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("Sharpe Ratio", f"{c_sharpe:.2f}")
    m2.metric("Sortino Ratio", f"{c_sortino:.2f}")
    
    if not np.isnan(c_beta):
        m3.metric("Beta (β)", f"{c_beta:.2f}")
        m4.metric("Alpha (α)", f"{c_alpha*100:.2f}%")
    else:
        m3.metric("Beta (β)", "N/A")
        m4.metric("Alpha (α)", "N/A")
        
    m5.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%")

    st.markdown("---")
    chart_col, data_col = st.columns([2, 1])

    with data_col:
        st.markdown("### Custom Weights")
        weights_df = pd.DataFrame.from_dict(custom_weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
        display_df = weights_df.copy()
        display_df['Weight'] = (display_df['Weight'] * 100).round(2).astype(str) + '%'
        st.dataframe(display_df, use_container_width=True)
        
        # NEW: Export to CSV Button
        csv = display_df.to_csv()
        st.download_button(
            label="📥 Download Weights as CSV",
            data=csv,
            file_name='custom_portfolio_weights.csv',
            mime='text/csv',
        )

    with chart_col:
        st.markdown("### Efficient Frontier")
        ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S)
        fig, ax = plt.subplots(figsize=(8, 5))
        plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
        
        # Updates the label of the red star based on the chosen strategy
        star_label = "Max Sharpe Optimum" if st.session_state.opt_target == "Max Sharpe" else "Min Volatility Optimum"
        ax.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r", label=star_label)
        
        ax.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Custom Allocation")
        ax.set_title("")
        ax.legend()
        st.pyplot(fig)
