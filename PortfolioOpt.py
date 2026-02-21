import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import io

st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Advanced Portfolio Optimizer & Forecaster")
st.markdown("Optimize weights, calculate trade values, backtest history, and forecast the future.")

if "optimized" not in st.session_state:
    st.session_state.optimized = False

# --- HELPER FUNCTION: ASSET METADATA ---
def get_asset_metadata(ticker):
    asset_class = 'Other'
    sector = 'Unknown'
    try:
        info = yf.Ticker(ticker).info
        q_type = info.get('quoteType', '').upper()
        country = info.get('country', 'Unknown').upper()
        category = info.get('category', '').upper()
        sector_info = info.get('sector', '')
        
        if sector_info: sector = sector_info
        elif category: sector = category.title()

        if q_type in ['MUTUALFUND', 'ETF']:
            if 'BOND' in category or 'FIXED INCOME' in category: asset_class = 'Fixed Income'
            elif 'MONEY MARKET' in category or 'CASH' in category: asset_class = 'Cash & Equivalents'
            elif 'CANADA' in category or ticker.endswith('.TO'): asset_class = 'Canadian Equities'
            elif 'FOREIGN' in category or 'EMERGING' in category or 'INTERNATIONAL' in category: asset_class = 'International Equities'
            else: asset_class = 'US Equities' 
        else:
            if country == 'CANADA' or ticker.endswith('.TO'): asset_class = 'Canadian Equities'
            elif country == 'UNITED STATES': asset_class = 'US Equities'
            elif country != 'UNKNOWN': asset_class = 'International Equities'
    except Exception:
        pass
    
    if asset_class == 'Other' and ticker.endswith('.TO'): asset_class = 'Canadian Equities'
    return asset_class, sector

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, GOOG, JNJ, SPY")
benchmark_ticker = st.sidebar.text_input("Benchmark (for Alpha, Beta & Backtest):", "SPY")

st.sidebar.header("2. Historical Horizon")
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

st.sidebar.header("3. Strategy & Constraints")
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_weight_pct = st.sidebar.slider("Max Weight per Asset", min_value=10, max_value=100, value=100, step=5, format="%d%%")
max_w = max_weight_pct / 100.0

st.sidebar.header("4. Trade & Forecast Settings")
portfolio_value = st.sidebar.number_input("Total Portfolio Value ($)", min_value=1000, value=100000, step=1000)
# NEW: Monte Carlo Inputs
mc_years = st.sidebar.slider("Years to Forecast (Monte Carlo)", min_value=1, max_value=30, value=10)
mc_sims = st.sidebar.selectbox("Number of Simulations", (100, 500, 1000, 5000), index=2)

optimize_button = st.sidebar.button("Run Analysis", type="primary")

# --- MAIN APP LOGIC ---
if optimize_button:
    tickers = []
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'Ticker' in df.columns: tickers = df['Ticker'].dropna().astype(str).tolist()
        except Exception:
            st.error("Failed to read Excel file.")
            st.stop()
    else:
        tickers = [t.strip().upper() for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2:
        st.warning("Please provide at least two valid tickers.")
        st.stop()
        
    if max_w < (1.0 / len(tickers)):
        st.error(f"Constraint mathematically impossible! Must be at least {np.ceil((1.0/len(tickers))*100)}%.")
        st.stop()
        
    bench_clean = benchmark_ticker.strip().upper()
    all_tickers = list(set(tickers + [bench_clean]))

    with st.spinner("Downloading market data and categorizing assets..."):
        raw_data = yf.download(all_tickers, start=start_date, end=end_date)
        st.session_state.asset_meta = {t: get_asset_metadata(t) for t in tickers}
        
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

    with st.spinner("Crunching the math..."):
        mu = expected_returns.mean_historical_return(port_data)
        S = risk_models.sample_cov(port_data)

        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
        raw_weights = ef.max_sharpe() if "Max Sharpe" in opt_metric else ef.min_volatility()
            
        st.session_state.opt_target = "Max Sharpe" if "Max Sharpe" in opt_metric else "Min Volatility"
        st.session_state.mu = mu
        st.session_state.S = S
        st.session_state.cleaned_weights = ef.clean_weights()
        st.session_state.ret, st.session_state.vol, st.session_state.sharpe = ef.portfolio_performance()
        st.session_state.asset_list = list(mu.index)
        st.session_state.daily_returns = port_data.pct_change().dropna()
        st.session_state.bench_returns = bench_data.pct_change().dropna() if not bench_data.empty else None
        st.session_state.optimized = True

# --- DASHBOARD VISUALS ---
if st.session_state.optimized:
    st.markdown("---")
    st.subheader(f"Interactive Adjustments ({st.session_state.opt_target} Baseline) 🎛️")
    
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
            p_ret, b_ret = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
            cov_matrix = np.cov(p_ret, b_ret)
            c_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            c_alpha = c_ret - (risk_free_rate + c_beta * ((b_ret.mean() * 252) - risk_free_rate))

    st.markdown("### Risk & Return Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Exp. Return", f"{c_ret*100:.2f}%")
    m2.metric("Sharpe Ratio", f"{c_sharpe:.2f}")
    m3.metric("Sortino Ratio", f"{c_sortino:.2f}")
    if not np.isnan(c_beta):
        m4.metric("Beta (β)", f"{c_beta:.2f}")
        m5.metric("Alpha (α)", f"{c_alpha*100:.2f}%")
    else:
        m4.metric("Beta", "N/A"); m5.metric("Alpha", "N/A")
    m6.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%")

    st.markdown("---")
    
    st.markdown("### Trade Calculator & Allocation Breakdown")
    trade_data = []
    for t, w in custom_weights.items():
        if w > 0.001:
            a_class, a_sector = st.session_state.asset_meta.get(t, ('Other', 'Unknown'))
            trade_data.append({
                'Ticker': t, 'Weight': w, 'Dollar Value ($)': w * portfolio_value,
                'Asset Class': a_class, 'Sector': a_sector
            })
            
    trade_df = pd.DataFrame(trade_data).sort_values(by='Weight', ascending=False).reset_index(drop=True)
    calc_col1, calc_col2 = st.columns([2, 1])
    with calc_col1:
        display_trade = trade_df.copy()
        display_trade['Weight'] = (display_trade['Weight'] * 100).round(2).astype(str) + '%'
        display_trade['Dollar Value ($)'] = display_trade['Dollar Value ($)'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_trade, use_container_width=True)
    with calc_col2:
        sector_totals = trade_df.groupby('Sector')['Weight'].sum()
        fig_sec, ax_sec = plt.subplots(figsize=(4, 4))
        ax_sec.pie(sector_totals, labels=sector_totals.index, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
        fig_sec.patch.set_alpha(0.0)
        st.pyplot(fig_sec)

    st.markdown("---")
    
    # --- MONTE CARLO SIMULATION ---
    st.markdown(f"### Future Projection: Monte Carlo Simulation ({mc_years} Years)")
    st.markdown(f"Running **{mc_sims}** simulated lifetimes based on your custom portfolio's historical return and volatility.")
    
    # Monte Carlo Math (Geometric Brownian Motion)
    np.random.seed(42) # Keeps paths stable when moving sliders
    dt = 1 # Annual steps for speed and clarity
    
    sim_results = np.zeros((mc_sims, mc_years + 1))
    sim_results[:, 0] = portfolio_value
    
    for i in range(mc_sims):
        Z = np.random.standard_normal(mc_years)
        # W_t = W_{t-1} * exp((mu - (sigma^2)/2)dt + sigma * sqrt(dt) * Z)
        growth_factors = np.exp((c_ret - (c_vol**2)/2)*dt + c_vol * np.sqrt(dt) * Z)
        sim_results[i, 1:] = portfolio_value * np.cumprod(growth_factors)
        
    final_values = sim_results[:, -1]
    median_val = np.percentile(final_values, 50)
    pct_10 = np.percentile(final_values, 10)
    pct_90 = np.percentile(final_values, 90)
    
    mc_col1, mc_col2 = st.columns([3, 1])
    
    with mc_col1:
        fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
        # Plot a sample of paths (up to 100) for visual texture
        for i in range(min(100, mc_sims)):
            ax_mc.plot(sim_results[i, :], color='gray', alpha=0.1)
            
        median_path = np.percentile(sim_results, 50, axis=0)
        pct_10_path = np.percentile(sim_results, 10, axis=0)
        pct_90_path = np.percentile(sim_results, 90, axis=0)
        
        ax_mc.plot(median_path, color='blue', linewidth=2, label=f'Median: ${median_val:,.0f}')
        ax_mc.plot(pct_10_path, color='red', linewidth=2, linestyle='--', label=f'10th Percentile (Bear): ${pct_10:,.0f}')
        ax_mc.plot(pct_90_path, color='green', linewidth=2, linestyle='--', label=f'90th Percentile (Bull): ${pct_90:,.0f}')
        
        ax_mc.set_xlim(0, mc_years)
        ax_mc.set_xlabel("Years into the Future")
        ax_mc.set_ylabel("Projected Value ($)")
        ax_mc.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax_mc.legend()
        st.pyplot(fig_mc)
        
    with mc_col2:
        st.markdown("#### Expected Outcomes")
        st.info(f"**Bull Market (90th Pct):**\n${pct_90:,.2f}")
        st.success(f"**Median Expectation:**\n${median_val:,.2f}")
        st.error(f"**Bear Market (10th Pct):**\n${pct_10:,.2f}")
        st.markdown("*Note: Assumes historical risk/return patterns continue, and does not account for future deposits or inflation.*")

    st.markdown("---")
    
    # --- HISTORICAL BACKTESTING ---
    st.markdown("### Historical Backtest ($10,000 Growth & Drawdowns)")
    port_wealth = (1 + port_daily).cumprod() * 10000
    bench_wealth = (1 + st.session_state.bench_returns).cumprod() * 10000 if st.session_state.bench_returns is not None else None
    rolling_max = port_wealth.cummax()
    drawdown = (port_wealth - rolling_max) / rolling_max
    
    bt_col1, bt_col2 = st.columns(2)
    with bt_col1:
        fig_wealth, ax_wealth = plt.subplots(figsize=(8, 5))
        ax_wealth.plot(port_wealth.index, port_wealth, label="Custom Portfolio", color='blue', linewidth=2)
        if bench_wealth is not None:
            bench_wealth_aligned = bench_wealth.reindex(port_wealth.index).ffill()
            ax_wealth.plot(port_wealth.index, bench_wealth_aligned, label=f"Benchmark ({benchmark_ticker})", color='gray', alpha=0.7)
        ax_wealth.set_ylabel("Portfolio Value ($)")
        ax_wealth.grid(True, linestyle='--', alpha=0.6)
        ax_wealth.legend()
        st.pyplot(fig_wealth)
    with bt_col2:
        fig_dd, ax_dd = plt.subplots(figsize=(8, 5))
        ax_dd.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax_dd.plot(drawdown.index, drawdown, color='red', linewidth=1)
        ax_dd.set_ylabel("Drawdown (%)")
        vals = ax_dd.get_yticks()
        ax_dd.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        ax_dd.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_dd)

    st.markdown("---")
    
    st.markdown("### Optimization Charts")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S, weight_bounds=(0, max_w))
        fig_ef, ax_ef = plt.subplots(figsize=(8, 5))
        plotting.plot_efficient_frontier(ef_plot, ax=ax_ef, show_assets=True)
        ax_ef.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r", label=f"{st.session_state.opt_target} Optimum")
        ax_ef.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Custom Allocation")
        ax_ef.legend()
        st.pyplot(fig_ef)
    with chart_col2:
        fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
        corr_matrix = st.session_state.daily_returns.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr, fmt=".2f", cbar=False)
        st.pyplot(fig_corr)
