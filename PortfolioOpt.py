import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

# Configure the GUI layout
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Max Sharpe Portfolio Optimizer")
st.markdown("Upload your portfolio or enter tickers manually to find the optimal allocation.")

# Initialize Session State to prevent re-downloading data when moving sliders
if "optimized" not in st.session_state:
    st.session_state.optimized = False

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")

uploaded_file = st.sidebar.file_uploader("Upload Excel File (Must have a 'Ticker' column)", type=["xlsx", "xls"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually (comma separated):", "AAPL, MSFT, GOOG, SPY")
st.sidebar.info("💡 **Tip for Canadian Funds:** Append `.TO` to the ticker for Canadian mutual funds and TSX stocks (e.g., `TDB900.TO`, `RY.TO`).")

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

optimize_button = st.sidebar.button("Run Optimization", type="primary")

# --- MAIN APP LOGIC ---
if optimize_button:
    tickers = []
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'Ticker' in df.columns:
                tickers = df['Ticker'].dropna().astype(str).tolist()
            else:
                st.error("Error: Your Excel file must contain a column named 'Ticker'.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
            st.stop()
    else:
        cleaned_input = manual_tickers.replace(' ', ',')
        tickers = [t.strip().upper() for t in cleaned_input.split(',') if t.strip()]

    if len(tickers) < 2:
        st.warning("Please provide at least two valid tickers.")
        st.stop()

    with st.spinner(f"Fetching historical data for {len(tickers)} assets..."):
        raw_data = yf.download(tickers, start=start_date, end=end_date)
        
        if raw_data.empty:
            st.error("No data found.")
            st.stop()
            
        try: data = raw_data['Adj Close']
        except KeyError:
            try: data = raw_data['Close']
            except KeyError:
                st.error("Error: Pricing columns not found.")
                st.stop()

        if isinstance(data, pd.Series): data = data.to_frame()
        data = data.dropna(axis=1, thresh=int(len(data)*0.8)).ffill().bfill()
        
        if data.shape[1] < 2:
            st.error(f"Not enough valid data. Found usable data for {data.shape[1]} assets.")
            st.stop()

    with st.spinner("Calculating optimal weights..."):
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()
        
        # SAVE TO SESSION STATE
        st.session_state.mu = mu
        st.session_state.S = S
        st.session_state.cleaned_weights = cleaned_weights
        st.session_state.ret = ret
        st.session_state.vol = vol
        st.session_state.sharpe = sharpe
        st.session_state.asset_list = list(mu.index)
        st.session_state.corr_matrix = data.pct_change().corr()
        st.session_state.optimized = True

# --- DASHBOARD VISUALS ---
if st.session_state.optimized:
    st.markdown("---")
    st.subheader("Interactive What-If Analysis 🎛️")
    st.markdown("Adjust the weight of any individual holding. The app will proportionally scale the remaining assets so your portfolio still equals 100%, and plot your custom setup against the mathematical optimum.")
    
    # --- SLIDER LOGIC ---
    adj_col1, adj_col2 = st.columns([1, 2])
    
    with adj_col1:
        # Select an asset to override
        adj_asset = st.selectbox("Select Asset to Adjust:", st.session_state.asset_list)
        
        # Get its original optimized weight
        orig_w = st.session_state.cleaned_weights.get(adj_asset, 0.0)
        
        # Slider for new weight
        new_w_pct = st.slider(f"Target Weight for {adj_asset}", min_value=0.0, max_value=100.0, value=float(orig_w*100), step=1.0, format="%.0f%%")
        new_w = new_w_pct / 100.0
    
    # Calculate Custom Weights
    custom_weights = st.session_state.cleaned_weights.copy()
    for t in st.session_state.asset_list:
        if t not in custom_weights:
            custom_weights[t] = 0.0 # Fill missing with 0
            
    old_remaining = 1.0 - orig_w
    new_remaining = 1.0 - new_w
    
    # Pro-rata distribution of remaining weight
    for t in custom_weights:
        if t != adj_asset:
            if old_remaining > 0:
                custom_weights[t] = custom_weights[t] * (new_remaining / old_remaining)
            else:
                custom_weights[t] = new_remaining / (len(custom_weights) - 1)
    
    custom_weights[adj_asset] = new_w
    
    # Calculate Custom Performance
    w_array = np.array([custom_weights[t] for t in st.session_state.asset_list])
    mu_array = st.session_state.mu.values
    S_matrix = st.session_state.S.values
    
    c_ret = np.dot(w_array, mu_array)
    c_vol = np.sqrt(np.dot(w_array.T, np.dot(S_matrix, w_array)))
    c_sharpe = (c_ret - 0.02) / c_vol # pypfopt assumes 2% risk-free rate by default

    # --- DISPLAY METRICS ---
    st.markdown("### Performance Comparison")
    m1, m2, m3 = st.columns(3)
    
    # Function to show delta arrows
    m1.metric("Sharpe Ratio", f"{c_sharpe:.2f}", delta=f"{c_sharpe - st.session_state.sharpe:.2f} vs Max Sharpe")
    m2.metric("Expected Annual Return", f"{c_ret*100:.2f}%", delta=f"{(c_ret - st.session_state.ret)*100:.2f}%")
    m3.metric("Annual Volatility (Risk)", f"{c_vol*100:.2f}%", delta=f"{(c_vol - st.session_state.vol)*100:.2f}%", delta_color="inverse")

    st.markdown("---")
    
    # --- CHARTS & DATA ---
    chart_col, data_col = st.columns([2, 1])

    with data_col:
        st.markdown("### Current Weights")
        weights_df = pd.DataFrame.from_dict(custom_weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0.001].sort_values(by='Weight', ascending=False)
        
        display_df = weights_df.copy()
        display_df['Weight'] = (display_df['Weight'] * 100).round(2).astype(str) + '%'
        st.dataframe(display_df, use_container_width=True)

    with chart_col:
        st.markdown("### Efficient Frontier")
        ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S)
        fig, ax = plt.subplots(figsize=(8, 5))
        plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
        
        # Plot Original Max Sharpe (Red Star)
        ax.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r", label="Max Sharpe (Math Optimum)")
        
        # Plot Custom Portfolio (Blue Circle)
        ax.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Your Custom Allocation")
        
        ax.set_title("")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    
    st.markdown("### Asset Correlation Matrix")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(st.session_state.corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax2, fmt=".2f")
    st.pyplot(fig2)
