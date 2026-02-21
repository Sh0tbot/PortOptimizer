import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

# Configure the GUI layout
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Max Sharpe Portfolio Optimizer")
st.markdown("Upload your portfolio or enter tickers manually to find the optimal allocation.")

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")

# File Uploader for Excel
uploaded_file = st.sidebar.file_uploader("Upload Excel File (Must have a 'Ticker' column)", type=["xlsx", "xls"])

# Fallback manual input
manual_tickers = st.sidebar.text_input("Or enter tickers manually (comma separated):", "AAPL, MSFT, GOOG, SPY")

st.sidebar.header("2. Time Horizon")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

optimize_button = st.sidebar.button("Run Optimization")

# --- MAIN APP LOGIC ---
if optimize_button:
    # 1. Parse Tickers
    tickers = []
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'Ticker' in df.columns:
                tickers = df['Ticker'].dropna().astype(str).tolist()
                st.success(f"Successfully loaded {len(tickers)} tickers from Excel.")
            else:
                st.error("Error: Your Excel file must contain a column named 'Ticker'.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
            st.stop()
    else:
        tickers = [t.strip().upper() for t in manual_tickers.split(',') if t.strip()]

    if not tickers:
        st.warning("Please provide at least two valid tickers.")
        st.stop()

    # 2. Fetch Data
    with st.spinner(f"Fetching historical data for {len(tickers)} assets..."):
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        if data.empty:
            st.error("No data found. Please check your ticker symbols or date range.")
            st.stop()

        # Handle missing data (drop columns with too many NaNs, fill the rest)
        data = data.dropna(axis=1, thresh=int(len(data)*0.8)).fillna(method='ffill')

    # 3. Optimize Portfolio
    with st.spinner("Calculating optimal weights..."):
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        ef = EfficientFrontier(mu, S)
        raw_weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()

    # --- DASHBOARD VISUALS ---
    st.subheader("Optimization Results")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{ret*100:.2f}%")
    col2.metric("Annual Volatility (Risk)", f"{vol*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # Two-column layout for charts and data
    chart_col, data_col = st.columns([2, 1])

    with data_col:
        st.markdown("### Optimal Weights")
        weights_df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0].sort_values(by='Weight', ascending=False)
        
        # Format as percentage for display
        display_df = weights_df.copy()
        display_df['Weight'] = (display_df['Weight'] * 100).round(2).astype(str) + '%'
        st.dataframe(display_df, use_container_width=True)

    with chart_col:
        st.markdown("### Efficient Frontier")
        ef_plot = EfficientFrontier(mu, S)
        fig, ax = plt.subplots(figsize=(8, 5))
        plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
        ax.scatter(vol, ret, marker="*", s=200, c="r", label="Max Sharpe Portfolio")
        ax.set_title("") # Remove default title to keep it clean
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    
    st.markdown("### Asset Correlation Matrix")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    correlation_matrix = data.pct_change().corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax2, fmt=".2f")
    st.pyplot(fig2)