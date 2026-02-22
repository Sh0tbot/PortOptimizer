import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import tempfile
from fpdf import FPDF

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Pro Portfolio Optimizer", layout="wide", page_icon="📈")
sns.set_theme(style="whitegrid")

# --- SECURITY: PASSWORD PROTECTION ---
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Enterprise Portfolio Optimizer")
    st.text_input("Please enter your access password:", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect. Please try again.")
    return False

if not check_password():
    st.stop()

# --- PDF GENERATOR FUNCTION ---
def generate_pdf_report(weights_dict, ret, vol, sharpe, sortino, alpha, beta, fig_ef, fig_wealth, fig_mc):
    pdf = FPDF()
    
    # --- PAGE 1: Overview & Efficient Frontier ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Portfolio Optimization & Strategy Report", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="1. Core Performance Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(100, 8, txt=f"Expected Annual Return: {ret*100:.2f}%")
    pdf.cell(100, 8, txt=f"Annual Volatility (Risk): {vol*100:.2f}%", ln=True)
    pdf.cell(100, 8, txt=f"Sharpe Ratio: {sharpe:.2f}")
    pdf.cell(100, 8, txt=f"Sortino Ratio: {sortino:.2f}", ln=True)
    pdf.cell(100, 8, txt=f"Alpha: {alpha*100:.2f}%" if not np.isnan(alpha) else "Alpha: N/A")
    pdf.cell(100, 8, txt=f"Beta: {beta:.2f}" if not np.isnan(beta) else "Beta: N/A", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="2. Target Asset Allocation", ln=True)
    pdf.set_font("Arial", '', 11)
    for ticker, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
        if weight > 0.001:
            pdf.cell(200, 6, txt=f"{ticker}: {weight*100:.2f}%", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="3. Efficient Frontier Profile", ln=True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_ef:
        fig_ef.savefig(tmp_ef.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_ef.name, x=15, w=180)

    # --- PAGE 2: Backtest & Forecast ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="4. Historical Backtest ($10,000 Growth vs Benchmark)", ln=True)
    
    current_y = pdf.get_y()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_wealth:
        fig_wealth.savefig(tmp_wealth.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_wealth.name, x=15, y=current_y, w=180)
        
    pdf.set_y(current_y + 115) # Move cursor below the first image
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="5. Monte Carlo Forecast", ln=True)
    current_y2 = pdf.get_y()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_mc:
        fig_mc.savefig(tmp_mc.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_mc.name, x=15, y=current_y2, w=180)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            return f.read()

# --- HELPER FUNCTION: ASSET METADATA ---
def get_asset_metadata(ticker):
    asset_class, sector = 'Other', 'Unknown'
    try:
        info = yf.Ticker(ticker).info
        q_type, country, category = info.get('quoteType', '').upper(), info.get('country', 'Unknown').upper(), info.get('category', '').upper()
        sector_info = info.get('sector', '')
        
        if sector_info: sector = sector_info
        elif category: sector = category.title()

        if q_type in ['MUTUALFUND', 'ETF']:
            if 'BOND' in category or 'FIXED' in category: asset_class = 'Fixed Income'
            elif 'MONEY' in category or 'CASH' in category: asset_class = 'Cash & Equivalents'
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

# ==========================================
# --- APP HEADER ---
# ==========================================
st.title("📈 Pro Portfolio Optimizer & Forecaster")
st.markdown("Optimize allocations, forecast performance, and generate execution reports. Created by Nizar.")

if "optimized" not in st.session_state:
    st.session_state.optimized = False

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, GOOG, JNJ, SPY")
benchmark_ticker = st.sidebar.text_input("Benchmark:", "SPY")

st.sidebar.header("2. Historical Horizon")
time_range = st.sidebar.selectbox("Select Time Range", ("1 Year", "3 Years", "5 Years", "7 Years", "10 Years"), index=2)
end_date = pd.Timestamp.today()
start_date = end_date - pd.DateOffset(years=int(time_range.split()[0]))

st.sidebar.header("3. Strategy Settings")
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_w = st.sidebar.slider("Max Weight per Asset", 10, 100, 100, 5) / 100.0

st.sidebar.header("4. Trade & Forecast")
portfolio_value = st.sidebar.number_input("Total Portfolio Value ($)", min_value=1000, value=100000, step=1000)
mc_years = st.sidebar.slider("Monte Carlo Years", 1, 30, 10)
mc_sims = st.sidebar.selectbox("Simulations", (100, 500, 1000), index=1)

optimize_button = st.sidebar.button("Run Full Analysis", type="primary", use_container_width=True)

# --- MAIN APP LOGIC ---
if optimize_button:
    tickers = []
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            if 'Ticker' in df.columns: tickers = df['Ticker'].dropna().astype(str).tolist()
        except Exception: st.error("Failed to read Excel file."); st.stop()
    else:
        tickers = [t.strip().upper() for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2: st.warning("Provide at least two valid tickers."); st.stop()
    if max_w < (1.0 / len(tickers)): st.error(f"Constraint mathematically impossible."); st.stop()
        
    bench_clean = benchmark_ticker.strip().upper()
    all_tickers = list(set(tickers + [bench_clean]))

    with st.spinner("Downloading market data and categorizing assets..."):
        raw_data = yf.download(all_tickers, start=start_date, end=end_date)
        st.session_state.asset_meta = {t: get_asset_metadata(t) for t in tickers}
        
        if raw_data.empty: st.error("No data found."); st.stop()
        try: data = raw_data['Adj Close']
        except KeyError:
            try: data = raw_data['Close']
            except KeyError: st.error("Pricing columns not found."); st.stop()

        if isinstance(data, pd.Series): data = data.to_frame()
        data = data.dropna(axis=1, thresh=int(len(data)*0.8)).ffill().bfill()
        
        if bench_clean in data.columns:
            bench_data = data[bench_clean]
            port_data = data.drop(columns=[bench_clean], errors='ignore')
        else:
            bench_data = pd.Series(dtype=float)
            port_data = data

    # THE MISSING SAFETY CHECK
        if port_data.empty or len(port_data) < 2:
            st.error("Not enough trading days in this specific Time Range. Try selecting a longer Time Horizon.")
            st.stop()
            
        if port_data.shape[1] < 2:
            st.error("Not enough valid assets in this Time Range. At least 2 assets are required to optimize.")
            st.stop()

    with st.spinner("Crunching the math..."):
        mu = expected_returns.mean_historical_return(port_data)
        S = risk_models.sample_cov(port_data)
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
        raw_weights = ef.max_sharpe() if "Max Sharpe" in opt_metric else ef.min_volatility()
            
        st.session_state.opt_target = "Max Sharpe" if "Max Sharpe" in opt_metric else "Min Volatility"
        st.session_state.mu, st.session_state.S = mu, S
        st.session_state.cleaned_weights = ef.clean_weights()
        st.session_state.ret, st.session_state.vol, st.session_state.sharpe = ef.portfolio_performance()
        st.session_state.asset_list = list(mu.index)
        st.session_state.daily_returns = port_data.pct_change().dropna()
        st.session_state.bench_returns = bench_data.pct_change().dropna() if not bench_data.empty else None
        st.session_state.optimized = True

# --- DASHBOARD VISUALS ---
if st.session_state.optimized:
    st.markdown("---")
    
    # THE WHAT-IF SLIDER
    st.subheader(f"🎛️ Adjust Allocation ({st.session_state.opt_target} Baseline)")
    adj_col1, adj_col2 = st.columns([1, 2])
    with adj_col1:
        adj_asset = st.selectbox("Select Asset to Adjust:", st.session_state.asset_list)
        orig_w = st.session_state.cleaned_weights.get(adj_asset, 0.0)
        new_w = st.slider(f"Target Weight for {adj_asset}", 0.0, 100.0, float(orig_w*100), 1.0, format="%.0f%%") / 100.0
    
    custom_weights = st.session_state.cleaned_weights.copy()
    for t in st.session_state.asset_list:
        if t not in custom_weights: custom_weights[t] = 0.0
            
    old_rem, new_rem = 1.0 - orig_w, 1.0 - new_w
    for t in custom_weights:
        if t != adj_asset:
            if old_rem > 0: custom_weights[t] *= (new_rem / old_rem)
            else: custom_weights[t] = new_rem / (len(custom_weights) - 1)
    custom_weights[adj_asset] = new_w
    
    # MATH CALCULATIONS
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

    # --- TOP METRICS BOARD ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Exp. Return", f"{c_ret*100:.2f}%")
    m2.metric("Sharpe Ratio", f"{c_sharpe:.2f}")
    m3.metric("Sortino Ratio", f"{c_sortino:.2f}")
    if not np.isnan(c_beta):
        m4.metric("Beta (β)", f"{c_beta:.2f}"); m5.metric("Alpha (α)", f"{c_alpha*100:.2f}%")
    else: m4.metric("Beta", "N/A"); m5.metric("Alpha", "N/A")
    m6.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%")
    
    st.markdown("---")

    # ==========================================
    # --- PRE-GENERATE CHARTS FOR PDF ---
    # ==========================================
    # 1. Efficient Frontier
    ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S, weight_bounds=(0, max_w))
    fig_ef, ax_ef = plt.subplots(figsize=(6, 4))
    plotting.plot_efficient_frontier(ef_plot, ax=ax_ef, show_assets=True)
    ax_ef.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r", label=f"{st.session_state.opt_target}")
    ax_ef.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Custom Allocation")
    ax_ef.set_title("Efficient Frontier Profile")
    ax_ef.legend()

    # 2. Historical Backtest
    port_wealth = (1 + port_daily).cumprod() * 10000
    bench_wealth = (1 + st.session_state.bench_returns).cumprod() * 10000 if st.session_state.bench_returns is not None else None
    fig_wealth, ax_wealth = plt.subplots(figsize=(7, 4))
    ax_wealth.plot(port_wealth.index, port_wealth, label="Custom Portfolio", color='#1f77b4', linewidth=2)
    if bench_wealth is not None:
        bench_wealth_aligned = bench_wealth.reindex(port_wealth.index).ffill()
        ax_wealth.plot(port_wealth.index, bench_wealth_aligned, label=f"Benchmark", color='gray', alpha=0.7)
    ax_wealth.set_ylabel("Portfolio Value ($)")
    ax_wealth.legend()

    # 3. Monte Carlo Simulation
    np.random.seed(42)
    dt = 1
    sim_results = np.zeros((int(mc_sims), mc_years + 1))
    sim_results[:, 0] = portfolio_value
    for i in range(int(mc_sims)):
        Z = np.random.standard_normal(mc_years)
        growth_factors = np.exp((c_ret - (c_vol**2)/2)*dt + c_vol * np.sqrt(dt) * Z)
        sim_results[i, 1:] = portfolio_value * np.cumprod(growth_factors)
        
    final_values = sim_results[:, -1]
    median_val = np.percentile(final_values, 50)
    pct_10, pct_90 = np.percentile(final_values, 10), np.percentile(final_values, 90)
    
    fig_mc, ax_mc = plt.subplots(figsize=(8, 4))
    for i in range(min(100, int(mc_sims))): ax_mc.plot(sim_results[i, :], color='gray', alpha=0.1)
    ax_mc.plot(np.percentile(sim_results, 50, axis=0), color='#1f77b4', linewidth=2, label=f'Median: ${median_val:,.0f}')
    ax_mc.plot(np.percentile(sim_results, 10, axis=0), color='#d62728', linewidth=2, linestyle='--', label=f'Bear (10%): ${pct_10:,.0f}')
    ax_mc.plot(np.percentile(sim_results, 90, axis=0), color='#2ca02c', linewidth=2, linestyle='--', label=f'Bull (90%): ${pct_90:,.0f}')
    ax_mc.set_xlim(0, mc_years)
    ax_mc.set_ylabel("Projected Value ($)")
    ax_mc.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax_mc.legend()

    # --- PDF DOWNLOAD BUTTON (Now has access to all charts) ---
    st.markdown("### 📄 Export Strategy")
    pdf_bytes = generate_pdf_report(custom_weights, c_ret, c_vol, c_sharpe, c_sortino, c_alpha, c_beta, fig_ef, fig_wealth, fig_mc)
    st.download_button(
        label="Download Comprehensive PDF Report",
        data=pdf_bytes,
        file_name="Complete_Portfolio_Report.pdf",
        mime="application/pdf",
        type="primary"
    )
    st.markdown("---")

    # ==========================================
    # --- PRO TAPPED NAVIGATION ---
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio & Optimization", "💱 Trade Calculator", "📈 Historical Backtest", "🔮 Monte Carlo Forecast"])

    with tab1:
        col_charts1, col_charts2 = st.columns([1, 1])
        with col_charts1:
            st.markdown("**Efficient Frontier**")
            st.pyplot(fig_ef)
        with col_charts2:
            st.markdown("**Asset Correlation Matrix**")
            fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
            corr_matrix = st.session_state.daily_returns.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_corr, fmt=".2f", cbar=False)
            st.pyplot(fig_corr)

    with tab2:
        trade_data = []
        for t, w in custom_weights.items():
            if w > 0.001:
                a_class, a_sector = st.session_state.asset_meta.get(t, ('Other', 'Unknown'))
                trade_data.append({'Ticker': t, 'Weight': w, 'Dollar Value ($)': w * portfolio_value, 'Asset Class': a_class, 'Sector': a_sector})
                
        trade_df = pd.DataFrame(trade_data).sort_values(by='Weight', ascending=False).reset_index(drop=True)
        calc_col1, calc_col2 = st.columns([2, 1])
        
        with calc_col1:
            display_trade = trade_df.copy()
            display_trade['Weight'] = (display_trade['Weight'] * 100).round(2).astype(str) + '%'
            display_trade['Dollar Value ($)'] = display_trade['Dollar Value ($)'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_trade, use_container_width=True)
            csv = display_trade.to_csv(index=False)
            st.download_button("📥 Download Trade Sheet (CSV)", data=csv, file_name='trade_calculator.csv', mime='text/csv')
            
        with calc_col2:
            sector_totals = trade_df.groupby('Sector')['Weight'].sum()
            fig_sec, ax_sec = plt.subplots(figsize=(4, 4))
            ax_sec.pie(sector_totals, labels=sector_totals.index, autopct='%1.1f%%', colors=sns.color_palette("muted"))
            ax_sec.set_title("Sector Exposure")
            st.pyplot(fig_sec)

    with tab3:
        bt_col1, bt_col2 = st.columns(2)
        with bt_col1:
            st.markdown("**$10,000 Growth vs Benchmark**")
            st.pyplot(fig_wealth)
        with bt_col2:
            st.markdown("**Historical Drawdowns**")
            rolling_max = port_wealth.cummax()
            drawdown = (port_wealth - rolling_max) / rolling_max
            fig_dd, ax_dd = plt.subplots(figsize=(7, 4))
            ax_dd.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
            ax_dd.plot(drawdown.index, drawdown, color='#d62728', linewidth=1)
            ax_dd.set_ylabel("Drawdown (%)")
            ax_dd.set_yticklabels(['{:,.0%}'.format(x) for x in ax_dd.get_yticks()])
            st.pyplot(fig_dd)

    with tab4:
        mc_col1, mc_col2 = st.columns([3, 1])
        with mc_col1:
            st.markdown(f"**Projected Value ({mc_years} Years)**")
            st.pyplot(fig_mc)
        with mc_col2:
            st.info(f"**Bull Market (90th Pct):**\n${pct_90:,.2f}")
            st.success(f"**Median Expectation:**\n${median_val:,.2f}")
            st.error(f"**Bear Market (10th Pct):**\n${pct_10:,.2f}")

  # --- LEGAL DISCLAIMER ---
    st.markdown("---")
    with st.expander("⚠️ Legal Disclaimer & Terms of Use"):
        st.caption("""
        **Informational Purposes Only:** This application is provided for educational and informational purposes only. It does not constitute financial, investment, legal, or tax advice. 
        
        **No Guarantee of Accuracy:** The pricing data and asset metadata are sourced from free public APIs (Yahoo Finance) which may contain errors, omissions, or delays. The creator of this tool makes no representations or warranties regarding the accuracy or completeness of the data.
        
        **Inherent Risks:** Financial markets are volatile. The "Optimal" portfolios, Sharpe Ratios, and Monte Carlo forecasts are based purely on historical mathematical models. **Past performance is not indicative of future results.** The projections do not account for trading fees, slippage, taxes, or future market shocks. 
        
        **Use at Your Own Risk:** By using this tool, you acknowledge that you are solely responsible for your own investment decisions. The creator of this application accepts no liability whatsoever for any losses or damages arising from the use of this software or its outputs. Always consult with a licensed and registered financial advisor before making investment decisions.
        """)

