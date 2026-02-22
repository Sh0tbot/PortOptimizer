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
st.set_page_config(page_title="Enterprise Portfolio Manager", layout="wide", page_icon="📈", initial_sidebar_state="expanded")
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

# --- CUSTOM CSS FOR SAAS AESTHETIC ---
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #1f77b4; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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

# --- HELPER FUNCTION: ASSET METADATA & DIVIDENDS ---
@st.cache_data(ttl=86400)
def get_asset_metadata(ticker):
    asset_class, sector, div_yield, mcap = 'Other', 'Unknown', 0.0, 1e9
    try:
        info = yf.Ticker(ticker).info
        q_type = info.get('quoteType', '').upper()
        country = info.get('country', 'Unknown').upper()
        category = info.get('category', '').upper()
        sector_info = info.get('sector', '')
        long_name = (info.get('longName') or '').upper()
        
        div_yield = info.get('trailingAnnualDividendYield', info.get('dividendYield', 0.0))
        if div_yield is None: div_yield = 0.0
        
        mcap = info.get('marketCap', info.get('totalAssets', 1e9))
        if mcap is None: mcap = 1e9

        if sector_info: sector = sector_info
        elif category: sector = category.title()

        is_fixed_income = any(word in long_name for word in ['BOND', 'FIXED INCOME', 'TREASURY', 'DEBT', 'YIELD']) or \
                          any(word in category for word in ['BOND', 'FIXED'])
        is_cash = any(word in category for word in ['MONEY', 'CASH']) or 'CASH' in long_name

        if q_type in ['MUTUALFUND', 'ETF']:
            if is_fixed_income: 
                asset_class, sector = 'Fixed Income', 'Bonds'
            elif is_cash: 
                asset_class = 'Cash & Equivalents'
            elif 'CANADA' in category or 'CANADA' in long_name or ticker.endswith('.TO'): 
                asset_class = 'Canadian Equities'
            elif 'FOREIGN' in category or 'EMERGING' in category or 'INTERNATIONAL' in category: 
                asset_class = 'International Equities'
            else: 
                asset_class = 'US Equities' 
        else:
            if country == 'CANADA' or ticker.endswith('.TO'): asset_class = 'Canadian Equities'
            elif country == 'UNITED STATES': asset_class = 'US Equities'
            elif country != 'UNKNOWN': asset_class = 'International Equities'
    except Exception:
        pass
    if asset_class == 'Other' and ticker.endswith('.TO'): asset_class = 'Canadian Equities'
    return asset_class, sector, div_yield, mcap

# --- PDF GENERATOR FUNCTION ---
def generate_pdf_report(weights_dict, ret, vol, sharpe, sortino, alpha, beta, port_yield, income, stress_results, rebalance_df, fig_ef, fig_wealth, fig_mc, is_bl=False, bench_label="Benchmark"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    title = "Portfolio Strategy & Execution Report" if not is_bl else "Portfolio Strategy Report (Black-Litterman)"
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="1. Core Performance & Income Metrics", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(95, 8, txt=f"Expected Annual Return: {ret*100:.2f}%")
    pdf.cell(95, 8, txt=f"Annual Volatility (Risk): {vol*100:.2f}%", ln=True)
    pdf.cell(95, 8, txt=f"Sharpe Ratio: {sharpe:.2f}")
    pdf.cell(95, 8, txt=f"Sortino Ratio: {sortino:.2f}", ln=True)
    pdf.cell(95, 8, txt=f"Alpha: {alpha*100:.2f}%" if not np.isnan(alpha) else "Alpha: N/A")
    pdf.cell(95, 8, txt=f"Beta: {beta:.2f}" if not np.isnan(beta) else "Beta: N/A", ln=True)
    pdf.cell(95, 8, txt=f"Portfolio Dividend Yield: {port_yield*100:.2f}%")
    pdf.cell(95, 8, txt=f"Proj. Annual Income: ${income:,.2f}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt=f"2. Historical Scenario Analysis ({bench_label})", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(80, 8, "Historical Event", border=1, align='C')
    pdf.cell(55, 8, "Portfolio Return", border=1, align='C')
    pdf.cell(55, 8, "Benchmark Return", border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 9)
    for res in stress_results:
        pdf.cell(80, 8, res['Event'], border=1)
        pdf.cell(55, 8, f"{res['Portfolio Return']*100:.2f}%" if pd.notnull(res['Portfolio Return']) else "N/A", border=1, align='C')
        pdf.cell(55, 8, f"{res['Benchmark Return']*100:.2f}%" if pd.notnull(res['Benchmark Return']) else "N/A", border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="3. Efficient Frontier Profile", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_ef:
        fig_ef.savefig(tmp_ef.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_ef.name, x=15, w=160)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="4. Target Allocation & Rebalancing Actions", ln=True)
    pdf.set_font("Arial", 'B', 9)
    pdf.cell(30, 8, "Ticker", border=1, align='C')
    pdf.cell(25, 8, "Target %", border=1, align='C')
    pdf.cell(40, 8, "Current Val ($)", border=1, align='C')
    pdf.cell(40, 8, "Target Val ($)", border=1, align='C')
    pdf.cell(50, 8, "Action Required", border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", '', 9)
    for _, row in rebalance_df.iterrows():
        pdf.cell(30, 8, str(row['Ticker']), border=1)
        pdf.cell(25, 8, f"{row['Weight']*100:.2f}%", border=1, align='C')
        pdf.cell(40, 8, f"${row['Current Value ($)']:,.2f}", border=1, align='R')
        pdf.cell(40, 8, f"${row['Target Value ($)']:,.2f}", border=1, align='R')
        pdf.cell(50, 8, str(row['Trade Action']), border=1, align='C')
        pdf.ln()
    pdf.ln(5)

    current_y = pdf.get_y()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt=f"5. Historical Backtest ($10,000 Growth vs {bench_label})", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_wealth:
        fig_wealth.savefig(tmp_wealth.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_wealth.name, x=15, w=160)
        
    pdf.ln(85)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="6. Monte Carlo Forecast", ln=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_mc:
        fig_mc.savefig(tmp_mc.name, format="png", bbox_inches="tight", dpi=150)
        pdf.image(tmp_mc.name, x=15, w=160)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            return f.read()

# ==========================================
# --- APP HEADER ---
# ==========================================
st.title("📈 Enterprise Portfolio Manager")
st.markdown("Optimize allocations, compare against current holdings, forecast income, and generate execution reports.")

if "optimized" not in st.session_state:
    st.session_state.optimized = False

# --- CONSTANTS ---
BENCH_MAP = {'US Equities': 'SPY', 'Canadian Equities': 'XIU.TO', 'International Equities': 'EFA', 'Fixed Income': 'AGG', 'Cash & Equivalents': 'BIL', 'Other': 'SPY'}

# --- SIDEBAR GUI ---
st.sidebar.header("1. Input Securities")
uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV File (Supports Current Weights)", type=["xlsx", "xls", "csv"])
manual_tickers = st.sidebar.text_input("Or enter tickers manually:", "AAPL, MSFT, GOOG, XIU.TO, XBB.TO")

autobench = st.sidebar.toggle("Auto-Bench by Asset Allocation", value=False, help="Dynamically creates a blended benchmark to match your custom asset allocation using proxy ETFs.")
if autobench:
    st.sidebar.info("📊 Benchmark: Dynamic Allocation Blend")
    benchmark_ticker = "AUTO"
else:
    benchmark_ticker = st.sidebar.text_input("Static Benchmark:", "SPY")

st.sidebar.header("2. Historical Horizon")
time_range = st.sidebar.selectbox("Select Time Range", ("1 Year", "3 Years", "5 Years", "7 Years", "10 Years", "Custom Dates"), index=2)

if time_range == "Custom Dates":
    col_d1, col_d2 = st.sidebar.columns(2)
    with col_d1: start_date = pd.to_datetime(st.date_input("Start Date", pd.Timestamp.today() - pd.DateOffset(years=5)))
    with col_d2: end_date = pd.to_datetime(st.date_input("End Date", pd.Timestamp.today()))
else:
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=int(time_range.split()[0]))

st.sidebar.header("3. Strategy Settings")
opt_metric = st.sidebar.selectbox("Optimize For:", ("Max Sharpe Ratio", "Minimum Volatility"))
max_w = st.sidebar.slider("Max Weight per Asset", 5, 100, 100, 5) / 100.0

st.sidebar.header("4. Black-Litterman (Views)")
use_bl = st.sidebar.toggle("Enable Black-Litterman Model")
bl_views_input = ""
if use_bl:
    bl_views_input = st.sidebar.text_input("Enter target returns (e.g., AAPL:0.15, SPY:-0.05)")

st.sidebar.header("5. Trade & Forecast")
portfolio_value = st.sidebar.number_input("Total Portfolio Target Value ($)", min_value=1000, value=100000, step=1000)
mc_years = st.sidebar.slider("Monte Carlo Years", 1, 30, 10)
mc_sims = st.sidebar.selectbox("Simulations", (100, 500, 1000), index=1)

optimize_button = st.sidebar.button("Run Full Analysis", type="primary", use_container_width=True)

# --- MAIN APP LOGIC ---
if optimize_button:
    tickers = []
    st.session_state.imported_weights = None
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            
            # --- CUSTOM EXCEL PARSER FOR CURRENT PORTFOLIO COMPARISON ---
            if 'Symbol' in df.columns and 'MV (%)' in df.columns:
                def parse_ticker(row):
                    t = str(row['Symbol']).strip().upper()
                    r = str(row.get('Region', '')).strip().upper()
                    if r == 'CA' and not t.endswith('.TO') and not t.endswith('.V'):
                        if len(t) > 5 and any(char.isdigit() for char in t): pass # Mutual Fund protection
                        else: t = t.replace('.', '-') + '.TO' # Fix dual class shares
                    return t
                
                df['Clean_Ticker'] = df.apply(parse_ticker, axis=1)
                agg_df = df.groupby('Clean_Ticker')['MV (%)'].sum().reset_index()
                
                agg_df['MV (%)'] = agg_df['MV (%)'] / 100.0
                agg_df['MV (%)'] = agg_df['MV (%)'] / agg_df['MV (%)'].sum()
                
                tickers = agg_df['Clean_Ticker'].tolist()
                st.session_state.imported_weights = dict(zip(agg_df['Clean_Ticker'], agg_df['MV (%)']))
                
                if 'Market Value' in df.columns:
                    portfolio_value = float(df['Market Value'].sum())
            elif 'Ticker' in df.columns:
                tickers = df['Ticker'].dropna().astype(str).tolist()
        except Exception: 
            st.error("Failed to read imported file. Ensure it has 'Symbol' and 'MV (%)' columns."); st.stop()
    else:
        tickers = [t.strip().upper() for t in manual_tickers.replace(' ', ',').split(',') if t.strip()]

    if len(tickers) < 2: st.warning("Provide at least two valid tickers."); st.stop()
    if max_w < (1.0 / len(tickers)): st.error("Constraint mathematically impossible."); st.stop()
    if start_date >= end_date: st.error("Start date must be before end date."); st.stop()
        
    bench_clean = benchmark_ticker.strip().upper()
    if autobench: all_tickers = list(set(tickers + list(BENCH_MAP.values())))
    else: all_tickers = list(set(tickers + [bench_clean]))

    with st.spinner("Validating symbols and auto-correcting exchanges..."):
        invalid_tickers = []
        corrected_tickers = {}
        
        for t in all_tickers:
            if yf.Ticker(t).history(period="1mo").empty: 
                # AUTO-CORRECT: If TSX (.TO) fails, try NEO Exchange (.NE)
                if t.endswith('.TO'):
                    ne_t = t.replace('.TO', '.NE')
                    if not yf.Ticker(ne_t).history(period="1mo").empty:
                        corrected_tickers[t] = ne_t
                        continue
                invalid_tickers.append(t)
                
        # APPLY NEO EXCHANGE CORRECTIONS
        if corrected_tickers:
            st.toast(f"🔄 Auto-corrected NEO Exchange ETFs: {', '.join([f'{k} → {v}' for k,v in corrected_tickers.items()])}")
            all_tickers = [corrected_tickers.get(t, t) for t in all_tickers]
            tickers = [corrected_tickers.get(t, t) for t in tickers]
            if st.session_state.imported_weights:
                for old_t, new_t in corrected_tickers.items():
                    if old_t in st.session_state.imported_weights:
                        st.session_state.imported_weights[new_t] = st.session_state.imported_weights.pop(old_t)
                
        # HANDLE REMAINING INVALID TICKERS (Like Mutual Funds)
        if invalid_tickers:
            mf_suspects = [t for t in invalid_tickers if len(t) >= 5 and any(c.isdigit() for c in t)]
            if mf_suspects:
                st.warning(f"⚠️ **Canadian Mutual Funds Detected:** {', '.join(mf_suspects)}")
                st.caption("Free APIs do not track Canadian Mutual Funds. Please replace them with Proxy ETFs (e.g., use `XBB.TO` for a Bond Fund) in your CSV and re-upload.")
            
            other_invalid = [t for t in invalid_tickers if t not in mf_suspects]
            if other_invalid:
                st.warning(f"⚠️ Omitting unreadable or delisted symbols: **{', '.join(other_invalid)}**")
            
            # Drop them but keep the app running for comparative analysis
            all_tickers = [t for t in all_tickers if t not in invalid_tickers]
            tickers = [t for t in tickers if t not in invalid_tickers]
            
            if st.session_state.imported_weights:
                for t in invalid_tickers: st.session_state.imported_weights.pop(t, None)
                tot_w = sum(st.session_state.imported_weights.values())
                if tot_w > 0: st.session_state.imported_weights = {k: v/tot_w for k,v in st.session_state.imported_weights.items()}

        if len(tickers) < 2: 
            st.error("Not enough valid tickers remaining to optimize.")
            st.stop()
            
        # FORCE DEEP DOWNLOAD FOR STRESS TESTS (Back to 2007 for GFC)
        fetch_start = min(start_date, pd.to_datetime("2007-01-01"))
        st.session_state.full_historical_data = yf.download(all_tickers, start=fetch_start, end=end_date)
        
        get_asset_metadata.clear()
        st.session_state.asset_meta = {t: get_asset_metadata(t) for t in tickers}
        
        if st.session_state.full_historical_data.empty: st.error("No data found."); st.stop()
            
        raw_data = st.session_state.full_historical_data
        try: data = raw_data['Adj Close']
        except KeyError:
            try: data = raw_data['Close']
            except KeyError: st.error("Pricing columns not found."); st.stop()

        if isinstance(data, pd.Series): 
            data = data.to_frame()
            if len(all_tickers) == 1: data.columns = [all_tickers[0]]

        data = data.dropna(axis=1, thresh=int(len(data)*0.8)).ffill().bfill()
        
        opt_data = data.loc[start_date:end_date]
        valid_tickers = [t for t in tickers if t in opt_data.columns]
        port_data = opt_data[valid_tickers]
        
        if autobench:
            st.session_state.proxy_data = data[[p for p in BENCH_MAP.values() if p in data.columns]]
            bench_data = pd.Series(dtype=float)
        elif bench_clean in opt_data.columns:
            bench_data = opt_data[bench_clean]
        else:
            bench_data = pd.Series(dtype=float)
            
        if port_data.empty or len(port_data) < 2:
            st.error("Not enough trading days/assets in this Time Range."); st.stop()

    with st.spinner("Crunching optimization matrices..."):
        mu = expected_returns.mean_historical_return(port_data)
        S = risk_models.sample_cov(port_data)
        
        if use_bl:
            from pypfopt import black_litterman, BlackLittermanModel
            views_dict = {}
            if bl_views_input.strip():
                for item in bl_views_input.split(','):
                    if ':' in item:
                        t, v = item.split(':')
                        try: views_dict[t.strip().upper()] = float(v.strip())
                        except ValueError: pass
            
            mcaps = {t: st.session_state.asset_meta[t][3] for t in port_data.columns if t in st.session_state.asset_meta}
            try: delta = black_litterman.market_implied_risk_aversion(bench_data) if not bench_data.empty else 2.5
            except Exception: delta = 2.5
                
            market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
            if views_dict:
                bl = BlackLittermanModel(S, pi=market_prior, absolute_views=views_dict)
                mu = bl.bl_returns()
                S = bl.bl_cov()
            else: mu = market_prior
            st.session_state.opt_target = f"Black-Litterman ({'Max Sharpe' if 'Max Sharpe' in opt_metric else 'Min Vol'})"
        else:
            st.session_state.opt_target = "Max Sharpe" if "Max Sharpe" in opt_metric else "Min Volatility"

        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
        raw_weights = ef.max_sharpe() if "Max Sharpe" in opt_metric else ef.min_volatility()
            
        st.session_state.mu, st.session_state.S = mu, S
        st.session_state.cleaned_weights = ef.clean_weights()
        st.session_state.ret, st.session_state.vol, st.session_state.sharpe = ef.portfolio_performance()
        st.session_state.asset_list = list(mu.index)
        st.session_state.daily_returns = port_data.pct_change().dropna()
        
        st.session_state.bench_returns_static = bench_data.pct_change().dropna() if not bench_data.empty else None
        st.session_state.stress_data = data
        st.session_state.bench_clean = bench_clean
        st.session_state.is_bl = use_bl
        st.session_state.autobench = autobench
        st.session_state.portfolio_value_target = portfolio_value
        st.session_state.optimized = True

# --- DASHBOARD VISUALS ---
if st.session_state.optimized:
    st.markdown("---")
    
    with st.container():
        st.subheader(f"🎛️ Adjust Target Allocation ({st.session_state.opt_target} Baseline)")
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
    
    # CALCULATE CUSTOM / OPTIMIZED PORTFOLIO
    w_array = np.array([custom_weights[t] for t in st.session_state.asset_list])
    c_ret = np.dot(w_array, st.session_state.mu.values)
    c_vol = np.sqrt(np.dot(w_array.T, np.dot(st.session_state.S.values, w_array)))
    risk_free_rate = 0.02 
    c_sharpe = (c_ret - risk_free_rate) / c_vol
    
    port_daily = st.session_state.daily_returns.dot(w_array)
    downside_returns = port_daily[port_daily < 0]
    down_stdev = np.sqrt(252) * downside_returns.std()
    c_sortino = (c_ret - risk_free_rate) / down_stdev if down_stdev > 0 else 0
    
    port_yield = sum(custom_weights[t] * st.session_state.asset_meta.get(t, ('', '', 0.0, 1e9))[2] for t in custom_weights)
    proj_income = port_yield * st.session_state.portfolio_value_target

    # CALCULATE BASELINE / CURRENT IMPORTED PORTFOLIO (If Available)
    curr_ret, curr_vol, curr_sharpe, curr_sortino, curr_yield, curr_income = 0, 0, 0, 0, 0, 0
    if st.session_state.imported_weights:
        curr_w_array = np.array([st.session_state.imported_weights.get(t, 0.0) for t in st.session_state.asset_list])
        curr_ret = np.dot(curr_w_array, st.session_state.mu.values)
        curr_vol = np.sqrt(np.dot(curr_w_array.T, np.dot(st.session_state.S.values, curr_w_array)))
        curr_sharpe = (curr_ret - risk_free_rate) / curr_vol if curr_vol > 0 else 0
        
        curr_port_daily = st.session_state.daily_returns.dot(curr_w_array)
        curr_downside = curr_port_daily[curr_port_daily < 0]
        curr_down_stdev = np.sqrt(252) * curr_downside.std()
        curr_sortino = (curr_ret - risk_free_rate) / curr_down_stdev if curr_down_stdev > 0 else 0
        
        curr_yield = sum(st.session_state.imported_weights.get(t, 0.0) * st.session_state.asset_meta.get(t, ('', '', 0.0, 1e9))[2] for t in st.session_state.asset_list)
        curr_income = curr_yield * st.session_state.portfolio_value_target

    # CALCULATE BENCHMARK DYNAMICS
    if st.session_state.autobench:
        ac_weights = {'US Equities': 0.0, 'Canadian Equities': 0.0, 'International Equities': 0.0, 'Fixed Income': 0.0, 'Cash & Equivalents': 0.0, 'Other': 0.0}
        for t, w in custom_weights.items():
            meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
            ac_weights[meta[0]] += w
            
        proxy_returns = st.session_state.proxy_data.pct_change().dropna()
        aligned_proxies = proxy_returns.reindex(port_daily.index).fillna(0)
        
        bench_daily = pd.Series(0.0, index=port_daily.index)
        for ac, w in ac_weights.items():
            if w > 0:
                proxy_ticker = BENCH_MAP[ac]
                if proxy_ticker in aligned_proxies.columns:
                    proxy_data = aligned_proxies[proxy_ticker]
                    if isinstance(proxy_data, pd.DataFrame): proxy_data = proxy_data.iloc[:, 0]
                    bench_daily = bench_daily + (proxy_data * w)
                    
        active_bench_returns = bench_daily
        bench_label = "Auto-Blended Benchmark"
    else:
        active_bench_returns = st.session_state.bench_returns_static
        bench_label = st.session_state.bench_clean

    c_beta, c_alpha = np.nan, np.nan
    if active_bench_returns is not None and not active_bench_returns.empty:
        aligned_data = pd.concat([port_daily, active_bench_returns], axis=1).dropna()
        if len(aligned_data) > 0:
            p_ret, b_ret = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
            cov_matrix = np.cov(p_ret, b_ret)
            c_beta = cov_matrix[0, 1] / cov_matrix[1, 1]
            c_alpha = c_ret - (risk_free_rate + c_beta * ((b_ret.mean() * 252) - risk_free_rate))

    st.markdown("---")
    
    title_text = "### 📊 Strategy Performance Overview" if not st.session_state.imported_weights else "### 📊 Target vs Current Portfolio Performance"
    st.markdown(title_text)
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    if st.session_state.imported_weights:
        kpi_col1.metric("Exp. Return", f"{c_ret*100:.2f}%", f"{(c_ret - curr_ret)*100:.2f}% vs Current")
        kpi_col2.metric("Sharpe Ratio", f"{c_sharpe:.2f}", f"{(c_sharpe - curr_sharpe):.2f} vs Current")
        kpi_col3.metric("Dividend Yield", f"{port_yield*100:.2f}%", f"{(port_yield - curr_yield)*100:.2f}% vs Current")
        kpi_col4.metric("Annual Income", f"${proj_income:,.2f}", f"${proj_income - curr_income:,.2f} vs Current")
    else:
        kpi_col1.metric("Exp. Return", f"{c_ret*100:.2f}%")
        kpi_col2.metric("Sharpe Ratio", f"{c_sharpe:.2f}")
        kpi_col3.metric("Dividend Yield", f"{port_yield*100:.2f}%")
        kpi_col4.metric("Annual Income", f"${proj_income:,.2f}")
    
    kpi_col5, kpi_col6, kpi_col7, kpi_col8 = st.columns(4)
    if st.session_state.imported_weights:
        # Lower risk is better, so if optimized risk is lower, show green arrow by multiplying by -1 for the UI delta
        kpi_col5.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%", f"{(c_vol - curr_vol)*100:.2f}% vs Current", delta_color="inverse")
        kpi_col6.metric("Sortino Ratio", f"{c_sortino:.2f}", f"{(c_sortino - curr_sortino):.2f} vs Current")
    else:
        kpi_col5.metric("Std Dev (Risk)", f"{c_vol*100:.2f}%")
        kpi_col6.metric("Sortino Ratio", f"{c_sortino:.2f}")
        
    if not np.isnan(c_alpha):
        kpi_col7.metric("Alpha (α)", f"{c_alpha*100:.2f}%")
        kpi_col8.metric("Beta (β)", f"{c_beta:.2f}")
    else:
        kpi_col7.metric("Alpha", "N/A"); kpi_col8.metric("Beta", "N/A")
    
    if st.session_state.autobench:
        st.caption(f"**Current Benchmark Blend:** " + ", ".join([f"{BENCH_MAP[k]} ({v*100:.1f}%)" for k,v in ac_weights.items() if v > 0.01]))
    
    st.markdown("<br>", unsafe_allow_html=True)

    ef_plot = EfficientFrontier(st.session_state.mu, st.session_state.S, weight_bounds=(0, max_w))
    fig_ef, ax_ef = plt.subplots(figsize=(10, 5))
    plotting.plot_efficient_frontier(ef_plot, ax=ax_ef, show_assets=True)
    ax_ef.scatter(st.session_state.vol, st.session_state.ret, marker="*", s=200, c="r", label=f"{st.session_state.opt_target}")
    ax_ef.scatter(c_vol, c_ret, marker="o", s=150, c="b", edgecolors='black', label="Custom Allocation")
    if st.session_state.imported_weights:
        ax_ef.scatter(curr_vol, curr_ret, marker="X", s=150, c="green", edgecolors='black', label="Current Allocation")
    ax_ef.set_title("Efficient Frontier Profile")
    ax_ef.legend()

    port_wealth = (1 + port_daily).cumprod() * 10000
    bench_wealth = (1 + active_bench_returns).cumprod() * 10000 if active_bench_returns is not None else None
    
    fig_wealth, ax_wealth = plt.subplots(figsize=(10, 5))
    ax_wealth.plot(port_wealth.index, port_wealth, label="Target Portfolio", color='#1f77b4', linewidth=2)
    if st.session_state.imported_weights:
        curr_wealth = (1 + curr_port_daily).cumprod() * 10000
        ax_wealth.plot(curr_wealth.index, curr_wealth, label="Current Portfolio", color='green', linewidth=2, linestyle='--')
    if bench_wealth is not None:
        bench_wealth_aligned = bench_wealth.reindex(port_wealth.index).ffill()
        ax_wealth.plot(port_wealth.index, bench_wealth_aligned, label=bench_label, color='gray', alpha=0.7)
    ax_wealth.set_ylabel("Portfolio Value ($)")
    ax_wealth.legend()

    np.random.seed(42)
    dt = 1
    sim_results = np.zeros((int(mc_sims), mc_years + 1))
    sim_results[:, 0] = st.session_state.portfolio_value_target
    for i in range(int(mc_sims)):
        Z = np.random.standard_normal(mc_years)
        growth_factors = np.exp((c_ret - (c_vol**2)/2)*dt + c_vol * np.sqrt(dt) * Z)
        sim_results[i, 1:] = st.session_state.portfolio_value_target * np.cumprod(growth_factors)
        
    final_values = sim_results[:, -1]
    median_val = np.percentile(final_values, 50)
    pct_10, pct_90 = np.percentile(final_values, 10), np.percentile(final_values, 90)
    
    fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
    for i in range(min(100, int(mc_sims))): ax_mc.plot(sim_results[i, :], color='gray', alpha=0.1)
    ax_mc.plot(np.percentile(sim_results, 50, axis=0), color='#1f77b4', linewidth=2, label=f'Median: ${median_val:,.0f}')
    ax_mc.plot(np.percentile(sim_results, 10, axis=0), color='#d62728', linewidth=2, linestyle='--', label=f'Bear (10%): ${pct_10:,.0f}')
    ax_mc.plot(np.percentile(sim_results, 90, axis=0), color='#2ca02c', linewidth=2, linestyle='--', label=f'Bull (90%): ${pct_90:,.0f}')
    ax_mc.set_xlim(0, mc_years)
    ax_mc.set_ylabel("Projected Value ($)")
    ax_mc.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax_mc.legend()

    stress_events = {
        "2008 Financial Crisis (Oct '07 - Mar '09)": ("2007-10-09", "2009-03-09"),
        "2018 Q4 Selloff (Sep '18 - Dec '18)": ("2018-09-20", "2018-12-24"),
        "COVID-19 Crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
        "2022 Bear Market (Jan - Oct 2022)": ("2022-01-03", "2022-10-12")
    }
    stress_results = []
    hist_data = st.session_state.stress_data
    for event_name, (s_date, e_date) in stress_events.items():
        try:
            window_data = hist_data.loc[s_date:e_date]
            if not window_data.empty and len(window_data) > 5:
                asset_rets = (window_data.iloc[-1] / window_data.iloc[0]) - 1
                p_ret = sum(custom_weights.get(t, 0) * asset_rets.get(t, 0) for t in custom_weights)
                
                if st.session_state.autobench:
                    b_ret = 0.0
                    for ac, w in ac_weights.items():
                        proxy = BENCH_MAP[ac]
                        if proxy in asset_rets and pd.notnull(asset_rets[proxy]):
                            b_ret += asset_rets[proxy] * w
                else:
                    b_ret = asset_rets.get(st.session_state.bench_clean, np.nan) if st.session_state.bench_clean in asset_rets else np.nan
                    
                stress_results.append({'Event': event_name, 'Portfolio Return': p_ret, 'Benchmark Return': b_ret})
            else:
                stress_results.append({'Event': event_name, 'Portfolio Return': np.nan, 'Benchmark Return': np.nan})
        except Exception: pass

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Allocation & Risk", "⚖️ Rebalancing", "📉 Stress Tests", "📈 Backtest", "🔮 Monte Carlo"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        ac_totals, sec_totals = {}, {}
        for t, w in custom_weights.items():
            if w > 0.001:
                meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
                ac_totals[meta[0]] = ac_totals.get(meta[0], 0) + w
                sec_totals[meta[1]] = sec_totals.get(meta[1], 0) + w
                
        pie_col1, pie_col2, pie_col3 = st.columns(3)
        with pie_col1:
            st.markdown("**Target Asset Class**")
            fig_ac, ax_ac = plt.subplots(figsize=(6, 6))
            ax_ac.pie(ac_totals.values(), labels=ac_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("pastel"))
            st.pyplot(fig_ac, use_container_width=True, clear_figure=True)
        with pie_col2:
            st.markdown("**Target Sector Exposure**")
            fig_sec, ax_sec = plt.subplots(figsize=(6, 6))
            ax_sec.pie(sec_totals.values(), labels=sec_totals.keys(), autopct='%1.1f%%', colors=sns.color_palette("muted"))
            st.pyplot(fig_sec, use_container_width=True, clear_figure=True)
        with pie_col3:
            st.markdown("**Asset Correlation Matrix**")
            corr_matrix = st.session_state.daily_returns.corr()
            num_assets = len(corr_matrix.columns)
            
            # DYNAMIC RENDERING: Scale font sizes and toggle numbers based on asset count
            show_numbers = num_assets <= 12
            font_size = max(6, 10 - (num_assets // 8))
            
            fig_corr, ax_corr = plt.subplots(figsize=(7, 6))
            sns.heatmap(
                corr_matrix, 
                annot=show_numbers, 
                cmap='coolwarm', 
                vmin=-1, vmax=1, 
                ax=ax_corr, 
                fmt=".2f", 
                cbar=not show_numbers, # Turn on color legend if numbers are hidden
                annot_kws={"size": 9},
                xticklabels=True, 
                yticklabels=True
            )
            
            # Rotate labels so ticker symbols don't crash into each other
            ax_corr.tick_params(axis='x', rotation=90, labelsize=font_size)
            ax_corr.tick_params(axis='y', rotation=0, labelsize=font_size)
            
            st.pyplot(fig_corr, use_container_width=True, clear_figure=True)
            
        st.markdown("---")
        st.markdown("**The Efficient Frontier**")
        st.pyplot(fig_ef, use_container_width=True, clear_figure=False)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        rebal_data = []
        for t, w in custom_weights.items():
            if w > 0.001:
                meta = st.session_state.asset_meta.get(t, ('Other', 'Unknown', 0.0, 1e9))
                rebal_data.append({
                    'Ticker': t, 'Weight': w, 'Target Value ($)': w * st.session_state.portfolio_value_target, 
                    'Asset Class': meta[0], 'Sector': meta[1], 'Yield': f"{meta[2]*100:.2f}%"
                })
        trade_df = pd.DataFrame(rebal_data).sort_values(by='Weight', ascending=False).reset_index(drop=True)
        
        # Populate the Current Value list automatically if we imported weights
        current_vals = [0.0]*len(trade_df)
        if st.session_state.imported_weights:
            for idx, row in trade_df.iterrows():
                t = row['Ticker']
                curr_w = st.session_state.imported_weights.get(t, 0.0)
                current_vals[idx] = curr_w * st.session_state.portfolio_value_target

        editable_df = pd.DataFrame({'Ticker': trade_df['Ticker'], 'Current Value ($)': current_vals})
        edited_df = st.data_editor(editable_df, hide_index=True, use_container_width=True)
        
        merged_df = pd.merge(trade_df, edited_df, on='Ticker', how='left')
        merged_df['Action ($)'] = merged_df['Target Value ($)'] - merged_df['Current Value ($)']
        merged_df['Trade Action'] = merged_df['Action ($)'].apply(lambda x: f"BUY ${x:,.2f}" if x > 1 else (f"SELL ${abs(x):,.2f}" if x < -1 else "HOLD"))
        
        st.markdown("**Final Execution List:**")
        display_trade = merged_df[['Ticker', 'Asset Class', 'Yield', 'Current Value ($)', 'Target Value ($)', 'Trade Action']].copy()
        display_trade['Target Value ($)'] = display_trade['Target Value ($)'].apply(lambda x: f"${x:,.2f}")
        display_trade['Current Value ($)'] = display_trade['Current Value ($)'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(display_trade, use_container_width=True)

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        stress_df = pd.DataFrame(stress_results)
        if not stress_df.empty:
            display_stress = stress_df.copy()
            display_stress['Portfolio Return'] = display_stress['Portfolio Return'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            display_stress['Benchmark Return'] = display_stress['Benchmark Return'].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
            st.table(display_stress)
        else:
            st.info("Insufficient historical data to run stress tests.")

    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(fig_wealth, use_container_width=True, clear_figure=False)
        st.markdown("---")
        st.markdown("**Historical Drawdowns**")
        rolling_max = port_wealth.cummax()
        drawdown = (port_wealth - rolling_max) / rolling_max
        fig_dd, ax_dd = plt.subplots(figsize=(10, 4))
        ax_dd.fill_between(drawdown.index, drawdown, 0, color='#d62728', alpha=0.3)
        ax_dd.plot(drawdown.index, drawdown, color='#d62728', linewidth=1)
        ax_dd.set_ylabel("Drawdown (%)")
        ax_dd.set_yticklabels(['{:,.0%}'.format(x) for x in ax_dd.get_yticks()])
        st.pyplot(fig_dd, use_container_width=True, clear_figure=True)

    with tab5:
        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(fig_mc, use_container_width=True, clear_figure=False)
        mc_col1, mc_col2, mc_col3 = st.columns(3)
        mc_col1.error(f"**Bear Market (10th Pct):**\n${pct_10:,.2f}")
        mc_col2.success(f"**Median Expectation:**\n${median_val:,.2f}")
        mc_col3.info(f"**Bull Market (90th Pct):**\n${pct_90:,.2f}")

    # --- 4. EXPORT AND LEGAL ---
    st.markdown("---")
    pdf_bytes = generate_pdf_report(custom_weights, c_ret, c_vol, c_sharpe, c_sortino, c_alpha, c_beta, port_yield, proj_income, stress_results, merged_df, fig_ef, fig_wealth, fig_mc, st.session_state.is_bl, bench_label)
    st.download_button(
        label="📄 Download Comprehensive Client PDF",
        data=pdf_bytes,
        file_name="Complete_Portfolio_Execution_Plan.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )

    st.markdown("---")
    with st.expander("⚠️ Legal Disclaimer & Terms of Use"):
        st.caption("""**Informational Purposes Only:** This software is provided for educational and illustrative purposes. The creator accepts no liability for investment decisions. Past performance is not indicative of future results.""")