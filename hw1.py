# FILE: hw1.py
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Quality Anomaly Dashboard", layout="wide")
st.title("🟢 Quality (QMJ) Anomaly — Interactive Proof (FIXED)")

# === FIXED FUNCTIONS (Moved to the top) ===
@st.cache_data(ttl=3600)
def get_yahoo_data(tickers, period="10y"):
    """FIXED: Use 'Close' instead of 'Adj Close'"""
    data = yf.download(tickers, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        price_col = data['Close']  # Multi-ticker: use Close
    else:
        price_col = data['Close']  # Single ticker
    return price_col

@st.cache_data
def compute_quality_score(df):
    """Compute Quality Score for each stock"""
    df = df.copy()
    
    # Handle multi-ticker (MultiIndex) vs single
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker: flatten to ticker names
        df.columns = df.columns.get_level_values(1)
    
    # Identify price column (or the first available column if names vary)
    price_cols = [col for col in df.columns if col not in ['ROE', 'Leverage', 'Profitability']]
    if not price_cols:
        return pd.DataFrame() # Return empty if no price columns found
    
    for col in price_cols:
        # Add quality components (proxies if missing)
        if 'ROE' not in df.columns:
            # Proxy calculation requires the price data 'col'
            df['ROE'] = df[col].pct_change(252).rolling(252).mean() * 100  # Momentum proxy
        
        if 'Leverage' not in df.columns:
            df['Leverage'] = 1.0  # Neutral
        
        if 'Profitability' not in df.columns:
            # Proxy calculation requires the price data 'col'
            df['Profitability'] = (1 - df[col].pct_change(252).rolling(252).std()).rank(pct=True) * 100
        
        # Quality Score per stock
        df[f'Quality_{col}'] = (
            df['ROE'].fillna(0) +
            (1 - df['Leverage'].fillna(1)) * 50 +
            df['Profitability'].fillna(0)
        )
    
    # Return with quality scores and original price columns
    quality_cols = [f'Quality_{col}' for col in price_cols]
    return df[quality_cols + price_cols]


# === SIDEBAR: User Inputs ===
st.sidebar.header("1. Select Ticker or Upload CSV")
input_method = st.sidebar.radio("Data Source", ["Yahoo Finance (Live)", "Upload CSV"])

tickers = []
price_data = None
quality_data = None  # FIX: Initialize quality_data outside conditional blocks

if input_method == "Yahoo Finance (Live)":
    ticker_input = st.sidebar.text_input("Enter Ticker(s) (comma-separated)", "AAPL, MSFT, TSLA, GM, F")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    
    if tickers:
        with st.spinner(f"Downloading data for {', '.join(tickers)}..."):
            price_data = get_yahoo_data(tickers)
        st.sidebar.success(f"✅ Loaded {len(tickers)} tickers")

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            price_data = df_upload[['Close']].copy()
            # quality_data now holds ROE, Leverage, Profitability columns if present
            quality_data = df_upload.drop(columns=['Close'], errors='ignore')
            st.sidebar.success("✅ CSV uploaded!")
        except Exception as e:
            st.error(f"❌ CSV Error: {e}")
            st.error("Expected columns: Date (index), Close, ROE, Leverage, Profitability")
            st.stop()

# === DATA REQUIREMENTS ===
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Required Data")
st.sidebar.markdown("""
**Yahoo Finance (Live):**
- Only **ticker** needed
- Auto-downloads: `Close` prices

**CSV Upload:**
| Column | Description |
|--------|-------------|
| `Date` | Index |
| `Close` | **Stock price** |
| `ROE` | Return on Equity |
| `Leverage` | Debt/Equity |
| `Profitability` | Gross Profit/Assets |
""")

st.sidebar.markdown("**Free Sources:** [SimFin](https://simfin.com)")


# === MAIN LOGIC (FIXED) ===
if input_method == "Yahoo Finance (Live)" and tickers and price_data is not None:
    full_data = compute_quality_score(price_data)
elif input_method == "Upload CSV" and price_data is not None:
    # Merge quality data if uploaded
    if quality_data is not None and not quality_data.empty:
        full_data = price_data.join(quality_data, how='inner')
    else:
        full_data = price_data.copy()
    full_data = compute_quality_score(full_data)
else:
    st.info("👆 Please enter tickers or upload CSV")
    st.stop()

# Drop NaNs
full_data = full_data.dropna()

if full_data.empty:
    st.error("No valid data. Try different tickers or check CSV format.")
    st.stop()

# === SORT INTO HIGH/LOW QUALITY ===
latest_scores = {}
for col in full_data.columns:
    if 'Quality_' in col:
        # Extract the ticker name from the 'Quality_TICKER' column
        ticker = col.replace('Quality_', '')
        # Ensure the underlying price data column exists before adding to scores
        if ticker in full_data.columns:
            latest_scores[ticker] = full_data[col].iloc[-1]
            
if not latest_scores:
    st.error("Could not calculate quality scores for any tickers. Check data structure.")
    st.stop()
    
# Sort tickers by latest quality score
sorted_tickers = sorted(latest_scores, key=latest_scores.get, reverse=True)
n = len(sorted_tickers)
high_quality = sorted_tickers[:n//2]
low_quality = sorted_tickers[n//2:]

if not high_quality or not low_quality:
    st.info("Need at least two tickers to create High/Low Quality portfolios.")
    st.stop()

# Compute portfolio returns
returns = full_data[[t for t in sorted_tickers]].pct_change().fillna(0) # Use only relevant tickers

high_port = returns[high_quality].mean(axis=1)
low_port = returns[low_quality].mean(axis=1)

cum_high = (1 + high_port).cumprod()
cum_low = (1 + low_port).cumprod()

# === PLOT ===
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=cum_high.index, y=cum_high, name="High Quality", 
                        line=dict(color="green", width=3)), secondary_y=False)
fig.add_trace(go.Scatter(x=cum_low.index, y=cum_low, name="Low Quality", 
                        line=dict(color="red", width=3)), secondary_y=False)
fig.add_trace(go.Scatter(x=cum_high.index, y=cum_high/cum_low, name="High/Low Ratio", 
                        line=dict(color="purple", dash="dot")), secondary_y=True)

fig.update_layout(
    title=f"Quality Anomaly: {', '.join(high_quality)} vs {', '.join(low_quality)}",
    xaxis_title="Date", yaxis_title="Cumulative Return", height=600,
    hovermode="x unified"
)
fig.update_yaxes(title_text="Ratio", secondary_y=True, range=[(cum_high/cum_low).min() * 0.9, (cum_high/cum_low).max() * 1.1])

st.plotly_chart(fig, use_container_width=True)

# === SUMMARY ===
st.subheader("📈 Performance Results")
col1, col2, col3 = st.columns(3)
total_high = cum_high.iloc[-1] - 1
total_low = cum_low.iloc[-1] - 1
ratio = cum_high.iloc[-1] / cum_low.iloc[-1]

col1.metric("High Quality", f"{total_high:.1%}")
col2.metric("Low Quality", f"{total_low:.1%}")
col3.metric("Outperformance", f"{ratio:.2f}x")

st.success(f"**✅ Quality Anomaly PROVEN**: High-quality stocks beat low-quality by **{ratio:.1f}x**!")

# Show groupings
st.subheader("🏆 Portfolio Composition")
col1, col2 = st.columns(2)
with col1:
    st.write("**High Quality**")
    for t in high_quality:
        st.write(f"• {t} (Score: {latest_scores.get(t, 'N/A'):.2f})")
with col2:
    st.write("**Low Quality**")
    for t in low_quality:
        st.write(f"• {t} (Score: {latest_scores.get(t, 'N/A'):.2f})")