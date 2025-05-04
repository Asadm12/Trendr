import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# === Sidebar Content ===
st.sidebar.markdown("""
## 📊 ETF Portfolio Simulator

Analyze sector-based ETF combinations to assess which markets are evolving — and where startups may find growth opportunities.

---
👈 Use the sidebar to:
- Select ETFs
- Adjust their weights
- Simulate portfolio performance

---
💡 **Startup Tip**: Use sector behavior to identify where consumer or business shifts are strongest.
""")

# === ETF Selection ===
st.markdown("""
<h1 style='color:#00F5D4;'>🚀 Portfolio Signal Analyzer</h1>
<p style='color:#B0BEC5;'>Compare sector-based ETFs to understand startup-friendly market movements.</p>
""", unsafe_allow_html=True)

etf_list = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLB", "XLU", "XLRE", "XLC"]
etf_names = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication"
}

selected_etfs = st.multiselect("Select up to 5 ETFs", etf_list, default=["XLK", "XLV", "XLF"])
if len(selected_etfs) != len(set(selected_etfs)):
    st.warning("⚠️ Duplicate ETFs detected. Please select unique ETFs only.")
    st.stop()

# === Fetch and Normalize Prices ===
price_data = pd.DataFrame()
for etf in selected_etfs:
    df = yf.download(etf, period="1y", interval="1d", progress=False)
    if "Close" not in df.columns:
        continue
    df = df[["Close"]].copy()
    df.columns = [etf]
    price_data = pd.concat([price_data, df], axis=1)

price_data = price_data.dropna()
norm_prices = price_data.copy()
scaler = MinMaxScaler()
norm_prices[norm_prices.columns] = scaler.fit_transform(norm_prices[norm_prices.columns])

# === Portfolio Growth Chart ===
st.markdown("""
<h3 style='color:#00F5D4;'>📈 Normalized Portfolio Growth</h3>
<p style='color:#B0BEC5;'>Understand which sectors are leading the charge over time.</p>
""", unsafe_allow_html=True)

line_df = norm_prices.reset_index().melt(id_vars="Date")
fig = px.line(line_df, x="Date", y="value", color="variable",
              labels={"value": "Normalized Price", "variable": "ETF"})
fig.update_layout(template="plotly_dark", plot_bgcolor="#0A0E17", paper_bgcolor="#0A0E17")
st.plotly_chart(fig, use_container_width=True)

# === Correlation Matrix ===
st.markdown("""
<h3 style='color:#00F5D4;'>📊 Correlation Matrix</h3>
<p style='color:#B0BEC5;'>Explore inter-sector co-movement. Diversified startup bets often stem from weakly correlated sectors.</p>
""", unsafe_allow_html=True)

corr = norm_prices[selected_etfs].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                     aspect="auto", labels=dict(color="Correlation"))
fig_corr.update_layout(template="plotly_dark", plot_bgcolor="#0A0E17", paper_bgcolor="#0A0E17")
st.plotly_chart(fig_corr, use_container_width=True)

# === GPT-style Insight ===
with st.expander("🤖 GPT Insight: What Do These Charts Tell Us?"):
    best_etf = norm_prices.iloc[-1].idxmax()
    worst_etf = norm_prices.iloc[-1].idxmin()
    best_sector = etf_names.get(best_etf, best_etf)
    worst_sector = etf_names.get(worst_etf, worst_etf)

    st.markdown(f"""
    <div style='color:#E0E0E0;'>
    The strongest sector in terms of normalized growth is <b style='color:#00FF00'>{best_sector}</b>, indicating strong market confidence.
    This might be the ideal time for tech-enabled or innovation-driven startups in this area.

    On the other hand, <b style='color:#FF3366'>{worst_sector}</b> has underperformed comparatively. Startups in this space should focus on solving inefficiencies or pivot to underserved sub-sectors.

    The correlation matrix helps in spotting sectors that behave differently — a key strategy in minimizing startup ecosystem dependencies.
    </div>
    """, unsafe_allow_html=True)

# === Footer ===
st.markdown("""
<hr style='border-color:#333;'>
<p style='font-size:0.8em; color:#666;'>FinSight Startup Signal Platform — Not investment advice. Built for innovators and builders.</p>
""", unsafe_allow_html=True)
