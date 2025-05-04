import streamlit as st
from utils.sentiment import fetch_news, analyze_sentiment
from streamlit_extras.stylable_container import stylable_container
import plotly.express as px
import pandas as pd

# === CONFIG ===
st.set_page_config(page_title="🧠 Sector Sentiment Scanner", layout="wide")

# === PAGE HEADER ===
st.markdown("""
    <h1 style='color:#00F5D4; font-family:Orbitron; text-align:center;'>🧠 Sector Sentiment Scanner</h1>
    <p style='text-align:center; color:#B0BEC5;'>Real-time news analysis to gauge public mood across startup sectors.</p>
""", unsafe_allow_html=True)

# === API KEY (Ideally keep secure) ===
API_KEY = "fa2967e49bcf404dbc92943bc214e797"  # For testing only

# === Sidebar – Sector selection ===
with st.sidebar:
    st.markdown("## 🔎 Choose a Sector")
    sector = st.selectbox("Which sector’s sentiment do you want to analyze?", [
        "Technology", "Healthcare", "Financials", "Energy", "Consumer Discretionary",
        "Industrials", "Utilities", "Real Estate", "Materials", "Communication"
    ])
    max_articles = st.slider("📰 Number of News Headlines", 5, 20, 10)

    st.markdown("---")
    st.info("This page uses NLP to analyze the latest news about your selected sector in real-time.")

# === Main area – Sentiment results ===
with st.spinner("Fetching live news and analyzing sentiment..."):
    try:
        news = fetch_news(sector, api_key=API_KEY, max_articles=max_articles)
        results = []

        for title, desc in news:
            full_text = f"{title}. {desc}"
            sentiment = analyze_sentiment(full_text)
            results.append({
                "Headline": title,
                "Description": desc,
                "Compound": sentiment["compound"],
                "Positive": sentiment["pos"],
                "Neutral": sentiment["neu"],
                "Negative": sentiment["neg"]
            })

        df = pd.DataFrame(results)

        st.markdown(f"### 📰 Latest Headlines for **{sector}**")
        st.dataframe(df[["Headline", "Compound", "Positive", "Neutral", "Negative"]], use_container_width=True)

        # === Sentiment Distribution Chart ===
        st.markdown("### 📊 Sentiment Distribution")
        fig = px.histogram(df, x="Compound", nbins=10, color_discrete_sequence=["#00F5D4"])
        fig.update_layout(template="plotly_dark", height=400, xaxis_title="Sentiment Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # === Aggregated Score Card ===
        avg_score = df["Compound"].mean()
        st.markdown("### 📈 Overall Sector Sentiment")

        with stylable_container(
            key="sentiment_summary",
            css_styles="""
                {
                    background-color: rgba(0, 245, 212, 0.05);
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid #00F5D4;
                    text-align: center;
                    margin-top: 20px;
                }
            """
        ):
            emoji = "😊" if avg_score > 0.1 else "😐" if avg_score > -0.1 else "😟"
            sentiment_tag = "Positive" if avg_score > 0.1 else "Neutral" if avg_score > -0.1 else "Negative"
            st.markdown(f"""
                <h2 style='color:#00F5D4;'>{emoji} {sentiment_tag} Sentiment</h2>
                <p style='color:#B0BEC5;'>Average Sentiment Score: <strong>{avg_score:.2f}</strong></p>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Failed to fetch sentiment data: {e}")
