import streamlit as st
from pathlib import Path
import random


import warnings
warnings.filterwarnings("ignore")


# === Page Config ===
st.set_page_config(
    page_title="Trendr: Market Signal Navigator",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Custom CSS for Enhanced Cyberpunk Aesthetic ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Futura:wght@400;700&family=Orbitron:wght@400;700&family=Roboto+Mono:wght@300;400;600&display=swap');

    .main {
        background: #0A0E1A;
        position: relative;
        overflow: hidden;
        font-family: 'Roboto Mono', monospace;
    }
    .stApp {
        background: transparent;
    }
    /* Background Neon Animation with GIF */
    .background-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
    }
    @keyframes neonPulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    /* Holographic Sector Cards */
    .sector-card {
        background: rgba(17, 25, 40, 0.5);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(0, 245, 212, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: hologram 2s infinite alternate;
    }
    .sector-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 0 30px rgba(0, 245, 212, 0.4);
    }
    @keyframes hologram {
        0% { border-color: rgba(0, 245, 212, 0.3); box-shadow: 0 0 20px rgba(0, 245, 212, 0.2); }
        100% { border-color: rgba(255, 0, 255, 0.3); box-shadow: 0 0 20px rgba(255, 0, 255, 0.2); }
    }
    /* Navigation Tiles */
    .nav-tile {
        background: rgba(17, 25, 40, 0.6);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(0, 245, 212, 0.2);
        backdrop-filter: blur(10px);
        text-align: center;
        color: #B0BEC5;
        font-size: 1.2rem;
        font-family: 'Futura', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .nav-tile:hover {
        background: rgba(0, 245, 212, 0.1);
        border-color: #00F5D4;
        color: #00F5D4;
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.3);
    }
    /* Highlight Strip */
    .highlight-strip {
        background: rgba(17, 25, 40, 0.5);
        border-left: 5px solid #00F5D4;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        color: #B0BEC5;
        font-size: 1rem;
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.2);
    }
    /* Sidebar Styling */
    .sidebar-neon {
        background: rgba(17, 25, 40, 0.7);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(0, 245, 212, 0.2);
        padding: 20px;
        display: flex;
        flex-direction: column;
        align-items: center;
        min-height: 100vh;
    }
    .sidebar-icon {
        font-size: 2.5rem;
        color: #B0BEC5;
        margin: 15px 0;
        transition: all 0.3s ease;
        cursor: pointer;
        animation: iconPulse 2s infinite alternate;
    }
    @keyframes iconPulse {
        0% { text-shadow: 0 0 5px #B0BEC5; }
        100% { text-shadow: 0 0 15px #00F5D4; }
    }
    .sidebar-icon:hover {
        color: #00F5D4;
        text-shadow: 0 0 20px #00F5D4;
    }
    /* Header Styling with Logo */
    .header-title {
        font-family: 'Orbitron', sans-serif;
        color: #00F5D4;
        font-size: 3.5rem;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0, 245, 212, 0.4);
        letter-spacing: 2px;
        display: flex;
        align-items: center;
    }
    .header-logo {
        width: 140px;
        height: 110px;
        margin-right: 15px;
    }
    .header-subtitle {
        color: #B0BEC5;
        font-size: 2rem;
        font-weight: 300;
        text-shadow: 0 0 15px rgba(176, 190, 197, 0.6), 0 0 25px rgba(176, 190, 197, 0.4);
        text-align: center;
    }
    /* Quote Styling */
    .quote-box {
        background: rgba(17, 25, 40, 0.5);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 0, 255, 0.2);
        backdrop-filter: blur(10px);
        text-align: center;
        color: #B0BEC5;
        font-style: italic;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.2);
    }
    /* Infographic */
    .infographic {
        background: rgba(17, 25, 40, 0.5);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(0, 245, 212, 0.2);
        backdrop-filter: blur(10px);
        color: #B0BEC5;
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.2);
    }
    /* Article/Video Section */
    .resource-box {
        background: rgba(17, 25, 40, 0.5);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid rgba(255, 0, 255, 0.2);
        backdrop-filter: blur(10px);
        color: #B0BEC5;
        font-size: 1rem;
        box-shadow: 0 0 15px rgba(255, 0, 255, 0.2);
    }
    a {
        color: #00F5D4;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    a:hover {
        color: #FF00FF;
        text-shadow: 0 0 10px #FF00FF;
    }
</style>
""", unsafe_allow_html=True)

# === Background Animation ===
st.markdown("""
<div class="background-container">
    <canvas id="neonGridCanvas"></canvas>
</div>
<script>
    const canvas = document.getElementById('neonGridCanvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext('2d');

    function drawGrid() {
        ctx.strokeStyle = 'rgba(0, 245, 212, 0.1)';
        ctx.lineWidth = 0.5;
        for (let x = 0; x <= canvas.width; x += 50) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        for (let y = 0; y <= canvas.height; y += 50) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(canvas.width, y);
            ctx.stroke();
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        drawGrid();
        requestAnimationFrame(animate);
    }
    animate();
</script>
""", unsafe_allow_html=True)

with st.sidebar:
    # Sidebar Background and Header
    st.markdown("""
        <style>
            .sidebar-welcome {
                text-align: center;
                font-family: 'Futura', sans-serif;
                font-weight: 700;
                font-size: 1.6rem;
                color: #00F5D4;
                text-shadow: 0 0 10px #00F5D4;
                margin-top: 10px;
                margin-bottom: 20px;
            }
            .sidebar-icon-label {
                font-family: 'Futura', sans-serif;
                font-size: 1rem;
                color: #B0BEC5;
                margin-top: 5px;
            }
            .sidebar-theme-select {
                background: #1C2526;
                color: #00F5D4;
                border: 2px solid #00F5D4;
                border-radius: 8px;
                padding: 5px 10px;
                font-family: 'Futura', sans-serif;
                margin-top: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Welcome GIF
    st.image("https://media0.giphy.com/media/Qiru6DuC7tZuOgjtBF/giphy.gif?cid=6c09b952xoxqg6s3c2ild3loy7igd1lq5xfc6pdziqo4umip&ep=v1_gifs_search&rid=giphy.gif&ct=g", use_container_width=True)

    # Welcome Title
    st.markdown('<div class="sidebar-welcome">Welcome, Strategist!</div>', unsafe_allow_html=True)

    # Navigation with Icons
    st.page_link("main.py", label=" Home", icon="🏠")
    st.page_link("pages/ETF_dashboard.py", label=" ETF Dashboard", icon="📊")
    st.page_link("pages/ML_forecasting.py", label=" ML Forecasting", icon="🧠")
    st.page_link("pages/portfolio_simulator.py", label=" Portfolio Sim", icon="📈")
    st.page_link("pages/sentiment_analysis.py", label=" Sentiment", icon="💬")

    # Theme Selector (Non-functional UI mockup dropdown)
    st.markdown("""
        <div style="text-align:center;">
            <select class="sidebar-theme-select">
                <option>Neon</option>
                <option>Dark</option>
                <option>Light</option>
            </select>
        </div>
    """, unsafe_allow_html=True)

    # Tagline
    st.markdown("""
        <div style="text-align:center; color:#B0BEC5; font-size:1.2rem; font-family:'Futura', sans-serif; margin-top:20px;">
            Navigate. Predict. Launch.
        </div>
    """, unsafe_allow_html=True)


# === Main Interface with Logo ===
st.markdown("""
    <div style="text-align:center; margin-bottom:50px; display:flex; align-items:center; justify-content:center;">
        <img src="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/401dde00-760d-406d-92ac-86ebaab75368/dbkp9bn-b742f397-4b29-4b96-901d-5dc638eee275.gif?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzQwMWRkZTAwLTc2MGQtNDA2ZC05MmFjLTg2ZWJhYWI3NTM2OFwvZGJrcDlibi1iNzQyZjM5Ny00YjI5LTRiOTYtOTAxZC01ZGM2MzhlZWUyNzUuZ2lmIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.hZD6eeEunqAO9QOann6l4kOCDGMTd8QJMHtwttuShq8" class="header-logo">
        <h1 class="header-title"> Trendr: Market Signal Navigator</h1>
    </div>
    <p class="header-subtitle">Empowering Startups with Financial Intelligence</p>
""", unsafe_allow_html=True)

# === Market Snapshot ===
st.markdown("<h2 style='color:#00F5D4; font-size:2.5rem; text-shadow:0 0 20px #00F5D4;'>Latest Market Snapshot</h2>", unsafe_allow_html=True)

# Mock data for top 3 sectors by momentum
etfs = {
    "XLK": {"sector": "Technology", "momentum": "Upward", "score": 3.2, "current_price": 215.43},
    "XLE": {"sector": "Energy", "momentum": "Upward", "score": 2.8, "current_price": 92.17},
    "XLF": {"sector": "Financials", "momentum": "Downward", "score": 1.5, "current_price": 43.89},
    "XLV": {"sector": "Healthcare", "momentum": "Upward", "score": 2.1, "current_price": 145.67},
    "XLY": {"sector": "Consumer Discretionary", "momentum": "Downward", "score": 1.8, "current_price": 182.34}
}
top_sectors = sorted(etfs.items(), key=lambda x: x[1]["score"], reverse=True)[:3]

cols = st.columns(3)
for i, (ticker, info) in enumerate(top_sectors):
    with cols[i]:
        trend_icon = "↑" if info["momentum"] == "Upward" else "↓"
        trend_color = "trend-up" if info["momentum"] == "Upward" else "trend-down"
        st.markdown(f"""
            <div class="sector-card">
                <h3 style="color:#00F5D4; font-size:1.5rem;">{info['sector']}</h3>
                <div style="font-size:1rem; color:#B0BEC5;">{ticker}</div>
                <div style="font-size:2rem; font-weight:600;">${info['current_price']}</div>
                <div style="font-size:1rem; color:#B0BEC5;">Score: {info['score']}</div>
                <div class="{trend_color}" style="font-size:1.2rem;">{trend_icon} {info['momentum']}</div>
            </div>
        """, unsafe_allow_html=True)

# === App Purpose and Infographic ===
st.markdown("<h2 style='color:#00F5D4; font-size:2.5rem; text-shadow:0 0 20px #00F5D4;'>What is Trendr?</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class="infographic">
        <p style="color:#B0BEC5; font-size:1.1rem;">
            Trendr is your ultimate financial intelligence dashboard designed for aspiring entrepreneurs and startup strategists. By leveraging real-time ETF sector data, machine learning forecasting, sentiment analysis, and portfolio simulation, Trendr helps you identify the most promising sectors for launching new business ventures. 
        </p>
        <p style="color:#B0BEC5; font-size:1.1rem; margin-top:10px;">
            <strong>Why ETFs?</strong> Exchange-Traded Funds (ETFs) track the performance of entire sectors, providing a reliable indicator of market trends, momentum, and stability. By analyzing ETF data, Trendr reveals which sectors are thriving, helping you make informed decisions for your startup's focus and investment strategy.
        </p>
    </div>
""", unsafe_allow_html=True)

# === Highlight Strip ===
st.markdown("<h2 style='color:#00F5D4; font-size:2.5rem; text-shadow:0 0 20px #00F5D4;'>Key Capabilities</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class='highlight-strip'>
        📊 <strong>Visualize Sector Performance</strong> – Dive into ETF price trends, momentum scores, and volatility with interactive charts.
    </div>
    <div class='highlight-strip'>
        🧠 <strong>Forecast with ML</strong> – Predict sector trends using advanced machine learning models like Random Forest and Gradient Boosting.
    </div>
    <div class='highlight-strip'>
        📈 <strong>Simulate Portfolios</strong> – Test ETF-based portfolios to optimize returns and minimize risk.
    </div>
    <div class='highlight-strip'>
        💬 <strong>Analyze Sentiment</strong> – Gauge market sentiment with real-time news analysis and NLP-powered insights.
    </div>
""", unsafe_allow_html=True)

# === Navigation Tiles ===
st.markdown("<h2 style='color:#00F5D4; font-size:2.5rem; text-shadow:0 0 20px #00F5D4;'>Navigate to Your Tools</h2>", unsafe_allow_html=True)
cols = st.columns(4)
pages = [
    {"title": "ETF Dashboard", "icon": "📊", "url": "pages/ETF_dashboard.py"},
    {"title": "ML Forecasting", "icon": "🧠", "url": "pages/ML_forecasting.py"},
    {"title": "Portfolio Simulator", "icon": "📈", "url": "pages/portfolio_simulator.py"},
    {"title": "Sentiment Analysis", "icon": "💬", "url": "pages/sentiment_analysis.py"}
]
for i, page in enumerate(pages):
    with cols[i]:
        st.markdown(f"""
            <a href="{page['url']}" style="text-decoration:none;">
                <div class="nav-tile">
                    <div style="font-size:2.5rem;">{page['icon']}</div>
                    <div>{page['title']}</div>
                </div>
            </a>
        """, unsafe_allow_html=True)

# === Daily Startup Insight ===
st.markdown("<h2 style='color:#00F5D4; font-size:2.5rem; text-shadow:0 0 20px #00F5D4;'>Daily Startup Insight</h2>", unsafe_allow_html=True)
quotes = [
    "‘The best way to predict the future is to create it.’ – Peter Drucker",
    "‘Every problem is a gift—without problems, we would not grow.’ – Tony Robbins",
    "‘Success is not the absence of obstacles, but the courage to push through.’ – Anonymous",
    "‘Innovation distinguishes between a leader and a follower.’ – Steve Jobs",
    "‘The only limit to our realization of tomorrow is our doubts of today.’ – Franklin D. Roosevelt"
]
daily_quote = random.choice(quotes)
st.markdown(f"""
    <div class="quote-box">
        {daily_quote}
    </div>
""", unsafe_allow_html=True)

# === ETF Resources Section ===
st.markdown("<h2 style='color:#00F5D4; font-size:2.5rem; text-shadow:0 0 20px #00F5D4;'>Learn About ETFs</h2>", unsafe_allow_html=True)
st.markdown("""
    <div class="resource-box">
        <strong>Articles:</strong><br>
        - <a href="https://www.investopedia.com/terms/e/etf.asp" target="_blank">What Are ETFs? (Investopedia)</a><br>
        - <a href="https://www.nerdwallet.com/article/investing/what-are-etfs" target="_blank">ETFs Explained (NerdWallet)</a><br>
        - <a href="https://www.fidelity.com/learning-center/investment-products/etf/overview" target="_blank">ETF Overview (Fidelity)</a><br>
        <strong>Video:</strong><br>
        - <a href="https://www.youtube.com/watch?v=3u8D5XzT3UU" target="_blank">What Are ETFs? (YouTube - Fidelity)</a>
    </div>
""", unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.markdown("""
    <div style="text-align:center; color:#B0BEC5; font-size:0.9rem;">
        © 2025 Trendr 
    </div>
""", unsafe_allow_html=True)