import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_extras.stylable_container import stylable_container
import yfinance as yf
import time
from datetime import datetime, timedelta
import uuid
import base64
import io
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Apply Streamlit page configuration
st.set_page_config(
    page_title="Trendr ML Pipeline",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state at the top
if 'pipeline_step' not in st.session_state:
    st.session_state.pipeline_step = 0
if 'sector_data' not in st.session_state:
    st.session_state.sector_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None

# Custom CSS with static progress bar styles
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0A0E1A, #1C2526);
        position: relative;
        overflow: hidden;
        animation: pulse 6s infinite;
    }
    @keyframes pulse {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: radial-gradient(circle, rgba(10, 14, 26, 0.85), rgba(28, 37, 38, 0.85));
    }
    .css-18e3th9 {
        padding: 2rem;
    }
    .stButton>button {
        background: linear-gradient(45deg, #00F5D4, #FF00FF);
        color: white;
        border: none;
        padding: 1.2rem 2.5rem;
        border-radius: 15px;
        font-size: 1.3rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.6), inset 0 0 10px rgba(255, 0, 255, 0.3);
        animation: neonGlow 1.5sinfinite alternate;
    }
    @keyframes neonGlow {
        from { box-shadow: 0 0 10px #00F5D4, inset 0 0 5px #FF00FF; }
        to { box-shadow: 0 0 30px #FF00FF, inset 0 0 15px #00F5D4; }
    }
    .stButton>button:hover {
        transform: scale(1.1);
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.8), inset 0 0 20px rgba(255, 0, 255, 0.5);
    }
    .metric-card {
        background: rgba(17, 25, 40, 0.7);
        border-radius: 20px;
        padding: 25px;
        border: 3px solid rgba(0, 245, 212, 0.3);
        backdrop-filter: blur(20px);
        box-shadow: 0 0 30px rgba(0, 245, 212, 0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 0 40px rgba(0, 245, 212, 0.3);
    }
    .header-title {
        color: #00F5D4;
        font-size: 4rem;
        font-weight: bold;
        text-shadow: 0 0 30px #00F5D4, 0 0 10px #FF00FF;
    }
    .header-subtitle {
        color: #B0BEC5;
        font-size: 1.7rem;
        text-shadow: 0 0 15px rgba(176, 190, 197, 0.4);
    }
    .etf-badge {
        padding: 8px 15px;
        background: rgba(0, 245, 212, 0.4);
        border-radius: 10px;
        font-size: 1.2rem;
        color: #00F5D4;
        text-shadow: 0 0 8px #00F5D4;
        box-shadow: inset 0 0 10px rgba(0, 245, 212, 0.3);
    }
    .trend-up {
        color: #00FF9D !important;
        text-shadow: 0 0 8px #00FF9D;
    }
    .trend-down {
        color: #FF416C !important;
        text-shadow: 0 0 8px #FF416C;
    }
    .carousel-container {
        position: relative;
        width: 900px;
        margin: 0 auto;
        overflow: hidden;
    }
    .carousel-step {
        display: none;
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.25), rgba(255, 0, 255, 0.25));
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        border: 3px solid rgba(0, 245, 212, 0.6);
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.2);
        width: 100%;
        animation: slideIn 0.6s ease-out;
    }
    .carousel-step.active {
        display: block;
    }
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    .progress-bar {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        height: 450px;
        width: 25px;
        position: fixed;
        left: 50px;
        top: 150px;
        z-index: 10;
        box-shadow: 0 0 15px rgba(0, 245, 212, 0.3);
    }
    .progress-fill {
        background: linear-gradient(45deg, #00F5D4, #FF00FF);
        height: 0%;
        width: 100%;
        border-radius: 10px;
        transition: height 0.6s ease;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
    }
    h4 {
        color: #00F5D4;
        font-size: 2rem;
        text-shadow: 0 0 15px #00F5D4, 0 0 5px #FF00FF;
    }
    .download-button {
        background: linear-gradient(45deg, #00F5D4, #FF00FF);
        color: white;
        padding: 1rem 2rem;
        border-radius: 15px;
        font-size: 1.2rem;
        border: none;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
        transition: all 0.3s ease;
    }
    .download-button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Particle background with neon trails
st.markdown("""
<div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;">
    <canvas id="particleCanvas"></canvas>
    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: radial-gradient(circle, rgba(10, 14, 26, 0.4), rgba(28, 37, 38, 0.4));"></div>
</div>
<script>
    const canvas = document.getElementById('particleCanvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const ctx = canvas.getContext('2d');
    const particles = [];
    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 4 + 1;
            this.speedX = Math.random() * 0.7 - 0.35;
            this.speedY = Math.random() * 0.7 - 0.35;
            this.trail = [];
            this.maxTrail = 10;
        }
        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            if (this.size > 0.3) this.size -= 0.015;
            this.trail.push({ x: this.x, y: this.y });
            if (this.trail.length > this.maxTrail) this.trail.shift();
            if (this.x < 0 || this.x > canvas.width || this.y < 0 || this.y > canvas.height) {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.trail = [];
            }
        }
        draw() {
            ctx.fillStyle = 'rgba(0, 245, 212, 0.7)';
            for (let i = 0; i < this.trail.length; i++) {
                ctx.beginPath();
                ctx.arc(this.trail[i].x, this.trail[i].y, this.size * (1 - i / this.maxTrail), 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    function init() {
        for (let i = 0; i < 200; i++) {
            particles.push(new Particle());
        }
    }
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach((particle, index) => {
            particle.update();
            particle.draw();
            if (particle.size <= 0.3) {
                particles.splice(index, 1);
                particles.push(new Particle());
            }
        });
        requestAnimationFrame(animate);
    }
    init();
    animate();
</script>
""", unsafe_allow_html=True)

# JavaScript to dynamically update progress bar height
progress_percentage = st.session_state.get('pipeline_step', 0) * 14.28  # 14.28% per step (100/7 steps)
st.markdown(f"""
<script>
    document.addEventListener('DOMContentLoaded', function() {{
        const progressFill = document.querySelector('.progress-fill');
        if (progressFill) {{
            progressFill.style.height = '{progress_percentage}%';
        }}
    }});
</script>
""", unsafe_allow_html=True)

# Sector Mapping
SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication"
}
TICKER_MAP = {v: k for k, v in SECTOR_MAP.items()}
SECTOR_COLORS = {
    "Technology": "#00F5D4", "Financials": "#5D5FEF", "Energy": "#FF7700",
    "Healthcare": "#00CFFF", "Consumer Discretionary": "#FF5CAC",
    "Industrials": "#FFC107", "Materials": "#8BC34A", "Utilities": "#00BCD4",
    "Real Estate": "#FF416C", "Communication": "#8A2BE2"
}

# Data Loading Functions
def fetch_yahoo_data(tickers, period="1y", interval="1d"):
    sector_data = {}
    with st.spinner("Fetching live data from Yahoo Finance..."):
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period=period, interval=interval)
                if df.empty or len(df) < 15:
                    st.warning(f"Insufficient data for {ticker} (less than 15 days). Skipping.")
                    continue
                df = df.reset_index()
                df.columns = [col.lower().replace(" ", "_") for col in df.columns]
                df["return"] = df["close"].pct_change().fillna(0)
                df["day"] = np.arange(len(df))
                df['5_ma'] = df['close'].rolling(window=5).mean()
                df['10_ma'] = df['close'].rolling(window=10).mean()
                df = df.dropna()
                if len(df) < 5:
                    st.warning(f"Insufficient data for {ticker} after preprocessing. Skipping.")
                    continue
                mean_return = df["return"].mean() * 100
                volatility = df["return"].std() * 100
                if volatility == 0:
                    st.warning(f"Zero volatility for {ticker}. Skipping.")
                    continue
                score = round(mean_return / volatility, 2)
                momentum = "Upward" if df['5_ma'].iloc[-1] > df['10_ma'].iloc[-1] else "Downward"
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
                sector = SECTOR_MAP.get(ticker, "Unknown")
                sector_data[sector] = {
                    "score": score,
                    "mean_return": round(mean_return, 2),
                    "volatility": round(volatility, 2),
                    "momentum": momentum,
                    "price_change": round(price_change, 2),
                    "current_price": round(df['close'].iloc[-1], 2),
                    "df": df,
                    "prediction": {
                        "prices": [],
                        "trend": "Unknown"
                    }
                }
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {e}")
                continue
        if not sector_data or len(sector_data) < 3:
            st.error("Not enough sectors with valid data (minimum 3 required).")
            return None
        return sector_data

def process_uploaded_data(file):
    try:
        df = pd.read_csv(file)
        required_columns = ["date", "close", "sector", "ticker"]
        if not all(col in df.columns for col in required_columns):
            st.error("Uploaded CSV must contain 'date', 'close', 'sector', and 'ticker' columns.")
            return None
        df["date"] = pd.to_datetime(df["date"])
        sector_data = {}
        for sector in df["sector"].unique():
            sector_df = df[df["sector"] == sector].copy()
            sector_df = sector_df.sort_values("date")
            if len(sector_df) < 15:
                st.warning(f"Insufficient data for sector {sector} (less than 15 days). Skipping.")
                continue
            sector_df["return"] = sector_df["close"].pct_change().fillna(0)
            sector_df["day"] = np.arange(len(sector_df))
            sector_df['5_ma'] = sector_df['close'].rolling(window=5).mean()
            sector_df['10_ma'] = sector_df['close'].rolling(window=10).mean()
            sector_df = sector_df.dropna()
            if len(sector_df) < 5:
                st.warning(f"Insufficient data for sector {sector} after preprocessing. Skipping.")
                continue
            mean_return = sector_df["return"].mean() * 100
            volatility = sector_df["return"].std() * 100
            if volatility == 0:
                st.warning(f"Zero volatility for sector {sector}. Skipping.")
                continue
            score = round(mean_return / volatility, 2)
            momentum = "Upward" if sector_df['5_ma'].iloc[-1] > sector_df['10_ma'].iloc[-1] else "Downward"
            price_change = ((sector_df['close'].iloc[-1] - sector_df['close'].iloc[0]) / sector_df['close'].iloc[0]) * 100
            sector_data[sector] = {
                "score": score,
                "mean_return": round(mean_return, 2),
                "volatility": round(volatility, 2),
                "momentum": momentum,
                "price_change": round(price_change, 2),
                "current_price": round(sector_df['close'].iloc[-1], 2),
                "df": sector_df,
                "prediction": {
                    "prices": [],
                    "trend": "Unknown"
                }
            }
        if not sector_data or len(sector_data) < 3:
            st.error("Not enough sectors with valid data (minimum 3 required).")
            return None
        return sector_data
    except Exception as e:
        st.error(f"Error processing uploaded CSV: {str(e)}. Please ensure the file is a valid CSV with the correct format.")
        return None

# ML Pipeline Functions
def load_data(data_source, file=None, tickers=None, period="1y"):
    if data_source == "Upload CSV":
        if file is None:
            st.error("Please upload a CSV file.")
            return None
        sector_data = process_uploaded_data(file)
    else:
        if not tickers:
            st.error("Please select at least one sector ticker.")
            return None
        sector_data = fetch_yahoo_data(tickers, period)
    if sector_data:
        st.session_state.sector_data = sector_data
        st.session_state.pipeline_step = 1
        st.success("Data loaded successfully!")
        st.dataframe(pd.DataFrame({
            "Sector": list(sector_data.keys()),
            "Current Price": [v['current_price'] for v in sector_data.values()],
            "Signal Score": [v['score'] for v in sector_data.values()]
        }))
    return sector_data

def preprocess_data(data):
    with st.spinner("Preprocessing data..."):
        time.sleep(1)
        processed_data = {}
        missing_stats = []
        for sector, info in data.items():
            df = info['df'].copy()
            missing = df.isnull().sum().sum()
            df = df.fillna(method='ffill').fillna(method='bfill')
            if df['return'].std() == 0:
                st.warning(f"Zero variance in returns for sector {sector}. Skipping.")
                continue
            processed_data[sector] = info.copy()
            processed_data[sector]['df'] = df
            missing_stats.append({"Sector": sector, "Missing Values": missing})
        if not processed_data or len(processed_data) < 3:
            st.error("Not enough sectors with valid data after preprocessing (minimum 3 required).")
            return None
        st.session_state.processed_data = processed_data
        st.session_state.pipeline_step = 2
        st.success("Data preprocessing completed!")
        st.dataframe(pd.DataFrame(missing_stats))
        return processed_data

def feature_engineering(data):
    with st.spinner("Engineering features..."):
        time.sleep(1)
        features = []
        for sector, info in data.items():
            df = info['df']
            df['volatility'] = df['return'].rolling(window=5).std() * 100
            df['momentum'] = df['5_ma'] / df['10_ma']
            df = df.dropna()
            if df.empty:
                st.warning(f"No valid data for sector {sector} after feature engineering. Skipping.")
                continue
            if np.any(df[['volatility', 'momentum']].isna()):
                st.warning(f"Invalid feature values for sector {sector}. Skipping.")
                continue
            features.append({
                "Sector": sector,
                "Mean Return": info['mean_return'],
                "Volatility": df['volatility'].iloc[-1],
                "Momentum": df['momentum'].iloc[-1],
                "Signal Score": info['score'],
                "Day": df['day'].iloc[-1]
            })
        feature_df = pd.DataFrame(features)
        if len(feature_df) < 3:
            st.error("Not enough sectors with valid features (minimum 3 required).")
            return None
        st.session_state.features = feature_df
        st.session_state.pipeline_step = 3
        st.success("Feature engineering completed!")
        fig = px.bar(feature_df, x="Sector", y=["Mean Return", "Volatility", "Momentum"], title="Feature Importance")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        return feature_df

def train_test_split_data(features):
    with st.spinner("Splitting data..."):
        time.sleep(1)
        X = features[["Mean Return", "Volatility", "Momentum", "Day"]]
        y = features["Signal Score"]
        if len(X) < 5:
            st.error("Not enough data for train/test split (minimum 5 sectors required).")
            return None
        test_size = max(0.3, 3/len(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if len(X_test) < 3:
            st.error("Test set too small (minimum 3 samples required).")
            return None
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.pipeline_step = 4
        st.success("Train/test split completed!")
        fig = go.Figure(data=[
            go.Pie(labels=["Training Set", "Test Set"], values=[len(X_train), len(X_test)], hole=0.4)
        ])
        fig.update_layout(template="plotly_dark", title="Train/Test Split")
        st.plotly_chart(fig, use_container_width=True)
        return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type="RandomForest"):
    with st.spinner("Training model..."):
        time.sleep(1)
        if model_type == "RandomForest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "Linear":
            model = LinearRegression()
        elif model_type == "KNN":
            model = KNeighborsRegressor(n_neighbors=5)
        else:
            model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        st.session_state.model = model
        st.session_state.pipeline_step = 5
        st.success(f"{model_type} model training completed!")
        return model

def evaluate_model(model, X_test, y_test):
    with st.spinner("Evaluating model..."):
        time.sleep(1)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        y_mean = np.mean(y_test)
        ss_tot = np.sum((y_test - y_mean) ** 2)
        ss_res = np.sum((y_test - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else float('nan')
        if np.any(np.isnan([mse, r2])) or np.any(np.isinf([mse, r2])):
            st.error("Invalid evaluation metrics (NaN or Inf). Check data for issues.")
            return None
        st.session_state.pipeline_step = 6
        st.success("Model evaluation completed!")
        st.metric("Mean Squared Error", f"{mse:.4f}")
        st.metric("R² Score", f"{r2:.4f}" if not np.isnan(r2) else "N/A")
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual Score", "y": "Predicted Score"}, title="Prediction vs Actual")
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines", name="Ideal"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        return y_pred

def forecast(model, features, sector_data):
    with st.spinner("Generating forecast..."):
        time.sleep(1)
        forecast_data = {}
        for sector in features["Sector"].unique():
            sector_info = sector_data[sector]
            df = sector_info['df']
            last_close = df['close'].iloc[-1]
            last_day = df['day'].iloc[-1]
            # Calculate trend based on 5-day moving average slope
            recent_data = df['5_ma'].tail(5)
            if len(recent_data) < 2:
                trend = 0
            else:
                trend = (recent_data.iloc[-1] - recent_data.iloc[0]) / (len(recent_data) - 1)
            future_days = np.arange(last_day + 1, last_day + 31)  # 30-day forecast
            future_prices = [last_close + trend * i for i in range(1, 31)]
            forecast_data[sector] = {
                "days": future_days,
                "prices": future_prices
            }
        st.session_state.forecast = forecast_data
        st.session_state.pipeline_step = 7
        st.success("Forecast generated!")
        for sector, data in forecast_data.items():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data["days"], y=data["prices"], mode='lines+markers', name=sector, line=dict(color=SECTOR_COLORS[sector])))
            fig.update_layout(title=f"30-Day Price Forecast for {sector}", template="plotly_dark", height=400, yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)
        return forecast_data

def prepare_download_data(sector_data, predictions, forecast):
    result_df = pd.DataFrame({
        "Sector": list(sector_data.keys()),
        "Current Price": [v['current_price'] for v in sector_data.values()],
        "Signal Score": [v['score'] for v in sector_data.values()],
        "Mean Return (%)": [v['mean_return'] for v in sector_data.values()],
        "Volatility (%)": [v['volatility'] for v in sector_data.values()],
        "Momentum": [v['momentum'] for v in sector_data.values()],
        "Predicted Score": [p for p in predictions] if predictions is not None else [None] * len(sector_data)
    })
    for sector in forecast.keys():
        forecast_df = pd.DataFrame({
            f"{sector}_Forecast_Day": forecast[sector]["days"],
            f"{sector}_Forecast_Price": forecast[sector]["prices"]
        })
        result_df = result_df.merge(forecast_df, how='left', left_index=True, right_index=True)
    return result_df

# Sector Performance Chart
def create_sector_performance_chart(data):
    chart_df = pd.DataFrame({
        "Sector": list(data.keys()),
        "Signal Score": [v['score'] for v in data.values()],
        "Return": [v['mean_return'] for v in data.values()],
        "Risk": [v['volatility'] for v in data.values()],
        "Momentum": [v['momentum'] for v in data.values()]
    }).sort_values("Signal Score", ascending=False)
    fig = px.bar(
        chart_df,
        x="Sector",
        y="Signal Score",
        color="Sector",
        color_discrete_map=SECTOR_COLORS,
        text="Signal Score"
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        title="Sector Signal Scores",
        template="plotly_dark",
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Main UI
def render_cosmic_navigator():
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color:#00F5D4; font-size: 2.2rem; text-shadow: 0 0 20px #00F5D4;'>Cosmic Controls</h2>", unsafe_allow_html=True)
        data_source = st.radio("Data Source", ["Upload CSV", "Fetch from Yahoo Finance"])
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        else:
            available_tickers = list(SECTOR_MAP.keys())
            selected_tickers = st.multiselect("Select Sectors", available_tickers, default=["XLK", "XLF", "XLE"])
            period = st.selectbox("Time Period", ["6mo", "1y", "3y"], index=1)
        model_type = st.selectbox("ML Model", ["RandomForest", "Linear", "KNN"], index=0)
        top_k = st.slider("Top Sectors", 1, 5, 3)
        st.markdown("<div style='margin-top:40px; text-align:center; color:#B0BEC5; font-size: 1.3rem;'>FinSight v2.1</div>", unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div style="text-align:center; margin-bottom:60px;">
            <h1 class="header-title">FinSight: Cosmic Market Navigator</h1>
            <p class="header-subtitle">AI-Powered Sector Insights with ML & Forecasting</p>
        </div>
    """, unsafe_allow_html=True)

    # Vertical Progress Bar
    st.markdown("""
        <div class="progress-bar">
            <div class="progress-fill"></div>
        </div>
    """, unsafe_allow_html=True)

    # Carousel Pipeline Section
    st.markdown("<h2 style='color:#00F5D4; font-size: 2.8rem; text-align:center; text-shadow: 0 0 20px #00F5D4;'>Machine Learning Pipeline</h2>", unsafe_allow_html=True)
    steps = ["Load Data", "Preprocess Data", "Feature Engineering", "Train/Test Split", "Train Model", "Evaluate Model", "Generate Forecast"]
    current_step = st.session_state.pipeline_step - 1 if st.session_state.pipeline_step > 0 else 0

    st.markdown("<div class='carousel-container'>", unsafe_allow_html=True)
    for i, step in enumerate(steps):
        with st.container():
            st.markdown(f"<div class='carousel-step {'active' if i == current_step else ''}' id='step_{i}'>", unsafe_allow_html=True)
            st.markdown(f"<h4>{step}</h4>", unsafe_allow_html=True)
            if i == 0 and st.session_state.pipeline_step == 0 and st.button(step, key=f"btn_{i}"):
                if data_source == "Upload CSV":
                    sector_data = load_data(data_source, file=uploaded_file)
                else:
                    sector_data = load_data(data_source, tickers=selected_tickers, period=period)
            elif i == 1 and st.session_state.pipeline_step == 1 and st.button(step, key=f"btn_{i}"):
                if st.session_state.sector_data:
                    processed_data = preprocess_data(st.session_state.sector_data)
                    if processed_data is None:
                        st.session_state.pipeline_step = 1
            elif i == 2 and st.session_state.pipeline_step == 2 and st.button(step, key=f"btn_{i}"):
                if st.session_state.processed_data:
                    features = feature_engineering(st.session_state.processed_data)
                    if features is None:
                        st.session_state.pipeline_step = 2
            elif i == 3 and st.session_state.pipeline_step == 3 and st.button(step, key=f"btn_{i}"):
                if st.session_state.features is not None:
                    result = train_test_split_data(st.session_state.features)
                    if result is None:
                        st.session_state.pipeline_step = 3
            elif i == 4 and st.session_state.pipeline_step == 4 and st.button(step, key=f"btn_{i}"):
                if st.session_state.X_train is not None and st.session_state.y_train is not None:
                    train_model(st.session_state.X_train, st.session_state.y_train, model_type)
            elif i == 5 and st.session_state.pipeline_step == 5 and st.button(step, key=f"btn_{i}"):
                if st.session_state.model is not None and st.session_state.X_test is not None:
                    result = evaluate_model(st.session_state.model, st.session_state.X_test, st.session_state.y_test)
                    if result is None:
                        st.session_state.pipeline_step = 5
            elif i == 6 and st.session_state.pipeline_step == 6 and st.button(step, key=f"btn_{i}"):
                if st.session_state.model is not None and st.session_state.features is not None:
                    forecast(st.session_state.model, st.session_state.features, st.session_state.sector_data)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Dashboard Section
    if st.session_state.sector_data and st.session_state.pipeline_step >= 7:
        st.markdown("<h2 style='color:#00F5D4; font-size: 2.8rem; text-align:center; text-shadow: 0 0 20px #00F5D4;'>Market Signal Dashboard</h2>", unsafe_allow_html=True)
        top_sectors = dict(sorted(st.session_state.sector_data.items(), key=lambda x: x[1]['score'], reverse=True)[:top_k])
        cols = st.columns(len(top_sectors))
        for i, (sector, info) in enumerate(top_sectors.items()):
            with cols[i]:
                trend_icon = "↑" if info['momentum'] == "Upward" else "↓"
                trend_color = "trend-up" if info['momentum'] == "Upward" else "trend-down"
                st.markdown(f"""
                    <div class="metric-card">
                        <span class="etf-badge">{TICKER_MAP.get(sector)}</span>
                        <h3 style="color:{SECTOR_COLORS.get(sector)}; font-size: 1.7rem;">{sector}</h3>
                        <div style="font-size: 2.8rem">{info['score']}</div>
                        <div>Signal Score</div>
                        <div class="{trend_color}">{trend_icon} {info['momentum']}</div>
                        <div>Return: {info['mean_return']}% | Risk: {info['volatility']}%</div>
                    </div>
                """, unsafe_allow_html=True)
        st.plotly_chart(create_sector_performance_chart(st.session_state.sector_data), use_container_width=True)

        # Download Results as Excel
        if st.session_state.predictions is not None and st.session_state.forecast is not None:
            download_data = prepare_download_data(st.session_state.sector_data, st.session_state.predictions, st.session_state.forecast)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                download_data.to_excel(writer, index=False, sheet_name='Results')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="finsight_results.xlsx" class="download-button">Download Excel Results</a>'
            st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    render_cosmic_navigator()

