import yfinance as yf
import pandas as pd
import streamlit as st
from io import StringIO

@st.cache_data(ttl=3600)
def load_yahoo_data(ticker, period='6mo', interval='1d'):
    try:
        st.info(f"Fetching data for {ticker}...")
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            st.error("No data returned. Check the ticker symbol or internet connection.")
            return None
        data.reset_index(inplace=True)
        data['Ticker'] = ticker
        st.success(f"Successfully fetched {len(data)} rows for {ticker}.")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def load_kragle_data(uploaded_file):
    try:
        st.info("Loading uploaded dataset...")
        file_content = uploaded_file.read().decode("utf-8")
        df = pd.read_csv(StringIO(file_content))
        df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
        st.success(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except UnicodeDecodeError:
        st.error("Encoding issue: Try saving your CSV as UTF-8.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

def get_etf_summary(df):
    if df is None or 'Close' not in df.columns:
        return None
    summary = {
        "Mean Return (%)": df['Close'].pct_change().mean() * 100,
        "Volatility (%)": df['Close'].pct_change().std() * 100,
        "Max Drawdown (%)": (df['Close'].cummax() - df['Close']).max() / df['Close'].cummax().max() * 100
    }
    return {k: round(v, 2) for k, v in summary.items()}
