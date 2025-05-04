import pandas as pd
from utils import data_loader

SECTOR_MAP = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy", "XLV": "Healthcare",
    "XLY": "Consumer Discretionary", "XLI": "Industrials", "XLB": "Materials",
    "XLU": "Utilities", "XLRE": "Real Estate", "XLC": "Communication"
}

def get_top_sectors(horizon="1y"):
    interval = "1d" if horizon in ["1y", "6mo"] else "1wk"  # Wider intervals for longer horizons
    scores = {}

    for ticker, sector in SECTOR_MAP.items():
        try:
            df = data_loader.load_yahoo_data(ticker, period=horizon, interval=interval)

            # Ensure valid DataFrame
            if df is None or df.empty or "Close" not in df.columns:
                print(f"[No Data] {ticker} - DataFrame is empty or missing 'Close'")
                continue

            df = df.dropna(subset=["Close"])
            if len(df) < 10:
                print(f"[Insufficient Data] {ticker} - Less than 10 data points")
                continue

            df["Return"] = df["Close"].pct_change()
            df = df.dropna(subset=["Return"])

            mean = df["Return"].mean() * 100
            std = df["Return"].std() * 100

            score = round(mean / std, 2) if std else 0
            scores[sector] = score

        except Exception as e:
            print(f"[Error] {ticker}: {e}")

    return scores
