# === sentiment_utils.py ===

import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon', quiet=True)

# Fetch recent news headlines related to a sector
def fetch_news(query="technology", api_key=None, max_articles=10):
    if api_key is None:
        raise ValueError("API key must be provided.")

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize={max_articles}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()

    articles = data.get("articles", [])
    return [(a["title"], a.get("description", "")) for a in articles]

# Analyze sentiment of given text
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)
