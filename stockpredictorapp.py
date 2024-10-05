import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objs as go
import requests  # Required for making API calls to Polygon.io

# Define sector stock tickers for the Personalized Investment Page
sector_stocks = {
    "Tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
    "Pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
    "Finance": ["JPM", "GS", "BAC", "C", "WFC"],
    "Energy": ["XOM", "CVX", "BP", "COP", "PSX"]
}

# API Key Initialization
NEWS_API_KEY = 'f8f49c3781944e259ce310d89915cbff'
POLYGON_API_KEY = 'c8bESrns7hXq5zuDnE04OwDSzxrNRp4j'  # Replace with your Polygon.io API key

# Streamlit App Title
st.title("📈 Stock Price Analysis & Personalized Investment Recommendations")

# Create a Sidebar Navigation
option = st.sidebar.radio("Choose a Page", ["Stock Analysis", "Personalized Recommendations"])

# Function to fetch historical stock data from Polygon.io
def fetch_polygon_data(ticker, start_date, end_date):
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                df = pd.DataFrame(data['results'])
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('t', inplace=True)
                df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
                return df
            else:
                st.error(f"No historical data found for {ticker}")
                return pd.DataFrame()
        else:
            st.error(f"Polygon.io request failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.warning(f"Polygon.io request failed with error: {e}")
        return None

# Function to fetch stock data using yfinance (Backup)
def fetch_yfinance_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No historical data found for {ticker} using yfinance.")
            return pd.DataFrame()
        else:
            return data
    except Exception as e:
        st.error(f"yfinance request failed with error: {e}")
        return pd.DataFrame()

# Function to fetch historical stock data with fallback mechanism
def fetch_data(ticker, start_date, end_date):
    st.info(f"Fetching data for {ticker} from Polygon.io...")
    data = fetch_polygon_data(ticker, start_date, end_date)
    if data is None or data.empty:
        st.warning("Polygon.io request failed or limit reached. Switching to yfinance...")
        st.info(f"Fetching data for {ticker} from yfinance as a fallback...")
        data = fetch_yfinance_data(ticker, start_date, end_date)
    return data

# Define the Stock Analysis Page
def stock_analysis():
    st.sidebar.header("Stock Analysis Input")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01")).strftime('%Y-%m-%d')
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31")).strftime('%Y-%m-%d')
    
    if st.sidebar.button("Analyze Stock"):
        if ticker:
            try:
                data = fetch_data(ticker, start_date, end_date)
                if data.empty:
                    st.error(f"Could not retrieve data for {ticker}. Check the ticker symbol or date range and try again.")
                else:
                    # Display Stock Data and Summary
                    st.write(f"### Stock Price Data for {ticker} ({start_date} to {end_date})")
                    st.write(data.head())
                    st.write("#### Statistical Summary")
                    st.write(data.describe())

                    # Visualize Stock Prices with Candlestick Chart
                    st.write(f"### {ticker} Stock Prices Candlestick Chart")
                    fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], increasing_line_color='green', decreasing_line_color='red')])
                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Define the Personalized Recommendation Page
def personalized_recommendation():
    investment_amount = st.number_input("Enter the amount you're willing to invest (USD):", min_value=1000, value=5000, step=500)
    investment_strategy = st.selectbox("Select your investment strategy:", ["Long-term", "Short-term", "Open to Either"])
    preferred_sectors = st.multiselect("Select preferred sectors to invest in (you can choose multiple):", list(sector_stocks.keys()), default=["Tech"])
    
    if st.button("Generate Investment Recommendations"):
        if investment_amount <= 0 or len(preferred_sectors) == 0:
            st.error("Please enter a valid investment amount and select at least one sector.")
        else:
            selected_stocks = [ticker for sector in preferred_sectors for ticker in sector_stocks[sector]]
            st.write(f"Analyzing the following stocks: {', '.join(selected_stocks)}")
            stock_data = fetch_data(",".join(selected_stocks), "2020-01-01", "2023-12-31")['Close']
            st.dataframe(stock_data.tail())

# Page Routing
if option == "Stock Analysis":
    stock_analysis()
else:
    personalized_recommendation()
