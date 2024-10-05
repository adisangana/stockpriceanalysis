import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objs as go  # Importing Plotly for candlestick charts

# Define sector stock tickers for the Personalized Investment Page
sector_stocks = {
    "Tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"],
    "Pharma": ["JNJ", "PFE", "MRK", "ABBV", "LLY"],
    "Finance": ["JPM", "GS", "BAC", "C", "WFC"],
    "Energy": ["XOM", "CVX", "BP", "COP", "PSX"]
}

# NewsAPI Client Initialization
NEWS_API_KEY = 'f8f49c3781944e259ce310d89915cbff'  # Replace with your NewsAPI key

# Streamlit App Title
st.title("📈 Stock Price Analysis & Personalized Investment Recommendations")

# Create a Sidebar Navigation
option = st.sidebar.radio("Choose a Page", ["Stock Analysis", "Personalized Recommendations"])

# Define Functions for Each Page
def stock_analysis():
    # Sidebar: User Input for Stock Ticker
    st.sidebar.header("Stock Analysis Input")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL").upper()

    # Sidebar: Date Range Input
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

    # NewsAPI Client Initialization
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    analyzer = SentimentIntensityAnalyzer()

    # Fetch data when the button is pressed
    if st.sidebar.button("Analyze Stock"):
        if ticker:
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                if data.empty:
                    st.error(f"Could not retrieve data for {ticker}. Check the ticker symbol or date range and try again.")
                else:
                    # Display Stock Data
                    st.write(f"### Stock Price Data for {ticker} ({start_date} to {end_date})")
                    st.write(data.head())

                    st.write("#### Statistical Summary")
                    st.write(data.describe())

                    # Visualize the Stock Prices with Candlestick Chart
                    st.write(f"### {ticker} Stock Prices Candlestick Chart")
                    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                         open=data['Open'],
                                                         high=data['High'],
                                                         low=data['Low'],
                                                         close=data['Close'],
                                                         increasing_line_color='green',
                                                         decreasing_line_color='red')])

                    # Add moving averages to the candlestick chart
                    data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    data['SMA_200'] = data['Close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-Day SMA'))

                    # Customize chart layout
                    fig.update_layout(title=f'{ticker} Stock Price with Moving Averages',
                                      yaxis_title='Stock Price (USD)',
                                      xaxis_title='Date')
                    st.plotly_chart(fig)

                    # Prepare Data for Machine Learning
                    data['Future Price'] = data['Close'].shift(-30)
                    data = data.dropna()

                    features = ['Close', 'SMA_50', 'SMA_200']
                    X = data[features]
                    y = data['Future Price']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train a Linear Regression Model
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    # Make Predictions and Evaluate the Model
                    predictions = model.predict(X_test)
                    r2_score = model.score(X_test, y_test)
                    st.write(f"### Model R² Score for {ticker}: {r2_score:.4f}")

                    # Plot Actual vs Predicted Prices
                    st.write(f"### Actual vs. Predicted Prices for {ticker}")
                    plt.figure(figsize=(14, 7))
                    plt.plot(np.arange(len(y_test)), y_test, label='Actual Prices')
                    plt.plot(np.arange(len(predictions)), predictions, label='Predicted Prices')
                    plt.title(f'{ticker} Stock Price Prediction')
                    plt.xlabel('Days')
                    plt.ylabel('Price in USD')
                    plt.legend()
                    st.pyplot(plt)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please enter a valid stock ticker symbol.")


def personalized_recommendation():
    # User Input: Investment Amount
    investment_amount = st.number_input("Enter the amount you're willing to invest (USD):", min_value=1000, value=5000, step=500)

    # User Input: Investment Strategy
    investment_strategy = st.selectbox("Select your investment strategy:", ["Long-term", "Short-term", "Open to Either"])

    # User Input: Preferred Sectors
    preferred_sectors = st.multiselect("Select preferred sectors to invest in (you can choose multiple):",
                                       list(sector_stocks.keys()), default=["Tech"])

    if st.button("Generate Investment Recommendations"):
        if investment_amount <= 0 or len(preferred_sectors) == 0:
            st.error("Please enter a valid investment amount and select at least one sector.")
        else:
            st.write(f"### Analysis for an investment of ${investment_amount} with a {investment_strategy} strategy in {', '.join(preferred_sectors)} sectors.")

            # Aggregate Stock Tickers Based on User's Sector Preference
            selected_stocks = [ticker for sector in preferred_sectors for ticker in sector_stocks[sector]]
            st.write(f"Analyzing the following stocks: {', '.join(selected_stocks)}")

            # Download Historical Stock Data
            stock_data = yf.download(selected_stocks, start="2020-01-01", end="2023-12-31")['Close']
            st.write("### Historical Stock Data (Close Prices)")
            st.dataframe(stock_data.tail())

            # Calculate Performance Metrics
            expected_returns = stock_data.pct_change().mean() * 252
            volatilities = stock_data.pct_change().std() * np.sqrt(252)

            # Generate Recommendations Based on Strategy
            st.write("### Recommended Portfolio Allocation")
            recommended_stocks = expected_returns.sort_values(ascending=False).index[:3]
            recommended_allocation = {stock: investment_amount / len(recommended_stocks) for stock in recommended_stocks}

            for stock, amount in recommended_allocation.items():
                st.write(f"**{stock}**: Invest **${amount:.2f}** - Expected Annual Return: **{expected_returns[stock]:.2%}** | Risk: **{volatilities[stock]:.2%}**")

            st.write("### Expected Portfolio Performance")
            portfolio_returns = stock_data[recommended_stocks].pct_change().mean(axis=1).cumsum() * investment_amount
            st.line_chart(portfolio_returns)

# Route to the Selected Page
if option == "Stock Analysis":
    stock_analysis()
else:
    personalized_recommendation()
