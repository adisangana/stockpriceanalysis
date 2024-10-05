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

# Custom CSS Styling for a More Professional Look
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; }
    .header { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .section-title { font-size: 22px; font-weight: bold; color: #4CAF50; margin-top: 10px; text-align: center; }
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; box-shadow: 2px 2px 8px rgba(0,0,0,0.1); margin: 10px; }
    .metric-card h3 { font-size: 18px; margin: 0; }
    .metric-card p { font-size: 16px; margin: 5px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 1. Create a Custom Header Section with a Search Bar
st.markdown("<div class='header'><h1>📈 Stock Price Analysis with News Sentiment</h1></div>", unsafe_allow_html=True)

# Stock Search Section
st.markdown("### Search for a Stock to Start Your Analysis")
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, TSLA):", value="AAPL").upper()

# 2. Stock Indices Cards (Example Indices)
st.markdown("### Global Stock Indices")
col1, col2, col3, col4 = st.columns(4)
col1.metric("S&P 500", "4,500", "+0.91%")
col2.metric("Nasdaq 100", "15,000", "+1.19%")
col3.metric("Dow Jones", "34,500", "+0.82%")
col4.metric("Russell 2000", "2,200", "+1.40%")

# Display Trending Stocks as Buttons
st.markdown("### Trending Stocks: [AAPL](#), [TSLA](#), [GOOGL](#), [AMZN](#)")

# Sidebar for Start and End Date
st.sidebar.title("User Input")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# NewsAPI and Sentiment Analyzer Setup
newsapi = NewsApiClient(api_key='f8f49c3781944e259ce310d89915cbff')
analyzer = SentimentIntensityAnalyzer()

# 3. Market Overview Section: Top Gainers and Top Losers
st.markdown("### Market Overview")
gainers, losers = st.columns(2)

with gainers:
    st.markdown("<div class='section-title'>Top Gainers 📈</div>", unsafe_allow_html=True)
    gainers_data = {"Symbol": ["AAPL", "TSLA", "NVDA"], "Name": ["Apple", "Tesla", "Nvidia"], "Price": ["$150", "$700", "$320"], "Change": ["+3.5%", "+5.2%", "+2.8%"]}
    gainers_df = pd.DataFrame(gainers_data)
    st.dataframe(gainers_df, use_container_width=True)

with losers:
    st.markdown("<div class='section-title'>Top Losers 📉</div>", unsafe_allow_html=True)
    losers_data = {"Symbol": ["GME", "AMC", "NIO"], "Name": ["GameStop", "AMC Entertainment", "NIO Inc."], "Price": ["$200", "$40", "$30"], "Change": ["-4.2%", "-3.5%", "-1.5%"]}
    losers_df = pd.DataFrame(losers_data)
    st.dataframe(losers_df, use_container_width=True)

# 4. Stock Analysis Section: Display Candlestick Chart and Data Analysis
if st.button("Analyze Stock"):
    st.markdown(f"### Stock Price Analysis for {ticker}")
    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        # Display Stock Data
        st.write(f"#### Stock Price Data for {ticker} ({start_date} to {end_date})")
        st.dataframe(data.head())

        st.write("#### Statistical Summary")
        st.dataframe(data.describe())

        # Create and Display a Candlestick Chart
        st.markdown(f"### {ticker} Stock Prices Candlestick Chart")
        fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], increasing_line_color='green', decreasing_line_color='red')])
        
        # Add Moving Averages
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA'))
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-Day SMA'))
        fig.update_layout(title=f'{ticker} Stock Price with Moving Averages', yaxis_title='Stock Price (USD)', xaxis_title='Date')
        st.plotly_chart(fig)

        # Prepare Data for Prediction
        data['Future Price'] = data['Close'].shift(-30)
        data = data.dropna()

        features = ['Close', 'SMA_50', 'SMA_200']
        X = data[features]
        y = data['Future Price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and Evaluate Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        r2_score = model.score(X_test, y_test)

        # Display Model Performance
        st.markdown(f"### Model R² Score for {ticker}: {r2_score:.4f}")

        # Actual vs. Predicted Prices
        st.markdown("### Actual vs. Predicted Prices")
        plt.figure(figsize=(14, 7))
        plt.plot(np.arange(len(y_test)), y_test, label='Actual Prices')
        plt.plot(np.arange(len(predictions)), predictions, label='Predicted Prices')
        plt.xlabel('Days')
        plt.ylabel('Price in USD')
        plt.legend()
        st.pyplot(plt)

        # Market News Section
        st.markdown(f"### 📰 Recent News and Sentiment Analysis for {ticker}")
        news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=5)

        if news['totalResults'] > 0:
            for article in news['articles']:
                title = article['title']
                description = article['description']
                url = article['url']
                sentiment = analyzer.polarity_scores(title + " " + (description if description else ""))

                st.markdown(f"**[{title}]({url})**")
                st.write(f"Sentiment Score: **{sentiment['compound']}** | Positive: {sentiment['pos']} | Neutral: {sentiment['neu']} | Negative: {sentiment['neg']}")
        else:
            st.write(f"No news articles found for {ticker}.")
