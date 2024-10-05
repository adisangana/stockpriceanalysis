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

# Apply custom CSS to style the app more like the reference image
st.markdown(
    """
    <style>
    .main { background-color: #F0F2F6; }
    .stButton>button { background-color: #4CAF50; color: white; }
    .stSidebar { background-color: #333; color: white; }
    .block-container { padding: 20px; }
    .stMarkdown h2 { color: #4CAF50; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App Title with a Custom Header
st.markdown("""
    <div style="background-color: #4CAF50; padding: 10px; border-radius: 10px;">
        <h1 style="text-align: center; color: white;">📈 Stock Price Analysis with News Sentiment</h1>
    </div>
""", unsafe_allow_html=True)

# Sidebar for User Input
st.sidebar.title("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# NewsAPI Client Initialization
newsapi = NewsApiClient(api_key='f8f49c3781944e259ce310d89915cbff')
analyzer = SentimentIntensityAnalyzer()

# Fetch and analyze data when the button is pressed
if st.sidebar.button("Analyze Stock"):
    if ticker:
        try:
            # Step 1: Download Stock Data
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error(f"Could not retrieve data for {ticker}. Check the ticker symbol or date range and try again.")
            else:
                # Use Columns to Create a More Structured Layout
                col1, col2 = st.columns([2, 3])

                with col1:
                    # Display DataFrame and Summary
                    st.markdown(f"## 📅 Stock Price Data for {ticker}")
                    st.dataframe(data.head())

                    st.markdown("## 📊 Statistical Summary")
                    st.dataframe(data.describe())

                with col2:
                    # Candlestick Chart
                    st.markdown(f"## 📈 {ticker} Stock Prices Candlestick Chart")
                    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                         open=data['Open'],
                                                         high=data['High'],
                                                         low=data['Low'],
                                                         close=data['Close'],
                                                         increasing_line_color='green',
                                                         decreasing_line_color='red')])

                    # Add Moving Averages
                    data['SMA_50'] = data['Close'].rolling(window=50).mean()
                    data['SMA_200'] = data['Close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA'))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-Day SMA'))
                    fig.update_layout(title=f'{ticker} Stock Price with Moving Averages',
                                      yaxis_title='Stock Price (USD)',
                                      xaxis_title='Date')
                    st.plotly_chart(fig)

                # Step 4: Prepare Data for Prediction
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

                # Step 8: Fetch and Display News Sentiment
                st.markdown(f"## 📰 Recent News and Sentiment Analysis for {ticker}")
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
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid stock ticker symbol.")
