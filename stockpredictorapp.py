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

# Streamlit App Title
st.title("📈 Stock Price Analysis with News Sentiment")

# Sidebar: User Input for Stock Ticker
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL").upper()

# Sidebar: Date Range Input
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))

# NewsAPI Client Initialization
newsapi = NewsApiClient(api_key='your_newsapi_key_here')  # Replace with your NewsAPI key
analyzer = SentimentIntensityAnalyzer()

# Fetch data when the button is pressed
if st.sidebar.button("Analyze Stock"):
    if ticker:
        try:
            # Step 1: Download Stock Data
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error(f"Could not retrieve data for {ticker}. Check the ticker symbol or date range and try again.")
            else:
                # Step 2: Display the DataFrame and Statistical Summary
                st.write(f"### Stock Price Data for {ticker} ({start_date} to {end_date})")
                st.write(data.head())

                st.write("#### Statistical Summary")
                st.write(data.describe())

                # Step 3: Visualize the Stock Prices with Candlestick Chart
                st.write(f"### {ticker} Stock Prices Candlestick Chart")

                # Create a Plotly candlestick chart
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
                
                # Render the candlestick chart in Streamlit
                st.plotly_chart(fig)

                # Step 4: Prepare Data for Machine Learning
                data['Future Price'] = data['Close'].shift(-30)
                data = data.dropna()

                # Select features and target
                features = ['Close', 'SMA_50', 'SMA_200']
                X = data[features]
                y = data['Future Price']

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Step 5: Train a Linear Regression Model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Step 6: Make Predictions and Evaluate the Model
                predictions = model.predict(X_test)
                r2_score = model.score(X_test, y_test)
                
                # Display R² Score
                st.write(f"### Model R² Score for {ticker}: {r2_score:.4f}")

                # Step 7: Plot Actual vs Predicted Prices
                st.write(f"### Actual vs. Predicted Prices for {ticker}")
                plt.figure(figsize=(14, 7))
                plt.plot(np.arange(len(y_test)), y_test, label='Actual Prices')
                plt.plot(np.arange(len(predictions)), predictions, label='Predicted Prices')
                plt.title(f'{ticker} Stock Price Prediction')
                plt.xlabel('Days')
                plt.ylabel('Price in USD')
                plt.legend()
                st.pyplot(plt)

                # Step 8: Fetch and Analyze Recent News Articles
                st.write(f"### Recent News and Sentiment Analysis for {ticker}")
                news = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=5)

                # Display News Headlines and Sentiment Scores
                if news['totalResults'] > 0:
                    for article in news['articles']:
                        title = article['title']
                        description = article['description']
                        url = article['url']
                        sentiment = analyzer.polarity_scores(title + " " + (description if description else ""))

                        st.write(f"**[{title}]({url})**")
                        st.write(f"Sentiment Score: **{sentiment['compound']}** | Positive: {sentiment['pos']} | Neutral: {sentiment['neu']} | Negative: {sentiment['neg']}")
                else:
                    st.write(f"No news articles found for {ticker}.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid stock ticker symbol.")
