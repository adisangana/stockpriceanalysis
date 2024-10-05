import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

def validate_ticker(ticker):
    # Attempt to download a small amount of data to check validity
    try:
        test_data = yf.Ticker(ticker).history(period='1d')
        if test_data.empty:
            return False
        return True
    except:
        return False

# Step 1: Loop to Prompt the User for a Valid Stock Ticker Symbol
while True:
    ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple, MSFT for Microsoft): ").upper()
    
    # Validate the ticker before attempting to download full data
    if not validate_ticker(ticker):
        print(f"Ticker '{ticker}' is not a valid or tradable symbol. Please enter a valid stock ticker symbol.")
        continue
    
    print(f"Fetching data for {ticker}...")
    
    # Step 2: Download Stock Data based on user input
    data = yf.download(ticker, start="2018-01-01", end="2023-12-31")
    break  # Break loop if the ticker is valid and data is successfully downloade

# Step 3: Display the first few rows of the data
print(f"\nFirst few rows of the dataset for {ticker}:")
print(data.head())

# Step 4: Basic Information and Summary Statistics
print("\nBasic information about the dataset:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Step 5: Visualize the Closing Prices Over Time
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label=f'{ticker} Closing Prices')
plt.title(f'{ticker} Stock Price (2018-2023)')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.legend()
plt.show()

# Step 6: Calculate Moving Averages
data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day Simple Moving Average

# Plot Moving Averages
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Closing Prices')
plt.plot(data['SMA_50'], label='50-Day SMA')
plt.plot(data['SMA_200'], label='200-Day SMA')
plt.title(f'{ticker} Stock Price with Moving Averages (2018-2023)')
plt.xlabel('Date')
plt.ylabel('Price in USD')
plt.legend()
plt.show()

# Step 7: Prepare the data for modeling
data['Future Price'] = data['Close'].shift(-30)  # Predict 30 days into the future
data = data.dropna()  # Drop rows with NaN values

# Select features and target
features = ['Close', 'SMA_50', 'SMA_200']
X = data[features]
y = data['Future Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 9: Evaluate the Model
predictions = model.predict(X_test)
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(y_test)), y_test, label='Actual Prices')
plt.plot(np.arange(len(predictions)), predictions, label='Predicted Prices')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price in USD')
plt.legend()
plt.show()

# Display Model Performance
print(f"\nModel R^2 Score for {ticker}: {model.score(X_test, y_test):.4f}")
