#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Define helper functions
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), verbose=1)
    return model

# Streamlit app
st.title("📈 Stock Price Prediction with LSTM")

stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.button("Predict"):
    # Fetch stock data
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        st.error("No data available for the given stock symbol and date range.")
    else:
        # Preprocess data
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce').dropna()
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        # Create sequences
        sequence_length = 60
        X, y = create_sequences(scaled_data, sequence_length)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Train LSTM model
        model = train_lstm_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        train_loss = model.evaluate(X_train, y_train)
        test_loss = model.evaluate(X_test, y_test)
        st.write(f"Train Loss: {train_loss}")
        st.write(f"Test Loss: {test_loss}")

        # Predict and plot
        test_predictions = model.predict(X_test)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        fig, ax = plt.subplots()
        ax.plot(range(len(y_test_actual)), y_test_actual, label='Actual Prices', color='blue')
        ax.plot(range(len(test_predictions)), test_predictions, label='Predicted Prices', color='red')
        ax.set_title('LSTM Stock Price Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)

        # Forecast next 7 days
        last_sequence = scaled_data[-sequence_length:]
        future_predictions = []
        for _ in range(7):
            next_prediction = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
            future_predictions.append(next_prediction)
            last_sequence = np.append(last_sequence[1:], [[next_prediction]], axis=0)
        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        st.write("### Forecasted Prices for the Next 7 Days:")
        for i, price in enumerate(future_predictions, 1):
            st.write(f"Day {i}: ${price[0]:.2f}")


# In[ ]:




