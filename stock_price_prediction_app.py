

import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)

# Function to preprocess data
def preprocess_data(data):
    data = data[['Close']].copy()  # Keep only the 'Close' price column
    data.dropna(inplace=True)
    return data

# Enhanced LSTM model
def create_lstm_model(sequence_length):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.3),
        LSTM(units=100, return_sequences=False),
        Dropout(0.3),
        Dense(units=50),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit app
st.title("📈 Real-Time Stock Price Prediction App")

# Input fields
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
prediction_date = st.date_input("Enter the Date:", value=pd.to_datetime("today"))

# Automatically set training range
train_start_date = "2020-01-01"
train_end_date = pd.to_datetime("today").strftime('%Y-%m-%d')

if st.button("Predict"):
    # Fetch data from Yahoo Finance
    data = yf.download(stock_symbol, start=train_start_date, end=train_end_date)
    
    if data.empty:
        st.error("No data available for the specified stock symbol.")
    else:
        # Preprocess data
        data = preprocess_data(data)

        # Prepare the data for training
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        sequence_length = 60
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(scaled_data[i + sequence_length])
        X, y = np.array(X), np.array(y)

        # Check if a saved model exists
        try:
            model = load_model('stock_price_model.h5')
        except:
            # Train the model and save it
            model = create_lstm_model(sequence_length)
            model.fit(X, y, batch_size=32, epochs=100, verbose=0)
            early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(X, y, batch_size=32, epochs=100, callbacks=[early_stop], verbose=0)
            model.save('stock_price_model.h5')

        # Predict on the entire dataset (for accuracy metrics)
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y.reshape(-1, 1))

        # Calculate accuracy metrics
        mae = mean_absolute_error(actual, predictions)
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        accuracy_rate = 100 - mape

        # Use the model to predict the stock price for the user-specified date
        prediction_date_str = prediction_date.strftime('%Y-%m-%d')

        # Ensure the prediction date is in the data range
        if prediction_date_str in data.index:
            date_index = data.index.get_loc(prediction_date_str)
            if date_index >= sequence_length:
                last_sequence = scaled_data[date_index - sequence_length:date_index]
                prediction = model.predict(last_sequence[np.newaxis, :, :])[0, 0]
                prediction = scaler.inverse_transform([[prediction]])[0][0]
                st.write(f"### Predicted Stock Price for {stock_symbol} on {prediction_date_str}: ${prediction:.2f}")
            else:
                st.error("Not enough data available before the prediction date to create input sequence.")
        else:
            st.error("Prediction date is not in the dataset.")

        # Display accuracy metrics
        st.write("### Model Accuracy Metrics:")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"Accuracy Rate: {accuracy_rate:.2f}%")

        # Plot actual vs predicted prices
        st.write("### Actual vs Predicted Prices")
        plt.figure(figsize=(14, 5))
        plt.plot(actual, label="Actual Prices", color="blue")
        plt.plot(predictions, label="Predicted Prices", color="red")
        plt.title("Actual vs Predicted Prices")
        plt.xlabel("Time Steps")
        plt.ylabel("Stock Price")
        plt.legend()
        st.pyplot(plt)





