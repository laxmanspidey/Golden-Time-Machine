import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(x), np.array(y)

def build_lstm_model(seq_length):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm(model, x_train, y_train, epochs=20, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def evaluate_lstm(model, x_test, y_test, scaler):
    predicted_scaled = model.predict(x_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return predicted, actual, mae, mse, rmse, r2

def plot_lstm_results(test_series, actual, predicted, seq_length):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(test_series.index[seq_length:], actual, label="Actual Test Data", color='green')
    ax.plot(test_series.index[seq_length:], predicted, label="Predicted Test Data", color='red')
    ax.set_title("Actual vs Predicted - LSTM Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Evening_Differenced_1")
    ax.legend()
    return fig