import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_arima(train_series, order=(1, 0, 1)):
    model = ARIMA(train_series, order=order)
    arima_result = model.fit()
    return arima_result

def evaluate_arima(model, test_series):
    forecast = model.forecast(steps=len(test_series))
    mae = mean_absolute_error(test_series, forecast)
    mse = mean_squared_error(test_series, forecast)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_series, forecast)
    return forecast, mae, mse, rmse, r2

def plot_arima_results(train_series, test_series, forecast):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_series.index, train_series, label="Training Data", color='blue')
    ax.plot(test_series.index, test_series, label="Actual Test Data", color='green')
    ax.plot(test_series.index, forecast, label="Predicted Test Data", color='red')
    ax.set_title("Actual vs Predicted - ARIMA Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Evening_Differenced_1")
    ax.legend()
    return fig
    #st.pyplot(fig)

def revert_forecast_to_original_scale(forecast, original_series):
    last_original_value = original_series.iloc[-1]
    forecast_original = [last_original_value + forecast[0]]
    for i in range(1, len(forecast)):
        forecast_original.append(forecast_original[-1] + forecast[i])
    return pd.Series(forecast_original, index=forecast.index)

def plot_reverted_forecast(test_series, forecast_original_series):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_series.index, test_series, label="Actual Data", color="blue")
    ax.plot(test_series.index, forecast_original_series, label="Reverted Forecast", color="red")
    ax.set_title("Actual vs. Forecast (Original Scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price")
    ax.legend()
    return fig
    #st.pyplot(fig)
