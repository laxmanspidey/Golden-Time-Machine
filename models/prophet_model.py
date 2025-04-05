import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_prophet(train_df):
    model = Prophet()
    model.fit(train_df)
    return model

def evaluate_prophet(model, test_df):
    future = pd.DataFrame({'ds': test_df['ds']})
    forecast = model.predict(future)
    forecasted_values = forecast['yhat'].values
    mae = mean_absolute_error(test_df['y'], forecasted_values)
    mse = mean_squared_error(test_df['y'], forecasted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_df['y'], forecasted_values)
    return forecast, forecasted_values, mae, mse, rmse, r2

def plot_prophet_results(test_df, forecasted_values):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_df['ds'], test_df['y'], label="Actual Test Data", color='green')
    ax.plot(test_df['ds'], forecasted_values, label="Predicted Test Data", color='red')
    ax.set_title("Actual vs Predicted - Prophet Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Evening_Differenced_1")
    ax.legend()
    return fig

def reconstruct_forecast(forecasted_values, train_data, test_df):
    last_original_value = train_data['Evening'].iloc[-1]
    reconstructed_forecast = [last_original_value]
    for diff_value in forecasted_values:
        next_value = reconstructed_forecast[-1] + diff_value
        reconstructed_forecast.append(next_value)
    reconstructed_forecast = reconstructed_forecast[1:]
    return pd.DataFrame({
        'ds': test_df['ds'],
        'actual': test_df['y'].values,
        'predicted': reconstructed_forecast
    })

def plot_reconstructed_forecast(reconstructed_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(reconstructed_df['ds'], reconstructed_df['actual'], label="Actual Data", color='green')
    ax.plot(reconstructed_df['ds'], reconstructed_df['predicted'], label="Reconstructed Predicted Data", color='red')
    ax.set_title("Actual vs Predicted - Reconstructed")
    ax.set_xlabel("Date")
    ax.set_ylabel("Evening Gold Prices")
    ax.legend()
    return fig

def find_optimal_purchase_dates(forecast, start_date, end_date):
    """
    Finds the optimal day(s) for purchasing gold within the given date range.

    Args:
        forecast (pd.DataFrame): Forecasted values from the Prophet model.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing the optimal purchase dates and prices.
    """

    # Convert start_date and end_date to datetime64[ns] for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the forecasted data for the selected date range
    mask = (forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)
    filtered_forecast = forecast.loc[mask]

    # Find the minimum price and corresponding dates
    min_price = filtered_forecast['yhat'].min()
    optimal_dates = filtered_forecast[filtered_forecast['yhat'] == min_price]

    #optimal_dates = optimal_dates[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Expected Price'})
    #optimal_dates['Date'] = optimal_dates['Date'].dt.strftime('%d-%m-%Y')  # Format date
    #optimal_dates['Expected Price'] = optimal_dates['Expected Price'].astype(int)

    optimal_dates = optimal_dates[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', 'yhat': 'Expected Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
    )
    optimal_dates['Date'] = optimal_dates['Date'].dt.strftime('%d-%m-%Y')  # Format date
    optimal_dates['Expected Price'] = optimal_dates['Expected Price'].astype(int)  # Convert price to integer
    optimal_dates['Lower Bound'] = optimal_dates['Lower Bound'].astype(int)  # Convert lower bound to integer
    optimal_dates['Upper Bound'] = optimal_dates['Upper Bound'].astype(int)  # Convert upper bound to integer

    # Calculate the reliability percentage (based on the confidence interval)
    optimal_dates['Reliability (%)'] = (
        (1 - (optimal_dates['Upper Bound'] - optimal_dates['Lower Bound']) / optimal_dates['Expected Price']) * 100
    ).round(2)

    return optimal_dates