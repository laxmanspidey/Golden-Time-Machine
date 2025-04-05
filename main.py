import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from config import CITIES, DB_PATH

from database.db_handler import GoldPriceDB
from data_pipeline.scraper import GoldPriceScraper

from eda.data_analysis import preprocess_data, calculate_statistics
from eda.visualization import plot_boxplots, plot_time_series, plot_rolling_statistics, plot_decomposition
from eda.stationarity import * #difference_data, plot_stationarity_comparison, print_stationarity_stats, plot_scatter_comparison, plot_autocorrelation

from models.arima_model import train_arima, evaluate_arima, plot_arima_results, revert_forecast_to_original_scale, plot_reverted_forecast
from models.lstm_model import create_sequences, build_lstm_model, train_lstm, evaluate_lstm, plot_lstm_results
from models.prophet_model import train_prophet, evaluate_prophet, plot_prophet_results, reconstruct_forecast, plot_reconstructed_forecast, find_optimal_purchase_dates

from sklearn.preprocessing import MinMaxScaler

def data_collection(city):
    # Database Initialization
    db = GoldPriceDB(DB_PATH)
    scraper = GoldPriceScraper(city)

    today = datetime.today()
    previous_day = today - timedelta(days=1)
    previous_day = previous_day.replace(hour=0, minute=0, second=0, microsecond=0)

    if not db.check_city_data(city):
        st.warning(f"No historical data found for {city}. Initializing data collection...")
        
        start_date = datetime(2021, 8, 1)
        with st.spinner(f"Scraping data from {start_date.strftime('%Y-%m-%d')} to {previous_day.strftime('%Y-%m-%d')}..."):
            try:
                complete_data = scraper.scrape_range(start_date, previous_day)
                db.update_data(city, complete_data)
                st.success(f"Successfully added {len(complete_data)} records for {city}!")
                st.session_state.data_collected = True
            except Exception as e:
                st.error(f"Error scraping initial data: {str(e)}")
                st.session_state.data_collected = False
                return                

    else:
        latest_date_str = db.get_latest_date(city)
        latest_date = datetime.strptime(latest_date_str, "%Y-%m-%d") # strptime stands for "string parse time." It is a method in Python used to convert a string representation of a date and time into a datetime object, based on a specified format.
        print("Latest Date: ",latest_date)
        print("Previous day: ",previous_day)
        if latest_date < previous_day:
            # Calculate the start date for updating (next day after latest date)
            update_start_date = latest_date + timedelta(days=1)

            st.info(f"Updating data from {update_start_date.strftime('%Y-%m-%d')} to {previous_day.strftime('%Y-%m-%d')}...") # In Python, strftime stands for "string format time." It is a method used to format datetime objects into readable strings according to a specified format.
    
            with st.spinner("Fetching latest prices..."):
                try:
                    new_data = scraper.scrape_range(update_start_date, previous_day)
                            
                    if not new_data.empty:
                        db.update_data(city, new_data)
                        st.success(f"Successfully updated {len(new_data)} new records!")
                        st.session_state.data_collected = True
                    else:
                        #st.warning("No new data found to update.")
                        st.success("Database is already up to date!")
                        st.session_state.data_collected = True

                except Exception as e:
                    st.error(f"Error updating data: {str(e)}")
                    st.session_state.data_collected = False
                    return
        else:
            st.success("Database is already up to date!")
            st.session_state.data_collected = True

def perform_eda(city):
    """Perform Exploratory Data Analysis and display results."""
    db = GoldPriceDB(DB_PATH)
    data = db.get_all_data(city)
    data = preprocess_data(data)
    
    st.subheader("Data Statistics")
    stats = calculate_statistics(data)
    st.write(f"Total null values: {stats['null_values']}")
    #st.write(f"Total duplicate rows: {stats['duplicates']}")
    st.write(f"Outliers - Morning: {stats['outliers']['Morning']}, Evening: {stats['outliers']['Evening']}")
    
    st.subheader("Visualizations")
    
    st.write("### Boxplots for Morning and Evening Prices")
    st.pyplot(plot_boxplots(data))
    
    st.write("### Time Series of Gold Prices")
    st.pyplot(plot_time_series(data))
    
    st.write("### Rolling Mean and Standard Deviation")
    st.pyplot(plot_rolling_statistics(data))
    
    st.write("### Time Series Decomposition using additive model")
    st.pyplot(plot_decomposition(data, model='additive'))

    st.write("### Time Series Decomposition using multiplicative model")
    st.pyplot(plot_decomposition(data, model='multiplicative'))

    #st.write("### Autocorrelation and Partial Autocorrelation")
    #st.pyplot(plot_autocorrelation(data))

    # Stationarity Analysis
    st.subheader("Stationarity Analysis")
    data = difference_data(data, 'Evening')
    
    st.write("### Scatter Plots")
    st.pyplot(plot_scatter_comparison(data, 'Evening'))
    st.pyplot(plot_lagged_scatter_comparison(data, 'Evening'))
    
    st.write("### Time Series and Rolling Statistics")
    st.pyplot(plot_time_series_comparison(data, 'Evening'))
    
    st.write("### Autocorrelation Analysis")
    st.pyplot(plot_autocorrelation_comparison(data, 'Evening'))
    
    st.write("### Statistics")
    print_statistics(data, 'Evening')
    print_mean_comparison(data, 'Evening')

    st.session_state.eda_performed = True

def perform_arima_analysis(city):
    db = GoldPriceDB(DB_PATH)
    data = db.get_all_data(city)
    data = preprocess_data(data)
    data = difference_data(data, 'Evening')
    
    train_series = pd.to_numeric(data['Evening_Differenced_1'], errors='coerce').dropna()
    train_series.index.freq = pd.infer_freq(train_series.index)
    
    test_series = pd.to_numeric(data['Evening_Differenced_1'], errors='coerce').dropna()
    test_series.index.freq = pd.infer_freq(test_series.index)
    
    model = train_arima(train_series)
    forecast, mae, mse, rmse, r2 = evaluate_arima(model, test_series)
    
    st.subheader("ARIMA Model Results")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R-squared (R2 Score): {r2}")
    
    st.pyplot(plot_arima_results(train_series, test_series, forecast))
    
    #forecast_original_series = revert_forecast_to_original_scale(forecast, data['Evening'])
    #st.pyplot(plot_reverted_forecast(test_series, forecast_original_series))

def perform_lstm_analysis(city):
    db = GoldPriceDB(DB_PATH)
    data = db.get_all_data(city)
    data = preprocess_data(data)
    data = difference_data(data, 'Evening')
    
    train_series = pd.to_numeric(data['Evening_Differenced_1'], errors='coerce').dropna()
    train_series.index.freq = pd.infer_freq(train_series.index)
    
    test_series = pd.to_numeric(data['Evening_Differenced_1'], errors='coerce').dropna()
    test_series.index.freq = pd.infer_freq(test_series.index)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
    test_scaled = scaler.transform(test_series.values.reshape(-1, 1))
    
    seq_length = 10
    x_train, y_train = create_sequences(train_scaled, seq_length)
    x_test, y_test = create_sequences(test_scaled, seq_length)
    
    model = build_lstm_model(seq_length)
    model = train_lstm(model, x_train, y_train)
    
    predicted, actual, mae, mse, rmse, r2 = evaluate_lstm(model, x_test, y_test, scaler)
    
    st.subheader("LSTM Model Results")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R-squared (R2 Score): {r2}")
    
    st.pyplot(plot_lstm_results(test_series, actual, predicted, seq_length))

def perform_prophet_analysis(city):
    db = GoldPriceDB(DB_PATH)
    data = db.get_all_data(city)
    data = preprocess_data(data)
    data = difference_data(data, 'Evening')
    
    train_series = pd.to_numeric(data['Evening_Differenced_1'], errors='coerce').dropna()
    train_series.index.freq = pd.infer_freq(train_series.index)
    
    test_series = pd.to_numeric(data['Evening_Differenced_1'], errors='coerce').dropna()
    test_series.index.freq = pd.infer_freq(test_series.index)
    
    train_df = train_series.reset_index()
    test_df = test_series.reset_index()
    
    train_df.columns = ['ds', 'y']
    test_df.columns = ['ds', 'y']
    
    model = train_prophet(train_df)
    forecast, forecasted_values, mae, mse, rmse, r2 = evaluate_prophet(model, test_df)
    
    st.subheader("Prophet Model Results")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R-squared (R2 Score): {r2}")
    
    st.pyplot(plot_prophet_results(test_df, forecasted_values))
    
    #reconstructed_df = reconstruct_forecast(forecasted_values, data, test_df)
    #st.pyplot(plot_reconstructed_forecast(reconstructed_df))

def find_optimal_purchase_date(city, start_date, end_date):
    """Finds the optimal day(s) for purchasing gold within the given date range."""
    db = GoldPriceDB(DB_PATH)
    data = db.get_all_data(city)
    data = preprocess_data(data)
    
    data = data.reset_index()

    train_df = data[['Date', 'Evening']].rename(columns={'Date': 'ds', 'Evening': 'y'})
    model = train_prophet(train_df)
    
    future = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date)})
    forecast = model.predict(future)
    
    optimal_dates = find_optimal_purchase_dates(forecast, start_date, end_date)
    
    st.subheader("Optimal Purchase Dates")
    if not optimal_dates.empty:
        st.write("The best day(s) to purchase gold are:")
        st.dataframe(optimal_dates)
    else:
        st.warning("No optimal dates found in the selected range.")
    
    # Display the full forecasted prices
    st.subheader("Forecasted Gold Prices")
    #forecast_display = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Expected Price'})
    #forecast_display['Date'] = forecast_display['Date'].dt.strftime('%d-%m-%Y')  # Format date
    #forecast_display['Expected Price'] = forecast_display['Expected Price'].astype(int)
    #st.write(forecast_display)

    forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', 'yhat': 'Expected Price', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
    )
    forecast_display['Date'] = forecast_display['Date'].dt.strftime('%d-%m-%Y')  # Format date
    forecast_display['Expected Price'] = forecast_display['Expected Price'].astype(int)  # Convert price to integer
    forecast_display['Lower Bound'] = forecast_display['Lower Bound'].astype(int)  # Convert lower bound to integer
    forecast_display['Upper Bound'] = forecast_display['Upper Bound'].astype(int)  # Convert upper bound to integer

    # Calculate the reliability percentage (based on the confidence interval)
    forecast_display['Reliability (%)'] = (
        (1 - (forecast_display['Upper Bound'] - forecast_display['Lower Bound']) / forecast_display['Expected Price']) * 100
    ).round(2)

    st.write(forecast_display)
def main():
    st.set_page_config(page_title="Golden Time Machine", layout="wide")

    if 'data_collected' not in st.session_state:
        st.session_state.data_collected = False

    if 'eda_performed' not in st.session_state:
        st.session_state.eda_performed = False

    # Header Section
    st.title("💰 Golden Time Machine")
    st.markdown("---")
    
    # City Selection
    city = st.selectbox("Select a city from where you want to buy gold: ", CITIES)

    st.warning("Please confirm your selection to proceed.")
    if st.button("Confirm Selection"):
        data_collection(city)

    if st.button("Perform EDA"):
        if st.session_state.data_collected:
            perform_eda(city)
        else:
            st.warning("Please collect data first by clicking 'Confirm Selection'.")

    if st.button("Model the data using ARIMA, LSTM and Prophet models"):
        if st.session_state.eda_performed:
            perform_arima_analysis(city)
            perform_lstm_analysis(city)
            perform_prophet_analysis(city)
        else:
            st.warning("Please perform EDA first by clicking 'Perform EDA'.")        

    # Forecasting    
    st.subheader("Find Optimal Purchase Date")
    start_date = st.date_input("Select start date", datetime.today())
    end_date = st.date_input("Select end date", datetime.today() + timedelta(days=30))

    if start_date > end_date:
        st.error("End date must be greater than or equal to start date.")
    else:
        if st.button("Find Optimal Purchase Date"):
            if st.session_state.data_collected:
                find_optimal_purchase_date(city, start_date, end_date)
            else:
                st.warning("Please collect data first by clicking 'Confirm Selection'.")
    #else:
    #    st.stop()  
    
if __name__ == "__main__":
    main()
