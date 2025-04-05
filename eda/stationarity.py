import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def difference_data(data, column):
    """Difference the data to make it stationary."""
    data[f'{column}_Differenced_1'] = data[column].diff().dropna()
    return data

def plot_scatter_comparison(data, column):
    """Plot scatter plots for original and differenced data."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original Data: yt vs yt-1
    axes[0, 0].scatter(data.iloc[:-1][column], data.iloc[1:][column], color='blue')
    axes[0, 0].set_title("Original: yt vs yt-1")
    axes[0, 0].set_xlabel("yt-1")
    axes[0, 0].set_ylabel("yt")
    axes[0, 0].grid(True)
    
    # Differenced Data: yt vs yt-1
    diff_column = f'{column}_Differenced_1'
    axes[0, 1].scatter(data.iloc[1:-1][diff_column], data.iloc[2:][diff_column], color='orange')
    axes[0, 1].set_title("Differenced: yt vs yt-1")
    axes[0, 1].set_xlabel("yt-1")
    axes[0, 1].set_ylabel("yt")
    axes[0, 1].grid(True)
    
    # Original Data: z_t vs z_t+lag_k
    lag_k = 730
    z_t = data.iloc[:-lag_k][column]
    z_t_k = data.iloc[lag_k:][column]
    axes[1, 0].scatter(z_t, z_t_k, color='blue')
    axes[1, 0].set_title(f"Original: z_t vs z_t+{lag_k}")
    axes[1, 0].set_xlabel("z_t")
    axes[1, 0].set_ylabel(f"z_t+{lag_k}")
    axes[1, 0].grid(True)
    
    # Differenced Data: z_t vs z_t+lag_k
    z_t_diff = data.iloc[1:-lag_k][diff_column]
    z_t_k_diff = data.iloc[lag_k:-1][diff_column]
    axes[1, 1].scatter(z_t_diff, z_t_k_diff, color='orange')
    axes[1, 1].set_title(f"Differenced: z_t vs z_t+{lag_k}")
    axes[1, 1].set_xlabel("z_t")
    axes[1, 1].set_ylabel(f"z_t+{lag_k}")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.close()
    return fig

def plot_lagged_scatter_comparison(data, column):
    """Plot scatter plots for lagged data (yt vs yt+300 and yt vs yt-300)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original Data: yt vs yt+300
    axes[0, 0].scatter(data.iloc[500:800][column], data.iloc[500+300:800+300][column], color='blue')
    axes[0, 0].set_title("Original: yt vs yt+300")
    axes[0, 0].set_xlabel("yt")
    axes[0, 0].set_ylabel("yt+300")
    axes[0, 0].grid(True)
    
    # Differenced Data: yt vs yt+300
    diff_column = f'{column}_Differenced_1'
    axes[0, 1].scatter(data.iloc[500:800][diff_column], data.iloc[500+300:800+300][diff_column], color='orange')
    axes[0, 1].set_title("Differenced: yt vs yt+300")
    axes[0, 1].set_xlabel("yt")
    axes[0, 1].set_ylabel("yt+300")
    axes[0, 1].grid(True)
    
    # Original Data: yt vs yt-300
    axes[1, 0].scatter(data.iloc[500:800][column], data.iloc[500-300:800-300][column], color='blue')
    axes[1, 0].set_title("Original: yt vs yt-300")
    axes[1, 0].set_xlabel("yt")
    axes[1, 0].set_ylabel("yt-300")
    axes[1, 0].grid(True)
    
    # Differenced Data: yt vs yt-300
    axes[1, 1].scatter(data.iloc[500:800][diff_column], data.iloc[500-300:800-300][diff_column], color='orange')
    axes[1, 1].set_title("Differenced: yt vs yt-300")
    axes[1, 1].set_xlabel("yt")
    axes[1, 1].set_ylabel("yt-300")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.close()
    return fig

def plot_time_series_comparison(data, column):
    """Plot time series and rolling statistics for original and differenced data."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original Data: Time Series
    axes[0, 0].plot(data.index, data[column], color='blue', label='Original')
    axes[0, 0].set_title(f'Original {column} Prices')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Differenced Data: Time Series
    diff_column = f'{column}_Differenced_1'
    axes[0, 1].plot(data.index, data[diff_column], color='orange', label='Differenced')
    axes[0, 1].set_title(f'Differenced {column} Prices')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Original Data: Rolling Mean and Std Dev
    axes[1, 0].plot(data[column].rolling(window=30).mean(), label='Rolling Mean', color='red')
    axes[1, 0].plot(data[column].rolling(window=30).std(), label='Rolling Std Dev', color='green')
    axes[1, 0].set_title(f'Rolling Stats for {column}')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Differenced Data: Rolling Mean and Std Dev
    axes[1, 1].plot(data[diff_column].rolling(window=30).mean(), label='Rolling Mean', color='red')
    axes[1, 1].plot(data[diff_column].rolling(window=30).std(), label='Rolling Std Dev', color='green')
    axes[1, 1].set_title(f'Rolling Stats for {diff_column}')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.close()
    return fig

def plot_autocorrelation_comparison(data, column, lags=50):
    """Plot ACF and PACF for original and differenced data."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Original Data: ACF
    plot_acf(data[column].dropna(), lags=lags, ax=axes[0, 0])
    axes[0, 0].set_title(f'ACF for {column}')
    
    # Differenced Data: ACF
    diff_column = f'{column}_Differenced_1'
    plot_acf(data[diff_column].dropna(), lags=lags, ax=axes[0, 1])
    axes[0, 1].set_title(f'ACF for {diff_column}')
    
    # Original Data: PACF
    plot_pacf(data[column].dropna(), lags=lags, ax=axes[1, 0])
    axes[1, 0].set_title(f'PACF for {column}')
    
    # Differenced Data: PACF
    plot_pacf(data[diff_column].dropna(), lags=lags, ax=axes[1, 1])
    axes[1, 1].set_title(f'PACF for {diff_column}')
    
    plt.tight_layout()
    plt.close()
    return fig

def print_statistics(data, column):
    """Print statistics for original and differenced data."""
    diff_column = f'{column}_Differenced_1'
    
    st.write("### Original Data Statistics")
    st.write(f"Mean: {data[column].mean():.2f}")
    st.write(f"Standard Deviation: {data[column].std():.2f}")
    st.write(f"Rolling Mean (last 30 days): {data[column].rolling(window=30).mean().iloc[-1]:.2f}")
    st.write(f"Rolling Std Dev (last 30 days): {data[column].rolling(window=30).std().iloc[-1]:.2f}")
    
    st.write("### Differenced Data Statistics")
    st.write(f"Mean: {data[diff_column].mean():.2f}")
    st.write(f"Standard Deviation: {data[diff_column].std():.2f}")
    st.write(f"Rolling Mean (last 30 days): {data[diff_column].rolling(window=30).mean().iloc[-1]:.2f}")
    st.write(f"Rolling Std Dev (last 30 days): {data[diff_column].rolling(window=30).std().iloc[-1]:.2f}")

def print_mean_comparison(data, column):
    """Print mean values for original and differenced data."""
    diff_column = f'{column}_Differenced_1'
    
    st.write("### Mean Comparison")
    st.write("#### Original Data")
    st.write(f"Mean of yt-1: {data.iloc[:-1][column].mean():.2f}")
    st.write(f"Mean of yt: {data.iloc[1:][column].mean():.2f}")
    
    st.write("#### Differenced Data")
    st.write(f"Mean of yt-1: {data.iloc[1:-1][diff_column].mean():.2f}")
    st.write(f"Mean of yt: {data.iloc[2:][diff_column].mean():.2f}")