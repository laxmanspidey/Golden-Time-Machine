import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

def plot_boxplots(data):
    """Plot boxplots for Morning and Evening prices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, col in enumerate(['Morning', 'Evening']):
        stats = data[col].describe()
        axes[i].boxplot(data[col])
        axes[i].set_title(f'Boxplot for {col} Prices')
        axes[i].set_ylabel('Price')
        
        # Annotate min, max, and median
        axes[i].scatter(1, stats['max'], label=f'Max: {stats["max"]:.2f}', color='orange')
        axes[i].scatter(1, stats['min'], label=f'Min: {stats["min"]:.2f}', color='blue')
        axes[i].scatter(1, stats['50%'], label=f'Median: {stats["50%"]:.2f}', color='green')
        axes[i].legend()
    
    return fig

def plot_time_series(data):
    """Plot time series for Morning and Evening prices."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    data['Morning'].plot(ax=axes[0], title='Gold Price - Morning', grid=True)
    data['Evening'].plot(ax=axes[1], title='Gold Price - Evening', grid=True)
    
    return fig

def plot_rolling_statistics(data, window=30):
    """Plot rolling mean and standard deviation."""
    normalized_data = data.copy()
    for col in ['Morning', 'Evening']:
        min_val = normalized_data[col].min()
        max_val = normalized_data[col].max()
        normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    for i, col in enumerate(['Morning', 'Evening']):
        normalized_data[f'{col}_RollingMean'] = normalized_data[col].rolling(window=window).mean()
        normalized_data[f'{col}_RollingStd'] = normalized_data[col].rolling(window=window).std()
        
        axes[i].plot(normalized_data[col], label=f'{col} Rate')
        axes[i].plot(normalized_data[f'{col}_RollingMean'], label=f'{window}-Day Rolling Mean', color='orange')
        axes[i].plot(normalized_data[f'{col}_RollingStd'], label=f'{window}-Day Rolling Std Dev', color='green')
        axes[i].set_title(f"Rolling Mean & Std Deviation for {col} Rate")
        axes[i].legend()
    
    return fig

def plot_decomposition(data, model='additive'):
    """Decompose time series into trend, seasonal, and residual components."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    #if data.index.freq is None:
    #   data = data.asfreq('D')  # Set frequency to daily
    
    for i, col in enumerate(['Morning', 'Evening']):
        decomposition = sm.tsa.seasonal_decompose(data[col], model=model)
            # Manually plot the decomposition components
        decomposition.observed.plot(ax=axes[0], title='Observed')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.close()
    return fig
