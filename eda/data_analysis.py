import pandas as pd

def preprocess_data(data):
    """Preprocess the data for EDA."""
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df.set_index('Date', inplace=True)

    # Convert Morning and Evening columns to numeric
    for col in ['Morning', 'Evening']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.asfreq('D')
    return df

def calculate_statistics(data):
    """Calculate basic statistics and outliers."""
    stats = {
        "null_values": data.isnull().sum().sum(),
        "outliers": {}
    }

    # Calculate IQR and outliers for Morning and Evening prices
    for col in ['Morning', 'Evening']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
        stats["outliers"][col] = outliers.sum()

    return stats