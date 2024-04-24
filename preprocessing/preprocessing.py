import pandas as pd
import numpy as np
import statsmodels.api as sm
import yfinance as yf
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


def data_collection(stock_symbol, period, interval):
    try:
        start = (datetime.date.today() - datetime.timedelta(period))
        end = datetime.date.today()
        data = yf.Ticker(stock_symbol).history(start=start, end=end, interval=interval)
        return data
    except Exception as e:
        print(f"Error:{e}")


def data_cleansing(dataset):
    # Replace '0' values with 'nan'
    dataset[['Open', 'High', 'Low', 'Close']] = dataset[['Open', 'High', 'Low', 'Close']].replace(0, np.nan)

    # Define the rolling window size (n)
    n = 3

    # Transform and replace missing values
    dataset = dataset.fillna(dataset.rolling(window=n, min_periods=1, center=True).mean())

    return dataset


def log_transformation(dataset):
    # Apply a log transformation to stabilize non-constant variance and reduce the magnitude of values
    dataset['Close'] = np.log(dataset['Close'])
    # Drop NaN values
    dataset.dropna(inplace=True)
    return dataset


def differencing(dataset):
    # Compute the difference between consecutive observations to remove trends
    dataset['Close'] = dataset['Close'].diff()
    dataset.dropna(inplace=True)
    return dataset


def linear_regression(dataset):
    # Create a time variable
    dataset['Time'] = range(1, len(dataset) + 1)

    # Fit a linear regression model
    X = sm.add_constant(dataset['Time'])
    model = sm.OLS(dataset['Close'], X)
    results = model.fit()

    # Remove trend
    dataset['Close'] = dataset['Close'] - results.fittedvalues
    return dataset


def seasonal_decompose_ts(dataset, period):
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(dataset['Close'], model='additive', period=period)
    dataset['Close'] = dataset['Close'] - decomposition.seasonal
    return dataset


