import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.model_selection import train_test_split


def arima_predict(dataset, save_path):
    model2 = pm.auto_arima(dataset['Close'], max_p=14, max_d=2, max_q=5, trace=True, error_action='ignore',
                           suppress_warnings=True)
    # Make predictions for the next 10 days
    forecast_next_10_days, conf_int_next_10_days = model2.predict(n_periods=70, return_conf_int=True)

    # Plot the results
    x_axis = np.arange(dataset.shape[0] + forecast_next_10_days.shape[0])
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis[:dataset.shape[0]], np.exp(dataset['Close']), label='Training Data')
    plt.plot(x_axis[dataset.shape[0]:], np.exp(forecast_next_10_days), label='Forecast', color='green')
    plt.fill_between(x_axis[dataset.shape[0]:], np.exp(conf_int_next_10_days[:, 0]), np.exp(conf_int_next_10_days[:, 1]),
                     color='gray', alpha=0.2)
    plt.title('ARIMA Forecast')
    plt.xlabel('Time Step')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')

    # Save the plot as an image file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def arima_test(dataset, save_path):
    train, test = train_test_split(dataset, train_size=0.9)
    model = pm.auto_arima(train['Close'], max_p=14, max_d=2, max_q=5, trace=True, error_action='ignore', suppress_warnings=True)
    # Make predictions for the next 10 days
    last_index = dataset.index[-1]
    forecast, conf_int = model.predict(n_periods=len(test['Close']), return_conf_int=True, start=last_index)

    # Plot the results
    x_axis = np.arange(train.shape[0] + test.shape[0])

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis[:train.shape[0]], train['Close'], label='Training Data')
    plt.plot(x_axis[train.shape[0]:], test['Close'], label='Test Data')
    plt.plot(x_axis[train.shape[0]:], forecast, label='Forecast', color='green')
    plt.fill_between(x_axis[train.shape[0]:], conf_int[:, 0], conf_int[:, 1], color='gray', alpha=0.2)
    plt.title('ARIMA Forecast')
    plt.xlabel('Time Step')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')

    # Save the plot as an image file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')