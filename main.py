from preprocessing.preprocessing import data_collection, data_cleansing, log_transformation, differencing, linear_regression, seasonal_decompose_ts
from prediction.prediction import arima_test, arima_predict
import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stock_symbol = 'SPY'
    data = data_collection(stock_symbol, 365, '1h')
    data_clean = data_cleansing(data)
    log_transformation(data_clean)
    test_folder = 'testModel'
    pred_folder = 'predModel'
    arima_test(data_clean, os.path.join(test_folder, f'{stock_symbol}_verification_plot.png'))
    arima_predict(data_clean, os.path.join(pred_folder, f'{stock_symbol}_forecast_plot.png'))


