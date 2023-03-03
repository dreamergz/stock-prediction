import pandas as pd
from data_preprocessor import Scaler
import conf
import joblib
import urllib
import os
import json
import datetime as dt
import sys

SPLIT_RATE = 0.8


def retrieve_data(symbol, api_key):
    json_file = conf.JSON_SOURCE_DATA_PATH % symbol
    csv_file = conf.CSV_SOURCE_DATA_PATH % symbol

    if os.path.exists(json_file):
        return

    with urllib.request.urlopen(conf.DATA_API % (
            symbol, api_key)) as url:
        data = url.read().decode()
        with open(json_file, mode='w') as f:
            f.write(data)
        data = json.loads(data)
        # extract stock market data
        data = data['Time Series (Daily)']
        df = pd.DataFrame(
            columns=['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume'])
        for k, v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(), float(v['1. open']), float(v['2. high']), float(v['3. low']),
                        float(v['4. close']), float(v['5. adjusted close']), float(v['6. volume'])]
            df.loc[-1, :] = data_row
            df.index = df.index + 1
        print('Data saved to : %s' % csv_file)
        df.to_csv(csv_file)


def prepare(symbol):
    df = pd.read_csv(conf.CSV_SOURCE_DATA_PATH % symbol, index_col='date')
    df = df[conf.DATA_COLUMNS]
    df = df[df['volume'] != 0]
    df = df.sort_values('date')
    split = int(SPLIT_RATE * df.shape[0])
    train_data = df[:split]
    train_data.to_csv(conf.TRAIN_DATA_PATH)
    print('train data: ', conf.TRAIN_DATA_PATH)

    rest_data = df[split:]
    split = int(0.5 * rest_data.shape[0])

    validation_data = rest_data[:split]
    validation_data.to_csv(conf.VALIDATION_DATA_PATH)
    print('validation data: ', conf.VALIDATION_DATA_PATH)

    test_data = rest_data[split:]
    test_data.to_csv(conf.TEST_DATA_PATH)
    print('test data: ', conf.TEST_DATA_PATH)


if '__main__' == __name__:
    symbol = conf.SYMBOL_CMB
    api_key = sys.argv[1]

    retrieve_data(symbol, api_key)
    prepare(symbol)
