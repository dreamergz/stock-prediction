from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import pandas as pd
import data_preprocessor
import matplotlib.pyplot as plt
import conf
from model import Model


def draw_prediction_curve(predictions, labels, indices):
    plt.figure(figsize=(18, 9))
    plt.plot(range(indices.shape[0]), labels, label="real")
    plt.plot(range(conf.NUM_UNROLL, predictions.shape[0] + conf.NUM_UNROLL),
             predictions, label="prediction")
    plt.xticks(range(0, indices.shape[0], 20), indices[::20], rotation=45)
    plt.xlabel('Trade date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(conf.FIGURE_PATH + 'predictions.png')


def test():
    data = pd.read_csv(conf.TEST_DATA_PATH, index_col='date')
    indices = data.index.values
    labels = data[conf.LABEL_COLUMNS].values
    checkpoint = torch.load(conf.CHECKPOINT_PATH)
    model = Model(checkpoint=checkpoint)
    test_data, scaler = data_preprocessor.preprocess(data, batch_size=10)
    loss, predictions = model.test(test_data)
    predictions = scaler.inverse_transform(
        np.array(predictions))
    print('loss: ', loss)
    draw_prediction_curve(predictions, labels, indices)


if '__main__' == __name__:
    test()
