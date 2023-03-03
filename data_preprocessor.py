from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import conf
import joblib


class Scaler:
    def __init__(self, path=None) -> None:
        self.__data_scalers = None
        self.__label_scalers = None

    def fit(self, data, labels):
        self.__data_scalers = self.__fit(data, StandardScaler)
        self.__label_scalers = self.__fit(labels, MinMaxScaler)

    @staticmethod
    def __fit(data, scaler_type):
        ret = []
        for i in range(data.shape[1]):
            scaler = scaler_type()
            scaler.fit(
                data[:, i].reshape(-1, 1))
            ret.append(scaler)
        return ret

    @staticmethod
    def __transform(data, scalers):
        ret = np.empty(tuple(data.shape), dtype=np.float32)
        for i in range(data.shape[1]):
            ret[:, i] = scalers[i].transform(
                data[:, i].reshape(-1, 1)).reshape(-1)
        return ret

    def transform(self, data, labels):
        return self.__transform(data, self.__data_scalers), self.__transform(labels, self.__label_scalers)

    def inverse_transform(self, labels):

        ret = np.empty(tuple(labels.shape), dtype=np.float32)
        for i in range(labels.shape[1]):
            ret[:, i] = (self.__label_scalers[i].inverse_transform(
                labels[:, i].reshape(-1, 1)).reshape(-1))
        return ret


class DataSequence(Dataset):
    def __init__(self, data, labels, num_unroll=10):
        self.__data = data
        self.__labels = labels
        self.__num_unroll = num_unroll
        self.__len = data.shape[0] - num_unroll

    def __getitem__(self, index):
        label_index = index + 1
        return torch.tensor(self.__data[index:index + self.__num_unroll], dtype=torch.float32), torch.tensor(self.__labels[label_index:label_index + self.__num_unroll], dtype=torch.float32)

    def __len__(self):
        return self.__len


def fit_scaler(data):
    scaler = Scaler()
    scaler.fit(data[conf.DATA_COLUMNS].values,
               data[conf.LABEL_COLUMNS].values)
    joblib.dump(scaler, conf.DATA_SCALER_PATH)
    print('scaler: ', conf.DATA_SCALER_PATH)

def preprocess(data, *, batch_size):
    scaler = joblib.load(conf.DATA_SCALER_PATH)
    inputs = data[conf.DATA_COLUMNS].values
    labels = data[conf.LABEL_COLUMNS].values
    inputs, labels = scaler.transform(inputs, labels)
    data = DataSequence(
        inputs, labels, num_unroll=conf.NUM_UNROLL)
    loader = DataLoader(dataset=data, batch_size=batch_size,
                        shuffle=False, drop_last=True)
    return loader, scaler


def argument_labels(labels):
    length = labels.shape[0]
    for i in range(length):
        target_index = i + np.random.randint(1, 4)
        if length <= target_index:
            target_index = length - 1
        labels[i] = labels[target_index]

    return labels
