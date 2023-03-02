import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from data_preprocessor import DataSequence, Scaler
from torch.utils.data import DataLoader
from lstm import LSTM
from torch import optim, nn
import conf


class Model:
    def __init__(self, *, checkpoint=None) -> None:
        self.__model = LSTM(input_size=conf.FEATURES_COUNT, hidden_size=conf.HIDDEN_SIZE,
                            batch_first=True, dropout=0.1)
        self.__learning_rate = conf.LEARNING_RATE
        self.__optimizer = optim.Adam(
            self.__model.parameters(), lr=conf.LEARNING_RATE)
        if checkpoint:
            print('Checkpoint: epoch: ', checkpoint['epoch'], 'train loss: ', checkpoint['train_loss'],
                  'test loss: ', checkpoint['test_loss'], 'learning rate', checkpoint['learning_rate'])
            self.__model.load_state_dict(checkpoint['model_state_dict'])
            self.__optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])
        self.__loss_fn = nn.MSELoss()
        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.__model = self.__model.to(self.__device)

    def train(self, train_data, test_data):

        train_losses = []
        test_losses = []
        checkpoint = {}
        train_loss_non_decrease_count = 0
        test_loss_non_decrease_count = 0
        for epoch in range(conf.EPOCHS):
            train_loss = self.__train(train_data)
            print("epoch: ", epoch, "train loss: ", train_loss)
            train_losses.append(train_loss)
            test_loss, _ = self.test(test_data)
            print("test_data MSELoss:(pred-real)/real=", test_loss)
            test_losses.append(test_loss)

            train_loss_non_decrease_count = Model.__update_non_decrease_count(Model.__is_loss_decreasing(train_losses, train_loss),
                                                                              train_loss_non_decrease_count)

            train_loss_non_decrease_count = self.__decrease_learning_rate(
                train_loss_non_decrease_count)

            is_test_loss_decreasing = Model.__is_loss_decreasing(
                test_losses, test_loss)
            if is_test_loss_decreasing:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.__model.state_dict(),
                    'optimizer_state_dict': self.__optimizer.state_dict(),
                    'learning_rate': self.__learning_rate,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }

            test_loss_non_decrease_count = Model.__update_non_decrease_count(
                is_test_loss_decreasing, test_loss_non_decrease_count)

            if Model.__early_stop(test_loss_non_decrease_count):
                break

        return train_losses, test_losses, checkpoint

    def __decrease_learning_rate(self, loss_non_decrease_count):
        if self.__learning_rate < conf.MIN_LEARNING_RATE + 0.000001:
            return loss_non_decrease_count

        if conf.TRAIN_LOSS_NON_DECREASE_THRESHOLD <= loss_non_decrease_count:
            self.__learning_rate *= 0.5
            for param_group in self.__optimizer.param_groups:
                param_group['lr'] = self.__learning_rate
            return 0

        return loss_non_decrease_count

    @staticmethod
    def __update_non_decrease_count(is_decreasing, non_decrease_count):
        if is_decreasing:
            return 0

        return non_decrease_count + 1

    @staticmethod
    def __is_loss_decreasing(losses, current_loss):
        return not losses or (current_loss <= min(losses))

    @staticmethod
    def __early_stop(loss_non_decrease_count):
        return conf.EPOCHS / 10 < loss_non_decrease_count

    def __train(self, data):
        losses = []
        self.__model.train()
        for i, (data, label) in enumerate(data):
            data = data.to(self.__device)
            label = label.to(self.__device)
            self.__optimizer.zero_grad()
            output = self.__model.forward(data)
            loss = self.__loss_fn(output, label)
            loss.backward()
            self.__optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    def test(self, data):
        predictions = []
        losses = []
        self.__model.eval()
        for i, (data, label) in enumerate(data):
            label = label[:, -1, :]
            with torch.no_grad():
                data, label = data.to(self.__device), label.to(self.__device)
                self.__optimizer.zero_grad()
                output = self.__model.forward(data)
                predictions.extend(output[:, -1].tolist())
                loss = self.__loss_fn(output[:, -1], label)
                losses.append(loss.item())

        return np.mean(losses), predictions
