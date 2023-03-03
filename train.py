import torch
import pandas as pd
import matplotlib.pyplot as plt
import data_preprocessor
from model import Model
import conf
import timeit


def draw_loss_curve(train_losses, test_losses):
    plt.figure(figsize=(18, 9))
    plt.plot(range(len(train_losses)), train_losses, label="train losses")
    plt.plot(range(len(test_losses)), test_losses, label="test losses")
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(conf.FIGURE_PATH + 'train_losses.png')


def train():
    model = Model()
    train_data = pd.read_csv(conf.TRAIN_DATA_PATH, index_col='date')
    data_preprocessor.fit_scaler(train_data)
    train_data[conf.LABEL_COLUMNS] = data_preprocessor.argument_labels(
        train_data[conf.LABEL_COLUMNS].values)
    train_data, _ = data_preprocessor.preprocess(
        train_data, batch_size=conf.BATCH_SIZE)
    test_data = pd.read_csv(conf.VALIDATION_DATA_PATH, index_col='date')
    test_data, _ = data_preprocessor.preprocess(test_data, batch_size=10)
    print('start training...')
    start_time = timeit.default_timer()
    train_losses, test_losses, checkpoint = model.train(train_data, test_data)
    end_time = timeit.default_timer()
    print(f'training completed in {end_time - start_time}s')
    print('epoch: ', checkpoint['epoch'], 'learning rate: ', checkpoint['learning_rate'], 'train loss: ',
          checkpoint['train_loss'], 'test loss: ', checkpoint['test_loss'])
    draw_loss_curve(train_losses, test_losses)
    torch.save(checkpoint, conf.CHECKPOINT_PATH)


if '__main__' == __name__:
    train()
