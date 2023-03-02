import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, *, input_size=1, hidden_size=10, num_layers=3, batch_first=False, dropout=0 ):
        super(LSTM, self).__init__()
        self.__lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                              num_layers=num_layers, batch_first=batch_first, dropout=dropout)
        self.__fc = nn.Linear(hidden_size, 1)

    def forward(self, input):
        out, _ = self.__lstm(input)
        return self.__fc(out)
