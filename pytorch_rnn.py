import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, bidirectional=False, lstm=False, gru=False):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = lstm
        self.gru = gru
        if self.lstm:
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True)
        elif self.gru:
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.RNN(input_size=input_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # input is of shape (batch_size, seq_len, input_size)
        batch_size = x.shape[0]  # number of training samples
        seq_length = x.shape[1]  # number of steps in the sequence x1, x2,x_{seq_length}
        input_size = x.shape[2]  # number of dimensions for each x_t in the sequence
        # output is of shape (batch_size, seq_len, hidden_size)
        # hidden is of shape (num_layers, batch_size, hidden_size)
        hidden = self.init_hidden(batch_size)
        output, hidden = self.rnn(x, hidden)
        # now we convert output to shape (batch_size*seq_len, hidden_size) to feed to linear layer
        # this essentially stacks all hidden vectors for each batch and each item in the sequence row-wise
        output = output.contiguous().view(-1, self.hidden_size)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        # according to https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        # h_0 should be of size (S, N, H_out), where S = num_layers * num_directions
        # N = batch_size, H_out = hidden_size
        num_directions = 2 if self.bidirectional is True else 1
        h_zero = torch.zeros(num_directions * self.num_layers,
                             batch_size,
                             self.hidden_size)
        # if it is LSTM, we also need to include the cell states
        if self.lstm:
            return h_zero, h_zero
        else:
            h_zero = torch.zeros(num_directions * self.num_layers,
                                 batch_size,
                                 self.hidden_size)
            return h_zero
