################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.device = device
        self.input_dim = input_dim  # note: ths can only be 1 somehow, we input a digit per time
        # U input-to-hidden matrix
        self.U = nn.Parameter(torch.Tensor(num_hidden, input_dim))
        # hidden state bias
        self.bh = nn.Parameter(torch.Tensor(num_hidden))
        # V hidden-to-output matrix
        self.V = nn.Parameter(torch.Tensor(num_classes, num_hidden))
        # output bias
        self.bp = nn.Parameter(torch.Tensor(num_classes))
        # W hidden-to-hidden matrix
        self.W = nn.Parameter(torch.Tensor(num_hidden, num_hidden), requires_grad=True)
        # init
        # vanilla rnn seems to be sensitive to weights init, xavier normal seems to work well
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.W)
        nn.init.zeros_(self.bh)
        nn.init.zeros_(self.bp)

    def forward(self, x):
        # Implementation here ...
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        hidden_state_prev = torch.zeros_like(self.bh)
        for t in range(seq_length):
            # x_ts represents the value at specific time step t, with length of batch size
            x_ts = x[:, t:t + self.input_dim]
            # apply a linear transform with hidden state and bias
            linear_transformed = x_ts @ self.U.data.T + hidden_state_prev @ self.W.data + self.bh
            # apply tanh non-linearity
            hidden_state_prev = torch.tanh(linear_transformed)
        output = hidden_state_prev @ self.V.T + self.bp
        # p represents the ouput at the last time step here, note that the softmax part is done in the loss function
        # do not include it in the output layer
        # we also return the hidden-to-hidden weight matrix and hidden state for last step for later analysis
        return self.W.data, hidden_state_prev, output
