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

# In this implementation, you can also stack multiple layers of RNN!
# Observation: too many layers make it too hard to learn, problem at hand is too small for deep NN.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import time
from datetime import datetime
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from dataset import PalindromeDataset, TextDataset, one_hot_encode
from pytorch_rnn import RNN
from raw_rnn import VanillaRNN

from sklearn.metrics import accuracy_score
import numpy as np


# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################
def calc_accuracy(predictions, targets):
    """
    Calculate the training/test accuracy.

    :param Tensor predictions: tensor of shape (batch_size, vocab_size)
    :param Tensor targets: tensor of shape (batch_size)
    :return float: accuracy
    """
    # use argmax to get the index of max value in a vector
    y_pred = predictions.argmax(dim=1)
    # user sklearn utility to calculate accuracy
    accuracy = accuracy_score(y_true=targets, y_pred=y_pred)
    ########################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def predict(model, characters, char2int, int2char, dict_size):
    # One-hot encoding our input to fit into the model
    characters = np.array([[char2int[c] for c in characters]])
    characters = one_hot_encode(characters, dict_size, characters.shape[1], 1)
    characters = torch.from_numpy(characters)

    out, hidden = model(characters)

    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden


def sample(model, out_len, start, char2int, int2char, dict_size):
    model.eval()  # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars, char2int, int2char, dict_size)
        chars.append(char)

    return ''.join(chars)


def train(config):
    text = ['hey how are you', 'good i am fine', 'have a nice day']
    dataset = TextDataset(text)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    # Initialize the device which to run the model on
    device = torch.device(config.device)
    if config.model_type == "LSTM":
        model = RNN(input_size=dataset.dict_size, output_size=dataset.dict_size, hidden_size=12, num_layers=1,
                    lstm=True)
    elif config.model_type == "GRU":
        model = RNN(input_size=dataset.dict_size, output_size=dataset.dict_size, hidden_size=12, num_layers=1,
                    gru=True)
    else:
        model = RNN(input_size=dataset.dict_size, output_size=dataset.dict_size, hidden_size=12, num_layers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), config.learning_rate)
    # init csv file
    for d in ["results", "checkpoints", "assets"]:
        if not os.path.exists(d):
            os.mkdir(d)
    cvs_file = 'results/w_grad_{}_inputlength_{}_hiddenunits_{}_lr_{}_batchsize_{}_{}.csv'.format(config.model_type,
                                                                                                  config.input_length,
                                                                                                  config.num_hidden,
                                                                                                  config.learning_rate,
                                                                                                  config.batch_size,
                                                                                                  int(time.time()))
    cols_data = ['step', 'train_loss', 'train_accuracy']
    with open(cvs_file, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(cols_data)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Add more code here ...
        # voncert tensors to device for gpu training
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        # note that model_output are raw output before softmaxing
        # adding an extra input size dimension this is needed for pytorch implementation
        model_output, hidden = model.forward(batch_inputs)
        # flatten out the target output to match model output
        batch_targets = batch_targets.view(-1).long()
        # note: as we only care about the last output, aka the last digit in the pandidrome!
        loss = criterion(model_output, batch_targets)
        loss.backward()

        ############################################################################
        # QUESTION: what happens here and why?
        ############################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        ############################################################################
        # Add more code here ...
        optimizer.step()
        loss = loss.item()
        accuracy = calc_accuracy(model_output, batch_targets)

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if step % 10 == 0 and step > 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                config.train_steps, config.batch_size, examples_per_second,
                accuracy, loss
            ))
            print("sampling start with: ", sample(model=model,
                                                  out_len=15, start='hey',
                                                  char2int=dataset.char2int,
                                                  int2char=dataset.int2char,
                                                  dict_size=dataset.dict_size))
            csv_data = [step, loss, accuracy]
            with open(cvs_file, 'a') as fd:
                writer = csv.writer(fd)
                writer.writerow(csv_data)
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="GRU", help="Model type, should be 'RNN' or 'LSTM' or 'GRU'")
    parser.add_argument('--input_length', type=int, default=6, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    train(config)
