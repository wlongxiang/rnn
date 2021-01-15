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

import math
import random
import sys

import numpy as np
import torch
import torch.utils.data as data


def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features


class PalindromeDataset(data.Dataset):

    def __init__(self, seq_length):
        self.seq_length = seq_length

    def __len__(self):
        # Number of possible palindroms can be very big:
        # (10**(seq_length/2) or (10**((seq_length+1)/2)
        # Therefore we return the maximum integer value
        return sys.maxsize

    def __getitem__(self, idx):
        # Keep last digit as target label. Note: one-hot encoding for inputs is
        # more suitable for training, but this also works.
        full_palindrome = self.generate_palindrome()
        # Split palindrome into inputs (N-1 digits) and target (1 digit)
        return full_palindrome[0:-1], int(full_palindrome[-1])

    def generate_palindrome(self):
        # Generates a single, random palindrome number of 'length' digits.
        left = [np.random.randint(0, 10) for _ in range(math.ceil(self.seq_length / 2))]
        left = np.asarray(left, dtype=np.float32)
        right = np.flip(left, 0) if self.seq_length % 2 == 0 else np.flip(left[:-1], 0)
        return np.concatenate((left, right))


class TextDataset(data.Dataset):

    def __init__(self, text):
        self.input_seq, self.target_seq = self.generate_palindrome(text)
        self.batch_size = self.input_seq.shape[0]

    def __len__(self):
        # Number of possible palindroms can be very big:
        # (10**(seq_length/2) or (10**((seq_length+1)/2)
        # Therefore we return the maximum integer value
        return sys.maxsize

    def __getitem__(self, idx):
        # Keep last digit as target label. Note: one-hot encoding for inputs is
        # more suitable for training, but this also works.
        random_index = random.randint(0, self.batch_size - 1)
        # Split palindrome into inputs (N-1 digits) and target (1 digit)
        return self.input_seq[random_index], self.target_seq[random_index]

    def generate_palindrome(self, text):
        # Join all the sentences together and extract the unique characters from the combined sentences
        chars = set(''.join(text))

        # Creating a dictionary that maps integers to the characters
        self.int2char = dict(enumerate(chars))

        # Creating another dictionary that maps characters to integers
        self.char2int = {char: ind for ind, char in self.int2char.items()}
        print(self.int2char)
        maxlen = len(max(text, key=len))
        print("The longest string has {} characters".format(maxlen))
        for i in range(len(text)):
            while len(text[i]) < maxlen:
                text[i] += ' '
        print(text)
        # Creating lists that will hold our input and target sequences
        input_seq = []
        target_seq = []

        for i in range(len(text)):
            # Remove last character for input sequence
            input_seq.append(text[i][:-1])

            # Remove firsts character for target sequence
            target_seq.append(text[i][1:])
            print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

        for i in range(len(text)):
            input_seq[i] = [self.char2int[character] for character in input_seq[i]]
            target_seq[i] = [self.char2int[character] for character in target_seq[i]]
            print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))

        self.dict_size = len(self.char2int)
        self.seq_len = maxlen - 1
        batch_size = len(text)
        input_seq = one_hot_encode(input_seq, self.dict_size, self.seq_len, batch_size)
        input_seq = torch.from_numpy(input_seq)
        print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))
        target_seq = torch.Tensor(target_seq)
        print(target_seq.shape)
        return input_seq, target_seq
