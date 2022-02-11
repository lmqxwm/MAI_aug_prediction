# -*- coding: utf-8 _*_
# @Time : 20/1/2022 12:22 pm
# @Author: ZHA Mengyue
# @FileName: utils.py
# @Software: MAEI
# @Blog: https://github.com/Dolores2333

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copy

import torch

"""utils.py includes
    1. Args Loading
    2. Data Related
    3. Model Related"""
    

def sliding_window(args, ori_data):
    """ Slicing the ori_data by sliding window
        Args:
            args
            ori_data (len(csv), z_dim)
        Returns:
            ori_data (:, seq_len, z_dim)"""
    # Flipping the data to make chronological data
    ori_data = ori_data[::-1]  # (len(csv), z_dim)
    # Make (len(ori_data), z_dim) into (num_samples, seq_len, z_dim)
    samples = []
    print("You should not set a batch size larger than", len(ori_data)-args.ts_size)
    for i in range(len(ori_data)-args.ts_size):
        single_sample = ori_data[i:i + args.ts_size]  # (seq_len, z_dim)
        samples.append(single_sample)
    samples = np.array(samples)  # (bs, seq_len, z_dim)
    np.random.shuffle(samples)  # Make it more like i.i.d.
    return samples


def min_max_scalar(data):
    """Min-Max Normalizer.

    Args:
      - data: raw data

    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """
    min_val = np.min(np.min(data, axis=0), axis=0)  # (z_dim, ) min for each feature
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)  # (z_dim, ) max for each feature
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


def sine_data_generation(num_samples, seq_len, z_dim):
    """Sine data generation
       Remark: no args.min/max/var for sine_data
               no normalization
               no renormalization
    Args:
        - num_samples: the number of samples
        - seq_len: the sequence length of the time-series
        - dim: feature dimensions
    Returns:
        - data: generated data"""
    sine_data = list()
    for i in range(num_samples):
        single_sample = list()
        for k in range(z_dim):
            # Randomly drawn frequency and phase for each feature (column)
            freq = np.random.uniform(low=0.03, high=0.1)
            phase = np.random.uniform(low=0, high=0.1)
            sine_feature = [np.sin(freq * j + phase) for j in range(seq_len)]
            single_sample.append(sine_feature)
        single_sample = np.transpose(np.asarray(single_sample))  # (seq_len, z_dim)
        single_sample = (single_sample + 1) * 0.5
        # Stack the generated data
        sine_data.append(single_sample)
    sine_data = np.array(sine_data)  # (num_sample, seq_len, z_dim)
    return sine_data


def get_batch(args, data):
    idx = np.random.permutation(len(data))
    idx = idx[:args.batch_size]
    data_mini = data[idx, ...]  # (bs, seq_len, z_dim)
    return data_mini


def train_test_split(args, data):
    # Split ori_data
    idx = np.random.permutation(len(data))
    train_idx = idx[:int(args.train_test_ratio * len(data))]
    test_idx = idx[int(args.train_test_ratio * len(data)):]
    train_data = data[train_idx, ...]
    test_data = data[test_idx, ...]
    return train_data, test_data


"""Model Related"""


def save_model(args, model):
    file_dir = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), file_dir)


def save_metrics_results(args, results):
    file_dir = os.path.join(args.model_dir, 'metrics_results.npy')
    np.save(file_dir, results)


def save_args(args):
    file_dir = os.path.join(args.model_dir, 'args_dict.npy')
    np.save(file_dir, args.__dict__)


def load_model(args, model):
    model_dir = args.model_dir
    file_dir = os.path.join(model_dir, 'model.pth')

    model_state_dict = torch.load(file_dir)
    model.load_state_dict({f'model.{k}': v for k, v in model_state_dict.items()})
    return model


def load_dict_npy(file_path):
    file = np.load(file_path, allow_pickle=True)
    return file


"""For TimeGAN metrics"""


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.
    Args:
        - data_x: original data
        - data_x_hat: generated data
        - data_t: original time
        - data_t_hat: generated time
        - train_rate: ratio of training data from the original data"""
    # Divide train/test index (original data)
    # permute the indexies and split the first 0.8 percent to be training data
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    # Repeat it again for the synthetic data
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def batch_generator(data, time, batch_size):
    """Mini-batch generator. Slice the original data to the size a batch.

    Args:
        - data: time-series data
        - time: time series length for each sample
        - batch_size: the number of samples in each batch

    Returns:
        - X_mb: time-series data in each batch (bs, seq_len, dim)
        - T_mb: time series length of samples in that batch (bs, len of the sample)"""
    # randomly select a batch of idx
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    # picked the selected samples and their corresponding series length
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
    - data: original data (no, seq_len, dim)

    Returns:
    - time: a list for each sequence length
    - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))
    return time, max_seq_len


def extract_factors(n):
    if (n == 0) or (n == 1):
        return [n]

    factor_list = [1]
    i = 2
    while i < n:
        if n % i == 0:
            factor_list.append(i)
        i += 1

    return factor_list
