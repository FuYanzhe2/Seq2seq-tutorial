# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 设置合成数据的特征

def gen_data(size):
    """ 按照上图生成合成序列数据

    Arguments:
        size: input 和 output 序列的总长度

    Returns:
        X, Y: input 和 output 序列，rank-1的numpy array （即，vector)
    """

    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    """产生minibatch数据

    Arguments:
        raw_data: 所有的数据， (input, output) tuple
        batch_size: 一个minibatch包含的样本数量；每个样本是一个sequence
        num_step: 每个sequence样本的长度

    Returns:
        一个generator，在一个tuple里面包含一个minibatch的输入，输出序列
    """

    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)

def gen_epochs(num_epochs, total_size, batch_size, num_steps):
    for i in range(num_epochs):
        yield gen_batch(gen_data(total_size), batch_size, num_steps)

if __name__ == '__main__':
    gen = gen_epochs(num_epochs = 3, total_size = 10000, batch_size = 16, num_steps = 32)
