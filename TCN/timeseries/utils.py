import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

# Sine functions
def sine_data_generator(N, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    period = int(seq_length / 1.5)
    X = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])
    const = 2 * np.pi / period
    for i in range(N):
        phase = np.random.rand() * period
        data = np.sin(np.arange(period) * const + phase * const) + \
            np.random.normal(scale=0.2, size=period)
        #X[i, 0, :] = torch.FloatTensor(data[:-1])
        X[i, 0, :] = torch.FloatTensor(np.append(data, data[:seq_length - period]))
        Y[i,0] = data[seq_length - period]
    return Variable(X), Variable(Y)


def two_sine_data_generator(N, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    period1 = int(seq_length / 2)
    period2 = int(seq_length / 4 * 3)
    period = int(seq_length / 2 * 3)
    X = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])
    const1 = 2 * np.pi / period1
    const2 = 2 * np.pi / period2
    for i in range(N):
        phase = np.random.rand() * period
        data = (np.sin(np.arange(seq_length + 1) * const1 + phase * const1) + \
          np.sin(np.arange(seq_length + 1) * const2 + phase * const2) + \
            np.random.normal(scale=0.2, size=(seq_length + 1))) / 2
        #X[i, 0, :] = torch.FloatTensor(data[:-1])
        X[i, 0, :] = torch.FloatTensor(data[:-1])
        Y[i,0] = data[-1]
    return Variable(X), Variable(Y)

