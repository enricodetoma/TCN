import torch
from torchvision import datasets, transforms
from dataset import *
from torch.autograd import Variable
import numpy as np

# Sine functions
def sine_data_generator(N, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    period = seq_length / 2
    X = torch.zeros([N, 1, seq_length])
    Y = torch.zeros([N, 1])
    for i in range(N):
        phase = np.random.rand() * period
        data = np.sin(np.arange(seq_length + 1) * 2 * np.pi / period + phase )
        X[i, 0, :] = torch.FloatTensor(data[:-1])
        # if data[-1] > data[-2]:
        #     Y[i,0] = 0
        # else:
        #     Y[i,0] = 1
        Y[i,0] = data[-1]
    return Variable(X), Variable(Y)