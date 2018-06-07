import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.timeseries.utils import *
from TCN.timeseries.model import *
import numpy as np
import argparse


import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


parser = argparse.ArgumentParser(description='Time Series')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--win_size', type=int, default=100,
                    help='size of time series segment (default: 100)')
parser.add_argument('--outfile', type=str, default='predictions.txt',
                    help='output file name')
parser.add_argument('--lstm', action='store_true', default=False,
                    help='whether or not to use LSTM')
parser.add_argument('--hard', action='store_true', default=False,
                    help='easy to hard test case')
parser.add_argument('--kernel', action='store_true', default=False,
                    help='sweep kernel size instead of levels')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './'
batch_size = args.batch_size
input_channels = 1
n_classes = 1
seq_length = args.win_size
epochs = args.epochs
steps = 0

iterations = 0
test_losses = []

print(args)
print("Producing data...")
if args.hard:
    X_train, Y_train = two_sine_data_generator(50000, seq_length)
    X_test, Y_test = two_sine_data_generator(5000, seq_length)
else:
    X_train, Y_train = sine_data_generator(50000, seq_length)
    X_test, Y_test = sine_data_generator(5000, seq_length)
if args.cuda:
    X_train = X_train.cuda()
    Y_train = Y_train.cuda()
    X_test = X_test.cuda()
    Y_test = Y_test.cuda()
kernel_size = args.ksize
lr = args.lr

def model_init(channel_sizes):
    if args.lstm:
        model = LSTM(input_channels, channel_sizes, n_classes)
    else:
        model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

    if args.cuda:
        model.cuda()

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    return model, optimizer

def tcn_model_init(kernel_size):
    model = TCN(input_channels, n_classes, [args.nhid] * args.levels,
        kernel_size=kernel_size, dropout=args.dropout)

    if args.cuda:
        model.cuda()

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    return model, optimizer


def train(epoch):
    global steps, iterations, test_acc
    total_loss = 0
    model.train()
    batch_idx = 1
    for i in range(0, X_train.size()[0], batch_size):
        iterations += 1
        if i + batch_size > X_train.size()[0]:
            x, y = X_train[i:], Y_train[i:]
        else:
            x, y = X_train[i:(i+batch_size)], Y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x)
        loss = F.l1_loss(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()
        batch_idx += 1
        total_loss += loss.data[0]
        steps += seq_length
        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, X_train.size()[0])
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                epoch, processed, X_train.size()[0], 100.*processed/X_train.size()[0], lr, cur_loss))
            total_loss = 0
        # if iterations == 1 or iterations % args.log_interval == 0:
        #     loss = evaluate(is_output=False)
        #     test_losses.append([iterations, loss])
        #     print('Iter: {}, Loss: {}'.format(iterations, loss))


def evaluate(is_output=True):
    model.eval()
    output = model(X_test)
    test_loss = F.l1_loss(output, Y_test)
    if is_output:
        print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.data[0]))
    return test_loss.data[0]

def gen_sine():
    period = int(seq_length / 1.5)
    prediction_inputs = []
    const = 2 * np.pi / period
    for i in range(5):
        phase = np.random.rand() * period
        data = np.sin(np.arange(period) * const + phase * const) + \
            np.random.normal(scale=0.2, size=period)
        raw = np.append(data, data[:seq_length - period])
        prediction_inputs.append(raw)
        f.write('%s\n' %str(list(raw)))
    return prediction_inputs

def gen_two_sine():
    period1 = int(seq_length / 2)
    period2 = int(seq_length / 4 * 3)
    period = int(seq_length / 2 * 3)
    prediction_inputs = []
    const1 = 2 * np.pi / period1
    const2 = 2 * np.pi / period2
    for i in range(5):
        phase = np.random.rand() * period
        noise = np.random.normal(scale=0.2, size=(seq_length))
        noise = np.append(noise, np.zeros(period * 2 - seq_length))
        raw = (np.sin(np.arange(period * 2) * const1 + phase * const1) + \
          np.sin(np.arange(period * 2) * const2 + phase * const2) + \
          noise) / 2
        prediction_inputs.append(raw[:seq_length])
        f.write('%s\n' %str(list(raw)))
    return prediction_inputs

def prediction(prediction_inputs):
    model.eval()
    predition_length = seq_length
    if args.hard:
        predition_length = int(seq_length / 2 * 3)
    for i in range(len(prediction_inputs)):
        raw = prediction_inputs[i]
        predict_x = Variable(torch.FloatTensor(raw).unsqueeze(0).unsqueeze(0))
        predictions = []
        for i in range(predition_length):
            output = model(predict_x.cuda()).data.cpu().numpy()[0][0]
            predictions.append(output)
            raw = np.append(raw[1:], [output])
            predict_x = Variable(torch.FloatTensor(raw).unsqueeze(0).unsqueeze(0))
        f.write('%s\n' %str(predictions))

if __name__ == "__main__":
    f = open(args.outfile, "w")
    if args.hard:
        prediction_inputs = gen_two_sine()
    else:
        prediction_inputs = gen_sine()

    if not args.kernel:
        for levels in range(1, 9):
            print("Level: %d" % levels)
            if args.lstm:
                channel_sizes = levels * 5
            else:
                channel_sizes = [args.nhid] * levels
            model, optimizer = model_init(channel_sizes)
            for ep in range(1, epochs+1):
                train(ep)
                tloss = evaluate()

            f.write('level: %d\n' % levels)
            f.write("Model parameters: %d\n" % count_parameters(model))
            prediction(prediction_inputs)
            f.write('\n')
        f.close()
    else:
        # Sweep kernel size for fixed layers
        for ksize in range(2, 9):
            print("Kernel: %d" % ksize)
            model, optimizer = tcn_model_init(ksize)
            for ep in range(1, epochs+1):
                train(ep)
                tloss = evaluate()

            f.write('kernel: %d\n' % ksize)
            f.write("Model parameters: %d\n" % count_parameters(model))
            prediction(prediction_inputs)
            f.write('\n')
        f.close()

