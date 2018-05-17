import torch.nn.functional as F
from torch import nn
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.GRU(input_size, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, output_size)
        for names in self.lstm._all_weights:
        	for name in filter(lambda n: "bias" in n, names):
        		bias = getattr(self.lstm, name)
        		n = bias.size(0)
        		start, end = n//4, n//2
        		bias.data[start:end].fill_(1.)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        seqlen = inputs.size()[2]
        inputs = inputs.permute(2, 0, 1)
        output_seq, hidden = self.lstm(inputs) # [N, hidden]
        lstm_out = output_seq[-1, :, :]
        output = self.linear(lstm_out) # [N, class]
        return F.log_softmax(output, dim=1)
