__all__ = ['DCN']

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

def getchannel(a):
    n = 1
    while a/2 >= 1:
        n = n+1
        a = a/2
    return n

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation)#weight_norm()
        self.net = nn.Sequential(self.conv1) #, self.chomp1, self.relu1, self.dropout1
        self.relu = nn.ReLU()
        self.init_weights()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        if self.n_inputs == self.n_outputs:
            out = out + x
        return out

class DCN(nn.Module):
    def __init__(self, seq_len, enc_in, pred_len, tk):#configs.seq_len, configs.enc_in, configs.pred_len, configs.tk, configs.isd, configs.moving_avg
        super(DCN, self).__init__()
        self.window_size = seq_len
        self.num_channels = enc_in
        self.hidden_dim = pred_len
        self.num_levels = getchannel(self.num_channels)
        self.L = pred_len
        self.k = tk
        self.activation = 'gelu'
        self.activation = F.relu if self.activation == "relu" else F.gelu
        self.dropout = nn.Dropout(0.1)
        layerD = []
        if self.k:
            for i in range(self.num_levels):
                in_channels = self.hidden_dim #self.window_size if i == 0 else
                out_channels = self.hidden_dim
                kernel_size = int(2 * i + 1)
                layerD += [TemporalBlock(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,
                                         padding=int((kernel_size - 1) / 2), dropout=0.25)]
        else:
            for i in range(self.num_levels):
                in_channels = self.hidden_dim#self.window_size if i == 0 else
                out_channels = self.hidden_dim
                # kernel_size = int(2 * i + 1)
                layerD += [TemporalBlock(in_channels, out_channels, kernel_size=3, stride=1, dilation=i + 1,
                                         padding=(2 - 1) * (i + 1), dropout=0.25)]
        self.networkD = nn.Sequential(*layerD)
    def forward(self, x):
        x_d_output = self.activation(self.networkD(x))#self.dropout() #b,H,D
        return x_d_output