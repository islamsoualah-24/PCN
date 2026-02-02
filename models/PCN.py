import torch
from torch import nn
from layers.PITS_backbone import PITS_backbone
from layers.PITS_layers import series_decomp
from layers.DCN import DCN
from layers.RevIN import RevIN
from layers.Embed import PatchEmbedding
import torch.nn.functional as F

def getchannel(a):
    n = 1
    while 2*(2**n-1)+1 < a:
        n = n+1
    print(n)#2(2”-1)+1
    return n

def getchannel1(a):
    n = 1 #n*(2-1)+n*(n-1)//2+1
    while n*(n-1) + 1 < a:
        n = n+1
    return n


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):  # nf?
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
#
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation)#weight_norm()
#         self.dropout1 = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#         self.net = nn.Sequential(self.conv1, self.dropout1)
#         self.init_weights()
#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#
#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#     def forward(self, x):
#         out = self.net(x)
#         if self.n_inputs == self.n_outputs:
#             out = out + x
#         return out
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation)#weight_norm()
        self.dropout1 = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.net = nn.Sequential(self.conv1,self.gelu,self.dropout1) #, self.chomp1, self.relu1, self.dropout1

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

class Model(nn.Module):
    def __init__(self, configs,
                 verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        context_window = configs.seq_len
        self.pred_len = configs.pred_len

        self.d_model = configs.d_model
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        self.stride = configs.stride
        padding_patch = configs.padding_patch

        self.c_in = configs.c_in
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        shared_embedding = configs.shared_embedding
        self.patch_len = configs.patch_len
        self.patch_num = int((configs.seq_len - patch_len) / self.stride)+2
        self.head_nf = self.d_model * self.patch_num
        self.n_vars = self.c_in
        self.individual = individual
        # model
        # self.model = PITS_backbone(c_in=c_in,
        #                            context_window=context_window, target_window=target_window, patch_len=patch_len,
        #                            stride=stride,
        #                            d_model=d_model,
        #                            shared_embedding=shared_embedding,
        #                            head_dropout=head_dropout,
        #                            padding_patch=padding_patch,
        #                            individual=individual, revin=revin, affine=affine,
        #                            subtract_last=subtract_last, verbose=verbose, **kwargs)

        self.revin_layer = RevIN(self.c_in, affine=affine, subtract_last=subtract_last)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        # self.linearP_D = nn.Linear(self.patch_len, self.d_model)
        # self.linearS_Pre = nn.Linear(self.d_model*self.patch_num, self.pred_len)
        # self.DCN = DCN(configs.d_model, configs.c_in * self.patch_num, configs.d_model, tk=0)

        layerD = []
        cn = 2# 1 2 3
        if cn == 1:
            self.num_levelk = getchannel1(self.patch_num)
            kernel_size = 3
            for i in range(self.num_levelk):
                if i == 0:
                    in_channels = self.patch_len
                else:
                    in_channels = self.d_model  # self.window_size if i == 0 else
                out_channels = self.d_model
                layerD += [TemporalBlock(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,
                                         padding=int((kernel_size - 1) / 2), dropout=0.2)]
                kernel_size = kernel_size + 2
        elif cn == 2:
            self.num_levels = getchannel(self.patch_num)
            for i in range(self.num_levels):
                if i == 0:
                    in_channels = self.patch_len
                else:
                    in_channels = self.d_model
                # in_channels = self.d_model  # self.window_size if i == 0 else
                out_channels = self.d_model
                # kernel_size = int(2 * i + 1)
                kernel_size = 3
                dilation = 2 ** i
                # print((dilation*(kernel_size-1))/2)
                layerD += [
                    TemporalBlock(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=dilation,
                                  padding=int(dilation * (kernel_size - 1) / 2), dropout=0.2)]
        elif cn == 3:
            if self.patch_num % 2 != 0:
                kernel_size = self.patch_num
            else:
                kernel_size = self.patch_num + 1
            level = 1
            for i in range(level):
                if i == 0:
                    in_channels = self.patch_len
                else:
                    in_channels = self.d_model
                # in_channels = self.d_model  # self.window_size if i == 0 else
                out_channels = self.d_model
                # kernel_size = int(2 * i + 1)
                layerD += [TemporalBlock(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1,
                                         padding=(kernel_size - 1) // 2, dropout=0.2)]
            k = 7
            layerD += [TemporalBlock(n_inputs=self.d_model, n_outputs=self.d_model, kernel_size=k, stride=1, dilation=1,
                                     padding=(k - 1) // 2, dropout=0.2)]
        self.networkD = nn.Sequential(*layerD)

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, self.pred_len, head_dropout=head_dropout)
        self.activation = F.gelu
        # self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
    def forward(self, x):  # x: [Batch, Input length, Channel] # x: [Batch, Channel, Input length]
        x = self.revin_layer(x, 'norm')
        x = x.permute(0, 2, 1)
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x = self.linearP_D(x)
        x = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))
        x = x.permute(0, 2, 1)
        x = self.networkD(x)
        # x = self.activation(self.conv4(x)+x)
        # x = self.DCN(x)
        x = x.permute(0, 2, 1)
        x = torch.reshape(x, (-1, self.n_vars, self.patch_num, x.shape[-1]))
        x = x.permute(0, 1, 3, 2)
        x = self.head(x)
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]
        x = self.revin_layer(x, 'denorm')
        return x
