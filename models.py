# -------------------------------------------------------
# Copyright Kaan Cökerim (Ruhr-Universität Bochum)
# Mail: kaan.coekerim@rub.de
# Created on 13.01.2024
#
# DESCRIPTION
#
# -------------------------------------------------------
import logging

import torch
from pytorch_tcn import TCN
import numpy as np


class ModTCN(torch.nn.Module):
    def __init__(self, seq_len=100, num_seq_feat=3, num_aux=5, dropout_rate=0.1,
                 num_channels=(6, 12, 24), kernel_size=4, kernel_initializer='xavier_normal',
                 causal: bool = False, use_skip_connections: bool = True,
                 device='cuda:0'):
        super(ModTCN, self).__init__()

        self.device = device

        # TCN Block
        self.TCN = TCN(num_inputs=num_seq_feat, kernel_size=kernel_size, num_channels=num_channels,
                       kernel_initializer=kernel_initializer, activation='leaky_relu',
                       causal=causal, use_skip_connections=use_skip_connections)

        # aux convolutions
        self.aux_conv1 = torch.nn.ConvTranspose1d(in_channels=num_aux,
                                                  out_channels=num_aux,
                                                  kernel_size=seq_len // 2 if seq_len > 1 else 2)
        self.act1 = torch.nn.LeakyReLU()

        self.aux_conv2 = torch.nn.ConvTranspose1d(in_channels=num_aux,
                                                  out_channels=num_aux,
                                                  bias=False,  # performance speed up
                                                  kernel_size=(seq_len - (seq_len // 2)) + 1 if seq_len > 1 else 1)

        self.batchnorm_aux = torch.nn.BatchNorm1d(num_features=num_aux)

        self.act2 = torch.nn.LeakyReLU()

        self.postTCN_conv = torch.nn.Conv1d(in_channels=num_channels[-1], out_channels=32, kernel_size=seq_len)

        # post-cat conv -> make causal
        padding = (64 - 1) * 1  # if causal else 0

        self.merge_conv = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(in_channels=num_channels[-1] + num_aux,  # 32 + num_aux,
                            out_channels=10,
                            kernel_size=64,
                            padding=padding)
        )
        self.act3 = torch.nn.LeakyReLU()

        merge_out_length = np.floor(((seq_len + 2 * self.merge_conv.padding[0] - self.merge_conv.dilation[0] * (
                self.merge_conv.kernel_size[0] - 1) - 1) / self.merge_conv.stride[0]) + 1)

        merge_out_length -= (self.merge_conv.kernel_size[0] - 1)
        merge_flat_size = int(merge_out_length) * self.merge_conv.out_channels

        # fully connected
        self.fc1 = torch.nn.Linear(in_features=merge_flat_size,
                                   out_features=merge_flat_size // 2)  # in_features=self.merge_conv.out_channels
        # dropout
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.act4 = torch.nn.LeakyReLU()

        self.fc2 = torch.nn.Linear(in_features=self.fc1.out_features,
                                   out_features=self.fc1.out_features // 2)

        self.batchnorm = torch.nn.BatchNorm1d(num_features=self.fc2.out_features)
        self.act5 = torch.nn.LeakyReLU()

        self.fc3 = torch.nn.Linear(in_features=self.fc2.out_features, out_features=1)

    def forward(self, x, aux):
        in_size = x.size()
        x = self.TCN(x)
        # x = x.reset_buffers()
        # x = x.to(self.device)

        aux = self.aux_conv1(aux)
        aux = self.act1(aux)

        # print(aux.size())
        aux = self.aux_conv2(aux)
        aux = self.batchnorm_aux(aux)
        aux = self.act2(aux)
        # x = self.postTCN_conv(x)
        # x = self.act2(x)
        # print(aux.size())

        x = torch.cat([x, aux], dim=1)

        x = self.merge_conv(x)
        x = x[:, :, :-self.merge_conv.padding[0]].contiguous()  # causal left padding
        x = self.act3(x)

        # flatten
        x = x.view(x.size()[0], -1)  # torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.act4(x)

        x = self.fc2(x)
        x = self.batchnorm(x)
        x = self.act5(x)

        x = self.fc3(x)

        # reshape
        x = x.view(*x.size(), 1)

        # logging.info(f"\tIn Model: input size {in_size},
        #            "output size {x.size()})

        return x
