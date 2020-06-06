import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict


class Inception(nn.Module):
    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception, self).__init__()
        assert(co % 4 == 0)
        cos = [co//4]*4
        self.activa = nn.Sequential()
        if norm:
            self.activa.add_module('norm', nn.BatchNorm1d(co))
        if relu:
            self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[0], 1, stride=1)),
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[1], 1)),
            ('norm1', nn.BatchNorm1d(cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[1], cos[1], 3, stride=1, padding=1)),
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin, cos[2], 3, padding=1)),
            ('norm1', nn.BatchNorm1d(cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv1d(cos[2], cos[2], 5, stride=1, padding=2)),
        ]))
        self.branch4 = nn.Sequential(OrderedDict([
            # ('pool',nn.MaxPool1d(2)),
            ('conv3', nn.Conv1d(cin, cos[3], 3, stride=1, padding=1)),
        ]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = self.activa(
            torch.cat((branch1, branch2, branch3, branch4), 1))
        return result


class CNNText_inception(nn.Module):
    def __init__(self, config):
        super(CNNText_inception, self).__init__()
        incept_dim = 512

        self.label_size = config.Label[config.UseLabel]
        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
                                          config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)


        self.content_conv = nn.Sequential(
            # (batch_size,64,opt.content_seq_len)->(batch_size,64,(opt.content_seq_len)/2)
            Inception(config.WordVectorsDim, incept_dim),
            # Inception(incept_dim,incept_dim),#(batch_size,64,opt.content_seq_len/2)->(batch_size,32,(opt.content_seq_len)/4)
            Inception(incept_dim, incept_dim),
            nn.AdaptiveMaxPool1d(1)
        )
        hidden_size = 1024
        self.fc = nn.Sequential(
            nn.Linear(incept_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, self.label_size)
        )

    def forward(self, content):

        content = self.WordEmbedding(content)
        content_out = self.content_conv(content.permute(1, 2, 0))
        out = content_out.view(content_out.size(0), -1)
        out = self.fc(out)
        return out
