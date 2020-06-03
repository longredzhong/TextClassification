import torch
import numpy as np
import torch.nn as nn
from sklearn.utils import shuffle
from torch.autograd import Variable


class BGRU_Attention(torch.nn.Module):
    def __init__(self, config):

        super(BGRU_Attention, self).__init__()
        self.hidden_dim = 128
        self.batch_size = config.TrainBatchSize
        self.use_gpu = torch.cuda.is_available()

        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
                                          config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)
        self.label_size = config.Label[config.UseLabel]
        self.num_layers = 1
        #self.bidirectional = True
        self.dropout = 0.5
        self.bilstm = nn.GRU(config.WordVectorsDim, self.hidden_dim // 2, batch_first=True,
                             num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)
        self.hidden = self.init_hidden()
        self.mean = config.__dict__.get("lstm_mean", True)
        self.attn_fc = torch.nn.Linear(config.WordVectorsDim, 1)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(2*self.num_layers,
                                      batch_size, self.hidden_dim // 2).cuda())
            c0 = Variable(torch.zeros(2*self.num_layers,
                                      batch_size, self.hidden_dim // 2).cuda())
        else:
            h0 = Variable(torch.zeros(2*self.num_layers,
                                      batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2*self.num_layers,
                                      batch_size, self.hidden_dim // 2))
        return (h0, c0)

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        # (batch, seq_len, cell_size) * (batch, cell_size, 1) = (batch, seq_len, 1)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        # (batch, cell_size, seq_len) * (batch, seq_len, 1) = (batch, cell_size, 1)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)
    # end method attention

    def forward(self, X):
        embedded = self.WordEmbedding(X).permute(1, 0, 2)
        # print(embedded.size())
        hidden = self.init_hidden(embedded.size()[0])
        rnn_out, hidden = self.bilstm(embedded, hidden[0])
        h_n = hidden
        attn_out = self.attention(rnn_out, h_n)
        logits = self.hidden2label(attn_out)
        # print(logits.size())
        return logits
