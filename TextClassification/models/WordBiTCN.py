import torch
import torch.nn as nn
import torch.nn.functional as F
from TextClassification.models.lib.BiTemporalConvNet_v5 import BiTemporalConvNet


class WordBiTCN(nn.Module):
    def __init__(self, config):
        super(WordBiTCN, self).__init__()
        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
                                          config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)
        self.num_channels_word = [300, 300, 300, 300]
        self.label_size = config.Label[config.UseLabel]
        self.TCN_word = BiTemporalConvNet(
            num_inputs=300, num_channels=self.num_channels_word, kernel_size=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, self.label_size)
        )

    def forward(self, input):
        x = self.WordEmbedding(input)
        x = x.permute(1, 2, 0)
        x = self.TCN_word(x)
        x = torch.mean(x, dim=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
