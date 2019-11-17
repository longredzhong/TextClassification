import torch
import torch.nn as nn
import torch.nn.functional as F
from TextClassification.models.lib.BiTemporalConvNet_v3 import BiTemporalConvNet


class WordBiTCNMCNN(nn.Module):
    def __init__(self, config):
        super(WordBiTCNMCNN, self).__init__()
        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
                                          config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)
        self.num_channels_word = [300, 300, 300, 300]
        self.label_size = config.Label[config.UseLabel]
        self.TCN_word = BiTemporalConvNet(
            num_inputs=300, num_channels=self.num_channels_word, kernel_size=2, dropout=0.2)
        convs1 = [
            nn.Sequential(
                nn.Conv1d(in_channels=300,
                          out_channels=300,
                          kernel_size=kernel_size,
                          stride=3),
                nn.BatchNorm1d(300),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=300,
                          out_channels=300,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(300),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(1)
            )
            for kernel_size in [1, 2, 3, 4, 5]
        ]
        self.convs1 = nn.ModuleList(convs1)
        self.fc = nn.Sequential(
            nn.Linear(1500, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, self.label_size)
        )

    def forward(self, input):
        x = self.WordEmbedding(input)
        x = x.permute(1, 2, 0)
        x = self.TCN_word(x)
        x = [conv(x) for conv in self.convs1]
        x = torch.cat(x, dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
