import torch
import torch.nn as nn
import torch.nn.functional as F
from TextClassification.models.lib.TemporalConvNet import TemporalConvNet

kernal_sizes = [1, 2, 3, 4, 5]


class TCN_MCNN(nn.Module):
    """
    Some Information about TCN_SCNN
    TCN_singleCNN_ChannelAttention
    return Linear label size
    """

    def __init__(self, config):
        super(TCN_MCNN, self).__init__()
        self.CharEmbedding = nn.Embedding(config.CharVocabSize,
        config.CharVectorsDim)
        if config.CharVectors is not None:
            self.CharEmbedding.weight.data.copy_(config.CharVectors)

        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
        config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)


        self.num_channels = [300, 300, 300, 300]
        self.label_size = config.Label[config.UseLabel]

        self.TCN_char = TemporalConvNet(
            300, self.num_channels, kernel_size=2, dropout=0.2)
        self.TCN_word = TemporalConvNet(
            300, self.num_channels, kernel_size=2, dropout=0.2)
        convs1 = [
            nn.Sequential(
                nn.Conv1d(in_channels=300,
                          out_channels=256,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(256),
                nn.ReLU6(inplace=True),
                nn.Conv1d(in_channels=256,
                          out_channels=256,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(256),
                nn.ReLU6(inplace=True),
                nn.AdaptiveMaxPool1d(1)
            )
            for kernel_size in [1, 2, 3, 4, 5]
        ]
        self.convs1 = nn.ModuleList(convs1)
        self.fc = nn.Sequential(
            nn.Linear(256*10, 2048),
            nn.Dropout(0.2),
            nn.BatchNorm1d(2048),
            nn.ReLU6(inplace=True),
            nn.Linear(2048, self.label_size)
        )

    def forward(self, input):
        char = input[0]
        word = input[1]

        char = self.CharEmbedding(char)
        word = self.WordEmbedding(word)
        char = char.permute(1, 2, 0)
        word = word.permute(1, 2, 0)

        char = self.TCN_char(char)
        char = [conv(char) for conv in self.convs1]
        char = torch.cat(char, dim=1)
        char = char.view(char.size(0), -1)

        word = self.TCN_word(word)
        word = [conv(word) for conv in self.convs1]
        word = torch.cat(word, dim=1)
        word = word.view(word.size(0), -1)

        x = torch.cat([word, char], dim=1)

        x = self.fc(x)

        return x
