import torch
import torch.nn as nn
import torch.nn.functional as F
from TextClassification.models.lib.non_local_gaussian import NONLocalBlock1D
from TextClassification.models.lib.TemporalConvNet import TemporalConvNet



class CharTCN_NonLocal(nn.Module):
    def __init__(self, config):
        super(CharTCN_NonLocal, self).__init__()
        self.CharEmbedding = nn.Embedding(config.CharVocabSize,
                                          config.CharVectorsDim)
        if config.CharVectors is not None:
            self.CharEmbedding.weight.data.copy_(config.CharVectors)
        self.num_channels_Char = [300, 300, 300, 300]
        self.label_size = config.Label[config.UseLabel]
        self.TCN_Char = TemporalConvNet(
            num_inputs=300, num_channels=self.num_channels_Char, kernel_size=2, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, self.label_size)
        )
        self.NONLocal = NONLocalBlock1D(300, 100, True, True)

    def forward(self, input):
        x = self.CharEmbedding(input)
        x = x.permute(1, 2, 0)
        x = self.NONLocal(x)
        x = self.TCN_Char(x)
        x = torch.mean(x, dim=2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
