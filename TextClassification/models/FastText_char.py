import torch
import torch.nn as nn
import torch.nn.functional as F
class FastText_char(nn.Module):
    def __init__(self, config):
        super(FastText_char, self).__init__()
        self.label_size = config.Label[config.UseLabel]
        self.CharEmbedding = nn.Embedding(config.CharVocabSize,
        config.CharVectorsDim)
        if config.CharVectors is not None:
            self.CharEmbedding.weight.data.copy_(config.CharVectors)
        self.content_fc = nn.Sequential(
            nn.Linear(config.CharVectorsDim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.label_size)
        )

    def forward(self, input):
        content_ = torch.mean(self.CharEmbedding(input).permute(1, 0, 2), dim=1)

        out = self.content_fc(content_.view(content_.size(0), -1))

        return out
