import torch
import torch.nn as nn
import torch.nn.functional as F
class FastText_word(nn.Module):
    def __init__(self, config):
        super(FastText_word, self).__init__()
        self.label_size = config.Label[config.UseLabel]
        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
        config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)
        self.content_fc = nn.Sequential(
            nn.Linear(config.WordVectorsDim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.label_size)
        )

    def forward(self, input):
        content_ = torch.mean(self.WordEmbedding(input).permute(1, 0, 2), dim=1)

        out = self.content_fc(content_.view(content_.size(0), -1))

        return out
