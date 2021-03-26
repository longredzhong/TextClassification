import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.WordEmbedding = torch.load(config.embedding_path)
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, WordVectorsDim)) for k in [2,3,4]])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(256 * len([2,3,4]), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.WordEmbedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out