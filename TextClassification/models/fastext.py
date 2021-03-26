import torch
import torch.nn as nn
import torch.nn.functional as F
class FastText(nn.Module):
    def __init__(self, config):
        super(FastText, self).__init__()
        self.WordEmbedding = torch.load(config.embedding_path)
        self.dropout = nn.Dropout(config.dropout)
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.fc1 = nn.Linear(WordVectorsDim, 300)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(300, config.num_classes)

    def forward(self, x):
        out = self.WordEmbedding(x)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out