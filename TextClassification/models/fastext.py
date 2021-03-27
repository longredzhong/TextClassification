import torch
import torch.nn as nn
import torch.nn.functional as F
class FastText(nn.Module):
    def __init__(self,embedding_path,num_classes):
        super(FastText, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.dropout = nn.Dropout(0.5)
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.fc1 = nn.Linear(WordVectorsDim, num_classes)


    def forward(self, x):
        out = self.WordEmbedding(x)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc1(out)
        return out