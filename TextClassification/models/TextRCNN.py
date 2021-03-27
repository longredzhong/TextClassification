import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRCNN(nn.Module):

    def __init__(self, embedding_path, hidden_size , num_classes):
        super(TextRCNN, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.WordEmbedding.weight.requires_grad = False
        WordVectorsDim = self.WordEmbedding.embedding_dim

        self.lstm = nn.LSTM(WordVectorsDim, hidden_size , 2, bidirectional=True,
                            batch_first=True, dropout=0.5)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_size * 2 + WordVectorsDim, num_classes)

    def forward(self, x):
        embed = self.WordEmbedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out),2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # 各维度调换shape(这里应该是变成了转置)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out