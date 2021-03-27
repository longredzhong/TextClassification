import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRNNAttention(nn.Module):
    def __init__(self, embedding_path, hidden_size , num_classes):
        super(TextRNNAttention, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.WordEmbedding.weight.requires_grad = False
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.lstm = nn.LSTM(WordVectorsDim, hidden_size, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_size * 2))
        # self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(hidden_size * 2, num_classes)


    def forward(self, x):
        emb = self.WordEmbedding(x)
        lstmout, _ = self.lstm(emb)

        # attention # https://www.cnblogs.com/DjangoBlog/p/9504771.html
        M = self.tanh1(lstmout)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = lstmout * alpha
        out = torch.sum(out, axis=1)

        out = F.relu(out)
        out = self.fc1(out)

        return out