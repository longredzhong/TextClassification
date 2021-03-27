import torch
import torch.nn as nn
class TextRNN(nn.Module):
    '''BILSTM'''
    def __init__(self, embedding_path, num_filters, num_classes):
        super(TextRNN, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.WordEmbedding.weight.requires_grad = False
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.lstm = nn.LSTM(WordVectorsDim, num_filters, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(num_filters * 2, num_classes)


    def forward(self, x):
        embed = self.WordEmbedding(x)
        lstmout, _ = self.lstm(embed)    # https://blog.csdn.net/m0_45478865/article/details/104455978
        fc_input = lstmout[:, -1, :]     # 句子最后时刻的 hidden state
        out = self.fc(fc_input)
        return out