import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

kernal_sizes = [1, 2, 3, 4, 5]


class CRAN(nn.Module):
    def __init__(self, config):
        super(CRAN, self).__init__()
        self.hidden_dim = config.WordVectorsDim
        self.gru_layers = 2

        self.word_flatten_size = 1280

        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
                                          config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)
        # 双向GRU，//操作为了与后面的Attention操作维度匹配，hidden_dim要取偶数！
        self.bigru = nn.GRU(config.WordVectorsDim, self.hidden_dim //
                            2, num_layers=self.gru_layers, dropout=0.2, bidirectional=True)
        # 由nn.Parameter定义的变量都为requires_grad=True状态
        self.weight_W = nn.Parameter(torch.Tensor(
            self.hidden_dim, config.WordVectorsDim))
        # self.weight_proj = nn.Parameter(torch.Tensor(config.word_embedding_dim, 1))
        self.weight_proj = nn.Parameter(torch.Tensor(40, 1))
        self.label_size = config.Label[config.UseLabel]
        self.fc = nn.Linear(self.hidden_dim, self.label_size)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

        convs2 = [
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.WordVectorsDim,
                    out_channels=256,
                    kernel_size=kernel_size),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.3),
                nn.Conv1d(
                    in_channels=256, out_channels=256,
                    kernel_size=kernel_size),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.3),
                nn.AdaptiveMaxPool1d(1),
            ) for kernel_size in [1, 2, 3, 4, 5]
        ]
        self.convs2 = nn.ModuleList(convs2)

        self.fc_word = nn.Sequential(
            nn.Linear(self.word_flatten_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2048, self.label_size)
        )

    def forward(self, sentence):
        embeds = self.WordEmbedding(sentence)  # [seq_len, bs, emb_dim]
        # print(embeds.size())
        gru_out, _ = self.bigru(embeds)  # [seq_len, bs, hid_dim]
        # print(gru_out.size())
        x = gru_out.permute(1, 0, 2)
        # print(x.size())
        # # # Attention过程，与上图中三个公式对应
        u = torch.matmul(torch.matmul(x, self.weight_W), x.permute(0, 2, 1))
        u = torch.tanh(u)
        att = torch.matmul(u, self.weight_proj)  # [bs, seq_len, 1]
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        # # # Attention过程结束
        # print(scored_x.size())
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        conv_out_word = [
            conv(scored_x.permute(0, 2, 1)) for conv in self.convs2
        ]
        for i in range(len(conv_out_word)):
            conv_out_word[i] = conv_out_word[i].view(conv_out_word[i].size(0),
                                                     -1)
            # print(i)
            # print(conv_out_word[i].size())
        flatten_word = torch.cat(conv_out_word, dim=1)
        logits_word = self.fc_word(flatten_word)
        y = logits_word
        return y
