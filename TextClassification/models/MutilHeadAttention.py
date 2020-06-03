import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
kernal_sizes = [1, 2, 3, 4, 5]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask=None):

        # scores : [batch_size x 5 x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(10)
        # if attn_mask is not None:
        #     scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # print(context)
        return context, attn


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self):
        super(MultiHeadAttentionBlock, self).__init__()
        self.W_Q = nn.Linear(300, 300)
        self.W_K = nn.Linear(300, 300)
        self.W_V = nn.Linear(300, 300)
        self.line = nn.Linear(300, 300)
        self.LN = nn.LayerNorm(300)

    def forward(self, input):
        Q = input
        K = input
        V = input
        # q: [batch_size x len_q x 50], k: [batch_size x len_k x 50], v: [batch_size x len_k x 50]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # q_s: [batch_size x 5 x len_q x 10]
        q_s = self.W_Q(Q).view(batch_size, -1, 6, 50).transpose(1, 2)
        # k_s: [batch_size x 5 x len_k x 10]
        k_s = self.W_K(K).view(batch_size, -1, 6, 50).transpose(1, 2)
        # v_s: [batch_size x 5 x len_k x 10]
        v_s = self.W_V(V).view(batch_size, -1, 6, 50).transpose(1, 2)

        # if attn_mask is not None: # attn_mask : [batch_size x len_q x len_k]
        #     attn_mask = attn_mask.unsqueeze(1).repeat(1, 5, 1, 1) # attn_mask : [batch_size x 5 x len_q x len_k]
        # context: [batch_size x 5 x len_q x 10], attn: [batch_size x 5 x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, 300)  # context: [batch_size x len_q x 5 * 10]
        output = self.line(context)
        return self.LN(output + residual)  # output: [batch_size x len_q x 50]


class MutilHeadAttention(nn.Module):
    def __init__(self, config):
        super(MutilHeadAttention, self).__init__()
        self.MultiHeadAttentionBlock = MultiHeadAttentionBlock()
        '''Embedding Layer'''
        # 使用预训练的词向量
        self.WordEmbedding = nn.Embedding(config.WordVocabSize,
                                          config.WordVectorsDim)
        if config.WordVectors is not None:
            self.WordEmbedding.weight.data.copy_(config.WordVectors)

        convs = [
            nn.Sequential(
                nn.Conv1d(in_channels=config.WordVectorsDim,
                          out_channels=512,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels=512,
                          out_channels=512,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(1)
            )
            for kernel_size in kernal_sizes
        ]

        self.convs = nn.ModuleList(convs)
        self.label_size = config.Label[config.UseLabel]
        self.fc = nn.Sequential(
            nn.Linear(512*5, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.label_size)
        )
        self.drop = nn.Dropout(0.5)

    def forward(self, inputs):
        embeds = self.WordEmbedding(inputs)  # seq * batch * embed
        embeds = self.MultiHeadAttentionBlock(embeds)
        # 进入卷积层前需要将Tensor第二个维度变成emb_dim，作为卷积的通道数
        conv_out = [conv(embeds.permute(1, 2, 0)) for conv in self.convs]
        for i in range(len(conv_out)):
            conv_out[i] = conv_out[i].view(conv_out[i].size(0), -1)
        conv_out = torch.cat(conv_out, dim=1)
        flatten = conv_out.view(conv_out.size(0), -1)
        line = self.drop(flatten)
        logits = self.fc(line)
        return logits
