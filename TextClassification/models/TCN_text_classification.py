from TextClassification.models.tcn import TemporalConvNet, TemporalConvNet_gate

import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, embedding_path,num_classes,ch_n=[256,256,256,256],kernel_size=2,max_pool_size=1):
        super(TCN, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.WordEmbedding.weight.requires_grad = False
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.TCN_block = TemporalConvNet(WordVectorsDim,ch_n, kernel_size)
        self.a_max_pool = nn.AdaptiveMaxPool1d(max_pool_size)
        self.btachnorm = nn.BatchNorm1d(1024)
        self.conv = nn.Conv1d(ch_n[-1]+WordVectorsDim,1024,1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024*max_pool_size, num_classes)


    def forward(self, x):
        out = self.WordEmbedding(x)
        out = out.permute(0,2,1)
        out = torch.cat([self.TCN_block(out),out],dim=1)
        out = self.conv(out)
        out = self.a_max_pool(out)
        out = out.view(out.size(0),-1)
        out = self.btachnorm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    

from TextClassification.models.attentionBlock import AttentionBlock
class TCN_attention(nn.Module):
    def __init__(self, embedding_path,num_classes,ch_n=[256,256,256,256]):
        super(TCN_attention, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.WordEmbedding.weight.requires_grad = False
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.TCN_block = TemporalConvNet(WordVectorsDim,ch_n)
        self.att = AttentionBlock(WordVectorsDim,WordVectorsDim,WordVectorsDim)
        self.dropout = nn.Dropout(0.5)
        self.btachnorm = nn.BatchNorm1d(ch_n[-1])
        self.fc = nn.Linear(ch_n[-1], num_classes)
        self.conv = nn.Conv2d(ch_n[-1], ch_n[-1], (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom


    def forward(self, x):
        out = self.WordEmbedding(x)
        out = self.att(out)
        out = out.permute(0,2,1)
        out = self.TCN_block(out)
        
        out = out.unsqueeze(3) 
        while out.size()[2] > 1:
            out = self._block(out)
        out = out.squeeze()
        # out = F.adaptive_max_pool1d(out,output_size=1)
        # out = F.adaptive_avg_pool1d(out,output_size=1)
        # out = out.view(out.size(0),-1)
        # out = self.btachnorm(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc(out)
        return out
    
    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

class Gate_TCN(nn.Module):
    def __init__(self, embedding_path,num_classes,ch_n=[256,256,256,256],kernel_size=2,max_pool_size=1):
        super(Gate_TCN, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.WordEmbedding.weight.requires_grad = False
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.TCN_block = TemporalConvNet_gate(WordVectorsDim,ch_n, kernel_size)
        self.a_max_pool = nn.AdaptiveMaxPool1d(max_pool_size)
        self.dropout = nn.Dropout(0.2)
        self.btachnorm = nn.BatchNorm1d(1024)
        self.conv = nn.Conv1d(ch_n[-1]+WordVectorsDim,1024,1)
        self.fc = nn.Linear(1024*max_pool_size, num_classes)


    def forward(self, x):
        out = self.WordEmbedding(x)
        out = out.permute(0,2,1)
        out = torch.cat([self.TCN_block(out),out],dim=1)
        out = self.conv(out)
        out = self.a_max_pool(out)
        out = out.view(out.size(0),-1)
        out = F.relu(out)
        out = self.btachnorm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out
