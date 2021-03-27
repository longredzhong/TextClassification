from TextClassification.models.tcn import TemporalConvNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, embedding_path,num_classes,ch_n=[256,256,256,256]):
        super(TCN, self).__init__()
        self.WordEmbedding = torch.load(embedding_path)
        self.WordEmbedding.weight.requires_grad = False
        WordVectorsDim = self.WordEmbedding.embedding_dim
        self.TCN_block = TemporalConvNet(WordVectorsDim,ch_n)
        self.dropout = nn.Dropout(0.5)
        self.btachnorm = nn.BatchNorm1d(ch_n[-1])
        self.fc = nn.Linear(ch_n[-1], num_classes)
        self.conv = nn.Conv2d(ch_n[-1], ch_n[-1], (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom


    def forward(self, x):
        out = self.WordEmbedding(x)
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
        # out = self.dropout(out)
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


