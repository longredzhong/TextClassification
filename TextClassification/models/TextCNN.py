import torch
import torch.nn as nn
import torch.nn.functional as F
class TextCNN(nn.Module):
    def __init__(self,config):
        super(TextCNN,self).__init__()
        self.WordEmbedding = torch.load(r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\embedding.pt")

        self.conv3 = nn.Conv1d(config.WordVectorsDim, 256, 3)
        self.conv4 = nn.Conv1d(config.WordVectorsDim, 256, 4)
        self.conv5 = nn.Conv1d(config.WordVectorsDim, 256, 5)
        self.Max3_pool = nn.AdaptiveMaxPool1d(1)
        self.Max4_pool = nn.AdaptiveMaxPool1d(1)
        self.Max5_pool = nn.AdaptiveMaxPool1d(1)

        self.label_size = config.label_size

        self.fc = nn.Linear(256*3, self.label_size)

    def forward(self, input):
        x = self.WordEmbedding(input)
        # print(x.size())
        x = x.permute(0,2,1)
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), 1)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x