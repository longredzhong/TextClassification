#%%
from TextClassification.models.TCN import WordTCN
import torch
from config.BaseConfig import BaseConfig
config = BaseConfig()
net = WordTCN(config)
#%%
a = torch.randint(8,(3,3))
print(a.size(),a)
net(a)

# %%
