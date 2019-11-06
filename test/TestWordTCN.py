#%%
from TextClassification.models.WordTCN import WordTCN
import torch
from config.BaseConfig import BaseConfig
config = BaseConfig()
net = WordTCN(config)
#%%
a = torch.randint(8,(3,3))
print(a.size(),a)
net(a)

# %%
net._get_name()
#%%
import TextClassification.models
m  = __import__("TextClassification")
m = getattr(m,"models")
m = getattr(m,"WordTCN")
#%%
from config.BaseConfig import BaseConfig
config = BaseConfig()
import torch
a = torch.randint(8,(3,3))
print(a.size(),a)

# %%


# %%
from TextClassification.models import LoadModel
from config.TestConfig import TestConfig
config = TestConfig()
model = LoadModel("WordTCN")
net = model(config)

import torch
a = torch.randint(8,(3,3))
print(a.size(),a)
# %%
net = torch.nn.DataParallel(net)
# %%
optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)

# %%
config.Label[config.UseLabel]

# %%
