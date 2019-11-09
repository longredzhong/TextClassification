#%%
from TextClassification.utils.Metrics import metrics
#%%
pred = [1,2,3,4,5,6,0,2,3,4]
label = [1,2,3,4,3,6,0,2,3,4]
loss = [0]
#%%
a = metrics()
#%%
a.update(pred,label,loss)
a.GetAvgAccuracy()
#%%
a.reset()
#%%
a.GetAvgAccuracy()
#%%
from torch import nn
import torch
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

# %%
print(input,target)

# %%
[output.tolist()]

# %%
