#%%
import torch
#model = torch.load("run_log/WordTCN/2019-11-09 19:06:19.116539/WordTCN.pth")
model = torch.load("run_log/WordTCN_NonLocal/2019-11-09 19:05:47.345005/WordTCN_NonLocal.pth")
# %%
import tensorwatch

# %%
tensorwatch.draw_model(model=model,input_shape=[1000])

# %%
for param in model.parameters():
     print(type(param.data), param.size())

# %%
