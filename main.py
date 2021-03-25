#%%
from TextClassification.models.TextCNN import TextCNN
import torch
from transformers.optimization import AdamW
from TextClassification.utils.tokenizers import Tokenizer
from TextClassification.trainer import TextTrainer
from TextClassification.dataloader.tnewsDataloader import get_dataloader, tnewsDataset
tokenizer = Tokenizer(
    r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\vocab.txt")
dataset_path = r"C:\Users\LongRed\code\TextClassification\dataset\tnews_public"
train_dataloader, dev_dataloader = get_dataloader(dataset_path,tokenizer)
#%%
#%%
class config():
    WordVectorsDim = 128
    label_size = 15


model = TextCNN(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

trainer = TextTrainer(optimizer)
#%%
trainer.fit(model,train_dataloader,dev_dataloader)

# %%
