#%%
from TextClassification.models.TextCNN import TextCNN
import torch

from TextClassification.utils.tokenizers import Tokenizer
from TextClassification.trainer import TextTrainer
from TextClassification.dataloader.tnewsDataloader import get_dataloader, tnewsDataset
import fitlog
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fitlog.set_log_dir("logs/") 
fitlog.commit(__file__)
tokenizer = Tokenizer(
    r"/home/longred/code/TextClassification/dataset/albert_tiny_zh/vocab.txt")
dataset_path = r"/home/longred/code/TextClassification/dataset/tnews_public"
train_dataloader, dev_dataloader = get_dataloader(dataset_path,tokenizer)
#%%
#%%
class config():
    WordVectorsDim = 128
    label_size = 15
    embedding_path = r"/home/longred/code/TextClassification/dataset/albert_tiny_zh/embedding.pt"
    lr = 0.0001
    seed = 666

model = TextCNN(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

trainer = TextTrainer(optimizer,config.seed)
#%%
# fitlog.add_hyper(config)
# trainer.train(model,train_dataloader,device)
#%%
trainer.fit(model,train_dataloader,dev_dataloader,fitlog,device)

# %%
