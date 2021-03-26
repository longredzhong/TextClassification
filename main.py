# %%
import fitlog
import torch
from transformers import AutoTokenizer

from TextClassification.dataloader.tnewsDataloader import (get_dataloader,
                                                           tnewsDataset)
from TextClassification.dataloader.toutiaonews38wDataloader import get_dataloader
from TextClassification.models.TextCNN import TextCNN
from TextClassification.models.fastext import FastText
from TextClassification.trainer import TextTrainer
from TextClassification.utils.tokenizers import Tokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fitlog.set_log_dir("logs/")
fitlog.commit(__file__)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenizer = Tokenizer(r"/home/longred/code/TextClassification/dataset/embedding/vocab.txt")
# dataset_path = r"/home/longred/code/TextClassification/dataset/tnews_public"
# dataset_path = r"/home/longred/code/TextClassification/dataset/iflytek"
dataset_path = r"/home/longred/code/TextClassification/dataset/toutiaonews38w"
train_dataloader, dev_dataloader ,num_classes= get_dataloader(dataset_path, tokenizer,batch_size=128)
# %%
# %%


class config():
    num_classes = num_classes
    embedding_path = r"/home/longred/code/TextClassification/dataset/embedding/bert-base-chinese.pt"
    lr = 0.00001
    seed = 666
    dropout = 0.5

model = TextCNN(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

trainer = TextTrainer(optimizer, config.seed)
# %%
# fitlog.add_hyper(config)
# trainer.train(model,train_dataloader,device)
# %%
trainer.fit(model, train_dataloader, dev_dataloader, fitlog, device)

# %%
fitlog.finish()