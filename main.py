#%%
from transformers.models.albert import AlbertConfig
# from TextClassification.dataloader.iflytekDataloader import iflytekDataset, get_dataloader
from  TextClassification.dataloader.tnewsDataloader import tnewsDataset,get_dataloader
from TextClassification.models.AlbertForSequenceClassification import AlbertForSequenceClassification
from TextClassification.utils.tokenizers import Tokenizer
from transformers import AdamW
# %%
tokenizer = Tokenizer(
    r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\vocab.txt")

config = AlbertConfig.from_pretrained(
    r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\config.json")
train_dataloader, dev_dataloader = get_dataloader(
    r'C:\Users\LongRed\code\TextClassification\dataset\tnews_public', tokenizer, batch_size=32)
config.num_labels = 119
config.dropout = 0.5
net = AlbertForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\pytorch_model.bin",
    config=config)
learning_rate = 5e-4
no_decay = ["bias", "LayerNorm.weight"]
bert_param_optimizer = list(net.bert.named_parameters())
linear_param_optimizer = list(net.classifier.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01, 'lr': learning_rate},
    {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': learning_rate},
    {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01, 'lr': 0.001},
    {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
     'lr': 0.001}
]
optimizer = AdamW(params=optimizer_grouped_parameters,
                  lr=learning_rate, correct_bias=False)

# %%
x = iter(train_dataloader)
#%%
input,label = next(x)
# %%
net(input,labels=label)
# %%
label
# %%
