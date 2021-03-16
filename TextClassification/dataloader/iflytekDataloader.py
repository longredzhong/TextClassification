#%%
import json
import torch
from torch.utils.data import  Dataset,DataLoader
from torch.utils.data import dataset


class iflytekDataset(Dataset):
    def __init__(self,raw_path,tokenizer) -> None:
        self.sentence = []
        self.input = []
        self.ids_mask = []
        self.label = []
        self.load_data(raw_path)
        self.preprocess(tokenizer)
        self.n_class = len(set(self.label))
        super().__init__()
        
    def __getitem__(self, index):
        return self.input[index],self.label[index]

    def __len__(self):
        return len(self.label)

    def preprocess(self,tokenizer):
        for tmp in self.sentence:
            t= tokenizer.encode(tmp)
            self.input.append(toTensor(t[0]))
            self.ids_mask.append(toTensor(t[1]))
        self.label = toTensor(self.label)

    def load_data(self,raw_path):
        with open(raw_path,mode='r',encoding="UTF8") as f:
            texts = f.readlines()
        for tmp in texts:
            tmp = json.loads(tmp)
            self.sentence.append(tmp['sentence'])
            self.label.append(int(tmp['label']))


#%%
import sys
import os
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))))
raw_path = r"C:\Users\LongRed\code\TextClassification\dataset\iflytek\train.json"
from TextClassification.utils.tokenizers import Tokenizer
from TextClassification.utils.util import sequence_padding
from TextClassification.utils.util import toTensor
#%%
token = Tokenizer(r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_489k\vocab.txt")
# %%
dataset = iflytekDataset(raw_path,token)

# %%
dataset.__getitem__(9)
# %%
def collate_pad(batch):
    text,label = zip(*batch)
    text = sequence_padding(text)
    label = toTensor(label)
    return (text,label)
# %%
dataloader = DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_pad)
# %%
next(iter(dataloader))
# %%
