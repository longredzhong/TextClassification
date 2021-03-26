#%%
import json
import torch
from torch.utils.data import  Dataset,DataLoader
from torch.utils.data import dataset
import os
from TextClassification.utils.util import toTensor,sequence_padding

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
        target = []
        for tmp in texts:
            tmp = json.loads(tmp)
            self.sentence.append(tmp['sentence'])
            target.append(tmp['label'])
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.label = [target2id[label] for label in target]

def collate_pad(batch):
    text,label = zip(*batch)
    text = sequence_padding(text)
    label = toTensor(label)
    return (text,label)

def get_dataloader(raw_path,tokenizer,batch_size=32):
    train_path = os.path.join(raw_path,'train.json')
    dev_path = os.path.join(raw_path,'dev.json')
    test_path = os.path.join(raw_path,'test.json')
    train_dataset = iflytekDataset(train_path,tokenizer)
    dev_dataset = iflytekDataset(dev_path,tokenizer)
    train_dataloader =  DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_pad)
    dev_dataloader =  DataLoader(dev_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_pad)
    return train_dataloader,dev_dataloader
