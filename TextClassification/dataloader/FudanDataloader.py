#%%
import io
import os
import time
from TextClassification.utils.util import open
import torchtext.legacy as torchtext

class FudanDataset(torchtext.data.Dataset):
    def __init__(self,rawpath,tokenizer,fields):
        example = []
        label_list = os.listdir(rawpath)
        label_paths = []
        for i in label_list:
            label_paths.append(os.path.join(rawpath,i,'utf8'))
        docs_path = []
        for i in label_paths:
            docs_path.append(os.listdir(i))
        # docs = []
        # label = []
        for i in range(len(label_list)):
            for j in range(len(docs_path[i])):
                path = os.path.join(label_paths[i],docs_path[i][j])
                with open(path,encoding='UTF8') as f:
                    text = f.read()
                text = tokenizer.encode(text)[0]
                # docs.append(text)
                # label.append(label_list[i])
                example.append(torchtext.data.Example.fromlist([text,label_list[i]],fields))
        super().__init__(example,fields)


#%%

# %%

# %%
