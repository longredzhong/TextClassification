# %%
import os
import sys

sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))


from TextClassification.utils.util import sequence_padding, toTensor
from TextClassification.utils.tokenizers import Tokenizer
#%%
from TextClassification.dataloader.tnewsDataloader import tnewsDataset,get_dataloader

path = r"/home/longred/code/TextClassification/dataset/tnews_public"

token = Tokenizer(
    r"/home/longred/code/TextClassification/dataset/embedding/vocab.txt")

train,val,nclass = get_dataloader(path,token)
# %%
train_dataset = tnewsDataset(r"/home/longred/code/TextClassification/dataset/tnews_public/train.json",token)
# %%
set(train_dataset.label.tolist())
# %%
