# %%
import os
import sys

sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
import torchtext.legacy as torchtext

from TextClassification.utils.util import sequence_padding, toTensor
from TextClassification.utils.tokenizers import Tokenizer
#%%
from TextClassification.dataloader.tnewsDataloader import tnewsDataset,get_dataloader

path = r"C:\Users\LongRed\code\TextClassification\dataset\tnews_public"

token = Tokenizer(
    r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\vocab.txt")

train,val = get_dataloader(path,token)
# %%
for data in train:
    print(data[1])
# %%
class config:
    WordVectorsDim=128
    label_size=15
from TextClassification.models.TextCNN import TextCNN
model = TextCNN(config)
# %%
model(data[0])
# %%
