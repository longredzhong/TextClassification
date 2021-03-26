# %%
import os
import sys

sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
import torchtext.legacy as torchtext

from TextClassification.utils.util import sequence_padding, toTensor
from TextClassification.utils.tokenizers import Tokenizer

# %%

# %%
token = Tokenizer(
    r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_489k\vocab.txt")
# %%
rawpath = r"C:\Users\LongRed\code\TextClassification\dataset\Fudan\train"
# %%

fields = [
    ("input", torchtext.data.RawField(postprocessing=sequence_padding)),
    ("label", torchtext.data.RawField(postprocessing=toTensor))
]
# %%
from TextClassification.dataloader.FudanDataloader import FudanDataset
#%%
dataset = FudanDataset(rawpath,token,fields)
#%%
i = torchtext.data.Iterator(dataset,batch_size=4,shuffle=True)
# %%
i
# %%
for n in i:
    n
# %%
