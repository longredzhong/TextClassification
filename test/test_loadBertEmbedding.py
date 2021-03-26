#%%
import os
import sys
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
import torch
pred = torch.load(r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\pytorch_model.bin")
# %%
# net.word_embeddings
# %%
from transformers.models.albert.modeling_albert import AlbertEmbeddings
from transformers.models.albert import AlbertConfig,AlbertModel
config = AlbertConfig.from_pretrained(
        r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh\config.json")
# %%

net = AlbertModel.from_pretrained( r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_zh")
# %%
embedding = net.get_input_embeddings()
# %%
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
#%%
embedding = model.get_input_embeddings()
# %%
import torch
torch.save(embedding,r"/home/longred/code/TextClassification/dataset/embedding/bert-base-chinese.pt")
# %%

# %%
