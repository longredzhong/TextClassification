# %%
import torch
from config.TestConfig import TestConfig
from TextClassification.dataloader import GetLoader
config = TestConfig()
Loader = GetLoader(config.DatasetName)
TrainIter, ValIter = Loader(config)
# %%
batch = next(iter(TrainIter))
Word = batch.word_text
# %%
embedding = torch.nn.Embedding(config.WordVocabSize, config.WordVectorsDim)


# %%
x = embedding(Word)

# %%
x.size()
# %%
x = x.permute(1, 2, 0)

# %%
x = x.flip((1, 2))

# %%
