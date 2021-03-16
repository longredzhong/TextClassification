#%%
import sys
import os
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
from TextClassification.utils.tokenizers import Tokenizer
#%%
token = Tokenizer(r"C:\Users\LongRed\code\TextClassification\dataset\albert_tiny_489k\vocab.txt")
# %%
t= token._tokenize("的沙发沙发沙发all")
# %%

# %%
