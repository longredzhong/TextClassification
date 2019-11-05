#%%
from torchtext.data import Field,BucketIterator
from TextClassification.dataloader.ECommerceDataloader import ECommerceDataset
path = "/home/longred/TextClassification/dataset/preProcess/E_commerce_data/val_data.tsv"
char_text = Field(sequential=True, lower=True, fix_length=None)
word_text = Field(sequential=True,lower=True,fix_length=None)
label_last = Field(sequential=False,use_vocab=False)
label_middle = Field(sequential=False,use_vocab=False)
label_first = Field(sequential=False,use_vocab=False)
fields = [
            ("char_text",char_text),
            ("word_text",word_text),
            ("label_last",label_last),
            ("label_middle",label_middle),
            ("label_first",label_middle)
        ]

# %%
loader = ECommerceDataset(path=path,fields=fields)
char_text.build_vocab(loader)
word_text.build_vocab(loader)
# %%
val_iter = BucketIterator(
            train=False,
            dataset=loader,
            batch_size=1,
            shuffle=False,
            sort=False,
            repeat=False)

# %%
loader.examples[0].char_text

# %%
for i in val_iter:
    print(i)

# %%
next(iter(val_iter))
