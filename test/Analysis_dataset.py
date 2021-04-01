# %%
import os
import sys
sys.path.append(
    (os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))))
#%%
import json
label = []
sentence = []
def load_data(raw_path):
    with open(raw_path,mode='r',encoding="UTF8") as f:
        texts = f.readlines()
    target = []
    for tmp in texts:
        tmp = json.loads(tmp)
        sentence.append(tmp['sentence'])
        target.append(tmp['label'])
    target2id = {label: indx for indx, label in enumerate(set(target))}
    label = [target2id[label] for label in target]
    return label
label = load_data("/home/longred/code/TextClassification/dataset/iflytek/train.json")
#%%
import wandb
wandb.init("dataset")
# %%
dataset_len = [0]*101
#%%
dataset_len
#%%
for i in sentence:
    if len(i)<1000:
        dataset_len[int(len(i)/10)]+=1
    else:
        dataset_len[100]+=1
# %%
for i in range(len(dataset_len)):
    wandb.log({
        "num__":dataset_len[i],
        "len__":i*10
    })
# %%
num_label = [0]*119
for i in label:
    num_label[i]+=1
# %%
num_label = sorted(num_label)
# %%
for i in range(len(num_label)):
    wandb.log({
        "label_num_":num_label[i],
        "label_":i
    })
# %%
num_label
# %%
