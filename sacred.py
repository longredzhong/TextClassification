import os

import torch
import torch.nn as nn

import sacred
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from TextClassification.utils.tokenizers import Tokenizer

ex = Experiment("TextCNN")
ex.observers.append(MongoObserver.create(url='172.24.245.146:27017', db_name='sacred')
# ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def getconfig():
    batchsize=128
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs=500
    datasetname="tnews"
    learning_rate=0.001
    dataset_path=r"/ home/longred/code/TextClassification/dataset"
    embedding_path=r"/home/longred/code/TextClassification/dataset/embedding/bert-base-chinese.pt"
    seed=777

@ex.automain
def main(batchsize, device, num_epochs, datasetname, learning_rate, dataset_path, seed):
    tokenizer=Tokenizer(
        r"/home/longred/code/TextClassification/dataset/embedding/vocab.txt")
    if datasetname == "tnews":
        from TextClassification.dataloader.tnewsDataloader import (
            get_dataloader, tnewsDataset)
        path=os.path.join(dataset_path, "tnews_public")
        train_dataloader, dev_dataloader, num_classes=get_dataloader(
            path, tokenizer, batch_size=batchsize)
    model=TextCNN(embedding_path, num_classes).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer=TextTrainer(optimizer, seed)
    for i in range(num_epochs):
        train_loss, train_acc=trainer.train(model, train_dataloader, device)
        ex.log_scalar("train_loss", train_loss, i+1)
        ex.log_scalar("train_acc", train_acc, i+1)
        dev_loss, acc, f1, precision, recall=trainer.dev(
            model, dev_dataloader, device)
        ex.log_scalar("dev_loss", dev_loss, i+1)
        ex.log_scalar("dev_acc", acc, i+1)
        ex.log_scalar("dev_precision", precision, i+1)
        ex.log_scalar("dev_recall", recall, i+1)
        ex.log_scalar("dev_F1", f1, i+1)
