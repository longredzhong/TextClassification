import os

import torch
import torch.nn as nn
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from TextClassification.models.TextCNN import TextCNN
from TextClassification.trainer import TextTrainer
from TextClassification.utils.tokenizers import Tokenizer

ex = Experiment("TextCNN_sss")
ex.observers.append(MongoObserver(
    url='mongodb://172.24.245.146:27017/', db_name='sacred'))
ex.observers.append(FileStorageObserver('my_runs'))
# ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def myconfig():
    batchsize = 128
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_epochs = 500
    datasetname = "tnews"
    learning_rate = 0.001
    dataset_path = r"/home/longred/code/TextClassification/dataset"
    embedding_path = r"/home/longred/code/TextClassification/dataset/embedding/bert-base-chinese.pt"
    seed = 777


@ex.automain
def main(_run, batchsize, device, num_epochs, datasetname, learning_rate, dataset_path, seed, embedding_path):
    tokenizer = Tokenizer(
        r"/home/longred/code/TextClassification/dataset/embedding/vocab.txt")
    if datasetname == "tnews":
        from TextClassification.dataloader.tnewsDataloader import (
            get_dataloader, tnewsDataset)
        path = os.path.join(dataset_path, "tnews_public")
        train_dataloader, dev_dataloader, num_classes = get_dataloader(
            path, tokenizer, batch_size=batchsize)
    model = TextCNN(embedding_path, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = TextTrainer(optimizer, seed)

    for i in range(num_epochs):
        train_loss, train_acc = trainer.train(model, train_dataloader, device)
        _run.log_scalar(metric_name="train.loss", value=train_loss, step=i+1)
        _run.log_scalar(metric_name="train.acc", value=train_acc, step=i+1)
        dev_loss, acc, f1, precision, recall = trainer.dev(
            model, dev_dataloader, device)
        print(dev_loss, acc, f1, precision, recall)
        _run.log_scalar(metric_name="dev.loss", value=dev_loss, step=i+1)
        _run.log_scalar(metric_name="dev.acc", value=acc, step=i+1)
        _run.log_scalar(metric_name="dev.precision", value=precision, step=i+1)
        _run.log_scalar(metric_name="dev.recall", value=recall, step=i+1)
        _run.log_scalar(metric_name="dev.F1", value=f1, step=i+1)
    return acc
