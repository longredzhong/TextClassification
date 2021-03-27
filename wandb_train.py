import os

import torch
import torch.nn as nn
import wandb

from TextClassification.trainer import TextTrainer
from TextClassification.utils.tokenizers import Tokenizer

wandb.init(project="Textclassification_tnews")
config = wandb.config
config.datasetname = "tnews"
config.model = "FastText"
config.batchsize = 128
config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config.learning_rate = 0.00005
config.dataset_path = r"/home/longred/code/TextClassification/dataset"
config.embedding_path = r"/home/longred/code/TextClassification/dataset/embedding/bert-base-chinese.pt"
config.seed = 777
config.num_epochs = 500



def main(config):
    tokenizer = Tokenizer(
        r"/home/longred/code/TextClassification/dataset/embedding/vocab.txt")
    if config.datasetname == "tnews":
        from TextClassification.dataloader.tnewsDataloader import (
            get_dataloader, tnewsDataset)
        path = os.path.join(config.dataset_path, "tnews_public")
        train_dataloader, dev_dataloader, num_classes = get_dataloader(
            path, tokenizer, batch_size=config.batchsize)
    if config.datasetname == "tnews":
        from TextClassification.dataloader.iflytekDataloader import (
            get_dataloader)
        path = os.path.join(config.dataset_path, "iflytek")
        train_dataloader, dev_dataloader, num_classes = get_dataloader(
            path, tokenizer, batch_size=config.batchsize)
        
    if config.model == "TextCNN":
        from TextClassification.models.TextCNN import TextCNN
        model = TextCNN(config.embedding_path, num_classes).to(config.device)
    if config.model == "FastText":
        from TextClassification.models.fastext import FastText
        model = FastText(config.embedding_path, num_classes).to(config.device)
    if config.model == "TCN":
        pass
    wandb.watch(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    trainer = TextTrainer(optimizer, config.seed)

    for i in range(config.num_epochs):
        train_loss, train_acc = trainer.train(model, train_dataloader, config.device)
        dev_loss, acc, f1, precision, recall = trainer.dev(model, dev_dataloader, config.device)
        # print(dev_loss, acc, f1, precision, recall)
        wandb.log({
            "train loss":train_loss,
            "train acc":train_acc,
            "dev loss":dev_loss,
            "dev acc":acc,
            "dev precision":precision,
            "dev recall":recall,
            "dev F1":f1,
            "epoch":i+1
        })


main(config)
