import os

import torch
import torch.nn as nn
from torch.utils.data import dataloader
import wandb

from TextClassification.trainer import TextTrainer
from TextClassification.utils.tokenizers import Tokenizer

wandb.init(project="Textclassification")
config = wandb.config
config.datasetname = "tnews"
config.model = "TextRNNAttention"
config.batchsize = 32
config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
config.learning_rate = 0.0000005
config.dataset_path = r"/home/longred/code/TextClassification/dataset"
config.embedding_path = r"/home/longred/code/TextClassification/dataset/embedding/bert-base-chinese.pt"
config.seed = 777
config.num_epochs = 500
config.num_channels = [256, 256]  # TCN
config.num_filters = 256  # DPCNN


def main(config):
    tokenizer = Tokenizer(
        r"/home/longred/code/TextClassification/dataset/embedding/vocab.txt")
    if config.datasetname == "tnews":
        from TextClassification.dataloader.tnewsDataloader import (
            get_dataloader, tnewsDataset)
        path = os.path.join(config.dataset_path, "tnews_public")
        train_dataloader, dev_dataloader, num_classes = get_dataloader(
            path, tokenizer, batch_size=config.batchsize)
    if config.datasetname == "iflytek":
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
        from TextClassification.models.TCN_text_classification import TCN
        model = TCN(config.embedding_path, num_classes,
                    config.num_channels).to(config.device)
    if config.model == "DPCNN":
        from TextClassification.models.DPCNN import DPCNN
        model = DPCNN(config.embedding_path, config.num_filters,
                      num_classes).to(config.device)
    if config.model == "TextRNN":
        from TextClassification.models.TextRNN import TextRNN
        model = TextRNN(config.embedding_path, config.num_filters,
                      num_classes).to(config.device)
    if config.model == "TextRCNN":
        from TextClassification.models.TextRCNN import TextRCNN
        model = TextRCNN(config.embedding_path, config.num_filters,
                      num_classes).to(config.device)
    if config.model == "TextRNNAttention":
        from TextClassification.models.TextRNNAttention import TextRNNAttention
        model = TextRNNAttention(config.embedding_path, config.num_filters,
                      num_classes).to(config.device)
    wandb.watch(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    trainer = TextTrainer(optimizer, config.seed)
    print(num_classes)
    for i in range(config.num_epochs):
        train_loss, train_acc = trainer.train(
            model, train_dataloader, config.device)
        dev_loss, acc, f1, precision, recall = trainer.dev(
            model, dev_dataloader, config.device)
        # print(dev_loss, acc, f1, precision, recall)
        wandb.log({
            "train loss": train_loss,
            "train acc": train_acc,
            "dev loss": dev_loss,
            "dev acc": acc,
            "dev precision": precision,
            "dev recall": recall,
            "dev F1": f1,
            "epoch": i+1
        })


main(config)
