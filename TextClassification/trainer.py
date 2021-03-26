import torch
import numpy as np
from TextClassification.utils.util import accuracy,macro_f1
import fitlog
fitlog.set_log_dir("logs/") 
class TextTrainer:
    def __init__(self, optimizer, seed=777, max_epoch=1000):
        self.max_epoch = max_epoch
        self.seed = seed
        self.set_seed()
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def train(self, model, dataloader):
        model.train()
        self.optimizer.zero_grad()
        preds_list = []
        labels_list = []
        loss_sum = 0
        for data in dataloader:
            logits = model.forward(data[0])
            pred = torch.max(logits, 1)[1]
            loss = self.criterion(logits, data[1])
            loss_sum += loss
            loss.backward()
            self.optimizer.step()
            preds_list += pred.tolist()
            labels_list += data[1].tolist()
        
        acc = accuracy(preds_list, labels_list)
        loss = loss_sum/len(dataloader)
        return loss,acc

    def dev(self, model, dataloader):
        model.eval()
        with torch.no_grad():
            loss_sum = 0
            preds_list = []
            labels_list = []
            for data in dataloader:
                logits = model.forward(data[0])
                pred = torch.max(logits, 1)[1]
                loss = self.criterion(logits, data[1])
                loss_sum += loss
                loss.backward()
                self.optimizer.step()
                preds_list += pred.tolist()
                labels_list += data[1].tolist()
            acc = accuracy(preds_list, labels_list)
            f1, precision, recall = macro_f1(preds_list, labels_list)
            loss = loss_sum/len(dataloader)
        return loss, acc, f1, precision, recall


    def save(self):
        pass

    def fit(self, model, train_dataloader,dev_dataloader):
        for i in range(self.max_epoch):
            train_loss,train_acc = self.train(model,train_dataloader)
            fitlog.add_loss(train_loss,name="Train Loss",step=i+1)
            fitlog.add_metric({"train":{"Acc":train_acc}}, step=i+1)
            dev_loss, acc, f1, precision, recall = self.dev(model,dev_dataloader)
            fitlog.add_loss(dev_loss,name="Dev Loss",step=i+1)
            fitlog.add_metric({"Dev":{"Acc":acc,"precision":precision,"recall":recall,"F1":f1}}, step=i+1)
            print(train_acc,acc)