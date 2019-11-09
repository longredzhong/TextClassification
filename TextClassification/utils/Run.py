from TextClassification.models import LoadModel
from TextClassification.utils.Metrics import metrics
import numpy as np
import torch
import torch.nn as nn
from TextClassification.dataloader import GetLoader
from tqdm import tqdm
import os
def Run(config,writer):
    # set constant
    torch.manual_seed(7777)
    torch.cuda.manual_seed_all(7777)
    np.random.seed(7777)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # set Loader
    Loader = GetLoader(config.DatasetName)
    TrainIter, ValIter = Loader(config)
    # set model
    net = LoadModel(config.ModelName)
    net = net(config)
    net = net.to(device)
    # set optimzier
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',factor=0.5,patience=4,verbose=True)
    # set loss
    criterion = nn.CrossEntropyLoss()
    start_iter = 0
    epoch = 1
    best_acc = -100
    if config.resume is not None:
        start_iter = config.start_iter
        epoch = config.epoch_end
    i = start_iter
    train_epoch_metrics = metrics()
    train_batch_metrics = metrics()
    val_metrics = metrics()
    while i <= config.TrainIterAll :
        train_epoch_metrics.reset()
        for epoch_iter,TrainBatch in enumerate(TrainIter,epoch):
            train_batch_metrics.reset()
            net.train()
            i += 1
            # load data
            input, label = get_input_label(TrainBatch,config,device)
            # forward backward optimizer
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            pred = torch.max(output,1)[1].cpu().numpy().tolist()
            label = label.cpu().numpy().tolist()
            loss = [loss.tolist()]
            train_epoch_metrics.update(pred,label,loss)
            train_batch_metrics.update(pred,label,loss)
            writer.add_scalar('Loss/train',train_batch_metrics.GetAvgLoss(),i)
            writer.add_scalar('Acc/train',train_batch_metrics.GetAvgAccuracy(),i)
            if (i+1) % config.ValInter == 0 :
                net.eval()
                val_metrics.reset()
                with torch.no_grad():
                    for ValBatch in tqdm(ValIter):
                        input, label = get_input_label(ValBatch,config,device)
                        output = net(input)
                        loss = criterion(output,label)
                        pred = torch.max(output,1)[1].cpu().numpy().tolist()
                        label = label.cpu().numpy().tolist()
                        loss = [loss.tolist()]
                        val_metrics.update(pred,label,loss)
                # val tensorboard
                writer.add_scalar('Loss/val',val_metrics.GetAvgLoss(),epoch_iter)
                writer.add_scalar('Acc/val',val_metrics.GetAvgAccuracy(),epoch_iter)
                writer.add_scalar('F1/val',val_metrics.GetAvgF1(),epoch_iter)
                writer.add_scalar('Recall/val',val_metrics.GetAvgRecall(),epoch_iter)
                writer.add_scalar('Precision/val',val_metrics.GetAvgPrecision(),epoch_iter)
                scheduler.step(val_metrics.GetAvgAccuracy())
                if val_metrics.GetAvgAccuracy() > best_acc :
                    best_acc = val_metrics.GetAvgAccuracy()
                    torch.save(net,os.path.join(config.LogPath,config.ModelName))
                    config.resume = os.path.join(config.LogPath,config.ModelName)
                    config.start_iter = i
                    config.epoch_end = epoch_iter
            # epoch tensorboard
            writer.add_scalar('Loss/val',train_epoch_metrics.GetAvgLoss(),epoch_iter)
            writer.add_scalar('Acc/val',train_epoch_metrics.GetAvgAccuracy(),epoch_iter)
            writer.add_scalar('F1/val',train_epoch_metrics.GetAvgF1(),epoch_iter)
            writer.add_scalar('Recall/val',train_epoch_metrics.GetAvgRecall(),epoch_iter)
            writer.add_scalar('Precision/val',train_epoch_metrics.GetAvgPrecision(),epoch_iter)


def get_input_label(Batch,config,device):
    char,word = Batch.char_text.to(device), Batch.word_text.to(device)
    if config.UseInput == "word":
        input = word
    elif config.UseInput == "char":
        input = char
    elif config.UseInput == "all":
        input = [char,word]
    
    if config.UseLabel == "last":
        label = Batch.label_last.to(device)
    elif config.UseLabel == "middle":
        label = Batch.label_middle.to(device)
    elif config.UseLabel == "first":
        label = Batch.label_first.to(device)
    return input, label