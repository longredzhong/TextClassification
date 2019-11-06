from TextClassification.models import LoadModel
import numpy as np
import torch
import torch.nn as nn
from TextClassification.dataloader import GetLoader
from tqdm import tqdm
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
    # set DataParallel
    # DeviceIdList = config.DeviceIds
    # net = torch.nn.DataParallel(net,device_ids=DeviceIdList)
    net = net.to(device)
    # set optimzier
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # set loss
    criterion = nn.CrossEntropyLoss()
    start_iter = 0
    if config.resume is not None:
        start_iter = config.start_iter
    i = start_iter
    while i <= config.TrainIterAll :
        for TrainBatch in TrainIter:
            net.train()
            # load data
            input, label = get_input_label(TrainBatch,config,device)
            # forward backward optimizer
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            # TODO metrics
            # TODO tensorboard
            print(loss.item())

            if (i+1) % config.ValInter == 0 :
                net.eval()
                with torch.no_grad():
                    for ValBatch in tqdm(ValIter):
                        input, label = get_input_label(ValBatch,config,device)
                        output = net(input)
                        loss = criterion(output)
                        # TODO metrics
                        # TODO tensorboard


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