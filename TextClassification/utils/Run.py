from TextClassification.models import LoadModel
import numpy as np
import torch
import torch.nn as nn
from TextClassification.dataloader import GetLoader
def Run(config,writer,logger):
    # set constant
    torch.manual_seed(7777)
    torch.cuda.manual_seed_all(7777)
    np.random.seed(7777)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = config()
    # set model
    net = LoadModel(config.ModelName)
    # set Loader
    Loader = GetLoader(config.DatasetName)
    TrainIter, ValIter = Loader(config)
    # set DataParallel
    DeviceIdList = config.DeviceIds
    net = torch.nn.DataParallel(net,device_ids=DeviceIdList)
    net.to(device)
    # set optimzier
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    # set scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # set loss
    criterion = nn.NLLLoss()
    start_iter = 0
    if config.resume is not None:
        start_iter = config.start_iter
    i = start_iter
    while i <= config.TrainIterAll :
        for TrainBatch in TrainIter:
            net.train()
            # load data
            char,word = TrainBatch.char_text.to(device), TrainBatch.word_text.to(device)
            
            if config.UseInput == "word":
                input = word
            elif config.UseInput == "char":
                input = char
            elif config.UseInput == "all":
                input = [char,word]
            
            if config.UseLabel == "last":
                label = TrainBatch.label_last.to(device)
            elif config.UseLabel == "middle":
                label = TrainBatch.label_middle.to(device)
            elif config.UseLabel == "first":
                label = TrainBatch.label_first.to(device)
            
            # forward backward optimizer
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output)
            loss.backward()
            optimizer.step()
            # metrics
            

