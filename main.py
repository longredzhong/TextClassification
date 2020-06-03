# %%
from config.BaseConfig import BaseConfig
import os
import datetime
from TextClassification.utils.cp_file2log import cp_file2log
from torch.utils.tensorboard import SummaryWriter
from TextClassification.utils.Run import Run
from TextClassification.utils.Run_CB_loss import Run_CB_Loss
if __name__ == "__main__":
    config = BaseConfig()
    logdir = os.path.join(config.LogPath, config.ModelName,
                          str(datetime.datetime.now()))
    config.LogPath = logdir
    write = SummaryWriter(log_dir=logdir)
    cp_file2log(logPath=logdir, ModelName=config.ModelName)
    if config.UseLoss == "CrossEntropyLoss":
        Run(config, write)
    elif config.UseLoss == "CBLoss":
        Run_CB_Loss(config,write)
# %%
