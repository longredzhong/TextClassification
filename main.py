from config import BaseConfig 
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
if __name__=="__main__":
    config = BaseConfig.BaseConfig()
    logdir = os.path.join("run_log",config.ModelName,str(datetime.datetime.now()))
    write = SummaryWriter(log_dir=logdir)
    train(config,write,logger)