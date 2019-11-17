# %%
from config.BaseConfig import BaseConfig
import os
import datetime
from TextClassification.utils.cp_file2log import cp_file2log
from torch.utils.tensorboard import SummaryWriter
from TextClassification.utils.Run import Run
if __name__ == "__main__":
    config = BaseConfig()
    logdir = os.path.join("run_log", config.ModelName,
                          str(datetime.datetime.now()))
    config.LogPath = logdir
    write = SummaryWriter(log_dir=logdir)
    cp_file2log(logPath=logdir, ModelName=config.ModelName)
    Run(config, write)


# %%
