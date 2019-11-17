# %%
from torch.utils.tensorboard import SummaryWriter
from config.BaseConfig import BaseConfig
from TextClassification.utils.cp_file2log import cp_file2log
import datetime
import os
# %%

config = BaseConfig()
logdir = os.path.join("run_log", config.ModelName,
                      str(datetime.datetime.now()))
config.LogPath = logdir
# %%
cp_file2log(config)


# %%
write = SummaryWriter(log_dir=logdir)


# %%
