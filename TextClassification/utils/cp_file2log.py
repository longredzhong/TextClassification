from shutil import copyfile
import os

import datetime


def cp_file2log(logPath, ModelName):
    logPath = logPath
    # cp model
    ModelPathSrc = os.path.join(
        'TextClassification/models', ModelName+'.py')
    ModelPathDst = os.path.join(logPath, ModelName+'.py')
    copyfile(ModelPathSrc, ModelPathDst)
    # cp config.py
    configPathSrc = 'config/BaseConfig.py'
    configPathDst = os.path.join(logPath, 'BaseConfig.py')
    copyfile(configPathSrc, configPathDst)
