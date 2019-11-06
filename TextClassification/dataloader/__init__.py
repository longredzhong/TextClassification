import importlib

def GetLoader(DatasetName):
    model = importlib.import_module("TextClassification.dataloader."+DatasetName+"Dataloader")
    model = getattr(model,DatasetName+"Loader")
    return model