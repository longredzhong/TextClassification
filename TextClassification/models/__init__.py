import importlib

def LoadModel(ModelName):
    model = importlib.import_module("TextClassification.models."+ModelName)
    model = getattr(model,ModelName)
    return model
