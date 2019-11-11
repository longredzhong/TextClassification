from sklearn.metrics import *
import os
class metrics(object):
    def __init__(self,logPath=None):
        self.pred = []
        self.label = []
        self.loss = []
        # TODO
        # self.logPath = os.path.join(logPath ,'metrics.log')
    def update(self,pred=[],label=[],loss=[]):
        self.pred += pred
        self.label += label
        self.loss += loss
    def GetAvgLoss(self):
        return sum(self.loss)/len(self.loss)
    def GetAvgAccuracy(self):
        return accuracy_score(self.label,self.pred)
    def GetAvgPrecision(self):
        return precision_score(self.label,self.pred,average="macro")
    def GetAvgRecall(self):
        return recall_score(self.label,self.pred,average="macro")
    def GetAvgF1(self):
        return f1_score(self.label,self.pred,average="macro")
    def reset(self):
        self.pred = []
        self.label = []
        self.loss = []
    def save(self):
        if self.logPath is not None:
            with open(self.logPath) as f:
                f.writelines(self.pred)
                f.writelines(self.label)
                f.writelines(self.GetAvgAccuracy())
