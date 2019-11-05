import torchtext
import os
import time

class ECommerceDataset(torchtext.data.Dataset):
    def __init__(self, examples, fields, filter_pred=None):
        
        super().__init__(examples, fields, filter_pred=filter_pred)
        