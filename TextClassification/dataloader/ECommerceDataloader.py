import io
import os
import time

from torchtext.data import BucketIterator, Dataset, Example, Field
from torchtext.utils import unicode_csv_reader
from torchtext.vocab import Vectors
import torch
from config.BaseConfig import BaseConfig


class ECommerceDataset(Dataset):
    """
    
    """
    def __init__(self, path,fields):
        examples = []
        with open(path, 'r') as f:
            csv_data = f.readlines()
        csv_data = csv_data[1:]
        for i in range(len(csv_data)):
            csv_data[i] = csv_data[i].strip().split('\t')
        print('read data from {}'.format(path))
        for i in csv_data:
            label = i[2]
            middle_cls = i[3]
            first_cls = i[4]
            char_text = i[5]
            word_text = i[6]
            examples.append(Example.fromlist([
                            char_text, word_text, label, middle_cls, first_cls
                        ], fields))
        super().__init__(examples, fields)


def ECommerceLoader(config):
    """
    input config
    output trainiter and valiter
    """
    char_text = Field(sequential=True, lower=True, fix_length=None)
    word_text = Field(sequential=True,lower=True,fix_length=None)
    label_last = Field(sequential=False,use_vocab=False)
    label_middle = Field(sequential=False,use_vocab=False)
    label_first = Field(sequential=False,use_vocab=False)
    fields = [
                ("char_text",char_text),
                ("word_text",word_text),
                ("label_last",label_last),
                ("label_middle",label_middle),
                ("label_first",label_middle)
               ]
    TrainDataset = ECommerceDataset(config.TrainPath,fields)
    ValDataset = ECommerceDataset(config.ValPath,fields)
    # load vector
    char_vectors = Vectors(config.CharVectorsPath)
    word_vectors = Vectors(config.WordVectorsPath)
    # build vocab
    char_text.build_vocab(TrainDataset, ValDataset, vectors=char_vectors)
    word_text.build_vocab(TrainDataset, ValDataset, vectors=word_vectors)
    # add config
    config.CharVocabSize = len(char_text.vocab)
    config.WordVocabSize = len(word_text.vocab)
    config.CharVectors = torch.tensor(char_text.vocab.vectors,requires_grad=True)
    config.WordVectors = torch.tensor(word_text.vocab.vectors,requires_grad=True)
    # iter 
    train_iter = BucketIterator(
            train=True,
            dataset=TrainDataset,
            batch_size=config.TrainBatchSize,
            shuffle=True,
            sort_within_batch=False,
            repeat=False)
    val_iter = BucketIterator(
            train=False,
            dataset=ValDataset,
            batch_size=config.ValBatchSize,
            shuffle=False,
            sort=False,
            repeat=False)
    
    return train_iter,val_iter
