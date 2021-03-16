
import json
import logging
import re
import sys
from collections import defaultdict

import numpy as np
import six
import torch

_open_ = open
is_py2 = six.PY2

if not is_py2:
    basestring = str


def sequence_padding(inputs, length=None, padding=0):
    """，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return torch.tensor(outputs, dtype=torch.long)


def toTensor(inputs):
    return torch.tensor(inputs)


def seq_len_to_mask(seq_len, max_len=None, batch_first=True):
    r"""

    将一个表示sequence length的一维数组转换为二维的mask，不包含的位置为0。
    转变 1-d seq_len到2-d mask.

    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param np.ndarray,torch.LongTensor seq_len: shape将是(B,)
    :param int max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有
        区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    :return: np.ndarray, torch.Tensor 。shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(
            seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        max_len = int(max_len) if max_len else int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
        if not batch_first:
            mask = mask.swapaxes(0, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim(
        ) == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(
            max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1)).byte()
        if not batch_first:
            mask = mask.permute(1, 0)
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

def truncate_sequences(maxlen, index, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences



def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)


def strQ2B(ustring):
    """全角符号转对应的半角符号
    """
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        # 全角空格直接转换
        if inside_code == 12288:
            inside_code = 32
        # 全角字符（除空格）根据关系转化
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


def string_matching(s, keywords):
    """判断s是否至少包含keywords中的至少一个字符串
    """
    for k in keywords:
        if re.search(k, s):
            return True
    return False


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, str):
            text = text.decode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def convert_to_str(text, encoding='utf-8', errors='ignore'):
    """字符串转换为str格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, unicode):
            text = text.encode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


class open:
    """模仿python自带的open函数，主要是为了同时兼容py2和py3
    """

    def __init__(self, name, mode='r', encoding=None, errors='ignore'):
        if is_py2:
            self.file = _open_(name, mode)
        else:
            self.file = _open_(name, mode, encoding=encoding, errors=errors)
        self.encoding = encoding
        self.errors = errors
        self.iterator = None

    def __iter__(self):
        for l in self.file:
            if self.encoding:
                l = convert_to_unicode(l, self.encoding, self.errors)
            yield l

    def next(self):
        if self.iterator is None:
            self.iterator = self.__iter__()
        return next(self.iterator)

    def __next__(self):
        return self.next()

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def readline(self):
        text = self.file.readline()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def readlines(self):
        if self.encoding:
            return [
                convert_to_unicode(text, self.encoding, self.errors)
                for text in self.file.readlines()
            ]
        else:
            return self.file.readlines()

    def write(self, text):
        if self.encoding:
            text = convert_to_str(text, self.encoding, self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def tell(self):
        return self.file.tell()

    def seek(self, whence=0):
        return self.file.seek(whence)

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
