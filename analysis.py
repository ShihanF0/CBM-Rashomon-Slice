
import pdb
import os
import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score, precision_score, recall_score, balanced_accuracy_score, classification_report

class Logger(object):
    """
    Log results to a file and flush() to view instant updates
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def binary_accuracy(output, target):
    """
    Computes the accuracy (in percent) for multiple binary predictions.
    Both `output` and `target` can live on any device.
    """
    if target.device != output.device:
        target = target.to(output.device)
    pred = (output >= 0.5).long()
    target = target.long()
    correct = pred.eq(target).sum()
    total   = target.numel()
    return correct * 100.0 / total

def multiclass_metric(output, target):
    """
    Return balanced accuracy score (average of recall for each class) in case of class imbalance,
    and classification report containing precision, recall, F1 score for each class
    """
    balanced_acc = balanced_accuracy_score(target, output)
    report = classification_report(target, output)
    return balanced_acc, report
