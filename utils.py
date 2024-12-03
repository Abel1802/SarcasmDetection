import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def cal_f1_score(logits, labels):
    '''
    Calculate the F1 score for the model.'''
    # to cup & numpy
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = np.argmax(logits, axis=1)
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return f1, precision, recall, accuracy
    
