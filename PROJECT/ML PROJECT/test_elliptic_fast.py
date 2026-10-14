import csv, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

p = r'C:\Users\Madhav\Downloads\elliptic_bitcoin_dataset'
t0 = time.time()

# 1. Load classes
classes = {}
with open(os.path.join(p, 'elliptic_txs_classes.csv'), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # '1' = illicit, '2' = licit, 'unknown' = unknown
        if row[1] == '1':
            classes[row[0]] = 1
        elif row[1] == '2':
            classes[row[0]] = 0

print(f"Loaded {len(classes)} labeled nodes in {time.time()-t0:.2f}s")
